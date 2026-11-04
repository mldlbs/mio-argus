"""
W29 采集：在**已验证的 W25/W26/W27 链**上挂录制器（按链轮转，保证多样性）

背景：`collect_inloop.py` 自己重写了 UI 导航，从未端到端跑通；`run_w25` 单独重复跑
虽然稳定，但每轮动作序列逐条相同（98 步 × N 份），加的是数据量不是覆盖度。
本脚本把三条已验证的链排成队列轮转：w25 单主题、w26 四个主题变体、w27 深色重绘，
每条链一个 task_id，示范自动落盘，并在汇总里报告**唯一步骤数**（不是样本数）。

用法:
    python experiments/computer_use/record_chain.py --rounds 1
    python experiments/computer_use/record_chain.py --rounds 4 --chains w25,w26,w27
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BR = ROOT / "experiments" / "qgis" / "qgis_bridge"
sys.path[:0] = [str(HERE), str(BR), str(ROOT / "mio"), str(ROOT)]

import collect_inloop as CI  # noqa: E402  (只用 ensure_qgis)
import run_w25  # noqa: E402
import run_w26  # noqa: E402
import run_w27  # noqa: E402
from client import call  # noqa: E402
from inloop_rec import Recorder  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "artifacts" / "computer_use_inloop"


def qgis_observation():
    """QGIS 侧观察：bridge truth / structure + 可见窗口。

    候选 = actions（QAction，带 text+rect，图标按钮的语义来源）
         + widgets（控件，带 class/path/rect/text）
    过滤：enabled、有 rect、有 text；不按面积裁剪（父容器可能是可操作目标），
    保留 path/class 作为层级信息。同时记录旧口径（前 60 条 text 控件）用于同帧对照。
    """
    truth, struct, windows = {}, [], []
    legacy, stats = [], {}
    try:
        truth = (call({"op": "truth"}, timeout=5) or {}).get("truth") or {}
    except Exception:
        pass
    try:
        st = call({"op": "structure"}, timeout=5) or {}
        widgets = st.get("widgets") or []
        actions = st.get("actions") or []
        legacy = [{"text": w.get("text", ""), "rect": w.get("rect"),
                   "cls": w.get("class", ""), "path": w.get("path", ""),
                   "enabled": w.get("enabled", True)}
                  for w in widgets if w.get("text") and w.get("rect")][:60]

        seen, cand = set(), []
        for a in actions:
            t = (a.get("text") or "").strip()
            r = a.get("rect")
            if not t or not r or not a.get("enabled", True):
                continue
            key = (t, tuple(r))
            if key in seen:
                continue
            seen.add(key)
            cand.append({"text": t, "rect": r, "cls": "QAction", "path": "",
                         "enabled": True, "kind": "action",
                         "shortcut": a.get("shortcut") or ""})
        for w in widgets:
            t = (w.get("text") or "").strip()
            r = w.get("rect")
            if not t or not r or not w.get("enabled", True):
                continue
            key = (t, tuple(r))
            if key in seen:
                continue
            seen.add(key)
            cand.append({"text": t, "rect": r, "cls": w.get("class", ""),
                         "path": w.get("path", ""), "enabled": True,
                         "kind": "widget", "objectName": w.get("objectName", "")})
        struct = cand
        stats = {"widgets": len(widgets), "actions": len(actions),
                 "kept": len(cand), "legacy": len(legacy),
                 "kinds": {"action": sum(1 for c in cand if c["kind"] == "action"),
                           "widget": sum(1 for c in cand if c["kind"] == "widget")}}
    except Exception:
        pass
    try:
        from pywinauto import Desktop
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                windows.append({"class": w.element_info.class_name or "",
                                "title": (w.window_text() or "")[:80],
                                "rect": list(w.rectangle())})
            except Exception:
                continue
    except Exception:
        pass

    # UIA 补充：bridge 给不到的菜单栏项 / 树项 / 标题栏按钮
    try:
        from pywinauto import Desktop
        qg = None
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                if (w.element_info.class_name or "") == "QgisApp":
                    qg = w
                    break
            except Exception:
                continue
        if qg is not None:
            seen2 = {(c["text"], tuple(c["rect"])) for c in struct}
            uia_add = 0
            for c in qg.descendants():
                try:
                    ct = c.element_info.control_type
                    if ct not in ("MenuItem", "TreeItem", "Button", "TabItem", "Edit", "ListItem"):
                        continue
                    t = (c.window_text() or "").strip()
                    r = c.rectangle()
                    if not t or r.width() < 4 or r.height() < 4:
                        continue
                    box = [int(r.left), int(r.top), int(r.width()), int(r.height())]
                    key = (t, tuple(box))
                    if key in seen2:
                        continue
                    seen2.add(key)
                    struct.append({"text": t, "rect": box, "cls": ct, "path": "",
                                   "enabled": bool(c.is_enabled()), "kind": "uia",
                                   "control_type": ct})
                    uia_add += 1
                except Exception:
                    continue
            stats["uia_added"] = uia_add
            stats["kept"] = len(struct)
    except Exception:
        pass

    return {"truth": truth, "affordances": struct, "affordances_legacy": legacy,
            "cand_stats": stats, "windows": windows[:20]}


def build_queue(chains):
    """把链名展开成 (task 名, 执行函数, reset 函数) 队列。"""
    q = []
    for c in chains:
        if c == "w25":
            q.append(("w25", lambda out_p, steps: run_w25.run_chain(out_p, steps),
                      run_w25.reset_env))
        elif c == "w26":
            for tid in run_w26.THEMES:
                q.append((f"w26_{tid}",
                          (lambda t: lambda out_p, steps: run_w26.run_chain(t, out_p, steps))(tid),
                          run_w26.reset_env))
        elif c.startswith("w26_"):
            tid = c[4:]
            if tid not in run_w26.THEMES:
                raise SystemExit(f"未知主题: {c}")
            q.append((c,
                      (lambda t: lambda out_p, steps: run_w26.run_chain(t, out_p, steps))(tid),
                      run_w26.reset_env))
        elif c == "w27":
            q.append(("w27", lambda out_p, steps: run_w27.run_chain(out_p, steps),
                      run_w27.reset_env))
        else:
            raise SystemExit(f"未知链: {c}")
    return q


def next_round(out_root, name):
    """已有 {name}_rNN 目录时从下一个编号继续，避免重复调用覆盖上一轮数据。"""
    import re
    mx = -1
    d = Path(out_root) / "qgis"
    if d.exists():
        for p in d.glob(f"{name}_r*"):
            m = re.match(rf"{re.escape(name)}_r(\d+)$", p.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def signature(demo):
    return (demo.get("action"), demo.get("instruction"),
            tuple(demo.get("target_rect") or ()), tuple(demo.get("norm_coord") or ()))


def write_splits(out_root, val_every=3):
    """按**链**划分 train/val（同链的所有轮次只落在一侧，避免同动作泄漏）。"""
    d = Path(out_root) / "qgis"
    tasks = sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "demo.json").exists())
    if not tasks:
        return None
    chains = []
    for t in tasks:
        base = "w25" if t.startswith("w25_chain_t") else t.rsplit("_r", 1)[0]
        if base not in chains:
            chains.append(base)
    val_chains = [c for i, c in enumerate(chains) if i % val_every == val_every - 1]
    val = [t for t in tasks
           if ("w25" if t.startswith("w25_chain_t") else t.rsplit("_r", 1)[0]) in val_chains]
    train = [t for t in tasks if t not in val]
    sigs, per_task = set(), {}
    for t in tasks:
        part = json.loads((d / t / "demo.json").read_text(encoding="utf-8"))
        per_task[t] = len(part)
        sigs.update(signature(x) for x in part)
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": f"按链划分：第 {val_every} 的倍数条链整体作为 val，同链各轮不跨侧",
        "val_chains": val_chains,
        "train": train, "val": val,
        "n_train_tasks": len(train), "n_val_tasks": len(val),
        "demos_train": sum(per_task[t] for t in train),
        "demos_val": sum(per_task[t] for t in val),
        "demos_total": sum(per_task.values()),
        "unique_steps": len(sigs),
        "per_task": per_task,
    }
    p = Path(out_root) / "qgis_splits.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--chains", default="w25,w26,w27")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shot-scale", type=float, default=0.6)
    ap.add_argument("--write-splits", action="store_true",
                    help="只扫描已有任务目录并写 qgis_splits.json（不采集）")
    args = ap.parse_args()

    if args.write_splits:
        res = write_splits(Path(args.out))
        if res is None:
            print("[splits] 没有找到任务目录")
            return 1
        p, payload = res
        print(f"[out] {p}  train={payload['n_train_tasks']}任务/{payload['demos_train']}条 "
              f"val={payload['n_val_tasks']}任务/{payload['demos_val']}条 "
              f"unique={payload['unique_steps']}")
        return 0

    chains = [c.strip() for c in args.chains.split(",") if c.strip()]
    out_root = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    CI.ensure_qgis()
    try:
        call({"op": "win", "maximize": True})
    except Exception:
        pass

    rec = Recorder("qgis", out_root, obs_provider=qgis_observation,
                   shot_scale=args.shot_scale)
    queue = build_queue(chains)
    print(f"[plan] {len(chains)} 链 × {args.rounds} 轮 = {len(queue) * args.rounds} 次执行")

    rows = []
    counters = {name: next_round(out_root, name) for name, _, _ in queue}
    print(f"[plan] 起始轮次 " + ", ".join(f"{k}=r{v:02d}" for k, v in counters.items()))
    rec.install()
    try:
        for r in range(args.rounds):
            for name, fn, reset in queue:
                task_id = f"{name}_r{counters[name]:02d}"
                counters[name] += 1
                out_p = OUT / "exports" / f"{task_id}.png"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                reset()
                rec.new_task(task_id, task_name=name, chain=name.split("_")[0])
                before = len(rec.demos)
                steps = []
                t0 = time.time()
                ok, fail = False, ""
                try:
                    ok, fail = fn(out_p, steps)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    ok, fail = False, f"exception: {e}"
                total = round(time.time() - t0, 1)
                rec.end_task(bool(ok), verify={"fail": fail, "steps": len(steps)})
                demos = len(rec.demos) - before
                rows.append({"round": r, "chain": name, "task_id": task_id,
                             "ok": bool(ok), "fail": fail, "total_s": total,
                             "demos": demos})
                print(f"  r{r} {name}: {'PASS' if ok else 'FAIL(' + fail + ')'} "
                      f"{total}s demos={demos}", flush=True)
    finally:
        rec.uninstall()

    sigs, all_demos = set(), []
    for r in rows:
        p = out_root / "qgis" / r["task_id"] / "demo.json"
        if p.exists():
            part = json.loads(p.read_text(encoding="utf-8"))
            all_demos.extend(part)
            sigs.update(signature(d) for d in part)
    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "run_w25/w26/w27 under inloop_rec.Recorder",
        "chains": chains, "rounds": args.rounds,
        "trials": rows,
        "n_demos": len(all_demos),
        "unique_steps": len(sigs),
        "pass": sum(1 for x in rows if x["ok"]),
        "n_runs": len(rows),
    }
    p = OUT / "record_chain_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {p}  demos={summary['n_demos']} unique={summary['unique_steps']} "
          f"pass={summary['pass']}/{summary['n_runs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
