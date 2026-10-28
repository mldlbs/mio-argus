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
    """QGIS 侧观察：bridge truth / structure + 可见窗口，用于遮挡与焦点归因。"""
    truth, struct, windows = {}, [], []
    try:
        truth = (call({"op": "truth"}, timeout=5) or {}).get("truth") or {}
    except Exception:
        pass
    try:
        st = call({"op": "structure"}, timeout=5) or {}
        widgets = st.get("widgets") or []
        struct = [{"text": w.get("text", ""), "rect": w.get("rect"),
                   "cls": w.get("class", ""), "path": w.get("path", ""),
                   "enabled": w.get("enabled", True)}
                  for w in widgets if w.get("text") and w.get("rect")]
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
    return {"truth": truth, "affordances": struct[:60], "windows": windows[:20]}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--chains", default="w25,w26,w27")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shot-scale", type=float, default=0.6)
    args = ap.parse_args()

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
