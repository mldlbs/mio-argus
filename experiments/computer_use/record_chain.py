"""
W29 采集（替代方案）：在**已验证的 W25 链**上挂录制器，而不是重写 UI 导航

背景：`collect_inloop.py` 自己实现了一遍 W23–W27 的菜单/对话框导航，端到端从未跑通，
失败点每轮漂移。这里改用 `run_w25.run_chain`（已有 w25 结果、链路稳定），
用 `inloop_rec.Recorder` 在 pyautogui / pywinauto 两层打补丁，链跑一遍示范自动落盘。

与 collect_inloop 的差别：
- 不再写 UI 逻辑，链的行为完全由 run_w25 决定；
- 观察 = bridge truth/structure + 可见窗口清单（供遮挡/焦点归因）；
- 指令由 Recorder 按动作生成（通用模板），相位/上下文由本脚本注入。

用法:
    python experiments/computer_use/record_chain.py --trials 1
    python experiments/computer_use/record_chain.py --trials 10 --out artifacts/computer_use_inloop
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

import collect_inloop as CI  # noqa: E402  (只用它的 ensure_qgis / main_qgis)
import run_w25  # noqa: E402
from client import call  # noqa: E402
from inloop_rec import Recorder  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "artifacts" / "computer_use_inloop"
TASK = "w25_chain"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--shot-scale", type=float, default=0.6)
    args = ap.parse_args()

    out_root = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    CI.ensure_qgis()
    try:
        call({"op": "win", "maximize": True})
    except Exception:
        pass

    rec = Recorder("qgis", out_root, obs_provider=qgis_observation,
                   shot_scale=args.shot_scale)
    rec.install()
    rows = []
    try:
        for i in range(args.trials):
            out_p = OUT / "exports" / f"w25_t{i}.png"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            task_id = f"{TASK}_t{i:02d}"
            rec.new_task(task_id, task_name="thematic_map", chain="run_w25")
            before = len(rec.demos)
            steps = []
            t0 = time.time()
            ok, fail = False, ""
            try:
                ok, fail = run_w25.run_chain(out_p, steps)
            except Exception as e:
                import traceback
                traceback.print_exc()
                ok, fail = False, f"exception: {e}"
            total = round(time.time() - t0, 1)
            rec.end_task(bool(ok), verify={"fail": fail, "steps": len(steps)})
            demos = len(rec.demos) - before
            rows.append({"trial": i, "task_id": task_id, "ok": bool(ok), "fail": fail,
                         "total_s": total, "demos": demos})
            print(f"  t{i}: {'PASS' if ok else 'FAIL(' + fail + ')'} {total}s demos={demos}",
                  flush=True)
    finally:
        rec.uninstall()

    summary = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "run_w25.run_chain under inloop_rec.Recorder",
        "trials": rows,
        "n_demos": sum(r["demos"] for r in rows),
        "pass": sum(1 for r in rows if r["ok"]),
    }
    p = OUT / "record_chain_summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {p}  demos={summary['n_demos']} pass={summary['pass']}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
