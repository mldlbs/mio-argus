"""M1：N 个合成触点 → 记录 WM_POINTER/WM_MOUSE 投递。

用法：python experiments/multipointer/run_m1.py [N]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]

from mp_core import (PointerWindow, SyntheticTouch, save_jsonl, touch_seq)  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
RECT = (120, 120, 1100, 760)
POINTS = [(200 + 85 * i, 420) for i in range(N)]

log: list[dict] = []
inject_log: list[dict] = []


def injector():
    dev = SyntheticTouch(max_count=N)
    try:
        seq = touch_seq(POINTS)
        for name, (ok, err) in seq(dev):
            inject_log.append({"phase": name, "ok": ok, "err": err})
    finally:
        dev.close()


win = PointerWindow(f"MP_M1_{N}", rect=RECT, legacy=False, log=log)
win.run(injector, delay=1.6, after_close=0.8)

out = ROOT / "results" / "multipointer" / f"m1_{N}.jsonl"
save_jsonl(out, log)
(out.parent / f"m1_{N}_inject.json").write_text(
    json.dumps(inject_log, ensure_ascii=False, indent=2), encoding="utf-8")

ptr = [r for r in log if r.get("msg") == "pointer"]
mouse = [r for r in log if r.get("msg") == "mouse"]
touch = [r for r in log if r.get("msg") == "touch"]
ids = sorted({r["pointer_id"] for r in ptr if r.get("pointer_id")})
print(f"N={N} points={POINTS}")
print(f"inject: {inject_log}")
print(f"pointer_msgs={len(ptr)} distinct_ids={ids}")
print(f"touch_msgs={len(touch)} mouse_msgs={len(mouse)}")
print(f"saved {out}")
