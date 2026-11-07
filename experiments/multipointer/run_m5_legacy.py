"""M5：传统窗口（不处理 WM_POINTER）在合成触点下的行为。

用法：python experiments/multipointer/run_m5_legacy.py [N]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]

from mp_core import PointerWindow, SyntheticTouch, save_jsonl, touch_seq  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RECT = (120, 120, 1100, 760)
POINTS = [(300 + 200 * i, 420) for i in range(N)]

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


win = PointerWindow(f"MP_M5_{N}", rect=RECT, legacy=True, log=log)
win.run(injector, delay=1.6, after_close=0.8)

out = ROOT / "results" / "multipointer" / f"m5_legacy_{N}.jsonl"
save_jsonl(out, log)
ptr = [r for r in log if r.get("msg") == "pointer"]
mouse = [r for r in log if r.get("msg") == "mouse"]
print(f"N={N} inject={inject_log}")
print(f"pointer_msgs={len(ptr)} mouse_msgs={len(mouse)}")
for r in mouse[:12]:
    print("  mouse", r["wm"], r["x"], r["y"])
print(f"saved {out}")
