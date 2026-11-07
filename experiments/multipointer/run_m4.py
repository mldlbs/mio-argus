"""M4：两个窗口，10 个触点（各 5 个），看是否各自收到独立指针。

用法：python experiments/multipointer/run_m4.py
"""
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]

from mp_core import (PointerWindow, SyntheticTouch, save_jsonl, touch_seq)  # noqa: E402

LEFT = (120, 120, 460, 760)
RIGHT = (600, 120, 1080, 760)
PTS_LEFT = [(180, 300), (240, 360), (300, 300), (360, 360), (420, 300)]
PTS_RIGHT = [(660, 300), (720, 360), (780, 300), (840, 360), (900, 300)]
ALL = PTS_LEFT + PTS_RIGHT

log_left: list[dict] = []
log_right: list[dict] = []
ready = threading.Barrier(2)


def make_worker(win):
    def worker():
        win.create()
        ready.wait()
        win.loop()
    return worker


w_left = PointerWindow("MP_M4_L", rect=LEFT, log=log_left)
w_right = PointerWindow("MP_M4_R", rect=RIGHT, log=log_right)
t1 = threading.Thread(target=make_worker(w_left), daemon=True)
t2 = threading.Thread(target=make_worker(w_right), daemon=True)
t1.start(); t2.start()
ready.wait()
time.sleep(1.2)

dev = SyntheticTouch(max_count=10)
try:
    seq = touch_seq(ALL)
    print("inject", seq(dev))
finally:
    dev.close()

time.sleep(0.8)
import win32gui  # noqa: E402
for w in (w_left, w_right):
    win32gui.PostMessage(w.hwnd, 0x0010, 0, 0)
t1.join(timeout=2); t2.join(timeout=2)

out = ROOT / "results" / "multipointer"
save_jsonl(out / "m4_left.jsonl", log_left)
save_jsonl(out / "m4_right.jsonl", log_right)
for name, log in (("left", log_left), ("right", log_right)):
    ptr = [r for r in log if r.get("msg") == "pointer"]
    ids = sorted({r["pointer_id"] for r in ptr if r.get("pointer_id")})
    xs = sorted({r["x"] for r in ptr if r.get("x") is not None})
    print(f"{name}: pointer={len(ptr)} ids={ids} xs={xs}")
