"""把 Bigemap 窗口摆到屏幕右上角（窗口化，不最大化）。"""
import sys
import time

import win32con
import win32gui

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCREEN_W, SCREEN_H = 2560, 1440
WIDTH, HEIGHT = 1200, 900
MARGIN = 16

h = None


def cb(x, _):
    global h
    try:
        if win32gui.GetClassName(x).startswith("Qt") and "Bigemap" in win32gui.GetWindowText(x):
            h = x
    except Exception:
        pass


win32gui.EnumWindows(cb, None)
print("[hwnd]", h)
if h is None:
    raise SystemExit("no bigemap window")

x = SCREEN_W - WIDTH - MARGIN
y = MARGIN
win32gui.ShowWindow(h, win32con.SW_RESTORE)
time.sleep(0.8)
win32gui.SetWindowPos(h, win32con.HWND_TOP, x, y, WIDTH, HEIGHT,
                      win32con.SWP_SHOWWINDOW)
time.sleep(1.0)
print("[rect]", win32gui.GetWindowRect(h), "-> expected", (x, y, x + WIDTH, y + HEIGHT))
