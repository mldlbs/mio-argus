"""双击导入的 AOI 矩形内部，打开下载对话框并抓树。

矩形边框是蓝色，先检测它的 bbox 再取中心双击，避免猜坐标。
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyautogui
import win32con
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\work\code\nanoGPT\artifacts\uia_capture\bigemap")
TMP = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode")
OUT.mkdir(parents=True, exist_ok=True)
DLG_TITLE = "\u4e0b\u8f7d\u5bf9\u8bdd\u6846"


def hwnds(pred):
    hits = []

    def cb(h, _):
        try:
            if pred(h):
                hits.append(h)
        except Exception:
            pass

    win32gui.EnumWindows(cb, hits)
    return hits


main = hwnds(lambda x: win32gui.GetClassName(x).startswith("Qt")
             and "Bigemap" in win32gui.GetWindowText(x))[0]
if win32gui.GetWindowRect(main)[0] <= -32000:
    win32gui.ShowWindow(main, win32con.SW_RESTORE)
    time.sleep(1.0)
win32gui.SetWindowPos(main, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
time.sleep(0.8)
L, T, R, B = win32gui.GetWindowRect(main)
print("[window]", (L, T, R, B))

# 画布里找蓝色矩形边框
canvas = (L + 280, T + 160, R - 20, B - 50)
shot = pyautogui.screenshot(region=canvas)
px = shot.load()
w_, h_ = shot.size
pts = []
for y in range(h_):
    for x in range(w_):
        r, g, b = px[x, y][:3]
        if b > 150 and b - r > 60 and b - g > 40:
            pts.append((x, y))
print("[blue pixels]", len(pts))
if len(pts) < 100:
    raise SystemExit("AOI rectangle not found")

xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
print("[bbox]", (min(xs), min(ys), max(xs), max(ys)))
cx = canvas[0] + (min(xs) + max(xs)) // 2
cy = canvas[1] + (min(ys) + max(ys)) // 2
print("[center]", (cx, cy))
pyautogui.doubleClick(cx, cy)

for i in range(12):
    time.sleep(1.0)
    d = hwnds(lambda x: win32gui.IsWindowVisible(x) and DLG_TITLE in win32gui.GetWindowText(x))
    if d:
        print(f"[dialog after {i + 1}s]", d[0])
        break
else:
    pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(TMP / "no_dl5.png")
    raise SystemExit("download dialog not opened")

win = Desktop(backend="uia").window(handle=d[0])
nodes = []
for _ in range(6):
    nodes = []
    for c in win.descendants():
        try:
            rr = c.rectangle()
            try:
                aid = c.element_info.auto_id or ""
            except Exception:
                aid = ""
            nodes.append({"type": c.element_info.control_type,
                          "name": (c.window_text() or "").strip(),
                          "class": c.element_info.class_name or "",
                          "auto_id": aid,
                          "rect": [int(rr.left), int(rr.top),
                                   int(rr.right - rr.left), int(rr.bottom - rr.top)]})
        except Exception:
            continue
    if nodes:
        break
    time.sleep(1.5)

r = win.rectangle()
pyautogui.screenshot(region=(int(r.left), int(r.top),
                             int(r.width()), int(r.height()))).save(OUT / "shot_download.png")
named = [n for n in nodes if n["name"]]
p = {"app": "bigemap", "state": "download", "title": win.window_text(),
     "window_rect": [int(r.left), int(r.top), int(r.width()), int(r.height())],
     "n_nodes": len(nodes), "n_named": len(named),
     "by_type": dict(Counter(n["type"] for n in nodes).most_common(20)),
     "nodes": nodes}
(OUT / "tree_download.json").write_text(json.dumps(p, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
print(f"[download] nodes={len(nodes)} named={len(named)}")
print(p["by_type"])
