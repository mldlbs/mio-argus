"""在下载对话框里勾选指定级别，然后点确定开始下载。

级别行是自绘 DataItem，无 pattern，只能按 rect 点最左侧的勾选框。
"""
import sys
import time
from pathlib import Path

import pyautogui
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode")
DLG_TITLE = "\u4e0b\u8f7d\u5bf9\u8bdd\u6846"
LEVEL = "\u7b2c18\u7ea7"


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


h = hwnds(lambda x: win32gui.IsWindowVisible(x) and DLG_TITLE in win32gui.GetWindowText(x))
if not h:
    raise SystemExit("download dialog not open")
win = Desktop(backend="uia").window(handle=h[0])
print("[dialog]", h[0], win.rectangle())


def rows():
    out = []
    for c in win.descendants():
        try:
            if c.element_info.control_type == "DataItem":
                r = c.rectangle()
                nm = (c.window_text() or "").strip()
                if r.left < win.rectangle().left + 200 and nm:
                    out.append((nm, r, c))
        except Exception:
            continue
    out.sort(key=lambda x: x[1].top)
    return out


# 滚到目标级别
target = None
for attempt in range(12):
    rs = rows()
    names = [n for n, _, _ in rs]
    print(f"[scroll {attempt}] visible: {names}")
    for n, r, c in rs:
        if n == LEVEL:
            target = (n, r, c)
            break
    if target:
        break
    tab = None
    for c in win.descendants():
        try:
            if c.element_info.control_type == "Table":
                tab = c
                break
        except Exception:
            continue
    if tab is None:
        raise SystemExit("table missing")
    tr = tab.rectangle()
    pyautogui.moveTo((tr.left + tr.right) // 2, (tr.top + tr.bottom) // 2)
    pyautogui.scroll(-240)
    time.sleep(1.0)

if target is None:
    raise SystemExit("level 18 not found")

n, r, _ = target
print("[target row]", n, (r.left, r.top, r.right, r.bottom))
# 勾选框在行最左侧，垂直居中
cx = r.left + 10
cy = (r.top + r.bottom) // 2
print("[click checkbox]", (cx, cy))
pyautogui.click(cx, cy)
time.sleep(1.2)

# 截图核查勾选状态与理论大小
tab = None
for c in win.descendants():
    try:
        if c.element_info.control_type == "Table":
            tab = c
            break
    except Exception:
        continue
tr = tab.rectangle()
crop = pyautogui.screenshot(region=(tr.left, tr.top, tr.width(), tr.height()))
crop.resize((crop.width * 2, crop.height * 2)).save(TMP / "level_checked.png")
print("[saved] level_checked.png")

# 回读该行理论大小
for c in win.descendants():
    try:
        if c.element_info.control_type == "DataItem":
            rr = c.rectangle()
            if abs(rr.top - r.top) <= 4:
                print("   ", (c.window_text() or "").strip(), end=" | ")
    except Exception:
        continue
print()
