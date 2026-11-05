"""打开 设置 -> 代理，并 dump 代理设置内容。"""
import sys
import time

import pyautogui
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = r"C:\Users\gf191\AppData\Local\Temp\opencode"


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
print("[window]", win32gui.GetWindowRect(main))

w = Desktop(backend="uia").window(handle=main)
w.descendants()

# 打开「设置」菜单
item = None
for c in w.descendants(control_type="MenuItem"):
    try:
        if (c.window_text() or "").strip() == "\u8bbe\u7f6e":
            item = c
            break
    except Exception:
        continue
print("[设置]", item is not None, item.rectangle() if item else None)
item.click_input()
time.sleep(1.5)

r = item.rectangle()
pyautogui.screenshot(region=(r.left - 20, r.top - 10, 320, 160)).save(f"{TMP}\\menu_open.png")
print("[saved] menu_open.png")

# 弹窗是独立的 Qt 顶层窗口，找它并读出条目
popups = hwnds(lambda x: win32gui.IsWindowVisible(x)
               and win32gui.GetClassName(x).startswith("Qt")
               and win32gui.GetWindowText(x) == "")
print("[popups]", [(p, win32gui.GetWindowRect(p)) for p in popups])
