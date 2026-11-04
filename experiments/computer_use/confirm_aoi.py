"""选中区域（用 SelectionItemPattern 并回读）-> 确定。"""
import sys
import time

import pyautogui
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = r"C:\Users\gf191\AppData\Local\Temp\opencode"
IMPORT_T = "\u5bfc\u5165\u8fb9\u754c\u6570\u636e"
TARGET = "shijingshan_aoi_200m_buffer"


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


# 先关掉提示框
for h in hwnds(lambda x: win32gui.IsWindowVisible(x)
               and win32gui.GetClassName(x).startswith("Qt")
               and "\u63d0\u793a" == win32gui.GetWindowText(x)):
    try:
        t = Desktop(backend="uia").window(handle=h)
        for b in t.descendants(control_type="Button"):
            if (b.window_text() or "").strip().upper() in ("OK", "\u786e\u5b9a"):
                b.click_input()
                print("[tip closed]")
                break
    except Exception as ex:
        print("[tip]", type(ex).__name__, ex)
time.sleep(1.0)

imp = hwnds(lambda x: win32gui.IsWindowVisible(x) and IMPORT_T in win32gui.GetWindowText(x))
dlg = Desktop(backend="uia").window(handle=imp[0])

item = None
for c in dlg.descendants():
    try:
        if c.element_info.control_type in ("ListItem", "TreeItem") and \
                (c.window_text() or "").strip() == TARGET:
            item = c
            break
    except Exception:
        continue
print("[item]", item is not None)

try:
    item.select()
    print("[select()] ok")
except Exception as ex:
    print("[select()] fail", type(ex).__name__, ex)

time.sleep(0.8)
try:
    print("[is_selected]", item.is_selected())
except Exception as ex:
    print("[is_selected] fail", type(ex).__name__, ex)

pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(f"{TMP}\\aoi_sel2.png")

for b in dlg.descendants():
    try:
        if b.element_info.control_type in ("Button", "SplitButton") and \
                (b.window_text() or "").strip().startswith("\u786e\u5b9a"):
            b.click_input()
            print("[确定] clicked")
            break
    except Exception:
        continue
time.sleep(3.5)

print("[import dlg open]", bool(hwnds(
    lambda x: win32gui.IsWindowVisible(x) and IMPORT_T in win32gui.GetWindowText(x))))
pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(f"{TMP}\\aoi_confirm2.png")
print("[saved] aoi_sel2.png / aoi_confirm2.png")
