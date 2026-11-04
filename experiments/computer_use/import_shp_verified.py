"""导入 SHP（严格逐步验证）。

1. 剪贴板粘贴路径（pyautogui.write 会吃掉反斜杠，不能用）
2. 点下「打开(O)」前确认按钮文本
3. 打开后立刻读「打开文件」字段，确认路径真的进去了
4. 再读「选择区域」，确认多边形被列出来
"""
import sys
import time

import pyperclip
import pyautogui
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TMP = r"C:\Users\gf191\AppData\Local\Temp\opencode"
FILEDLG = "\u9009\u62e9\u6587\u4ef6"
IMPORT_T = "\u5bfc\u5165\u8fb9\u754c\u6570\u636e"
OPEN_O = "\u6253\u5f00(O)"
SHP = r"C:\Users\gf191\sjs_aoi.shp"


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


def visible(title):
    return hwnds(lambda x: win32gui.IsWindowVisible(x) and title in win32gui.GetWindowText(x))


# ---------- 1. 文件对话框：粘贴路径 ----------
fdh = visible(FILEDLG)
print("[step1] file dialog open:", bool(fdh))
if not fdh:
    imp0 = visible(IMPORT_T)
    if not imp0:
        raise SystemExit("import dialog not open either")
    d0 = Desktop(backend="uia").window(handle=imp0[0])
    for b in d0.descendants(control_type="Button"):
        try:
            if (b.window_text() or "").strip() == "...":
                b.click_input()
                print("[step1] clicked ...")
                break
        except Exception:
            continue
    for _ in range(20):
        time.sleep(0.5)
        fdh = visible(FILEDLG)
        if fdh:
            break
    if not fdh:
        raise SystemExit("file dialog did not open")

fd = Desktop(backend="uia").window(handle=fdh[0])
fr = fd.rectangle()
# 文件名输入框：最底部、最宽的那个 Edit
edits = []
for e in fd.descendants(control_type="Edit"):
    r = e.rectangle()
    if r.top > fr.top + 500 and (r.right - r.left) > 400:
        edits.append((r.top, r.left, e))
edits.sort()
name_edit = edits[-1][2]
print("[step1] name edit rect:", name_edit.rectangle())

pyperclip.copy(SHP)
name_edit.click_input()
time.sleep(0.3)
pyautogui.hotkey("ctrl", "a")
time.sleep(0.2)
pyautogui.hotkey("ctrl", "v")
time.sleep(0.8)

pyautogui.screenshot(region=(fr.left + 180, fr.bottom - 110, 700, 50)).save(f"{TMP}\\verify_path.png")

# ---------- 2. 打开(O) ----------
btn = None
for b in fd.descendants():
    try:
        if b.element_info.control_type not in ("Button", "SplitButton"):
            continue
        if (b.window_text() or "").strip() == OPEN_O:
            btn = b
            break
    except Exception:
        continue
print("[step2] open(O) found:", btn is not None, btn.rectangle() if btn else None)
if btn is None:
    raise SystemExit("no 打开(O) button")
btn.click_input()
time.sleep(3.5)

# ---------- 3. 打开文件字段 ----------
imp = visible(IMPORT_T)
print("[step3] import dialog open:", bool(imp))
if not imp:
    raise SystemExit("import dialog closed unexpectedly")
dlg = Desktop(backend="uia").window(handle=imp[0])

file_field = []
for e in dlg.descendants(control_type="Edit"):
    try:
        r = e.rectangle()
        file_field.append(((r.top, r.left), e.window_text() or ""))
    except Exception:
        continue
file_field.sort()
print("[step3] edit fields:", file_field)

# ---------- 4. 选择区域 ----------
rows = []
for c in dlg.descendants():
    try:
        if c.element_info.control_type in ("ListItem", "TreeItem"):
            n = (c.window_text() or "").strip()
            if n:
                rows.append(n)
    except Exception:
        continue
print("[step4] region list:", rows)

pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(f"{TMP}\\import_verified.png")
print("[saved] verify_path.png / import_verified.png")
