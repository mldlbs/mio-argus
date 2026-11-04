"""右上角窗口化状态下打开下载对话框：重画矩形 -> 双击 -> 抓 UIA 树。

所有坐标都从当前窗口矩形推导，换位置也不用改脚本。
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
RECT_TOOL = "\u77e9\u5f62\u6846"


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


def main_hwnd():
    for h in hwnds(lambda x: win32gui.GetClassName(x).startswith("Qt")
                   and "Bigemap" in win32gui.GetWindowText(x)):
        return h
    return None


def dlg_hwnd():
    for h in hwnds(lambda x: win32gui.IsWindowVisible(x)
                   and DLG_TITLE in win32gui.GetWindowText(x)):
        return h
    return None


def auto_id(ctrl):
    try:
        return ctrl.element_info.auto_id or ""
    except Exception:
        return ""


def node_dict(ctrl):
    try:
        rr = ctrl.rectangle()
        return {"type": ctrl.element_info.control_type,
                "name": (ctrl.window_text() or "").strip(),
                "class": ctrl.element_info.class_name or "",
                "auto_id": auto_id(ctrl),
                "rect": [int(rr.left), int(rr.top),
                         int(rr.right - rr.left), int(rr.bottom - rr.top)]}
    except Exception:
        return None


def capture(win, tag, retries=6):
    nodes = []
    for i in range(retries):
        nodes = [d for d in (node_dict(c) for c in win.descendants()) if d]
        if nodes:
            break
        time.sleep(1.5)
    r = win.rectangle()
    pyautogui.screenshot(region=(int(r.left), int(r.top),
                                 int(r.width()), int(r.height()))).save(OUT / f"shot_{tag}.png")
    named = [n for n in nodes if n["name"]]
    p = {"app": "bigemap", "state": tag, "title": win.window_text(),
         "window_rect": [int(r.left), int(r.top), int(r.width()), int(r.height())],
         "n_nodes": len(nodes), "n_named": len(named),
         "by_type": dict(Counter(n["type"] for n in nodes).most_common(20)),
         "nodes": nodes}
    (OUT / f"tree_{tag}.json").write_text(json.dumps(p, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[{tag}] nodes={len(nodes)} named={len(named)} {p['by_type']}")
    return p


def main():
    h = main_hwnd()
    if h is None:
        raise SystemExit("no main window")
    if win32gui.GetWindowRect(h)[0] <= -32000:
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
        time.sleep(1.0)
    try:
        win32gui.SetForegroundWindow(h)
    except Exception:
        pass
    time.sleep(1.0)
    rect = win32gui.GetWindowRect(h)
    # 抢前台不可靠（SetForegroundWindow 会被前台锁拒绝），改临时置顶：只改 z 序
    win32gui.SetWindowPos(h, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    time.sleep(0.6)
    print("[window]", rect, "fg", win32gui.GetForegroundWindow() == h)

    w = Desktop(backend="uia").window(handle=h)
    w.descendants()  # 预热

    radio = None
    for c in w.descendants(control_type="RadioButton"):
        try:
            if (c.window_text() or "").strip() == RECT_TOOL:
                radio = c
                break
        except Exception:
            continue
    print("[rect tool]", radio is not None)
    if radio is None:
        return 2
    radio.click_input()
    time.sleep(0.8)

    # 按窗口矩形推导地图内的一块区域（避开左侧图层树和右侧面板）
    left, top, right, bottom = rect
    mx1 = left + int((right - left) * 0.45)
    my1 = top + int((bottom - top) * 0.45)
    mx2 = mx1 + 260
    my2 = my1 + 200
    print("[draw]", (mx1, my1), "->", (mx2, my2))
    pyautogui.moveTo(mx1, my1)
    time.sleep(0.3)
    pyautogui.dragTo(mx2, my2, duration=0.8, button="left")
    time.sleep(1.2)

    cx, cy = (mx1 + mx2) // 2, (my1 + my2) // 2
    print("[double-click]", (cx, cy))
    pyautogui.doubleClick(cx, cy)
    for i in range(12):
        time.sleep(1.0)
        d = dlg_hwnd()
        if d:
            print(f"[dialog after {i + 1}s]", d)
            return 0 if capture(Desktop(backend="uia").window(handle=d), "download") else 1
    pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(TMP / "no_dl_tr.png")
    print("[fail] dialog not opened")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
