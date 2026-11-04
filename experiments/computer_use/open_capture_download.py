"""点 Rectangle 面板的下载图标 -> 等对话框 -> 预热 -> 抓 UIA 树。

踩过的坑（都已修）：
1. element_info.auto_id 在 Qt 节点上抛 AttributeError，逐节点 except 会把整棵树吞成 0；
2. 冷进程第一次查 Qt 弹窗可能拿到空树，先查一次主窗口预热；
3. 被遮挡时 UIA 报 offscreen，visible_only=True 会整窗过滤，改用 hwnd 枚举。
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyautogui
import win32gui
from pywinauto import Desktop

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\work\code\nanoGPT\artifacts\uia_capture\bigemap")
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


def main_hwnd():
    h = None
    for cand in hwnds(lambda x: win32gui.GetClassName(x).startswith("Qt")
                      and "Bigemap" in win32gui.GetWindowText(x)):
        h = cand
        break
    if h is None:
        return None
    # 失去焦点时这个窗口会自己最小化（rect 变成 -32000），先恢复
    if win32gui.GetWindowRect(h)[0] <= -32000:
        win32gui.ShowWindow(h, 3)
        time.sleep(1.0)
    return h


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
        print(f"[try {i + 1}] {len(nodes)} nodes")
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
    print(f"[{tag}] nodes={len(nodes)} named={len(named)}")
    print(f"[{tag}] by_type={p['by_type']}")
    return p


def open_dialog(icon_xy):
    h = main_hwnd()
    if h is None:
        raise SystemExit("no main window")
    win32gui.ShowWindow(h, 3)
    try:
        win32gui.SetForegroundWindow(h)
    except Exception:
        pass
    time.sleep(1.0)
    print("[main]", h, "fg", win32gui.GetForegroundWindow() == h)

    Desktop(backend="uia").window(handle=h).descendants()  # 预热

    pyautogui.click(*icon_xy)
    for i in range(12):
        time.sleep(1.0)
        for d in hwnds(lambda x: win32gui.IsWindowVisible(x)
                       and win32gui.GetClassName(x).startswith("Qt")
                       and DLG_TITLE in win32gui.GetWindowText(x)):
            print(f"[dialog found after {i + 1}s]", d)
            return d
    return None


def main():
    icon = (2407, 302)
    dh = open_dialog(icon)
    if dh is None:
        pyautogui.screenshot(region=(0, 0, 2560, 1440)).save(
            Path(r"C:\Users\gf191\AppData\Local\Temp\opencode\no_dl3.png"))
        print("[fail] dialog not opened")
        return 1
    p = capture(Desktop(backend="uia").window(handle=dh), "download")
    rows = sorted(p["nodes"], key=lambda n: (n["rect"][1], n["rect"][0]))
    print("--- nodes ---")
    for n in rows:
        print(f"  {n['type']:12s} [{n['rect'][0]:5d},{n['rect'][1]:5d},"
              f"{n['rect'][2]:4d}x{n['rect'][3]:3d}] {n['name'][:36]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
