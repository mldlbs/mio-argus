"""抓取已打开的 Bigemap「下载对话框」控件树（带 UIA 懒加载重试）。"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from pywinauto import Desktop
import pyautogui
import win32gui

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\work\code\nanoGPT\artifacts\uia_capture\bigemap")
OUT.mkdir(parents=True, exist_ok=True)


def _auto_id(ctrl):
    """Qt 节点上 element_info.auto_id 会抛 AttributeError，必须兜住。"""
    try:
        return ctrl.element_info.auto_id or ""
    except Exception:
        return ""


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


def find_dialog():
    for h in hwnds(lambda h: win32gui.IsWindowVisible(h)
                   and win32gui.GetClassName(h).startswith("Qt")):
        if "\u4e0b\u8f7d\u5bf9\u8bdd\u6846" in win32gui.GetWindowText(h):
            return h
    return None


def capture(win, tag, retries=12):
    nodes = []
    for i in range(retries):
        nodes = []
        for c in win.descendants():
            try:
                rr = c.rectangle()
                nodes.append({"type": c.element_info.control_type,
                              "name": (c.window_text() or "").strip(),
                              "class": c.element_info.class_name or "",
                              "auto_id": _auto_id(c),
                              "rect": [int(rr.left), int(rr.top),
                                       int(rr.right - rr.left), int(rr.bottom - rr.top)]})
            except Exception:
                continue
        if nodes:
            print(f"[lazy] attempt {i + 1} -> {len(nodes)} nodes")
            break
        time.sleep(1.0)
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


def warmup():
    """Qt 的 UIA 桥接在冷进程首次连接时会返回空树，先查一次主窗口预热。"""
    for h in hwnds(lambda h: win32gui.GetClassName(h).startswith("Qt")
                   and "Bigemap" in win32gui.GetWindowText(h)):
        try:
            n = len(Desktop(backend="uia").window(handle=h).descendants())
            print(f"[warmup] main {h} -> {n} nodes")
            return n
        except Exception as ex:
            print("[warmup-fail]", type(ex).__name__, ex)
    return 0


def main():
    warmup()
    h = find_dialog()
    print("[hwnd]", h)
    if h is None:
        return 1
    p = capture(Desktop(backend="uia").window(handle=h), "download")
    rows = sorted(p["nodes"], key=lambda n: (n["rect"][1], n["rect"][0]))
    print("--- nodes ---")
    for n in rows:
        print(f"  {n['type']:12s} [{n['rect'][0]:5d},{n['rect'][1]:5d},"
              f"{n['rect'][2]:4d}x{n['rect'][3]:3d}] {n['name'][:34]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
