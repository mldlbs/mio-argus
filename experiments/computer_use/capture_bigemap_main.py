"""直接抓 Bigemap 主窗口控件树（已验证可用的路径）。"""
import json
from collections import Counter
from pathlib import Path

import pyautogui
from pywinauto import Desktop

OUT = Path(r"D:\work\code\nanoGPT\artifacts\uia_capture\bigemap")
OUT.mkdir(parents=True, exist_ok=True)

ACTIONABLE = {"Button", "MenuItem", "TabItem", "Edit", "ComboBox", "ListItem",
              "TreeItem", "CheckBox", "RadioButton", "SplitButton", "Slider", "Hyperlink"}

win = None
for w in Desktop(backend="uia").windows(visible_only=True):
    try:
        cls = w.element_info.class_name or ""
        if "Bigemap" in (w.window_text() or "") and cls.startswith("Qt"):
            win = w
            break
    except Exception:
        continue
if win is None:
    raise SystemExit("no window")

r = win.rectangle()
pyautogui.screenshot(region=(int(r.left), int(r.top),
                             int(r.width()), int(r.height()))).save(OUT / "shot_main.png")

nodes = []
for c in win.descendants():
    try:
        ct = c.element_info.control_type
        name = (c.window_text() or "").strip()
        rr = c.rectangle()
        nodes.append({"type": ct, "name": name,
                      "class": c.element_info.class_name or "",
                      "automation_id": c.element_info.auto_id or "",
                      "rect": [int(rr.left), int(rr.top),
                               int(rr.right - rr.left), int(rr.bottom - rr.top)],
                      "enabled": True})
    except Exception:
        continue

named = [n for n in nodes if n["name"]]
act = [n for n in nodes if n["type"] in ACTIONABLE and n["name"]]
payload = {"app": "bigemap", "state": "main",
           "title": win.window_text(),
           "window_rect": [int(r.left), int(r.top), int(r.width()), int(r.height())],
           "n_nodes": len(nodes), "n_named": len(named),
           "n_actionable_named": len(act),
           "by_type": dict(Counter(n["type"] for n in nodes).most_common(15)),
           "nodes": nodes}
(OUT / "tree_main.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
print(f"[bigemap] nodes={len(nodes)} named={len(named)} actionable_named={len(act)}")
print(f"[bigemap] by_type={payload['by_type']}")
print(f"[out] {OUT / 'tree_main.json'}")
