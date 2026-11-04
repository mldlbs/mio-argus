"""
通用 UIA 候选采集：抓某个应用窗口的控件树 + 截图，作为候选生成基准（应用 #2）。

输出（每个应用一份）：
  artifacts/uia_capture/<app>/tree.json     每个控件：type/name/automation_id/class/rect/enabled/offscreen/depth/path
  artifacts/uia_capture/<app>/shot.png      整窗截图

统计口径（与 candidate_bench 一致）：
  - named：有名字的控件
  - actionable：Button/MenuItem/TabItem/Edit/ComboBox/ListItem/TreeItem/CheckBox/RadioButton/Slider
  - container：Window/Pane/Custom/Document 等结构性节点（不计入候选）
"""
import argparse
import json
import time
from pathlib import Path

import pyautogui
from pywinauto import Desktop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "uia_capture"

ACTIONABLE = {"Button", "MenuItem", "TabItem", "Edit", "ComboBox", "ListItem",
              "TreeItem", "CheckBox", "RadioButton", "SplitButton", "Slider", "Hyperlink"}
CONTAINER = {"Window", "Pane", "Custom", "Document", "Group", "ToolBar", "MenuBar",
             "TitleBar", "ScrollBar", "Tree", "List", "Table", "DataGrid", "StatusBar"}


def capture(title_sub, app_name, max_nodes=4000):
    win = None
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if title_sub in (w.window_text() or ""):
                win = w
                break
        except Exception:
            continue
    if win is None:
        raise SystemExit(f"window not found: {title_sub}")

    r = win.rectangle()
    out_dir = OUT / app_name
    out_dir.mkdir(parents=True, exist_ok=True)
    img = pyautogui.screenshot(region=(int(r.left), int(r.top), int(r.width()), int(r.height())))
    img.save(out_dir / "shot.png")

    nodes = []
    for c in win.descendants():
        try:
            ct = c.element_info.control_type
            name = (c.window_text() or "").strip()
            rr = c.rectangle()
            nodes.append({
                "type": ct, "name": name,
                "automation_id": c.element_info.auto_id or "",
                "class": c.element_info.class_name or "",
                "rect": [int(rr.left), int(rr.top), int(rr.right - rr.left), int(rr.bottom - rr.top)],
                "enabled": bool(c.is_enabled()),
                "offscreen": bool(c.is_offscreen()),
            })
        except Exception:
            continue
        if len(nodes) >= max_nodes:
            break

    named = [n for n in nodes if n["name"]]
    actionable = [n for n in nodes if n["type"] in ACTIONABLE]
    act_named = [n for n in actionable if n["name"] and n["enabled"]]
    containers = [n for n in nodes if n["type"] in CONTAINER]

    payload = {
        "app": app_name, "title": win.window_text(),
        "window_rect": [int(r.left), int(r.top), int(r.width()), int(r.height())],
        "n_nodes": len(nodes), "n_named": len(named),
        "n_actionable": len(actionable), "n_actionable_named_enabled": len(act_named),
        "n_container": len(containers),
        "by_type": {},
        "nodes": nodes,
    }
    from collections import Counter
    payload["by_type"] = dict(Counter(n["type"] for n in nodes).most_common(15))
    (out_dir / "tree.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{app_name}] nodes={len(nodes)} named={len(named)} "
          f"actionable={len(actionable)} actionable_named_enabled={len(act_named)} "
          f"containers={len(containers)}")
    print(f"[{app_name}] by_type={payload['by_type']}")
    print(f"[out] {out_dir}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Bigemap")
    ap.add_argument("--app", default="bigemap")
    ap.add_argument("--wait", type=float, default=0.0)
    args = ap.parse_args()
    if args.wait:
        time.sleep(args.wait)
    capture(args.title, args.app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
