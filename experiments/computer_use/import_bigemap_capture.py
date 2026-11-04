"""把早前抓到的 Bigemap 控件树（bigemap_controls.txt）转成标准 capture 格式。

来源：Bigemap 健康运行时的一次 UIA 导出（主窗口，97 个控件），
格式：`TYPE [l,t,r,b] TEXT`，本脚本解析为 {type,name,rect,...}。
"""
import json
import re
from pathlib import Path

SRC = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode\bigemap_controls.txt")
SHOT = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode\bigemap_focus.png")
OUT = Path(r"D:\work\code\nanoGPT\artifacts\uia_capture\bigemap")

ACTIONABLE = {"Button", "MenuItem", "TabItem", "Edit", "ComboBox", "ListItem",
              "TreeItem", "CheckBox", "RadioButton", "SplitButton", "Slider", "Hyperlink"}

PAT = re.compile(r"^(\w+)\s+\[(-?\d+),(-?\d+),(-?\d+),(-?\d+)\]\s*(.*)$")


def main():
    nodes = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        t, l, tp, r, b, name = m.groups()
        l, tp, r, b = map(int, (l, tp, r, b))
        nodes.append({"type": t, "name": name.strip(),
                      "class": "", "automation_id": "",
                      "rect": [l, tp, r - l, b - tp], "enabled": True})

    named = [n for n in nodes if n["name"]]
    act = [n for n in nodes if n["type"] in ACTIONABLE and n["name"]]
    from collections import Counter
    payload = {
        "title": "Bigemap GIS Office (健康运行时的 UIA 导出)",
        "window_rect": [0, 0, 2568, 1408],
        "n_nodes": len(nodes), "n_named": len(named),
        "n_actionable_named": len(act),
        "by_type": dict(Counter(n["type"] for n in nodes).most_common(12)),
        "source": "bigemap_controls.txt (较早一次健康会话的导出)",
        "nodes": nodes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tree_legacy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    if SHOT.exists():
        import shutil
        shutil.copy(SHOT, OUT / "shot_legacy.png")
    print(f"[bigemap] nodes={len(nodes)} named={len(named)} actionable_named={len(act)}")
    print(f"[bigemap] by_type={payload['by_type']}")
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
