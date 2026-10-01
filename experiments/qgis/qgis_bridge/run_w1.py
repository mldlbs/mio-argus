"""
QGIS W1：真桌面应用上的任务 / 通道对照

  1. oracle 冒烟：4 个 GUI 任务，由**结构化动作**执行、由 **PyQGIS 状态**判定
       - activate_zoom_in / activate_pan ：动作是"地图工具"，真值 = canvas.mapTool()
       - zoom_full                      ：先把视图缩小（特权预置），再用 Zoom Full 复原
       - toggle_console                 ：Plugins 工具栏按钮，真值 = Python 控制台可见性
  2. 候选统计：工具栏动作数 / 按钮尺寸
  3. 通道探针：结构（动作名）vs 像素（按钮外观）

用法:
    python experiments/qgis/qgis_bridge/run_w1.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402


def snap():
    r = call({"op": "truth"})
    if not r.get("ok"):
        raise RuntimeError(r)
    return r["truth"]


def trigger(text):
    return call({"op": "act", "kind": "action", "text": text})


def layer_extent(t):
    return t["layers"][0]["extent"] if t.get("layers") else None


def shrink_view():
    t = snap()
    e = layer_extent(t)
    if e is None:
        return
    x0, y0, x1, y1 = e
    call({"op": "set_extent", "box": [x0, y0, x0 + (x1 - x0) * 0.3, y0 + (y1 - y0) * 0.3]})


TASKS = [
    dict(id="activate_zoom_in", instr="activate the zoom in tool", act="Zoom In", pre=[],
         check=lambda b, a, r: (a.get("tool") or "").lower().startswith("zoom in")),
    dict(id="activate_pan", instr="activate the pan tool", act="Pan Map", pre=[],
         check=lambda b, a, r: "pan" in (a.get("tool") or "").lower()),
    dict(id="zoom_full", instr="zoom to the full extent of the cities layer",
         act="Zoom Full", pre=["shrink"],
         check=lambda b, a, r: abs(a["extent"][0] - r["extent"][0]) < 1.0),
    dict(id="toggle_console", instr="open the python console", act="Python Console", pre=[],
         check=lambda b, a, r: a.get("console_visible") != b.get("console_visible")),
]


def word_pick(cands, instr):
    words = [w for w in instr.lower().replace(",", " ").split() if len(w) > 2]
    best, bs = None, 0.0
    for c in cands:
        n = c["text"].lower()
        s = sum(1.0 for w in words if w in n)
        if s > bs:
            best, bs = c, s
    return best if bs > 0 else None


def main():
    call({"op": "win", "maximize": True})

    print("=== 1) oracle 冒烟（结构动作 + 状态判定）===")
    ok_all = True
    for rep in range(2):
        for t in TASKS:
            call({"op": "reset"})
            r0 = snap()
            for pre in t["pre"]:
                if pre == "shrink":
                    shrink_view()
            b = snap()
            res = trigger(t["act"])
            a = snap()
            good = bool(res.get("ok")) and bool(t["check"](b, a, r0))
            ok_all &= good
            print(f"  [{t['id']:<16}] rep{rep} trigger={res.get('ok')} "
                  f"truth_ok={t['check'](b, a, r0)} tool={a.get('tool')!r} "
                  f"console={a.get('console_visible')}")
    print(f"  => oracle 全部成功: {ok_all}")

    print("\n=== 2) 候选统计 ===")
    st = call({"op": "structure"})
    cands = [a for a in st["actions"] if a.get("rect") and a["rect"][2] > 0]
    ws = [a["rect"][2] for a in cands]
    hs = [a["rect"][3] for a in cands]
    print(f"  工具栏动作(带坐标) = {len(cands)}；按钮尺寸 median "
          f"{int(np.median(ws))}x{int(np.median(hs))}，min {min(ws)}x{min(hs)}")

    print("\n=== 3a) 结构通道：按动作名关键词选 ===")
    hit = 0
    for t in TASKS:
        p = word_pick(cands, t["instr"])
        good = p is not None and p["text"] == t["act"]
        hit += int(good)
        print(f"  [{t['id']:<16}] '{t['instr']}' -> "
              f"{p['text'] if p else None} (目标 {t['act']}) {'OK' if good else 'MISS'}")
    print(f"  结构关键词命中 = {hit}/{len(TASKS)}")

    print("\n=== 3b) 像素通道：按钮外观可分性（不读名字，只测外观）===")
    from PIL import Image
    shot = call({"op": "shot", "path": str(HERE / "_w1.png")})
    img = np.asarray(Image.open(shot["path"]).convert("RGB"))
    print(f"  QGIS 窗口 {shot['size']}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"  QGIS 窗口 {shot['size']}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    CH, CW = 32, 48
    crops, labels = [], []
    for i, a in enumerate(cands):
        x, y, w, h = a["rect"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crops.append(np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((CW, CH))))
        labels.append(i)
    X = torch.tensor(np.stack(crops)).permute(0, 3, 1, 2).float().div_(255)
    y = torch.tensor(labels)
    K = len(cands)
    perm = np.random.default_rng(0).permutation(len(crops))
    cut = int(len(crops) * 0.8)
    tr, te = perm[:cut], perm[cut:]

    class Net(nn.Module):
        def __init__(s, k):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                                nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, k))
        def forward(s, x):
            return s.f(x)

    torch.manual_seed(0)
    m = Net(K)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(60):
        loss = F.cross_entropy(m(X[tr]), y[tr])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        a_tr = float((m(X[tr]).argmax(-1) == y[tr]).float().mean())
        a_te = float((m(X[te]).argmax(-1) == y[te]).float().mean())
    print(f"  外观分类（{K} 类）：train {a_tr:.3f} | test {a_te:.3f} | chance {1/K:.3f}")
    print("  注：这是**上界探针**（只测外观可分性）；任务要的是"
          "'指令→按钮'，而像素侧读不到名字")

    (HERE / "w1_results.json").write_text(json.dumps({
        "oracle_all_ok": ok_all,
        "toolbar_actions": len(cands),
        "btn_size_median": [int(np.median(ws)), int(np.median(hs))],
        "struct_keyword_hit": f"{hit}/{len(TASKS)}",
        "pixel_appearance_train": a_tr, "pixel_appearance_test": a_te,
        "pixel_chance": 1.0 / K,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w1_results.json")


if __name__ == "__main__":
    main()
