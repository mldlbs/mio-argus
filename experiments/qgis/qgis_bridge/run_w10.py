"""
QGIS W10：依赖式任务（结果可见版）——修掉 W9 的 verifier 伪迹

W9 的教训（三组对照实测）：
  · 点图层上"选中 2 个"的视觉信号 diff≈0.0034，而失败变体（选 0 个）会弹
    "No matching features" 警告框 → V 学到的是**弹窗伪迹**（AUC 0.968/0.824），
    正确样本的 after 与 before 几乎逐像素相同（0.0001）——V 根本没读选择。
  · 换多边形后：正确/错误变体的画面差异 diff≈1.13/1.61（300 倍于噪声底），
    V(S1) AUC=1.000（同期对照 0.748/0.824）。

本轮的修正是**任务设计**：让每步的结果在画面上真实可见，且错误变体**不弹窗**：
  S1 select_expr('"area" > 300')  → 真值 selected==2（North, South 两大区）
      错误变体 '"area" <= 300' → 选 3 个小区（无警告框，纯视觉差异）
  S2 Zoom to Selection / 错误变体 Zoom In
  S3 导出 GeoJSON（仅选中）/ 错误变体 全量导出 → 真值 文件里恰好 {North, South}

对比：no-V / learned-V(S1) / full-V(S1+S2+S3) / truth-V（上界）
用法: python experiments/qgis/qgis_bridge/run_w10.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CW, CH = 256, 128
CROP = (340, 120, 1940, 1220)
GEOJSON = HERE / "data" / "sel.geojson"
SCENE = "regions"
RIGHT = {"North", "South"}
Q = 0.5
MAX_RETRY = 3
REF_X0 = None       # 由一次干净参考运行确定


def truth():
    return call({"op": "truth"})["truth"]


def selected():
    t = truth()
    return t["layers"][0]["selected"] if t.get("layers") else None


def extent_x0():
    return truth()["extent"][0]


def grab(path):
    call({"op": "shot", "path": str(path)})
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = CROP
    img = img.crop((x0, y0, min(img.width, x1), min(img.height, y1)))
    return np.asarray(img.resize((CW, CH), Image.BILINEAR))


def do_s1(correct):
    expr = '"area" > 300' if correct else '"area" <= 300'
    return call({"op": "select_expr", "expr": expr, "layer": SCENE})["ok"]


def do_s2(correct):
    if correct:
        return call({"op": "zoom_to_selection"})["ok"]
    return call({"op": "act", "kind": "action", "text": "Zoom In"})["ok"]


def do_s3(correct):
    return call({"op": "export", "path": str(GEOJSON),
                 "only_selected": bool(correct), "layer": SCENE})["ok"]


def artifact_names():
    try:
        d = json.load(open(GEOJSON, encoding="utf-8"))
        return {f["properties"]["name"] for f in d["features"]}
    except Exception:
        return set()


class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(6, 16, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
    def forward(s, x):
        return s.f(x)


def auc_of(prob, y):
    order = np.argsort(prob); ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob)+1)
    pos, neg = y == 1, y == 0
    return float((ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))


def train_v(step, rng, n=100, epochs=40):
    tmp = HERE / f"_w10_{step}.png"
    data, labels, diffs = [], [], [0.0, 0.0]
    for i in range(n):
        call({"op": "reset", "scene": SCENE})
        time.sleep(0.15)
        if step == "s2":
            do_s1(True)
            time.sleep(0.2)
        before = grab(tmp)
        (do_s1 if step == "s1" else do_s2)(i % 2 == 0)
        time.sleep(0.25)
        after = grab(tmp)
        diffs[i % 2] += float(np.abs(after.astype(float)-before.astype(float)).mean())
        data.append(np.concatenate([before, after], -1))
        ok = (selected() == 2) if step == "s1" else (abs(extent_x0()-REF_X0) < 1.0)
        labels.append(int(ok))
    X = torch.tensor(np.stack(data)).permute(0, 3, 1, 2).float()/255.0
    y = torch.tensor(labels)
    perm = rng.permutation(len(labels)); cut = int(len(labels)*0.75)
    tr, te = perm[:cut], perm[cut:]
    torch.manual_seed(0)
    m = V(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            idx = p[i:i+32]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = F.softmax(m(X), -1)[:, 1].numpy()
    auc = auc_of(prob, y.numpy())
    pv = prob[te]; yv = y[te].numpy()
    cand = np.unique(np.concatenate([[0.0, 1.0], (np.sort(pv)[:-1]+np.sort(pv)[1:])/2]))
    bt, bb = 0.5, -1
    for t in cand:
        pr = (pv > t).astype(int)
        tp = ((pr == 1) & (yv == 1)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
        tn = ((pr == 0) & (yv == 0)).sum(); fp = ((pr == 1) & (yv == 0)).sum()
        bal = 0.5*(tp/max(1, tp+fn) + tn/max(1, tn+fp))
        if bal > bb:
            bb, bt = bal, float(t)
    acc = float(((pv > bt).astype(int) == yv).mean())
    print(f"V({step})：AUC={auc:.3f} 校准后 acc={acc:.3f} (阈值 {bt:.2f})  "
          f"视觉信号 diff(ok)={diffs[1]/(n//2):.3f} diff(bad)={diffs[0]/(n//2):.3f}")
    return m, bt


def eval_v(m, thr, before, after):
    x = torch.tensor(np.concatenate([before, after], -1))[None]
    x = x.permute(0, 3, 1, 2).float()/255.0
    with torch.no_grad():
        return float(F.softmax(m(x), -1)[0, 1]) > thr


STEPS = {1: [("s3", do_s3)], 2: [("s1", do_s1), ("s3", do_s3)],
         3: [("s1", do_s1), ("s2", do_s2), ("s3", do_s3)]}


def run_episode(vs, k, mode, rng, tmp):
    call({"op": "reset", "scene": SCENE})
    time.sleep(0.15)
    if k == 1:
        do_s1(True)
        time.sleep(0.2)
    for name, fn in STEPS[k]:
        got = False
        for _ in range(1 + MAX_RETRY):
            correct = rng.random() < Q
            if mode == "truth":
                fn(correct)
                if name == "s1":
                    hit = (selected() == 2)
                elif name == "s2":
                    hit = abs(extent_x0()-REF_X0) < 1.0
                else:
                    hit = artifact_names() == RIGHT
                if hit:
                    got = True; break
                continue
            if mode in ("learned", "full") and name in ("s1", "s2"):
                m, thr = vs[name]
                before = grab(tmp)
                fn(correct)
                time.sleep(0.2)
                after = grab(tmp)
                if eval_v(m, thr, before, after):
                    got = True; break
                continue
            if mode == "full" and name == "s3":
                fn(correct)
                if artifact_names() == RIGHT:
                    got = True; break
                continue
            fn(correct)
            got = True
            break
        if not got:
            return False
    ok = artifact_names() == RIGHT
    if k == 3:
        ok = ok and abs(extent_x0()-REF_X0) < 1.0
    return ok


def main():
    global REF_X0
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})
    tmp = HERE / "_w10_tmp.png"

    call({"op": "reset", "scene": SCENE})
    time.sleep(0.2)
    do_s1(True); time.sleep(0.2); call({"op": "zoom_to_selection"}); time.sleep(0.4)
    REF_X0 = extent_x0()
    full_x0 = truth()["layers"][0]["extent"][0]
    print(f"参考：正确 S1+S2 后 extent_x0 = {REF_X0:.2f}（图层全范围为 {full_x0:.2f}）")
    print(f"q={Q}  MAX_RETRY={MAX_RETRY}  正解产物 = {sorted(RIGHT)}")

    vs = {"s1": train_v("s1", rng), "s2": train_v("s2", rng)}

    print(f"\n{'k':>3}{'no-V':>9}{'learned-V(S1)':>15}{'full-V':>9}{'truth-V':>10}")
    print("-" * 46)
    out = {}
    for k in (1, 2, 3):
        row = {}
        for mode in ("nov", "learned", "full", "truth"):
            T = 12
            hit = sum(int(run_episode(vs, k, mode, rng, tmp)) for _ in range(T))
            row[mode] = hit / T
        out[k] = row
        print(f"{k:>3}{row['nov']:>9.3f}{row['learned']:>15.3f}"
              f"{row['full']:>9.3f}{row['truth']:>10.3f}")
    print("-" * 46)
    print(f"k=3：no-V {out[3]['nov']:.3f} → full-V {out[3]['full']:.3f} "
          f"（truth-V 上界 {out[3]['truth']:.3f}）")

    (HERE / "w10_results.json").write_text(json.dumps(
        {"scene": SCENE, "q": Q, "max_retry": MAX_RETRY, "ref_x0": REF_X0,
         "by_k": {str(k): v for k, v in out.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w10_results.json")


if __name__ == "__main__":
    main()
