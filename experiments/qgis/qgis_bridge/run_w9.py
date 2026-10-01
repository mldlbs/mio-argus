"""
QGIS W9：依赖式子任务上的 V（P3 形式，正确版）

任务（真依赖）：
  S1 按表达式选择 pop>1e6   → 真值 selected==2
  S2 按选择缩放             → 真值 extent ≈ 选择的 bbox
  S3 导出 GeoJSON（仅选中）  → 真值 文件里恰好 {London, Beijing}
错误会**带下去**：S1 错（选 0 个）→ S3 导出空文件；S3 错（导出全部）→ 5 个要素。

k = 参与依赖链的步数（1/2/3）；每步以 q 概率正确，否则触发错误变体。
模式：no-V（假定成功直接推进）/ learned-V（S1 用截图 verifier）/ truth-V（上界）

指标：任务成功率（最终产物正确）随 k 变化 —— V 的增益应随 k 上升。

用法:
    python experiments/qgis/qgis_bridge/run_w9.py
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
RIGHT = {"London", "Beijing"}
Q = 0.5
MAX_RETRY = 3


def truth():
    return call({"op": "truth"})["truth"]


def selected():
    t = truth()
    return t["layers"][0]["selected"] if t.get("layers") else None


def grab(path):
    call({"op": "shot", "path": str(path)})
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = CROP
    img = img.crop((x0, y0, min(img.width, x1), min(img.height, y1)))
    return np.asarray(img.resize((CW, CH), Image.BILINEAR))


# ---------- 各步的"正确/错误"变体 ----------

def do_s1(correct):
    expr = '"pop" > 1000000' if correct else '"pop" > 50000000'
    return call({"op": "select_expr", "expr": expr})["ok"]


def do_s2(correct):
    if correct:
        return call({"op": "zoom_to_selection"})["ok"]
    return call({"op": "act", "kind": "action", "text": "Zoom Full"})["ok"]


def do_s3(correct):
    r = call({"op": "export", "path": str(GEOJSON), "only_selected": bool(correct)})
    return r["ok"]


def artifact_ok():
    try:
        d = json.load(open(GEOJSON, encoding="utf-8"))
    except Exception:
        return False
    names = {f["properties"]["name"] for f in d["features"]}
    return names == RIGHT


STEPS = {1: [("s3", do_s3)], 2: [("s1", do_s1), ("s3", do_s3)],
         3: [("s1", do_s1), ("s2", do_s2), ("s3", do_s3)]}


class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(6, 16, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
    def forward(s, x):
        return s.f(x)


def train_v_s1(rng, n=100, epochs=40):
    """S1 的 verifier：看 (选择前, 选择后) 截图判断是否真的选中了。"""
    tmp = HERE / "_w9_tmp.png"
    data, labels = [], []
    for i in range(n):
        call({"op": "reset"})
        b = grab(tmp)
        do_s1(i % 2 == 0)
        time.sleep(0.2)
        a = grab(tmp)
        data.append(np.concatenate([b, a], -1))
        labels.append(int(selected() == 2))
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
    # AUC
    order = np.argsort(prob); ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob)+1)
    pos, neg = y.numpy() == 1, y.numpy() == 0
    auc = ((ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())
           if pos.sum() and neg.sum() else float("nan"))
    # 阈值校准
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
    print(f"V(S1)：AUC={auc:.3f} 校准后 acc={acc:.3f} (阈值 {bt:.2f}, n={len(labels)})")
    return m, bt


def run_episode(m, thr, k, mode, rng, tmp):
    call({"op": "reset"})
    if k == 1:                     # k=1 时预置正确选择，让 S3 单独决定产物
        do_s1(True)
    for name, fn in STEPS[k]:
        got = False
        for _ in range(1 + MAX_RETRY):
            correct = rng.random() < Q
            if mode == "learned" and name == "s1":
                before = grab(tmp)
                fn(correct)
                time.sleep(0.15)
                after = grab(tmp)
                x = torch.tensor(np.concatenate([before, after], -1))[None]
                x = x.permute(0, 3, 1, 2).float()/255.0
                with torch.no_grad():
                    p = float(F.softmax(m(x), -1)[0, 1])
                if p > thr:
                    got = True; break
                continue
            if mode == "truth":
                fn(correct)
                hit = (selected() == 2) if name == "s1" else True
                if name == "s2":
                    e = truth()["extent"]; hit = abs(e[0]-12.54) < 1.5
                if name == "s3":
                    hit = artifact_ok()
                if hit:
                    got = True; break
                continue
            # no-V / learned 的其它步：假定成功
            fn(correct)
            got = True
            break
        if not got:
            return False
    # 最终判据：产物正确（k=3 还要求视图确实缩放到选择范围）
    ok = artifact_ok()
    if k == 3:
        e = truth()["extent"]
        ok = ok and abs(e[0]-12.54) < 1.5
    return ok


def main():
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})
    tmp = HERE / "_w9_tmp.png"
    print(f"q={Q}  MAX_RETRY={MAX_RETRY}  正解产物 = {sorted(RIGHT)}")
    m, thr = train_v_s1(rng)

    print(f"\n{'k':>3}{'no-V':>9}{'learned-V':>11}{'truth-V':>10}")
    print("-" * 34)
    out = {}
    for k in (1, 2, 3):
        row = {}
        for mode in ("nov", "learned", "truth"):
            T = 12
            hit = 0
            for _ in range(T):
                hit += int(run_episode(m, thr, k, mode, rng, tmp))
            row[mode] = hit / T
        out[k] = row
        print(f"{k:>3}{row['nov']:>9.3f}{row['learned']:>11.3f}{row['truth']:>10.3f}")
    print("-" * 34)
    g1 = out[1]["learned"] - out[1]["nov"]
    g3 = out[3]["learned"] - out[3]["nov"]
    print(f"learned-V 相对 no-V 增益：k=1 {g1:+.3f} → k=3 {g3:+.3f}"
          f"（随 k 上升 = P3 的 crossing 形式）")

    (HERE / "w9_results.json").write_text(json.dumps(
        {"q": Q, "max_retry": MAX_RETRY, "by_k": {str(k): v for k, v in out.items()},
         "gain_k1": g1, "gain_k3": g3}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w9_results.json")


if __name__ == "__main__":
    main()
