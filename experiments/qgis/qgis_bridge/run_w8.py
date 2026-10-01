"""
QGIS W8：长 horizon + 步数成本 —— 让 V 的门控真正兑现（P3 形式）

任务：连续 k 个"把视图复原"的子目标。每次尝试：
  - 以概率 q 触发正确动作（Zoom Full），否则触发错误动作
  - 关键：**子目标是否达成，agent 看不到真值**，只能靠 V 看截图判断
    · no-V    ：假定成功，直接进入下一个子目标 → 错误会**带下去**（误差复利）
    · learned-V：用 (动作前,动作后) 截图训练的 verifier 判断 → 失败就重试
    · oracle-V ：直接读真值（上界）

指标：任务成功率（k 个子目标全部达成）随 k 的变化；步数消耗。

用法:
    python experiments/qgis/qgis_bridge/run_w8.py
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
WRONG = ["Zoom Out", "Zoom In", "Pan Map", "Zoom Next", "Zoom Last"]
Q = 0.5
MAX_RETRY = 4


def truth():
    return call({"op": "truth"})["truth"]


def ref_full():
    call({"op": "reset"})
    return truth()["extent"][0]


REF = None


def is_full():
    return abs(truth()["extent"][0] - REF) < 2.0


def shrink():
    t = truth()
    e = t["layers"][0]["extent"]
    call({"op": "set_extent",
          "box": [e[0], e[1], e[0] + (e[2]-e[0])*0.3, e[1] + (e[3]-e[1])*0.3]})


def grab(path):
    call({"op": "shot", "path": str(path)})
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = CROP
    img = img.crop((x0, y0, min(img.width, x1), min(img.height, y1)))
    return np.asarray(img.resize((CW, CH), Image.BILINEAR))


class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(
            nn.Conv2d(6, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))

    def forward(s, x):
        return s.f(x)


def train_v(rng, n=120, epochs=40):
    tmp = HERE / "_w8_tmp.png"
    data, labels = [], []
    for i in range(n):
        shrink()
        wrong = (i % 2 == 1)
        a = str(rng.choice(WRONG)) if wrong else "Zoom Full"
        b = grab(tmp)
        call({"op": "act", "kind": "action", "text": a})
        time.sleep(0.2)
        af = grab(tmp)
        data.append(np.concatenate([b, af], -1))
        labels.append(int(is_full()))
    X = torch.tensor(np.stack(data)).permute(0, 3, 1, 2).float()/255.0
    y = torch.tensor(labels)
    perm = rng.permutation(len(labels))
    cut = int(len(labels)*0.75)
    tr, te = perm[:cut], perm[cut:]
    torch.manual_seed(0)
    m = V()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            idx = p[i:i+32]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = F.softmax(m(X), -1)[:, 1].numpy()
    # 阈值校准（P6）
    pv = prob[te]
    best_t, best_bal = 0.5, -1
    for t in np.unique(np.concatenate([[0.0, 1.0], (np.sort(pv)[:-1]+np.sort(pv)[1:])/2])):
        pr = (pv > t).astype(int)
        tp = ((pr == 1) & (y[te].numpy() == 1)).sum(); fn = ((pr == 0) & (y[te].numpy() == 1)).sum()
        tn = ((pr == 0) & (y[te].numpy() == 0)).sum(); fp = ((pr == 1) & (y[te].numpy() == 0)).sum()
        bal = 0.5*(tp/max(1, tp+fn) + tn/max(1, tn+fp))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    pr = (pv > best_t).astype(int)
    acc = float((pr == y[te].numpy()).mean())
    print(f"V 训练：阈值 t={best_t:.2f}  验证 acc={acc:.3f}  "
          f"(n={len(labels)}, 正例 {sum(labels)})")
    return m, best_t


def run_episode(m, thr, k, mode, rng, tmp):
    """k 个子目标；返回 (成功, 步数)。"""
    for gi in range(k):
        shrink()
        got = False
        for attempt in range(1 + MAX_RETRY):
            ok = rng.random() < Q
            a = "Zoom Full" if ok else str(rng.choice(WRONG))
            before = grab(tmp) if mode == "learned" else None
            call({"op": "act", "kind": "action", "text": a})
            if mode == "oracle":
                if is_full():
                    got = True
                    break
                continue
            if mode == "learned":
                time.sleep(0.15)
                after = grab(tmp)
                x = torch.tensor(np.concatenate([before, after], -1))[None]
                x = x.permute(0, 3, 1, 2).float()/255.0
                with torch.no_grad():
                    p = float(F.softmax(m(x), -1)[0, 1])
                if p > thr:
                    got = True
                    break
                continue
            # no-V：假定成功，直接进入下一个子目标
            got = True
            break
        if not got:
            return False, 0
    return is_full(), 0


def main():
    global REF
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})
    REF = ref_full()
    print(f"参考 extent[0]={REF:.2f}；q={Q}，MAX_RETRY={MAX_RETRY}")
    tmp = HERE / "_w8_tmp.png"
    m, thr = train_v(rng)

    print(f"\n{'k':>3}{'no-V':>9}{'learned-V':>11}{'oracle-V':>10}")
    print("-" * 34)
    out = {}
    for k in (1, 2, 4):
        row = {}
        for mode in ("nov", "learned", "oracle"):
            hit = 0
            T = 24
            for _ in range(T):
                ok, _ = run_episode(m, thr, k, mode, rng, tmp)
                hit += int(ok)
            row[mode] = hit / T
        out[k] = row
        print(f"{k:>3}{row['nov']:>9.3f}{row['learned']:>11.3f}{row['oracle']:>10.3f}")
    print("-" * 34)
    g1 = out[4]["learned"] - out[4]["nov"]
    g0 = out[1]["learned"] - out[1]["nov"]
    print(f"learned-V 相对 no-V 的增益：k=1 {g0:+.3f} → k=4 {g1:+.3f}"
          f"（应随 k 上升 = P3 的 crossing 形式）")

    (HERE / "w8_results.json").write_text(json.dumps(
        {"q": Q, "max_retry": MAX_RETRY, "v_threshold": thr,
         "by_k": {str(k): v for k, v in out.items()},
         "gain_k1": g0, "gain_k4": g1}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w8_results.json")


if __name__ == "__main__":
    main()
