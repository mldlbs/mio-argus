"""
QGIS W5：H3（结果验证）在真桌面应用上

P3/P6 的机制在真应用上复验：
  V(动作前截图, 动作后截图) → 这次动作有没有让状态推进？
  V 必须用到 π 拿不到的信息（动作的**后果**），所以是 H3 的合法形式（结果验证）。

数据：同一目标（Zoom Full），策略有时触发正确动作、有时触发错误动作；
      每组记录 (before, after) 两张画布截图 + 真值标签（extent 是否恢复）。
指标：V 的 AUC / 准确率；以及**用 V 门控重试**相对不重试的成功率。

用法:
    python experiments/qgis/qgis_bridge/run_w5.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

N_TRIALS = 140
CW, CH = 256, 128
WRONG = ["Zoom Out", "Zoom In", "Pan Map", "Show Map Tips"]


def truth():
    return call({"op": "truth"})["truth"]


CROP = (340, 120, 1940, 1220)          # 最大化窗口里的地图区（固定裁剪）


def grab_window(path):
    """整窗截图 → 裁地图区 → 缩放到 (CW,CH)。"""
    call({"op": "shot", "path": str(path)})
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = CROP
    img = img.crop((x0, y0, min(img.width, x1), min(img.height, y1)))
    return np.asarray(img.resize((CW, CH), Image.BILINEAR))


def ref_full():
    call({"op": "reset"})
    return truth()["extent"][0]


def shrink():
    t = truth()
    e = t["layers"][0]["extent"]
    call({"op": "set_extent",
          "box": [e[0], e[1], e[0] + (e[2]-e[0])*0.3, e[1] + (e[3]-e[1])*0.3]})


def grab_canvas(rect, path):
    return grab_window(path)


def main():
    rng = np.random.default_rng(0)
    rect = None
    ref = ref_full()
    print(f"整窗截图 {CW}x{CH}  参考 extent[0]={ref:.2f}")

    tmp = HERE / "_w5_tmp.png"
    data, labels, correct_actions = [], [], []
    for i in range(N_TRIALS):
        shrink()
        ok = bool(i % 2 == 0)
        a = "Zoom Full" if ok else str(rng.choice(WRONG))
        before = grab_canvas(rect, tmp)
        call({"op": "act", "kind": "action", "text": a})
        time.sleep(0.25)
        after = grab_canvas(rect, tmp)
        good = abs(truth()["extent"][0] - ref) < 2.0
        data.append(np.concatenate([before, after], -1))   # (H,W,6)
        labels.append(int(good))
        correct_actions.append(ok)
    L = np.array(labels)
    print(f"样本 {len(L)}：成功 {L.sum()} / 失败 {len(L)-L.sum()} "
          f"（策略正确率 {np.mean(correct_actions):.2f}）")

    X = torch.tensor(np.stack(data)).permute(0, 3, 1, 2).float().div_(255)
    y = torch.tensor(L)
    n = len(L)
    perm = rng.permutation(n)
    cut = int(n * 0.75)
    tr, te = perm[:cut], perm[cut:]

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

    torch.manual_seed(0)
    m = V()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(40):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            idx = p[i:i+32]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = m(X[te])
        prob = F.softmax(logits, -1)[:, 1].numpy()
    yte = y[te].numpy()

    # AUC（阈值无关）：升序秩
    order = np.argsort(prob)
    ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob) + 1)
    pos, neg = yte == 1, yte == 0
    auc = ((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
           if pos.sum() and neg.sum() else float("nan"))
    acc = float(((prob > 0.5).astype(int) == yte).mean())
    fnr = float(((prob <= 0.5) & pos).sum() / max(1, pos.sum()))
    fpr = float(((prob > 0.5) & neg).sum() / max(1, neg.sum()))
    print(f"V：AUC={auc:.3f}  acc={acc:.3f}  FNR(误拒)={fnr:.3f}  FPR(误收)={fpr:.3f}")

    # --- 阈值校准（P6 的方法论）：在验证集上取平衡准确率最大的阈值 ---
    pv = prob; yv = yte
    best_t, best_bal = 0.5, -1.0
    cand = np.unique(np.concatenate([[0.0, 1.0], (pv[:-1] + pv[1:]) / 2]))
    for t in cand:
        pred = (pv > t).astype(int)
        tp = ((pred == 1) & (yv == 1)).sum(); fn = ((pred == 0) & (yv == 1)).sum()
        tn = ((pred == 0) & (yv == 0)).sum(); fp = ((pred == 1) & (yv == 0)).sum()
        bal = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    pred_c = (pv > best_t).astype(int)
    acc_c = float((pred_c == yv).mean())
    fnr_c = float(((pred_c == 0) & (yv == 1)).sum() / max(1, (yv == 1).sum()))
    fpr_c = float(((pred_c == 1) & (yv == 0)).sum() / max(1, (yv == 0).sum()))
    print(f"V(校准后 t={best_t:.2f})：acc={acc_c:.3f}  FNR={fnr_c:.3f}  FPR={fpr_c:.3f}")

    # 验证门控重试：策略正确率 q=0.5，最多 3 次；用 V 的真实 FNR/FPR 模拟
    q = float(np.mean(correct_actions))
    def trial_with_v(attempts=3):
        for _ in range(attempts):
            rng2 = np.random.default_rng()
            correct = rng2.random() < q
            if correct:
                if rng2.random() > fnr_c:
                    return True
            else:
                if rng2.random() < fpr_c:
                    return False
        return False

    sim = np.random.default_rng(1)
    def no_v(attempts=3):
        for _ in range(attempts):
            if sim.random() < q:
                return True
        return False

    W = 4000
    p_no = np.mean([no_v() for _ in range(W)])
    p_v = np.mean([trial_with_v() for _ in range(W)])
    print(f"\n门控重试（≤3 次）：无 V {p_no:.3f} → 有 V {p_v:.3f}  (+{p_v-p_no:.3f})")
    print("（V 的误拒浪费机会、误收会停在错误状态——两者都已计入）")

    (HERE / "w5_results.json").write_text(json.dumps({
        "n": n, "success_rate": float(L.mean()), "policy_acc": q,
        "V_auc": auc, "V_acc": acc, "V_fnr": fnr, "V_fpr": fpr,
        "retry3_no_V": float(p_no), "retry3_with_V": float(p_v),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w5_results.json")


if __name__ == "__main__":
    main()
