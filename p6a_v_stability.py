"""
P6a：诊断并修复 OutcomeVerifier 的训练不稳定。

已知缺陷（复现两次）
--------------------
P3 完整运行 3 次中 1 次、P6 quick 运行中 1 次，V 退化为**恒输出正类**：
    tpr = 1.0, fpr = 1.0, acc ≈ 正样本比例
此时 π+V 会崩到 0（因为 V 接受了所有错误动作）。

要区分两种病因
--------------
  A. **分数可分但阈值偏了** → 校准（选阈值）即可修复
  B. **根本没学到判别** → 需要改训练

诊断量：
  - AUC（阈值无关的可分性）
  - 正/负样本的分数均值差
  - 不同阈值下的最优准确率

用法: python p6a_v_stability.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import p3_verifier as P3

cfg = P3.cfg_p3()


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(len(scores))
    n1 = float(labels.sum())
    n0 = float(len(labels) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n1 * (n1 - 1) / 2) / (n0 * n1)


def train_logged(width, tr, epochs, device, lr=2e-3, batch=64, seed=0,
                 log=True):
    torch.manual_seed(seed)
    v = P3.OutcomeVerifier(width=width).to(device)
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    B = P3.to_t(tr["before"], device)
    A = P3.to_t(tr["after"], device)
    y = torch.from_numpy(tr["label"]).to(device)
    n = B.shape[0]
    spe = max(1, n // batch)
    hist = []
    v.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            logit = v(B[idx], A[idx])
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        # 记录正负分数均值
        v.eval()
        with torch.no_grad():
            sc = torch.sigmoid(v(B, A)).cpu().numpy()
        v.train()
        mp = float(sc[y.cpu().numpy() == 1].mean())
        mn = float(sc[y.cpu().numpy() == 0].mean())
        hist.append({"ep": ep + 1, "loss": tot / spe, "mean_pos": mp,
                     "mean_neg": mn, "gap": mp - mn})
        if log and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    ep{ep+1:>3} loss={tot/spe:.4f} pos={mp:.3f} "
                  f"neg={mn:.3f} gap={mp-mn:+.3f}")
    return v, hist


def best_threshold(scores, labels):
    """
    精确最优阈值：对准确率而言，最优阈值必落在排序分数的相邻两点之间。
    不能用 0→1 的均匀网格——分数尺度会漂移，网格会漏掉真正的最优点。
    """
    s = np.sort(np.unique(scores))
    if len(s) == 1:
        return float(s[0]), float((labels == (scores > s[0])).mean())
    cands = (s[:-1] + s[1:]) / 2.0
    best_t, best_acc = float(cands[0]), -1.0
    for t in cands:
        acc = float(((scores > t).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def evaluate(v, data, device):
    v.eval()
    B = P3.to_t(data["before"], device)
    A = P3.to_t(data["after"], device)
    with torch.no_grad():
        sc = torch.sigmoid(v(B, A)).cpu().numpy()
    y = data["label"].astype(int)
    a = auc(sc, y)
    p05 = (sc > 0.5).astype(int)
    acc05 = float((p05 == y).mean())
    best_t, best_acc = best_threshold(sc, y)
    return {"auc": float(a), "acc@0.5": acc05,
            "acc@best": best_acc, "best_thr": best_t,
            "score_max": float(sc.max()), "score_min": float(sc.min()),
            "pred_pos_rate@0.5": float(p05.mean()),
            "true_pos_rate": float(y.mean())}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n")
    vtr = P3.build_v_data(4000, 1, cfg)
    vte = P3.build_v_data(1000, 998, cfg)
    print(f"v-data: train={len(vtr['label'])} (pos {vtr['label'].mean():.3f})  "
          f"test={len(vte['label'])} (pos {vte['label'].mean():.3f})\n")

    variants = [
        ("baseline  lr=2e-3 ep=15", dict(lr=2e-3, epochs=15)),
        ("low-lr    lr=5e-4 ep=15", dict(lr=5e-4, epochs=15)),
        ("long      lr=2e-3 ep=40", dict(lr=2e-3, epochs=40)),
    ]
    results = []
    for name, kw in variants:
        print(f"### {name}")
        v, hist = train_logged(8, vtr, kw["epochs"], device, lr=kw["lr"])
        m = evaluate(v, vte, device)
        results.append({"name": name, **m,
                        "final_gap": hist[-1]["gap"],
                        "final_loss": hist[-1]["loss"]})
        print(f"    AUC={m['auc']:.4f}  acc@0.5={m['acc@0.5']:.4f}  "
              f"acc@best={m['acc@best']:.4f} (thr={m['best_thr']:.2f})  "
              f"pred_pos@0.5={m['pred_pos_rate@0.5']:.3f} "
              f"(true={m['true_pos_rate']:.3f})\n")

    # 多次运行看稳定性
    print("### 稳定性：同配置跑 6 次（seed 0..5，记录 AUC / acc@0.5）")
    aucs, accs05, accsbest = [], [], []
    for s in range(6):
        v, _ = train_logged(8, vtr, 15, device, lr=2e-3, seed=s, log=False)
        m = evaluate(v, vte, device)
        aucs.append(m["auc"]); accs05.append(m["acc@0.5"])
        accsbest.append(m["acc@best"])
        print(f"    seed={s}  AUC={m['auc']:.4f}  acc@0.5={m['acc@0.5']:.4f}  "
              f"acc@best={m['acc@best']:.4f}")
    print(f"\n    AUC        min={min(aucs):.4f} max={max(aucs):.4f}")
    print(f"    acc@0.5    min={min(accs05):.4f} max={max(accs05):.4f}")
    print(f"    acc@best   min={min(accsbest):.4f} max={max(accsbest):.4f}")

    out = {"device": str(device), "variants": results,
           "stability": {"auc": aucs, "acc@0.5": accs05, "acc@best": accsbest}}

    # ---- 修复验证：校准 + 失败重试 ----
    print("\n### 修复后：train_v_calibrated（验证集校准阈值 + 低 AUC 重试）")
    val = P3.build_v_data(1000, 555, cfg)          # 校准用验证集
    print(f"{'seed':>5}{'attempts':>10}{'val_auc':>10}{'thr':>8}"
          f"{'test@thr':>11}{'test@0.5':>11}")
    print("-" * 55)
    cal_rows = []
    for s in range(6):
        v, rec = P3.train_v_calibrated(8, vtr, val, 15, device, seed=s)
        thr = rec["thr"]
        sc = P3.v_scores_all(v, vte, device)
        y = vte["label"].astype(int)
        test_thr = float(((sc > thr).astype(int) == y).mean())
        test_05 = float(((sc > 0.5).astype(int) == y).mean())
        cal_rows.append({"seed": s, "attempts": rec["attempt"] + 1,
                         "val_auc": rec["val_auc"], "thr": thr,
                         "test@thr": test_thr, "test@0.5": test_05})
        print(f"{s:>5}{rec['attempt']+1:>10}{rec['val_auc']:>10.4f}"
              f"{thr:>8.3f}{test_thr:>11.4f}{test_05:>11.4f}")
    t_thr = [r["test@thr"] for r in cal_rows]
    t_05 = [r["test@0.5"] for r in cal_rows]
    print(f"\n    校准后 test acc: min={min(t_thr):.4f} max={max(t_thr):.4f}")
    print(f"    固定 0.5 test acc: min={min(t_05):.4f} max={max(t_05):.4f}")
    out["calibrated"] = cal_rows
    Path("p6a_v_stability_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p6a_v_stability_results.json")


if __name__ == "__main__":
    main()
