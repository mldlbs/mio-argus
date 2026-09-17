"""
P4a：多样性探针。

关键修正
--------
在单步任务上**行为多样性不可能存在**——最优动作唯一（找到那个 label 的格子）。
所以"10 个模型选不同动作"是个伪指标。

真正可能存在的多样性是**失败剖面的差异**：
  precision 专精者在粗元素上不失误、在细元素上失误；
  coarse 专精者反之。
这才可能被 verifier / 集成利用。

因此 P4a 测 **coverage**：

  population coverage@k   = P(真值 ∈ k 个成员 top-1 的并集)
  within-policy cover@k  = P(真值 ∈ 单个成员 top-k)          ← 对照

两者都用 k 个候选。若 population 覆盖更高 → 跨策略多样性确有价值。

用法:
    python p4_population.py --quick
    python p4_population.py --epochs 20
"""
import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from p0_env import EnvCfg, make_sample, instruction_onehot, LABELS
from p0_h2 import cfg_for, to_t
from p1_horizon import train_single_step

GRID = 6
IMG_SIZE = 160
CACHE = Path("p0_cache")


# ---------- 专精数据集（grid 固定为 6，保证动作空间一致 S=36）----------

def build_specialty(n, seed, fills, jitters=(4,), tag="spec"):
    """按 (fill, jitter) 切片生成数据。grid 固定 6 → S 恒为 36。"""
    rng = np.random.default_rng(seed)
    S = GRID * GRID
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    fills_used = np.zeros(n, dtype=np.float32)

    for i in range(n):
        fill = float(rng.choice(fills))
        jit = int(rng.choice(jitters))
        cfg = EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=S,
                     fill=fill, jitter=jit, color_by_label=True)
        s = make_sample(rng, cfg)
        imgs[i] = (s.image * 255).astype(np.uint8)
        inst[i] = instruction_onehot(s.instruction, len(LABELS))
        slots[i] = s.target_slot
        fills_used[i] = fill
    return {"images": imgs, "inst": inst, "slots": slots,
            "xys": np.zeros((n, 2), dtype=np.float32),
            "bbox": np.zeros((n, 4), dtype=np.int32),
            "centers": np.zeros((n, S, 2), dtype=np.float32),
            "_fills": fills_used}


def specialty_data(n, seed, key):
    if key == "mixed":
        return build_specialty(n, seed, [0.28, 0.40, 0.55, 0.75])
    if key == "precise":
        return build_specialty(n, seed, [0.26, 0.30, 0.34])
    if key == "coarse":
        return build_specialty(n, seed, [0.62, 0.70, 0.78])
    if key == "jittered":
        return build_specialty(n, seed, [0.40, 0.55], jitters=(8, 10))
    raise ValueError(key)


SPECIALTIES = ["mixed", "precise", "coarse", "jittered", "mixed"]  # 最后一个用不同 seed


# ---------- 评估 ----------

def topk_preds(model, data, device, k, batch=256):
    """返回 (n, k) 的 top-k cell 预测。"""
    model.eval()
    S = GRID * GRID
    images = to_t(data["images"], device)
    inst = to_t(data["inst"], device)
    out = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            logits = model(images[i:i + batch], inst[i:i + batch])
            o = torch.argsort(-logits, dim=-1)[:, :k].cpu().numpy()
            out.append(o)
    return np.concatenate(out, 0)


def coverage(preds_k, target):
    """preds_k: (n, k) → P(target ∈ top-k)。"""
    return float(np.mean([target[i] in preds_k[i] for i in range(len(target))]))


def union_coverage(top1_list, target):
    """top1_list: [(n,) ...] 每成员 top-1 → 并集覆盖。"""
    n = len(target)
    return float(np.mean([any(t1[i] == target[i] for t1 in top1_list)
                          for i in range(n)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1500)
    args = ap.parse_args()

    epochs = 8 if args.quick else args.epochs
    n_tr = 1200 if args.quick else args.n_train
    n_te = 400 if args.quick else args.n_test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg_for(GRID, fill=0.40)
    print(f"device={device} grid={GRID} S={GRID*GRID} width={args.width} "
          f"epochs={epochs} n_train={n_tr} n_test={n_te}")
    print(f"population = {SPECIALTIES}")

    # 共同混合测试集（含精度分层，便于看失败剖面）
    te = build_specialty(n_te, 4242, [0.28, 0.40, 0.55, 0.75])
    tgt = te["slots"]

    members, names = [], []
    for idx, sp in enumerate(SPECIALTIES):
        seed = 0 if idx < len(SPECIALTIES) - 1 else 1   # 最后一个做 seed 对照
        t0 = time.time()
        tr = specialty_data(n_tr, 100 + idx, sp)
        m, mt = train_single_step(args.width, "cls", "-", tr, te, mcfg,
                                  epochs, device, seed=seed)
        members.append(m); names.append(f"{sp}{'(s1)' if seed == 1 else ''}")
        print(f"  [{names[-1]:<10}] train_hit={mt['hit_rate']:.4f}  "
              f"({time.time()-t0:.0f}s)")

    # ---- 指标 ----
    top1s = [topk_preds(m, te, device, 1)[:, 0] for m in members]
    accs = [float((t1 == tgt).mean()) for t1 in top1s]

    print(f"\n{'member':<12}{'top1 acc':>10}{'err set':>10}")
    print("-" * 32)
    for nm, a, t1 in zip(names, accs, top1s):
        print(f"{nm:<12}{a:>10.4f}{int((t1 != tgt).sum()):>10}")

    # 两两错误集 Jaccard
    print(f"\n两两错误集 Jaccard（越低越互补）")
    print(f"{'pair':<24}{'jaccard':>9}")
    print("-" * 34)
    jac = []
    for i, j in combinations(range(len(members)), 2):
        ei = set(np.where(top1s[i] != tgt)[0])
        ej = set(np.where(top1s[j] != tgt)[0])
        jv = len(ei & ej) / max(1, len(ei | ej))
        jac.append(jv)
        print(f"{names[i]+' vs '+names[j]:<24}{jv:>9.4f}")

    # 两两 top-1 分歧率
    print(f"\n两两 top-1 分歧率（行为多样性，预期很低）")
    dis = []
    for i, j in combinations(range(len(members)), 2):
        d = float((top1s[i] != top1s[j]).mean())
        dis.append(d)
        print(f"{names[i]+' vs '+names[j]:<24}{d:>9.4f}")

    # ---- 核心：coverage@k（对所有 k-子集取平均，避免"只取前 k 个"的不公平）----
    print(f"\n{'k':>3}{'pop cover@k':>13}{'within-mean@k':>15}"
          f"{'within-max@k':>14}{'gap(mean)':>11}{'gap(max)':>10}")
    print("-" * 76)
    cov_rows = []
    for k in range(1, len(members) + 1):
        subs = list(combinations(range(len(members)), k))
        pop = float(np.mean([union_coverage([top1s[i] for i in s], tgt)
                             for s in subs]))
        within_all = [coverage(topk_preds(m, te, device, k), tgt) for m in members]
        within_mean = float(np.mean(within_all))
        within_max = float(np.max(within_all))
        cov_rows.append({"k": k, "pop": pop, "within_mean": within_mean,
                         "within_max": within_max,
                         "gap_mean": pop - within_mean,
                         "gap_max": pop - within_max})
        print(f"{k:>3}{pop:>13.4f}{within_mean:>15.4f}{within_max:>14.4f}"
              f"{pop-within_mean:>+11.4f}{pop-within_max:>+10.4f}")

    # 精度分层：看失败剖面是否真按专精分野
    print(f"\n按元素尺寸分层的 top-1 准确率")
    print(f"{'member':<12}" + "".join(f"fill={f:<7}" for f in [0.28, 0.40, 0.55, 0.75]))
    print("-" * 58)
    layer = {}
    for nm, t1 in zip(names, top1s):
        row = []
        for f in [0.28, 0.40, 0.55, 0.75]:
            mask = te["_fills"] == f
            row.append(float((t1[mask] == tgt[mask]).mean()) if mask.any() else float("nan"))
        layer[nm] = row
        print(f"{nm:<12}" + "".join(f"{v:<12.4f}" for v in row))

    out = {"device": str(device), "grid": GRID, "width": args.width,
           "epochs": epochs, "n_train": n_tr, "n_test": n_te,
           "names": names, "accs": accs,
           "pairwise_jaccard": jac, "pairwise_disagreement": dis,
           "coverage": cov_rows, "layer_acc": layer}
    Path("p4a_population_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p4a_population_results.json")


if __name__ == "__main__":
    main()
