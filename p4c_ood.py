"""
P4c-OOD：P4a' 的优势能否迁移到**从未见过的 domain**？

背景
----
P4a' 显示：在 D 个 domain（各自独立的 label→palette 置换）上，
population(专精者) + router 在参数匹配下大幅胜过通才（0.6887 vs 0.3273）。

但这可能只是"记住了 6 张表"，而非可迁移能力。
本实验把 domain 置换参数化，**留出从未参与训练的置换**：

  TRAIN:  domain 0..5
  OOD:    domain 6..9   ← 从未见过

四格对比：
                    seen domain      OOD domain
  generalist          ?                ?
  population+routed   ?                ?

预测（需实测）：OOD 上双方都掉到随机（1/S）。
若如此 → P4a' 的优势**不可迁移**，只是查表记忆。

用法:
    python p4c_ood.py
"""
import json
from pathlib import Path

import numpy as np

import p4b_capacity as B

GRID = 6
B.GRID = GRID
B.N_DOMAINS = 6            # router 只认训练域

N_TRAIN_D = 6
N_OOD_D = 4
EPOCHS = 25
N_TR = 6000
N_TE = 1200


def perms(n, seed):
    rng = np.random.default_rng(seed)
    V = GRID * GRID
    P = len(B.PALETTE)
    return [rng.permutation(P)[:V] for _ in range(n)]


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)

    all_perms = perms(N_TRAIN_D + N_OOD_D, seed=1234)
    train_d = list(range(N_TRAIN_D))
    ood_d = list(range(N_TRAIN_D, N_TRAIN_D + N_OOD_D))

    print(f"device={device} grid={GRID} S={GRID*GRID} "
          f"train_domains={train_d} ood_domains={ood_d}")
    print(f"n_train={N_TR} n_test={N_TE} epochs={EPOCHS}")
    print(f"chance = {1.0/(GRID*GRID):.4f}\n")

    # ---- 数据 ----
    tr = B.build_data(N_TR, 100, train_d, all_perms)
    te_seen = B.build_data(N_TE, 777, train_d, all_perms)
    te_ood = B.build_data(N_TE, 778, ood_d, all_perms)

    # ---- 通才 ----
    gens = {}
    for w in (8, 16, 32):
        gens[w] = B.train(w, B.build_data(N_TR, 100, train_d, all_perms),
                          EPOCHS, device)
    # ---- 专精者（每个训练域一个）----
    members = []
    for d in train_d:
        dtr = B.build_data(N_TR // N_TRAIN_D, 200 + d, [d], all_perms)
        members.append(B.train(12, dtr, EPOCHS, device, seed=d))
    # ---- router ----
    R = B.train_router(8, tr, EPOCHS // 2, device)

    def evaluate(data, name):
        tgt = data["slots"]
        preds = B.predict_all(members, data, device)
        rd_raw, racc = B.router_acc(R, data, device)
        rd = np.minimum(rd_raw, len(members) - 1)
        routed = float(np.mean([preds[rd[i], i] == tgt[i]
                                for i in range(len(tgt))]))
        o = B.oracle(preds, tgt)
        v = B.vote(preds, tgt)
        row = {"set": name, "router_acc": racc, "oracle": o, "vote": v,
               "routed": routed}
        for w in gens:
            g = B.predict_all([gens[w]], data, device)
            row[f"gen_w{w}"] = B.acc1(g[0], tgt)
        print(f"[{name}] router_acc={racc:.4f} oracle={o:.4f} "
              f"routed={routed:.4f} vote={v:.4f} | " +
              " ".join(f"gen_w{w}={row[f'gen_w{w}']:.4f}" for w in gens))
        return row

    print("=" * 92)
    seen = evaluate(te_seen, "seen  ")
    ood = evaluate(te_ood, "OOD   ")
    print("=" * 92)

    best_gen_seen = max(seen[f"gen_w{w}"] for w in gens)
    best_gen_ood = max(ood[f"gen_w{w}"] for w in gens)
    print(f"\n{'':<22}{'seen':>10}{'OOD':>10}{'drop':>10}")
    print("-" * 52)
    print(f"{'best generalist':<22}{best_gen_seen:>10.4f}{best_gen_ood:>10.4f}"
          f"{best_gen_seen-best_gen_ood:>+10.4f}")
    print(f"{'population + routed':<22}{seen['routed']:>10.4f}"
          f"{ood['routed']:>10.4f}{seen['routed']-ood['routed']:>+10.4f}")
    print(f"{'oracle (upper bound)':<22}{seen['oracle']:>10.4f}"
          f"{ood['oracle']:>10.4f}{seen['oracle']-ood['oracle']:>+10.4f}")
    print(f"\nrandom baseline = {1.0/(GRID*GRID):.4f}")

    out = {"device": str(device), "grid": GRID, "chance": 1.0 / (GRID * GRID),
           "train_domains": train_d, "ood_domains": ood_d,
           "epochs": EPOCHS, "n_train": N_TR, "n_test": N_TE,
           "seen": seen, "ood": ood}
    Path("p4c_ood_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p4c_ood_results.json")


if __name__ == "__main__":
    main()
