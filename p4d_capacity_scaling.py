"""
P4d：容量扫描 —— population 到底是不是"必要"，还是只是"容量不足的变通"？

P4a' 的事实
-----------
在 D=6 个互不相关 domain 的任务上：
    通才 width 8/16/32  →  0.3140 / 0.2973 / 0.3273（参数 9K/28K/100K）
    population(6×w12) + router, 109K 参数  →  0.6887

通才被卡在 ~0.33，而且加 10× 参数几乎不涨。**但"几乎不涨"≠"不会涨"。**

本实验
------
把通才宽度一直扫上去（直到参数量远超 population 总参数），回答：

  Q: 通才需要多少参数才能覆盖这 6 个 domain？
     - 若在某处追上 0.6887 → population **不必要**，只是容量变通
     - 若扫到 10× population 总参数仍追不上 → 专精确有结构性优势
       （但已被 P4c 证明不迁移）

同时跑一个"更多数据"的对照，排除数据量不足的干扰。

用法:
    python p4d_capacity_scaling.py
"""
import json
from pathlib import Path

import numpy as np

import p4b_capacity as B

GRID = 6
B.GRID = GRID
B.N_DOMAINS = 6
EPOCHS = 25

WIDTHS = [16, 24, 32, 48, 64, 96, 128]
N_TR_BASE = 6000
N_TR_BIG = 24000          # 大通才的数据对照
N_TE = 1200


def perms(n, seed):
    rng = np.random.default_rng(seed)
    V = GRID * GRID
    P = len(B.PALETTE)
    return [rng.permutation(P)[:V] for _ in range(n)]


def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    doms = list(range(B.N_DOMAINS))
    P = perms(B.N_DOMAINS, seed=1234)

    print(f"device={device} grid={GRID} S={GRID*GRID} domains={doms}")
    print(f"epochs={EPOCHS}  chance={1.0/(GRID*GRID):.4f}")
    print(f"\n参照线：population(6×w12)+router = 0.6887 @ 109,350 参数 "
          f"(见 P4B_RESULTS.md)")

    te = B.build_data(N_TE, 777, doms, P)

    # ---- 容量扫描（固定数据量）----
    print(f"\n{'width':>6}{'params':>12}{'acc':>9}{'per-domain':>40}")
    print("-" * 68)
    rows = []
    for w in WIDTHS:
        tr = B.build_data(N_TR_BASE, 100, doms, P)
        m = B.train(w, tr, EPOCHS, device)
        preds = B.predict_all([m], te, device)
        a = B.acc1(preds[0], te["slots"])
        per = [float((preds[0][te["dom"] == d] == te["slots"][te["dom"] == d]).mean())
               for d in doms]
        pp = B.params_of(w)
        rows.append({"width": w, "params": pp, "acc": a, "per_domain": per,
                     "n_train": N_TR_BASE})
        print(f"{w:>6}{pp:>12,}{a:>9.4f}   " + " ".join(f"{v:.3f}" for v in per))

    # ---- 大模型 + 更多数据（排除数据量干扰）----
    print(f"\n大通才 + 更多数据 (n_train={N_TR_BIG})：")
    print(f"{'width':>6}{'params':>12}{'acc':>9}")
    print("-" * 28)
    big_rows = []
    tr_big = B.build_data(N_TR_BIG, 101, doms, P)
    for w in (32, 64, 128):
        m = B.train(w, tr_big, EPOCHS, device, batch=128)
        preds = B.predict_all([m], te, device)
        a = B.acc1(preds[0], te["slots"])
        pp = B.params_of(w)
        big_rows.append({"width": w, "params": pp, "acc": a,
                         "n_train": N_TR_BIG})
        print(f"{w:>6}{pp:>12,}{a:>9.4f}")

    # ---- 判定 ----
    pop_ref, pop_params = 0.6887, 109350
    best = max(rows + big_rows, key=lambda r: r["acc"])
    print(f"\n{'='*68}")
    print(f"population 参照      : acc={pop_ref:.4f}  params={pop_params:,}")
    print(f"最强通才             : acc={best['acc']:.4f}  "
          f"params={best['params']:,}  (width {best['width']}, "
          f"n_train={best['n_train']})")
    ratio = best["params"] / pop_params
    print(f"参数比               : {ratio:.1f}×")
    if best["acc"] >= pop_ref:
        print(f"判定: 通才在 {ratio:.1f}× 参数下追上 population → "
              f"**population 不必要，只是容量变通**")
    else:
        print(f"判定: 通才在 {ratio:.1f}× 参数下仍未追上 → "
              f"专精有结构性优势（但已被 P4c 证明不迁移）")
    print("=" * 68)

    out = {"device": str(device), "grid": GRID, "epochs": EPOCHS,
           "n_test": N_TE, "population_ref": {"acc": pop_ref,
                                              "params": pop_params},
           "sweep": rows, "big": big_rows}
    Path("p4d_scaling_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p4d_scaling_results.json")


if __name__ == "__main__":
    main()
