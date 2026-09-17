"""
诚信核查：P0 扫描 B 用的是 gauss 监督的 reg，可能低估了 reg 基线。
这里在同一环境、同一参数量下，把 cls / reg-coord / reg-gauss 三方对齐比较。

用法: python p0_regloss.py --epochs 25 --seeds 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p0_env import EnvCfg, LABELS
from p0_h2 import (cfg_for, load_or_build, train_one, precision_demand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    grids = [3, 6] if args.quick else [3, 4, 5, 6, 8]
    epochs, seeds = (6, 1) if args.quick else (args.epochs, args.seeds)
    n_tr, n_te = (1200, 400) if args.quick else (4000, 1000)
    width = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} width={width} train={n_tr} test={n_te} "
          f"epochs={epochs} seeds={seeds}")
    tag = "q" if args.quick else "m"

    rows = []
    for g in grids:
        cfg = cfg_for(g, fill=0.40)
        tr = load_or_build(n_tr, 0, cfg, f"tr{tag}")
        te = load_or_build(n_te, 999, cfg, f"te{tag}")
        print(f"\n[grid={g} slots={cfg.num_slots} precision={precision_demand(cfg):.4f}]")
        for head, rl in [("cls", "-"), ("reg", "coord"), ("reg", "gauss")]:
            ms = [train_one(width, head, tr, te, cfg, epochs, device,
                            seed=s, reg_loss=rl) for s in range(seeds)]
            hit = float(np.mean([m["hit_rate"] for m in ms]))
            cell = float(np.mean([m["cell_acc"] for m in ms]))
            sd = float(np.std([m["hit_rate"] for m in ms]))
            rows.append({"grid": g, "slots": cfg.num_slots, "head": head,
                         "reg_loss": rl, "hit_rate": hit, "cell_acc": cell,
                         "hit_sd": sd, "params": ms[0]["params"]})
            print(f"  {head:>4}/{rl:<6} hit={hit:.4f}±{sd:.4f} cell={cell:.4f}")
        del tr, te

    print(f"\n{'-'*88}")
    print(f"{'grid':>5}{'slots':>7}{'cls':>9}{'reg coord':>12}{'reg gauss':>12}"
          f"{'Δ(coord)':>11}{'Δ(gauss)':>11}")
    print("-" * 88)
    for g in grids:
        def gv(h, rl):
            return next(r for r in rows if r["grid"] == g and r["head"] == h
                        and r["reg_loss"] == rl)
        c, rc, rg = gv("cls", "-"), gv("reg", "coord"), gv("reg", "gauss")
        print(f"{g:>5}{c['slots']:>7}{c['hit_rate']:>9.4f}"
              f"{rc['hit_rate']:>12.4f}{rg['hit_rate']:>12.4f}"
              f"{c['hit_rate']-rc['hit_rate']:>+11.4f}"
              f"{c['hit_rate']-rg['hit_rate']:>+11.4f}")
    print("-" * 88)

    Path("p0_regloss_results.json").write_text(
        json.dumps({"device": str(device), "epochs": epochs, "seeds": seeds,
                    "n_train": n_tr, "n_test": n_te, "width": width,
                    "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p0_regloss_results.json")


if __name__ == "__main__":
    main()
