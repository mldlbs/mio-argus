"""E1-L · M1b 机制诊断（A=progress0，B=混合 progress；2 次训练）。

协议：`results/paper/E1L_M1B_PREREG.md` v1.0
用法（需批准后）：
    python experiments/paper/e1l_m1b_diag.py --tasks 500 --seed 0 --rho 0.3
"""
import argparse
import json
from pathlib import Path

import numpy as np

import e1l_m1_prescan as M

W = 24
D = 1.0
HS = [1, 2, 4, 8]
ARMS = [("A", "zero"), ("B", "mixed")]


def bucket_progress(p):
    if p == 0:
        return "0"
    if p <= 3:
        return "1-3"
    return "4-7"


def mark_state(wrong, done):
    if any(wrong):
        return "wrong"
    if any(done):
        return "done_only"
    return "none"


def aggregate(policy, cfg, scripts, H, rho, seed):
    acc_t = [{"n": 0, "ok": 0} for _ in range(M.H_MAX)]
    prog = {}
    marks = {}
    succ = 0
    for ti, sc in enumerate(scripts):
        r = M.rollout(policy, cfg, sc, H, D, seed, ti, rho, keep_states=True)
        succ += int(r["success"])
        for t, ok in enumerate(r["step_ok"]):
            acc_t[t]["n"] += 1
            acc_t[t]["ok"] += int(ok)
            p, wrong, done = r["states"][t]
            b = bucket_progress(p)
            prog.setdefault(b, {"n": 0, "ok": 0})
            prog[b]["n"] += 1
            prog[b]["ok"] += int(ok)
            mk = mark_state(wrong, done)
            marks.setdefault(mk, {"n": 0, "ok": 0})
            marks[mk]["n"] += 1
            marks[mk]["ok"] += int(ok)
    return {"n": len(scripts), "succ": succ, "acc_by_t": acc_t,
            "by_progress": prog, "by_marks": marks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results/paper/e1l_m1b_diag.jsonl")
    args = ap.parse_args()

    import torch
    import h4_density as H4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = M.EnvCfg6()
    scripts = M.build_scripts(cfg, args.seed, args.rho, args.tasks)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():
        Path(args.out).unlink()

    for arm, mode in ARMS:
        tr = M.build_train_data(cfg, 3000, 100, D, args.seed, mode=mode)
        model = H4.train(W, tr, args.epochs, device, seed=args.seed)
        policy = M.NNPolicy(model, device)
        print(f"[train] arm={arm} mode={mode} w={W} d={D} seed={args.seed}", flush=True)
        for H in HS:
            agg = aggregate(policy, cfg, scripts, H, args.rho, args.seed)
            row = {"arm": arm, "mode": mode, "w": W, "d": D, "seed": args.seed,
                   "rho": args.rho, "H": H, **agg}
            with Path(args.out).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[eval] arm={arm} H{H}: {agg['succ']}/{agg['n']} = "
                  f"{agg['succ'] / agg['n']:.3f}  step0={agg['acc_by_t'][0]['ok'] / max(1, agg['acc_by_t'][0]['n']):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
