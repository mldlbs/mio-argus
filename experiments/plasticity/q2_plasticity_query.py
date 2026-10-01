"""
Q2：Q1 的负面是不是"均匀查询"造成的？—— 查询分布 + 巩固

Q1 的裁决：门控空转、衰减一致为负（见 results/plasticity/Q1_RESULTS.md）。
两个已知的辩护理由，Q2 各拆一刀：

  1. **查询距离均匀**，天然惩罚衰减 —— 若查询偏近因（更贴近真实交互），
     衰减是否转正？扫 `beta`：权重 w_d = beta^d（beta=1 均匀，越小越偏近因）。
     找 decay 效应由负转正的 beta*。

  2. **没有巩固**：复现的关联应被写得更强（"该记什么"的最小实现）。
     加 `consolidate` 臂：按出现次数放大写入强度，看是否在 repeat>0 时有净收益。

臂：
  A_recency / B_kv(FIFO) / C0(plain Hebbian)
  C_decay(λ=0.98) / C_consol(无衰减，重视复现) / C_decay_consol
  C_gate_consol（软门控 + 巩固）—— 门控在 Q1 空转，这里给它一个"值得跳过"的场景

用法:
    python experiments/plasticity/q2_plasticity_query.py --quick
    python experiments/plasticity/q2_plasticity_query.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

import q1_plasticity as Q1

N_DIM = Q1.N_DIM
M_DEF = Q1.M_DEF
EPS = Q1.EPS


def make_stream_ids(E, M, n, seed, repeat):
    """带 id 的流：复现项复用同一 id，便于按重要性巩固。"""
    rng = np.random.default_rng(seed)
    keys = Q1._unit_rows(rng.standard_normal((E, M, n)))
    vals = rng.standard_normal((E, M, n))
    ids = -np.ones((E, M), dtype=int)
    pool, nxt = [], 0
    for e in range(E):
        for i in range(M):
            if pool and rng.random() < repeat:
                k, v, j = pool[int(rng.integers(0, len(pool)))]
                keys[e, i], vals[e, i], ids[e, i] = k, v, j
            else:
                ids[e, i] = nxt
                pool.append((keys[e, i].copy(), vals[e, i].copy(), nxt))
                nxt += 1
    return keys, vals, ids


class FWConsolidate:
    """快速权重 + 重要性巩固（复现越多写得越强）+ 可选衰减/门控。"""

    def __init__(self, n, decay=1.0, eta=1.0, gated=False, cons_gain=1.0):
        self.n = n
        self.W = np.zeros((n, n))
        self.decay = decay
        self.eta = eta
        self.gated = gated
        self.cons_gain = cons_gain
        self.count = {}
        tag = []
        tag.append("gate" if gated else "nogate")
        tag.append("decay" if decay < 1.0 else "nodecay")
        tag.append("consol")
        self.name = "C_" + "_".join(tag)

    def write(self, k, v, item_id=None):
        kn = k / (np.linalg.norm(k) + EPS)
        pred = self.W @ kn
        surprise = np.linalg.norm(v - pred) / (np.linalg.norm(v) + EPS)
        g = min(1.0, float(surprise)) if self.gated else 1.0
        c = self.count.get(item_id, 0)
        eta_eff = self.eta * (1.0 + self.cons_gain * c)
        self.W = self.decay * self.W + eta_eff * g * np.outer(v, kn)
        if item_id is not None:
            self.count[item_id] = c + 1

    def read(self, q):
        qn = q / (np.linalg.norm(q) + EPS)
        return self.W @ qn


def make_arms(n, cap):
    return [
        Q1.RecencyWindow(n, cap=M_DEF),
        Q1.ExactKV(n, cap=cap),
        Q1.FastWeights(n, decay=1.0, gated=False),          # C0
        Q1.FastWeights(n, decay=0.98, gated=False),         # C_decay
        FWConsolidate(n, decay=1.0, gated=False),           # C_consol
        FWConsolidate(n, decay=0.98, gated=False),          # C_decay_consol
        FWConsolidate(n, decay=1.0, gated=True),            # C_gate_consol
    ]


def evaluate(E, M, n, cap, corruption, seed, repeat, beta):
    keys, vals, ids = make_stream_ids(E, M, n, seed, repeat)
    rng = np.random.default_rng(seed + 777)
    arms = make_arms(n, cap)
    for e in range(E):
        for i in range(M):
            for a in arms:
                if isinstance(a, FWConsolidate):
                    a.write(keys[e, i], vals[e, i], int(ids[e, i]))
                else:
                    a.write(keys[e, i], vals[e, i])
    acc = {a.name: {} for a in arms}
    for e in range(E):
        d = E - 1 - e
        for i in range(M):
            cue = keys[e, i].copy()
            if corruption > 0:
                cue = cue + corruption * rng.standard_normal(n)
            for a in arms:
                acc[a.name].setdefault(d, []).append(
                    Q1._cos(a.read(cue), vals[e, i]))
    per_d = {a.name: {d: float(np.mean(v)) for d, v in acc[a.name].items()}
             for a in arms}
    w = {d: beta ** d for d in per_d[arms[0].name]}
    tot = sum(w.values())
    weighted = {name: sum(w[d] * per_d[name][d] for d in w) / tot
                for name in per_d}
    return per_d, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--es", type=str, default=None)
    ap.add_argument("--corr", type=str, default=None)
    ap.add_argument("--repeat", type=str, default=None)
    ap.add_argument("--beta", type=str, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--cap", type=int, default=Q1.S_DEF)
    ap.add_argument("--m", type=int, default=M_DEF)
    args = ap.parse_args()

    Es = ([int(x) for x in args.es.split(",")] if args.es else [16])
    corrs = ([float(x) for x in args.corr.split(",")] if args.corr
             else ([0.0, 0.4] if not args.quick else [0.0]))
    reps = ([float(x) for x in args.repeat.split(",")] if args.repeat
            else ([0.5] if not args.quick else [0.5]))
    betas = ([float(x) for x in args.beta.split(",")] if args.beta
             else ([1.0, 0.9, 0.7, 0.5] if not args.quick else [1.0, 0.7]))
    seeds = args.seeds if args.seeds else (5 if args.quick else 20)

    print(f"n={N_DIM} M={args.m} cap={args.cap} seeds={seeds} "
          f"E={Es} corr={corrs} repeat={reps} beta={betas}")

    rows = []
    for E in Es:
        for c in corrs:
            for rep in reps:
                for beta in betas:
                    acc = {}
                    for s in range(seeds):
                        per_d, weighted = evaluate(E, args.m, N_DIM,
                                                   args.cap, c, 1000 + s,
                                                   rep, beta)
                        for name, val in weighted.items():
                            acc.setdefault(name, []).append(val)
                    mean = {k: float(np.mean(v)) for k, v in acc.items()}
                    names = list(mean.keys())
                    decay_names = [n for n in names if "decay" in n]
                    C0 = mean["C_nogate_nodecay"]
                    Cdec = mean.get("C_nogate_decay", C0)
                    cons = mean.get("C_nogate_nodecay_consol", C0)
                    gcons = mean.get("C_gate_nodecay_consol", C0)
                    B = mean["B_kv"]
                    decay_eff = Cdec - C0
                    cons_eff = cons - C0
                    gate_cons_eff = gcons - mean.get("C_nogate_nodecay_consol", C0)
                    print(f"E={E} corr={c} rep={rep} beta={beta}: "
                          f"A={mean['A_recency']:.3f} B={B:.3f} "
                          f"C0={C0:.3f} Cdecay={Cdec:.3f} "
                          f"Ccons={cons:.3f} Cgate_cons={gcons:.3f} | "
                          f"decay_eff={decay_eff:+.3f} cons_eff={cons_eff:+.3f} "
                          f"gate_cons_eff={gate_cons_eff:+.3f}")
                    for name in names:
                        rows.append(dict(E=E, corruption=c, repeat=rep,
                                         beta=beta, arm=name, acc=mean[name],
                                         decay_eff=decay_eff, cons_eff=cons_eff))

    print(f"\n{'='*72}")
    print("判据：")
    print("  decay 由负转正的 beta* —— 若存在，则衰减是机制，但只在偏近因查询下")
    print("  cons 效应 > 0 —— 则『巩固/该记什么』有净收益")
    for rep in reps:
        for c in corrs:
            sub = [r for r in rows if r["repeat"] == rep
                   and r["corruption"] == c and r["arm"] == "C_nogate_decay"]
            sub.sort(key=lambda r: -r["beta"])
            effs = [(r["beta"], r["decay_eff"]) for r in sub]
            pos = [b for b, e in effs if e > 0.02]
            print(f"  rep={rep} corr={c}: decay_eff(beta)="
                  f"{[(b, round(e,3)) for b, e in effs]} "
                  f"-> {'转正@'+str(pos) if pos else '始终为负'}")
    print("=" * 72)

    out = {"n": N_DIM, "m": args.m, "cap": args.cap, "seeds": seeds,
           "Es": Es, "corr": corrs, "repeat": reps, "beta": betas, "rows": rows}
    Path("results/plasticity/q2_plasticity_query_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 results/plasticity/q2_plasticity_query_results.json")


if __name__ == "__main__":
    main()
