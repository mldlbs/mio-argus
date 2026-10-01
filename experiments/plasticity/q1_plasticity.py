"""
Q1：可塑性 vs 上下文 —— 快速权重到底是不是注意力的换皮

设计见 docs/PLASTICITY.md。核心立场（§2）：
  快速权重与线性注意力同构（Schlag 2021；arXiv 2602.21204）。
  所以自变量不是"是否在线更新权重"，而是
  **"固定大小状态的生命周期与容量管理"**。

本实验是一版**无训练**的机制对照（刻意的裁剪，见 docs/PLASTICITY.md §6）：
在受控向量空间里，直接比较"精确有界缓冲（上下文/检索）"与
"叠加式压缩记忆（快速权重）"，并用 2×2 消融隔离
**门控（gate）** 与 **衰减（decay）** 各自是否是机制。

臂（等状态占用：C 的 W 是 n² floats，B 的 S=n/2 槽也是 n² floats）：

  A       截断上下文   ：只保留当前 episode（cap=M），按键最近邻读
  B       KV/FIFO     ：容量 S，FIFO 淘汰
  C_*     快速权重 2×2 ：W ← λW + η·g·(v⊗k)
            gate  ∈ {off, on}    g = min(1, surprise)（软门控）
            decay ∈ {1.0, 0.98}

流带**关联复现**率 p：复现项对门控才有意义（否则全新 → surprise≡1 → 门控空转）。

判据（docs/PLASTICITY.md §5.4/§5.5）：
  - sanity ：acc_C(d) > acc_A(d)（d≥1）
  - 分水岭 ：acc_C(d) vs acc_B(d)  —— C≈B 则可塑性=记忆淘汰策略
  - 机制   ：门控/衰减 任一单独成分能否提升（否则不是机制）
  - crossing：快速权重在哪个距离 d* 上反超精确 KV（若有）

用法:
    python experiments/plasticity/q1_plasticity.py --quick
    python experiments/plasticity/q1_plasticity.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_DIM = 64
M_DEF = 8
S_DEF = N_DIM // 2          # KV 容量槽数；与 W(n²) 占用匹配
EPS = 1e-9

FW_ARMS = ["C_nogate_nodecay", "C_gate_nodecay",
           "C_nogate_decay", "C_gate_decay"]
ALL_ARMS = ["A_recency", "B_kv"] + FW_ARMS


def _unit_rows(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def make_stream(E, M, n, seed, repeat=0.0):
    """生成 episodic 关联流；repeat 比例的行为对之前项的复现。"""
    rng = np.random.default_rng(seed)
    keys = _unit_rows(rng.standard_normal((E, M, n)))
    vals = rng.standard_normal((E, M, n))
    if repeat > 0:
        pool = []
        for e in range(E):
            for i in range(M):
                if pool and rng.random() < repeat:
                    k, v = pool[int(rng.integers(0, len(pool)))]
                    keys[e, i], vals[e, i] = k, v
                else:
                    pool.append((keys[e, i].copy(), vals[e, i].copy()))
    return keys, vals


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


# ---------------- 臂 ----------------

class RecencyWindow:
    """A：只保留当前 episode。"""
    name = "A_recency"

    def __init__(self, n, cap=S_DEF):
        self.cap = cap
        self.k, self.v = [], []

    def write(self, k, v):
        self.k.append(k.copy())
        self.v.append(v.copy())
        if len(self.k) > self.cap:
            self.k.pop(0)
            self.v.pop(0)

    def read(self, q):
        if not self.k:
            return np.zeros_like(q)
        K = np.stack(self.k)
        return self.v[int(np.argmax(K @ q))].copy()


class ExactKV:
    """B：精确有界缓冲（FIFO）。"""

    def __init__(self, n, cap=S_DEF):
        self.cap = cap
        self.k, self.v = [], []
        self.name = "B_kv"

    def write(self, k, v):
        self.k.append(k.copy())
        self.v.append(v.copy())
        if len(self.k) > self.cap:
            self.k.pop(0)
            self.v.pop(0)

    def read(self, q):
        if not self.k:
            return np.zeros_like(q)
        K = np.stack(self.k)
        return self.v[int(np.argmax(K @ q))].copy()


class FastWeights:
    """C_*：叠加式压缩记忆。2×2 门控×衰减消融。"""

    def __init__(self, n, decay=1.0, eta=1.0, gated=False):
        self.n = n
        self.W = np.zeros((n, n))
        self.decay = decay
        self.eta = eta
        self.gated = gated
        tag = "gate" if gated else "nogate"
        tag2 = "decay" if decay < 1.0 else "nodecay"
        self.name = f"C_{tag}_{tag2}"

    def write(self, k, v):
        kn = k / (np.linalg.norm(k) + EPS)
        pred = self.W @ kn
        surprise = np.linalg.norm(v - pred) / (np.linalg.norm(v) + EPS)
        g = min(1.0, float(surprise)) if self.gated else 1.0
        self.W = self.decay * self.W + self.eta * g * np.outer(v, kn)

    def read(self, q):
        qn = q / (np.linalg.norm(q) + EPS)
        return self.W @ qn


def make_arms(n, cap):
    return [
        RecencyWindow(n, cap=M_DEF),
        ExactKV(n, cap=cap),
        FastWeights(n, decay=1.0, gated=False),
        FastWeights(n, decay=1.0, gated=True),
        FastWeights(n, decay=0.98, gated=False),
        FastWeights(n, decay=0.98, gated=True),
    ]


# ---------------- 评测 ----------------

def evaluate(E, M, n, cap, corruption, seed, repeat=0.0):
    keys, vals = make_stream(E, M, n, seed, repeat)
    rng = np.random.default_rng(seed + 777)
    arms = make_arms(n, cap)
    for e in range(E):
        for i in range(M):
            for a in arms:
                a.write(keys[e, i], vals[e, i])
    acc = {a.name: {} for a in arms}
    for e in range(E):
        d = E - 1 - e
        for i in range(M):
            cue = keys[e, i].copy()
            if corruption > 0:
                cue = cue + corruption * rng.standard_normal(n)
            for a in arms:
                acc[a.name].setdefault(d, []).append(_cos(a.read(cue), vals[e, i]))
    return {a.name: {d: float(np.mean(v)) for d, v in acc[a.name].items()}
            for a in arms}


def _crossing(mean, d_keys, fw_name, mg=0.02):
    """快速权重在哪些距离上反超精确 KV；返回最大的那个 d。"""
    ds = [d for d in d_keys if mean[fw_name][d] > mean["B_kv"][d] + mg]
    return max(ds) if ds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--es", type=str, default=None, help="episode 数，如 8,16,32")
    ap.add_argument("--corr", type=str, default=None, help="cue 噪声，如 0,0.4,0.8")
    ap.add_argument("--repeat", type=str, default=None, help="复现率，如 0,0.5")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--cap", type=int, default=S_DEF)
    ap.add_argument("--m", type=int, default=M_DEF)
    args = ap.parse_args()

    Es = ([int(x) for x in args.es.split(",")] if args.es
          else ([8, 16] if not args.quick else [8]))
    corrs = ([float(x) for x in args.corr.split(",")] if args.corr
             else ([0.0, 0.4, 0.8] if not args.quick else [0.0, 0.6]))
    reps = ([float(x) for x in args.repeat.split(",")] if args.repeat
            else ([0.0, 0.5] if not args.quick else [0.0]))
    seeds = args.seeds if args.seeds else (5 if args.quick else 20)

    print(f"n={N_DIM} M={args.m} cap={args.cap} seeds={seeds} "
          f"E={Es} corruption={corrs} repeat={reps}")
    print(f"状态占用：C 的 W = {N_DIM * N_DIM} floats；"
          f"B 的 {args.cap} 槽 = {args.cap * 2 * N_DIM} floats\n")

    rows = []
    for E in Es:
        for c in corrs:
            for rep in reps:
                agg = {}
                for s in range(seeds):
                    r = evaluate(E, args.m, N_DIM, args.cap, c, 1000 + s, rep)
                    for name, dd in r.items():
                        for d, v in dd.items():
                            agg.setdefault(name, {}).setdefault(d, []).append(v)
                mean = {n: {d: float(np.mean(v)) for d, v in dd.items()}
                        for n, dd in agg.items()}
                ds = sorted(mean["A_recency"].keys())
                buckets = {"d=0": ds[0], "d=mid": ds[len(ds) // 2],
                           "d=max": ds[-1]}

                print(f"=== E={E}  corr={c}  repeat={rep} ===")
                print(f"{'arm':<18}" + "".join(f"{b:>9}" for b in buckets)
                      + f"{'d>=1':>9}")
                for name in ALL_ARMS:
                    dm = mean[name]
                    row = "".join(f"{dm[buckets[b]]:>9.3f}" for b in buckets)
                    old = float(np.mean([dm[d] for d in ds if d >= 1]))
                    print(f"{name:<18}{row}{old:>9.3f}")
                    rows.append(dict(E=E, corruption=c, repeat=rep, arm=name,
                                     d0=dm[ds[0]], d_mid=dm[ds[len(ds) // 2]],
                                     d_max=dm[ds[-1]], d_ge1=old))

                MG = 0.02
                C0 = mean["C_nogate_nodecay"]
                Cbest = max(FW_ARMS, key=lambda a: np.mean(
                    [mean[a][d] for d in ds if d >= 1]))
                fw_old = float(np.mean([mean[Cbest][d] for d in ds if d >= 1]))
                a_old = float(np.mean([mean["A_recency"][d] for d in ds if d >= 1]))
                sane = fw_old > a_old + MG
                b_old = float(np.mean([mean["B_kv"][d] for d in ds if d >= 1]))
                vsB = ("fw>kv" if fw_old > b_old + MG else
                       "fw<kv" if b_old > fw_old + MG else "fw≈kv")
                gate_eff = (float(np.mean([mean["C_gate_decay"][d] for d in ds if d >= 1]))
                            - float(np.mean([mean["C_nogate_decay"][d] for d in ds if d >= 1])))
                decay_eff = (float(np.mean([mean["C_nogate_decay"][d] for d in ds if d >= 1]))
                             - float(np.mean([mean["C_nogate_nodecay"][d] for d in ds if d >= 1])))
                dstar = _crossing(mean, ds, Cbest)
                print(f"  -> sanity={sane} bestFW={Cbest}(d>=1 {fw_old:.3f}) "
                      f"B_kv={b_old:.3f} C0={float(np.mean([C0[d] for d in ds if d>=1])):.3f} "
                      f"| gate={gate_eff:+.3f} decay={decay_eff:+.3f} "
                      f"| {vsB} crossing_d*={dstar}\n")

    out = {"n": N_DIM, "m": args.m, "cap": args.cap, "seeds": seeds,
           "Es": Es, "corruption": corrs, "repeat": reps, "rows": rows}
    Path("results/plasticity/q1_plasticity_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/plasticity/q1_plasticity_results.json")


if __name__ == "__main__":
    main()
