"""
Q3：学的写入门控能否兑现可塑性？—— 端到端

Q1/Q2（无训练）的结论：
  门控空转、衰减只在偏近因查询下有用、巩固有害（见 Q1/results/plasticity/Q2_RESULTS.md）。
Q3 给门控/衰减一个**可学**的版本，端到端训练后看结论是否被推翻。

可微联想记忆：
  k = W_k x_key， v = W_v y_val          （学习投影）
  W ← λ·W + g·(v ⊗ k)/d                   （g 可学门控，λ 可学衰减）
  y_hat = W_read · (W @ (W_q x_cue))      （学习读出）
  损失：1 − cos(y_hat, y_true)

2×2 消融（同参数/同算力，只差门控与衰减）：
  A nogate_nodecay     ：g≡1，λ=1（朴素 Hebbian + 学习投影/读出）
  B learnedgate_nodecay：g 可学，λ=1
  C nogate_ldecay      ：g≡1，λ 可学
  D learnedgate_ldecay ：g、λ 均可学
外加**非学习参照** ExactKV（raw 向量，容量 32）用于对照距离行为。

判据：
  - learned gate 是否在等参数下 > nogate（uniform 与 recency 两个查询分布）
  - learned decay 是否提升；若只在 recency 下提升，则与 Q2 一致
  - 上述任一为正 → 手工门控的否定被训练推翻；全为负 → 结论稳健

用法:
    python experiments/plasticity/q3_learned_gate.py --quick
    python experiments/plasticity/q3_learned_gate.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIM = 64
D_MEM = 64
M_DEF = 8
E_DEF = 16
CAP_DEF = N_DIM // 2


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def make_streams(n_streams, E, M, n, seed):
    rng = np.random.default_rng(seed)
    keys = unit(rng.standard_normal((n_streams, E, M, n)))
    vals = rng.standard_normal((n_streams, E, M, n))
    return (torch.tensor(keys, dtype=torch.float32),
            torch.tensor(vals, dtype=torch.float32))


class LearnedMem(nn.Module):
    def __init__(self, n, d, gate, decay, h=32):
        super().__init__()
        self.n, self.d = n, d
        self.use_gate = gate
        self.use_decay = decay
        self.wk = nn.Linear(n, d)
        self.wv = nn.Linear(n, d)
        self.wq = nn.Linear(n, d)
        self.readout = nn.Linear(d, n)
        self.gate_mlp = (nn.Sequential(nn.Linear(3, h), nn.ReLU(),
                                       nn.Linear(h, 1)) if gate else None)
        self.decay_raw = nn.Parameter(torch.tensor(2.0)) if decay else None

    def encode(self, keys, vals):
        return self.wk(keys), self.wv(vals)

    def write_all(self, K, V):
        """K,V: (B,T,d) -> 终态 W: (B,d,d)。"""
        B, T, d = K.shape
        W = torch.zeros(B, d, d, device=K.device)
        lam = torch.sigmoid(self.decay_raw) if self.decay_raw is not None else 1.0
        for t in range(T):
            k = F.normalize(K[:, t], dim=-1)
            v = V[:, t]
            pred = torch.einsum("bij,bj->bi", W, k)
            if self.use_gate:
                surprise = (v - pred).norm(dim=-1) / (v.norm(dim=-1) + 1e-6)
                feats = torch.stack([surprise, pred.norm(dim=-1),
                                     v.norm(dim=-1)], dim=-1)
                g = torch.sigmoid(self.gate_mlp(feats)).squeeze(-1)
            else:
                g = torch.ones(B, device=K.device)
            W = lam * W + g[:, None, None] * torch.einsum("bi,bj->bij", v, k) / d
        return W


def per_distance(pred, true, dist):
    """pred,true: (B,Q,n); dist: (B,Q) -> {d: mean cosine}."""
    cos = F.cosine_similarity(pred, true, dim=-1)     # (B,Q)
    out = {}
    for d in torch.unique(dist):
        out[int(d)] = float(cos[dist == d].mean())
    return out


def eval_mem(model, keys, vals, noise, cap):
    """keys,vals: (B,E,M,n) -> 每距离平均 cosine；返回 (model_curve, kv_curve)。"""
    dev = next(model.parameters()).device
    keys, vals = keys.to(dev), vals.to(dev)
    B, E, M, n = keys.shape
    with torch.no_grad():
        K, V = model.encode(keys.reshape(B, E * M, n), vals.reshape(B, E * M, n))
        W = model.write_all(K, V)

        cue = keys.reshape(B, E * M, n)
        if noise > 0:
            cue = cue + noise * torch.randn_like(cue)
        q = F.normalize(model.wq(cue), dim=-1)
        mem = torch.einsum("bij,bqj->bqi", W, q)
        yhat = model.readout(mem)
        true_v = vals.reshape(B, E * M, n)

        md = (E - 1 - torch.arange(E, device=keys.device)).repeat_interleave(M)
        dist = md.unsqueeze(0).expand(B, -1)
        curve = per_distance(yhat, true_v, dist)

        kv_curve = _eval_kv(keys, vals, cue, dist, cap)
    return curve, kv_curve


def _eval_kv(keys, vals, cue, dist, cap):
    B, E, M, n = keys.shape
    out = {int(d): [] for d in torch.unique(dist)}
    for b in range(B):
        allk = keys[b].reshape(E * M, n).detach().cpu().numpy()
        allv = vals[b].reshape(E * M, n).detach().cpu().numpy()
        kbuf = allk[-cap:]
        vbuf = allv[-cap:]
        qn = cue[b].detach().cpu().numpy()
        sim = qn @ kbuf.T
        idx = sim.argmax(1)
        pred = vbuf[idx]
        true = allv
        cos = (pred * true).sum(1) / (np.linalg.norm(pred, axis=1)
                                      * np.linalg.norm(true, axis=1) + 1e-9)
        dd = dist[b].detach().cpu().numpy()
        for d in np.unique(dd):
            out[int(d)].append(float(cos[dd == d].mean()))
    return {d: float(np.mean(v)) for d, v in out.items()}


def weighted(curve, beta):
    ds = sorted(curve)
    w = np.array([beta ** d for d in ds])
    vals = np.array([curve[d] for d in ds])
    return float((w * vals).sum() / w.sum())


def train_arm(gate, decay, tr, te, epochs, device, seed, noise_train=0.2, lr=3e-3, bs=64):
    torch.manual_seed(seed)
    model = LearnedMem(N_DIM, D_MEM, gate, decay).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keys, vals = tr
    B, E, M, n = keys.shape
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(B)
        for i in range(0, B, bs):
            idx = perm[i:i + bs]
            kb, vb = keys[idx].to(device), vals[idx].to(device)
            K, V = model.encode(kb.reshape(-1, E * M, n),
                                vb.reshape(-1, E * M, n))
            W = model.write_all(K, V)
            # 每流采样 Q 个查询
            Q = 16
            ei = torch.randint(0, E * M, (kb.shape[0], Q), device=device)
            flat = kb.reshape(-1, E * M, n)
            fv = vb.reshape(-1, E * M, n)
            cue = torch.gather(flat, 1, ei[..., None].expand(-1, -1, n))
            tgt = torch.gather(fv, 1, ei[..., None].expand(-1, -1, n))
            if noise_train > 0:
                cue = cue + noise_train * torch.randn_like(cue)
            q = F.normalize(model.wq(cue), dim=-1)
            mem = torch.einsum("bij,bqj->bqi", W, q)
            yhat = model.readout(mem)
            loss = (1 - F.cosine_similarity(yhat, tgt, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (40 if args.quick else 150)
    n_tr = args.n_train if args.n_train else (400 if args.quick else 1500)
    n_te = args.n_test if args.n_test else (150 if args.quick else 400)
    seeds = args.seeds if args.seeds else (1 if args.quick else 3)
    E, M = E_DEF, M_DEF
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} n={N_DIM} d={D_MEM} E={E} M={M} "
          f"epochs={epochs} n_tr={n_tr} n_te={n_te} seeds={seeds}")

    tr = make_streams(n_tr, E, M, N_DIM, 100)
    te = make_streams(n_te, E, M, N_DIM, 900)

    arms = [("A_nogate_nodecay", False, False),
            ("B_learnedgate_nodecay", True, False),
            ("C_nogate_ldecay", False, True),
            ("D_learnedgate_ldecay", True, True)]
    betas = [1.0, 0.7]
    noises = [0.0, 0.4]

    results = {}
    for seed in range(seeds):
        for name, gate, decay in arms:
            m = train_arm(gate, decay, tr, te, epochs, device, seed=seed)
            for noise in noises:
                curve, kv = eval_mem(m, te[0], te[1], noise, CAP_DEF)
                for beta in betas:
                    results.setdefault((name, noise, beta), []).append(
                        (weighted(curve, beta), weighted(kv, beta)))

    print(f"\n{'arm':<24}{'noise':>6}{'beta':>6}{'acc':>9}{'KvRef':>9}")
    print("-" * 54)
    summary = {}
    for (name, noise, beta), vals in results.items():
        acc = float(np.mean([v[0] for v in vals]))
        kv = float(np.mean([v[1] for v in vals]))
        summary[(name, noise, beta)] = (acc, kv)
        print(f"{name:<24}{noise:>6}{beta:>6}{acc:>9.4f}{kv:>9.4f}")

    print(f"\n{'='*66}")
    print("判据：learned gate / learned decay 相对 A_nogate_nodecay 的增量")
    for noise in noises:
        for beta in betas:
            base = summary[("A_nogate_nodecay", noise, beta)][0]
            b = summary[("B_learnedgate_nodecay", noise, beta)][0] - base
            c = summary[("C_nogate_ldecay", noise, beta)][0] - base
            d = summary[("D_learnedgate_ldecay", noise, beta)][0] - base
            kv = summary[("A_nogate_nodecay", noise, beta)][1]
            print(f"  noise={noise} beta={beta}: gate={b:+.4f} "
                  f"decay={c:+.4f} both={d:+.4f} | A={base:.4f} "
                  f"KVref={kv:.4f} ({'fw>kv' if base > kv + 0.02 else 'fw<=kv'})")
    print("=" * 66)

    out = {"n": N_DIM, "d": D_MEM, "E": E, "M": M, "epochs": epochs,
           "n_train": n_tr, "n_test": n_te, "seeds": seeds,
           "summary": {f"{k[0]}|noise{k[1]}|beta{k[2]}": v
                       for k, v in summary.items()}}
    Path("results/plasticity/q3_learned_gate_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/plasticity/q3_learned_gate_results.json")


if __name__ == "__main__":
    main()
