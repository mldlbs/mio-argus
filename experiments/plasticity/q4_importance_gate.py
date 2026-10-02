"""
Q4：任务相关重要性下，写入门控有没有"头"？—— oracle vs learned

Q3 的遗留问题（Q3_RESULTS §6 中优项）：learned gate ≈0，可能是因为
(1) 门控在本任务里根本没有 headroom，或 (2) 写入时刻的特征
（surprise/范数）不携带"未来会被查询"的信号。
Q4 用**重要性偏置的查询分布** + **oracle 门控**把这两种情况分开：

  - 查询权重 w_i ∝ imp_i（25% 条目为 hot，imp=8；分层抽样到每个 episode，
    与距离无关 → 不是 Q2 的"偏近因"的重述）；
  - 写入强度 g_i = imp_i / mean(imp)（总写入质量与 A 相同，只重新分配）。

三臂（同参数、同算力，只差写入时可用的信息）：
  A nogate        g≡1，查询按重要性采样（可学投影端到端训练）
  B learnedgate   g 由 [surprise, pred_norm, v_norm] MLP 预测（Q3 同款特征）
  O oraclegate    g 由真值 imp 给出（写入时刻的特权信息）

参照（等容量 cap=32 / 128 条目）：
  KvFifo      按位置保留最近 cap 条
  KvOracleEvict 按真值重要性保留 cap 条（查表侧的 oracle 淘汰）

判据：
  - O − A 明显为正 且 B − A ≈0 → headroom 存在但写时特征够不着
    （"该记什么"需要写入时刻不可得的边信息）；
  - O − A ≈0 → 即使完美重要性写入也没用 → 可塑性写入侧硬收线；
  - O vs KvOracleEvict：压缩+特权写 vs 精确+特权淘汰，谁是分水岭。

用法:
    python experiments/plasticity/q4_importance_gate.py --quick
    python experiments/plasticity/q4_importance_gate.py
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
HOT_FRAC = 0.25
HOT_IMP = 8.0


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def make_streams(n_streams, E, M, n, seed):
    """返回 (keys, vals, imp)；imp 分层：每个 episode 内 25% 为 hot（与距离无关）。"""
    rng = np.random.default_rng(seed)
    keys = unit(rng.standard_normal((n_streams, E, M, n)))
    vals = rng.standard_normal((n_streams, E, M, n))
    n_hot = max(1, int(round(M * HOT_FRAC)))
    imp = np.ones((n_streams, E, M), dtype=np.float32)
    for e in range(M):
        hot_idx = rng.choice(M, size=n_hot, replace=False)
        imp[:, e, hot_idx] = HOT_IMP
    return (torch.tensor(keys, dtype=torch.float32),
            torch.tensor(vals, dtype=torch.float32),
            torch.tensor(imp, dtype=torch.float32))


class LearnedMem(nn.Module):
    def __init__(self, n, d, gate, h=32):
        super().__init__()
        self.n, self.d = n, d
        self.use_gate = gate
        self.wk = nn.Linear(n, d)
        self.wv = nn.Linear(n, d)
        self.wq = nn.Linear(n, d)
        self.readout = nn.Linear(d, n)
        self.gate_mlp = (nn.Sequential(nn.Linear(3, h), nn.ReLU(),
                                       nn.Linear(h, 1)) if gate else None)

    def encode(self, keys, vals):
        return self.wk(keys), self.wv(vals)

    def write_all(self, K, V, g_override=None):
        """K,V: (B,T,d)；g_override: (B,T) 或 None → 返回 (B,d,d)。"""
        B, T, d = K.shape
        W = torch.zeros(B, d, d, device=K.device)
        for t in range(T):
            k = F.normalize(K[:, t], dim=-1)
            v = V[:, t]
            if g_override is not None:
                g = g_override[:, t]
            elif self.use_gate:
                pred = torch.einsum("bij,bj->bi", W, k)
                surprise = (v - pred).norm(dim=-1) / (v.norm(dim=-1) + 1e-6)
                feats = torch.stack([surprise, pred.norm(dim=-1),
                                     v.norm(dim=-1)], dim=-1)
                g = torch.sigmoid(self.gate_mlp(feats)).squeeze(-1)
            else:
                g = torch.ones(B, device=K.device)
            W = W + g[:, None, None] * torch.einsum("bi,bj->bij", v, k) / d
        return W


def stream_importance(imp):
    """(B,T) 归一化写入强度：总写入质量与 g≡1 相同。"""
    return imp / imp.mean(dim=-1, keepdim=True)


def recall_by_stream(yhat, true, imp):
    """逐流加权（重要性）与均匀余弦召回 → (weighted, uniform)。"""
    cos = F.cosine_similarity(yhat, true, dim=-1)          # (B,T)
    w = imp / imp.sum(dim=-1, keepdim=True)
    weighted = (cos * w).sum(dim=-1).mean()
    uniform = cos.mean()
    return float(weighted), float(uniform)


def eval_mem(model, keys, vals, imp, noise, mode):
    dev = next(model.parameters()).device
    keys, vals, imp = keys.to(dev), vals.to(dev), imp.to(dev)
    B, E, M, n = keys.shape
    T = E * M
    with torch.no_grad():
        flat_k = keys.reshape(B, T, n)
        flat_v = vals.reshape(B, T, n)
        flat_i = imp.reshape(B, T)
        K, V = model.encode(flat_k, flat_v)
        g = stream_importance(flat_i) if mode == "O" else None
        W = model.write_all(K, V, g_override=g)
        cue = flat_k + noise * torch.randn_like(flat_k) if noise > 0 else flat_k
        q = F.normalize(model.wq(cue), dim=-1)
        mem = torch.einsum("bij,bqj->bqi", W, q)
        yhat = model.readout(mem)
        md = (E - 1 - torch.arange(E, device=keys.device)).repeat_interleave(M)
        dist = md.unsqueeze(0).expand(B, -1)
        curve = {int(d): float((F.cosine_similarity(yhat, flat_v, dim=-1)
                                [dist == d]).mean()) for d in torch.unique(dist)}
        kv_f = _eval_kv(keys, vals, imp, cue, dist, CAP_DEF, mode="fifo")
        kv_o = _eval_kv(keys, vals, imp, cue, dist, CAP_DEF, mode="oracle")
    return curve, kv_f, kv_o, yhat, flat_v, flat_i


def _eval_kv(keys, vals, imp, cue, dist, cap, mode):
    """mode: fifo=最近 cap 条 / oracle=重要性最高 cap 条。返回 (dist_curve, w, u)。"""
    B, E, M, n = keys.shape
    out = {int(d): [] for d in torch.unique(dist)}
    ws, us = [], []
    for b in range(B):
        allk = keys[b].reshape(E * M, n).detach().cpu().numpy()
        allv = vals[b].reshape(E * M, n).detach().cpu().numpy()
        alli = imp[b].reshape(E * M).detach().cpu().numpy()
        if mode == "fifo":
            keep = np.arange(E * M)[-cap:]
        else:
            keep = np.argsort(alli)[-cap:]
        sim = cue[b].detach().cpu().numpy() @ allk[keep].T
        pred = allv[keep][sim.argmax(1)]
        cos = (pred * allv).sum(1) / (np.linalg.norm(pred, axis=1)
                                      * np.linalg.norm(allv, axis=1) + 1e-9)
        w = alli / alli.sum()
        ws.append(float((cos * w).sum()))
        us.append(float(cos.mean()))
        dd = dist[b].detach().cpu().numpy()
        for d in np.unique(dd):
            out[int(d)].append(float(cos[dd == d].mean()))
    return ({d: float(np.mean(v)) for d, v in out.items()},
            float(np.mean(ws)), float(np.mean(us)))


def train_arm(mode, tr, epochs, device, seed, noise_train=0.2, lr=3e-3, bs=64):
    """mode: 'A' | 'B' | 'O'。"""
    torch.manual_seed(seed)
    model = LearnedMem(N_DIM, D_MEM, gate=(mode == "B")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keys, vals, imp = tr
    B, E, M, n = keys.shape
    T = E * M
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(B)
        for i in range(0, B, bs):
            idx = perm[i:i + bs]
            kb, vb, ib = (keys[idx].to(device), vals[idx].to(device),
                          imp[idx].to(device))
            flat_k = kb.reshape(-1, T, n)
            flat_v = vb.reshape(-1, T, n)
            flat_i = ib.reshape(-1, T)
            K, V = model.encode(flat_k, flat_v)
            W = model.write_all(K, V, g_override=stream_importance(flat_i)
                                if mode == "O" else None)
            Q = 16
            w = flat_i / flat_i.sum(dim=-1, keepdim=True)
            ei = torch.multinomial(w, Q, replacement=True)
            b_idx = torch.arange(flat_k.shape[0], device=device).unsqueeze(1)
            cue = flat_k[b_idx, ei]
            tgt = flat_v[b_idx, ei]
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
          f"epochs={epochs} n_tr={n_tr} n_te={n_te} seeds={seeds} "
          f"hot={HOT_FRAC}*{HOT_IMP} cap={CAP_DEF}")

    tr = make_streams(n_tr, E, M, N_DIM, 200)
    te = make_streams(n_te, E, M, N_DIM, 900)

    modes = [("A_nogate", "A"), ("B_learnedgate", "B"), ("O_oraclegate", "O")]
    noises = [0.0, 0.4]

    acc = {k: [] for k in
           [(m, no, met) for m, _ in modes for no in noises
            for met in ("w", "u", "kw", "ku", "ko_w", "ko_u")]}
    curves = {m: {} for m, _ in modes}

    for seed in range(seeds):
        for name, mode in modes:
            m = train_arm(mode, tr, epochs, device, seed=seed)
            for noise in noises:
                curve, kvf, kvo, yhat, tv, ti = eval_mem(
                    m, te[0], te[1], te[2], noise, mode)
                w, u = recall_by_stream(yhat, tv, ti)
                acc[(name, noise, "w")].append(w)
                acc[(name, noise, "u")].append(u)
                acc[(name, noise, "kw")].append(kvf[1])
                acc[(name, noise, "ku")].append(kvf[2])
                acc[(name, noise, "ko_w")].append(kvo[1])
                acc[(name, noise, "ko_u")].append(kvo[2])
                for d, v in curve.items():
                    curves[name].setdefault(d, []).append(v)

    mean = lambda k: float(np.mean(acc[k]))
    print(f"\n{'arm':<16}{'noise':>6}{'wImp':>9}{'uUnif':>9}"
          f"{'KvFifo_w':>10}{'KvOrac_w':>10}")
    print("-" * 60)
    for name, _ in modes:
        for noise in noises:
            print(f"{name:<16}{noise:>6}{mean((name,noise,'w')):>9.4f}"
                  f"{mean((name,noise,'u')):>9.4f}"
                  f"{mean((name,noise,'kw')):>10.4f}"
                  f"{mean((name,noise,'ko_w')):>10.4f}")

    print(f"\n{'='*66}")
    print("判据（重要性加权召回）：learned / oracle 相对 A 的增量")
    verdict = {}
    for noise in noises:
        base = mean(("A_nogate", noise, "w"))
        b = mean(("B_learnedgate", noise, "w")) - base
        o = mean(("O_oraclegate", noise, "w")) - base
        ko = mean(("A_nogate", noise, "ko_w"))
        verdict[f"noise{noise}"] = {"A": base, "B-A": b, "O-A": o,
                                    "KvOracle": ko}
        print(f"  noise={noise}: B-A={b:+.4f} O-A={o:+.4f} | A={base:.4f} "
              f"KvOracle={ko:.4f}")
    print("=" * 66)

    out = {"n": N_DIM, "d": D_MEM, "E": E, "M": M, "hot_frac": HOT_FRAC,
           "hot_imp": HOT_IMP, "cap": CAP_DEF, "epochs": epochs,
           "n_train": n_tr, "n_test": n_te, "seeds": seeds,
           "mean": {f"{k[0]}|noise{k[1]}|{k[2]}": float(np.mean(v))
                    for k, v in acc.items()},
           "verdict": verdict,
           "dist_curve": {k: {str(d): float(np.mean(v))
                              for d, v in sorted(cv.items())}
                          for k, cv in curves.items()}}
    Path("results/plasticity/q4_importance_gate_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/plasticity/q4_importance_gate_results.json")


if __name__ == "__main__":
    main()
