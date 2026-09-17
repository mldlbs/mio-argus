"""
P2：H3 —— 验证能否替代策略精度。

核心问题
--------
等总参数量下，"小策略 + 小 verifier" 能否打败 "单个更大的策略"？

    π(w32) + V(w32)   总参 ≈ 212K
    π(w64)            总参 ≈ 392K

若前者更好，说明**拆分出验证器**比**继续放大策略**更划算 → H3 成立。

verifier 的定义
---------------
V 不是一个独立的二分类头，而是**以候选动作为条件**的打分器：
    V(screen, instruction, candidate_cell) → 该动作正确的 logit
它用 RoI-pool 取该格子处的特征（与 π 的 g×g 热力图池化是**不同**的机制），
因此 V 的排序与 π 的排序不同，re-rank 才有意义。

用途：π 给出 top-k 候选，V 在候选内重排 → 复合策略。

用法:
    python p2_verifier.py --quick
    python p2_verifier.py --epochs 25
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from p0_env import LABELS, instruction_onehot, hit
from p0_models import SpatialBackbone, P0Net
from p0_h2 import load_or_build, cfg_for, to_t, evaluate
from p1_horizon import (make_task, render_layout, predict, GRID, IMG_SIZE,
                        train_single_step)

CACHE = Path("p0_cache")


# ---------- Verifier ----------

class VerifierNet(nn.Module):
    """V(image, instruction, candidate_cell) → 正确性 logit。"""

    def __init__(self, width=32, num_labels=64, grid=GRID,
                 in_res=IMG_SIZE, stages=3, inst_dim=64):
        super().__init__()
        self.backbone = SpatialBackbone(width, in_res, stages)
        C = self.backbone.out_channels
        self.grid = grid
        self.C = C
        self.inst_emb = nn.Embedding(num_labels, inst_dim)
        self.query = nn.Linear(inst_dim, C)
        self.bias = nn.Parameter(torch.zeros(1))

    def _cell_boxes(self, cells, device, dtype):
        """cell index → 归一化 bbox (N,4)。"""
        g = self.grid
        r = (cells // g).float()
        c = (cells % g).float()
        return torch.stack([c / g, r / g, (c + 1) / g, (r + 1) / g], dim=-1).to(dtype)

    def forward(self, images, inst, cells):
        B = images.shape[0]
        feat = self.backbone(images)
        q = self.query(self.inst_emb(inst.argmax(dim=-1)))        # (B,C)
        boxes = self._cell_boxes(cells, images.device, feat.dtype)  # (B,4)
        scale = torch.tensor([feat.shape[3], feat.shape[2]] * 2,
                             dtype=feat.dtype, device=feat.device)
        rois = torch.cat([torch.arange(B, device=feat.device,
                                       dtype=feat.dtype).unsqueeze(1),
                          boxes * scale], dim=1)
        pooled = roi_align(feat, rois, output_size=(1, 1),
                           spatial_scale=1.0, aligned=True).reshape(B, self.C)
        return (pooled * q).sum(-1) / math.sqrt(self.C) + self.bias

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_verifier(width, tr, epochs, device, grid=GRID,
                   n_rand=2, n_hard=2, seed=0, lr=1e-3, batch=64,
                   pi_probs=None):
    """
    正样本 = 目标格；负样本 = 随机格 + **π 的高置信错误格**（困难负样本）。

    随机负样本太容易，V 学不到"区分 π 的混淆"。
    困难负样本是 H3 的公平版本：V 必须比 π 更会判断 π 自己犹豫的地方。
    """
    torch.manual_seed(seed)
    v = VerifierNet(width=width, grid=grid).to(device)
    opt = torch.optim.Adam(v.parameters(), lr=lr)

    images = to_t(tr["images"], device)
    inst = to_t(tr["inst"], device)
    slots = torch.from_numpy(tr["slots"]).to(device)
    S = grid * grid
    n = images.shape[0]
    spe = max(1, n // batch)
    K = 1 + n_rand + n_hard

    # 预取困难负样本：π 概率次高的错误格
    hard_pool = None
    if pi_probs is not None and n_hard > 0:
        order = np.argsort(-pi_probs, axis=1)          # (n, S) 降序
        tgt = tr["slots"]
        # 去掉正样本后的前 n_hard 个
        hp = []
        for i in range(n):
            row = [c for c in order[i] if c != tgt[i]][:n_hard]
            hp.append(row)
        hard_pool = torch.tensor(np.array(hp), device=device)   # (n, n_hard)

    v.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            b = idx.shape[0]
            pos = slots[idx]
            parts = [pos.unsqueeze(1)]
            if n_rand > 0:
                r = torch.randint(0, S, (b, n_rand), device=device)
                r = torch.where(r == pos.unsqueeze(1), (r + 1) % S, r)
                parts.append(r)
            if hard_pool is not None:
                parts.append(hard_pool[idx])
            cells = torch.cat(parts, dim=1)
            nc = cells.shape[1]
            cells = cells.reshape(-1)
            img_rep = images[idx].repeat_interleave(nc, dim=0)
            inst_rep = inst[idx].repeat_interleave(nc, dim=0)
            y = torch.zeros(b * nc, device=device)
            y[torch.arange(b, device=device) * nc] = 1.0

            logit = v(img_rep, inst_rep, cells)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return v


def pi_probs_on(pi, data, device, batch=256):
    """在给定集合上算 π 的完整概率分布 (n, S)。"""
    pi.eval()
    images = to_t(data["images"], device)
    inst = to_t(data["inst"], device)
    outs = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            outs.append(torch.softmax(pi(images[i:i + batch], inst[i:i + batch]),
                                      dim=-1).cpu().numpy())
    return np.concatenate(outs, 0)


def verify_diagnostics(pi, v, data, cfg, device, topk=3):
    """
    诊断：
      ceiling = P(正确格落在 π 的 top-k 内)   ← re-rank 的理论上限
      pick    = P(V 在 top-k 内选对 | 正确在 top-k 内)
    """
    probs = pi_probs_on(pi, data, device)
    n = probs.shape[0]
    order = np.argsort(-probs, axis=1)[:, :topk]
    tgt = data["slots"]

    ceiling = float(np.mean([tgt[i] in order[i] for i in range(n)]))

    good = [i for i in range(n) if tgt[i] in order[i]]
    imgs = to_t(data["images"], device)
    inst = to_t(data["inst"], device)
    picked = 0
    for i in good:
        cand = list(order[i])
        sc = v_scores(v, data["images"][i].astype(np.float32) / 255.0,
                      LABELS[int(np.argmax(data["inst"][i]))], cand, device)
        if cand[int(np.argmax(sc))] == tgt[i]:
            picked += 1
    pick = picked / max(1, len(good))
    return {"ceiling_topk": ceiling, "v_pick_given_in_topk": float(pick),
            "joint": ceiling * pick, "topk": topk}


# ---------- 复合策略 ----------

def pi_topk(pi, img_chw, label, cfg, device, k=3):
    """返回 π 的 top-k cell 及其概率。"""
    with torch.no_grad():
        it = torch.from_numpy(img_chw).unsqueeze(0).to(device)
        oh = torch.from_numpy(instruction_onehot(f"click {label}", len(LABELS)))
        oh = oh.unsqueeze(0).to(device)
        logits = pi(it, oh)
        p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    order = np.argsort(-p)[:k]
    return order, p[order]


def v_scores(v, img_chw, label, cells, device):
    with torch.no_grad():
        it = torch.from_numpy(img_chw).unsqueeze(0).repeat(len(cells), 1, 1, 1).to(device)
        oh = torch.from_numpy(instruction_onehot(f"click {label}", len(LABELS)))
        oh = oh.unsqueeze(0).repeat(len(cells), 1).to(device)
        c = torch.tensor(cells, device=device)
        return v(it, oh, c).cpu().numpy()


# ---------- rollout（带/不带 verifier）----------

def rollout_compound(pi, v, cfg, device, H, rho, k, n_tasks, seed,
                     topk=1, n_rerank=3):
    """
    topk=1 且 v=None      → 纯 π
    v 给定                 → π 的 top-n_rerank 内用 V 重排
    """
    rng = np.random.default_rng(seed)
    per_step, task_ok = [], []

    for _ in range(n_tasks):
        task = make_task(cfg, rng, H, rho)
        screens, prev = [], None
        for labels, _ in task:
            if prev is not None and labels == prev:
                screens.append(screens[-1])
            else:
                screens.append(render_layout(labels, cfg, rng))
            prev = labels

        obs_idx, step_ok = None, []
        for t, (labels_t, L_t) in enumerate(task):
            if (t % k == 0) or (obs_idx is None):
                obs_idx = t
            obs_img = screens[obs_idx][0]

            if v is None:
                kind, pred = predict(pi, "cls", obs_img, L_t, cfg, device)
                chosen = pred
            else:
                cand, _ = pi_topk(pi, obs_img, L_t, cfg, device, k=n_rerank)
                sc = v_scores(v, obs_img, L_t, list(cand), device)
                chosen = int(cand[int(np.argmax(sc))])

            ok = (labels_t[chosen] == L_t)
            step_ok.append(bool(ok)); per_step.append(bool(ok))
        task_ok.append(all(step_ok))

    return {"task_success": float(np.mean(task_ok)),
            "step_acc": float(np.mean(per_step))}


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--v-epochs", type=int, default=15)
    ap.add_argument("--tasks", type=int, default=300)
    args = ap.parse_args()

    epochs = 8 if args.quick else args.epochs
    v_ep = 5 if args.quick else args.v_epochs
    n_tr, n_te = (1500, 500) if args.quick else (4000, 1000)
    Hs = [1, 2, 4] if args.quick else [1, 2, 4, 8]
    rhos = [1.0] if args.quick else [0.3, 1.0]
    n_tasks = 80 if args.quick else args.tasks
    tag = "q" if args.quick else "m"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} epochs={epochs} v_epochs={v_ep} "
          f"Hs={Hs} rhos={rhos} tasks={n_tasks}")

    cfg = cfg_for(GRID, fill=0.40)
    tr = load_or_build(n_tr, 0, cfg, f"tr{tag}")
    te = load_or_build(n_te, 999, cfg, f"te{tag}")

    # 策略：小(w32) 与 大(w64)
    pi_small, m_small = train_single_step(32, "cls", "-", tr, te, cfg,
                                          epochs, device, seed=0)
    pi_big, m_big = train_single_step(64, "cls", "-", tr, te, cfg,
                                      epochs, device, seed=0)
    # verifier（用 π(w32) 的困难负样本训练）
    t0 = time.time()
    pi_train_probs = pi_probs_on(pi_small, tr, device)
    V = train_verifier(32, tr, v_ep, device, seed=0, pi_probs=pi_train_probs)
    v_params = V.n_params()
    V.eval()

    print(f"\n单步 top-1: π(w32)={m_small['hit_rate']:.4f} "
          f"({m_small['params']:,})  π(w64)={m_big['hit_rate']:.4f} "
          f"({m_big['params']:,})")
    print(f"V(w32) params={v_params:,}  "
          f"复合总参={m_small['params']+v_params:,}  "
          f"(π(w64)={m_big['params']:,})  训练耗时={time.time()-t0:.0f}s")

    # V 单独作为打分器（全格 argmax）
    hits = []
    for i in range(te["images"].shape[0]):
        img = te["images"][i].astype(np.float32) / 255.0
        lbl = LABELS[int(np.argmax(te["inst"][i]))]
        sc = v_scores(V, img, lbl, list(range(GRID * GRID)), device)
        hits.append(te["slots"][i] == int(np.argmax(sc)))
    v_alone = float(np.mean(hits))
    print(f"V 单独 argmax 单步准确率: {v_alone:.4f}")

    # 诊断：re-rank 的上限与 V 的实际选择能力
    diag = verify_diagnostics(pi_small, V, te, cfg, device, topk=3)
    print(f"诊断: P(正确∈π top3)={diag['ceiling_topk']:.4f}  "
          f"P(V 选对|在 top3)={diag['v_pick_given_in_topk']:.4f}  "
          f"联合={diag['joint']:.4f}  (π top1={m_small['hit_rate']:.4f})")

    # 多步 rollout
    rows = []
    print(f"\n{'rho':>5}{'H':>4}{'pi32':>9}{'pi64':>9}{'pi32+V':>9}"
          f"{'V gain':>8}{'vs pi64':>9}")
    print("-" * 56)
    for rho in rhos:
        for H in Hs:
            sd = 1000 + H * 7 + int(rho * 100)
            r_s = rollout_compound(pi_small, None, cfg, device, H, rho, 1,
                                   n_tasks, sd)
            r_b = rollout_compound(pi_big, None, cfg, device, H, rho, 1,
                                   n_tasks, sd)
            r_v = rollout_compound(pi_small, V, cfg, device, H, rho, 1,
                                   n_tasks, sd, n_rerank=3)
            rows.append({"rho": rho, "H": H, "pi_small": r_s["task_success"],
                         "pi_big": r_b["task_success"],
                         "pi_small_v": r_v["task_success"],
                         "step_pi_small": r_s["step_acc"],
                         "step_pi_big": r_b["step_acc"],
                         "step_pi_small_v": r_v["step_acc"]})
            print(f"{rho:>5.1f}{H:>4}{r_s['task_success']:>9.4f}"
                  f"{r_b['task_success']:>9.4f}{r_v['task_success']:>9.4f}"
                  f"{r_v['task_success']-r_s['task_success']:>+8.4f}"
                  f"{r_v['task_success']-r_b['task_success']:>+9.4f}")
        print()

    out = {"device": str(device), "grid": GRID, "epochs": epochs,
           "v_epochs": v_ep, "n_train": n_tr, "n_test": n_te,
           "n_tasks": n_tasks, "Hs": Hs, "rhos": rhos,
           "params": {"pi_w32": m_small["params"], "pi_w64": m_big["params"],
                      "V_w32": v_params},
           "single_step": {"pi_w32": m_small["hit_rate"],
                           "pi_w64": m_big["hit_rate"],
                           "v_alone": v_alone},
           "diagnostics": diag,
           "rows": rows}
    Path("p2_verifier_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 p2_verifier_results.json")


if __name__ == "__main__":
    main()
