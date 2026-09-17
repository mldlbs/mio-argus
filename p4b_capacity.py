"""
P4a'：让"通才覆盖不了"成为可测事实，再测 population 是否终于有增益。

P4a 的结论是：本任务空间**不需要专精**——通才在 1/5 参数预算下就饱和。
所以 population 无价值。P4a' 做两件事让结论有可能翻转：

1. **加宽任务空间**：引入 D 个 domain，每个 domain 有**不同的 label→颜色 映射**。
   通才必须同时装下 D 张查找表；专精者只需一张。
   （现实类比：不同 App 用不同配色方案。）
2. **压缩容量**：扫 width 4→32，找出通才**开始失败**的区间。

然后在**参数匹配**下比较：
    population(N 个 width=w 专精者)  vs  一个 width=W 的通才，params(W) ≈ N×params(w)

指标：
  generalist  单体
  oracle      N 个成员里任意一个对（verifier 可达到的上界）
  vote        多数投票（可部署的集成）

判据：若 oracle/vote 在匹配参数下 > 通才 → population 终于成立。
      若仍 ≤ → population 路线整体放弃。

用法:
    python p4b_capacity.py --quick
    python p4b_capacity.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p0_env import (EnvCfg, Element, render, PALETTE, LABEL_TO_ID, LABELS,
                    instruction_onehot)
from p0_models import P0Net, SpatialBackbone

GRID = 8
IMG_SIZE = 160
N_DOMAINS = 6
CACHE = Path("p0_cache")

def env_cfg(fill=0.35, jitter=4):
    return EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=GRID * GRID,
                  fill=fill, jitter=jitter, color_by_label=True)


def domain_perms(seed=0):
    """每个 domain 一张 label→palette 的随机置换。"""
    rng = np.random.default_rng(seed)
    V = GRID * GRID
    P = len(PALETTE)
    return [rng.permutation(P)[:V] for _ in range(N_DOMAINS)]


def build_elements(labels, perm, cfg, rng):
    els = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = cfg.slot_bbox(slot)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, cfg.img_size - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, cfg.img_size - 1))
        lab = labels[slot]
        color = tuple(int(c) for c in PALETTE[int(perm[LABEL_TO_ID[lab]])])
        els.append(Element(slot, (ex1, ey1, ex2, ey2), lab, "button", color))
    return els


def build_data(n, seed, domains, perms, fills=(0.35,), jitters=(4,)):
    """domains: 参与采样的 domain id 列表。"""
    rng = np.random.default_rng(seed)
    S = GRID * GRID
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    dom = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d = int(rng.choice(domains))
        fill = float(rng.choice(fills))
        jit = int(rng.choice(jitters))
        cfg = env_cfg(fill, jit)
        vocab = LABELS[:S]
        idx = rng.choice(len(vocab), size=S, replace=False)
        labels = [vocab[int(j)] for j in idx]
        slot = int(rng.integers(0, S))
        els = build_elements(labels, perms[d], cfg, rng)
        img = render(els, cfg)
        imgs[i] = (img * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {labels[slot]}", len(LABELS))
        slots[i] = slot
        dom[i] = d
    return {"images": imgs, "inst": inst, "slots": slots, "dom": dom,
            "xys": np.zeros((n, 2), dtype=np.float32),
            "bbox": np.zeros((n, 4), dtype=np.int32),
            "centers": np.zeros((n, S, 2), dtype=np.float32)}


def train(width, tr, epochs, device, seed=0, lr=3e-3, batch=64, inst_dim=32):
    torch.manual_seed(seed)
    m = P0Net(width=width, head="cls", num_slots=GRID * GRID, grid=GRID,
              num_labels=len(LABELS), inst_dim=inst_dim,
              in_res=IMG_SIZE, stages=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    imgs = tr["images"].astype(np.float32) / 255.0
    imgs = torch.from_numpy(imgs).to(device)
    inst = torch.from_numpy(tr["inst"]).to(device)
    tgt = torch.from_numpy(tr["slots"]).to(device)
    n = imgs.shape[0]; spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(m(imgs[idx], inst[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def predict_all(models, data, device, batch=256):
    """返回 (M, n) 的 top-1 预测。"""
    imgs = torch.from_numpy(data["images"].astype(np.float32) / 255.0).to(device)
    inst = torch.from_numpy(data["inst"]).to(device)
    outs = []
    for m in models:
        m.eval()
        p = []
        with torch.no_grad():
            for i in range(0, imgs.shape[0], batch):
                p.append(m(imgs[i:i + batch], inst[i:i + batch]).argmax(-1).cpu().numpy())
        outs.append(np.concatenate(p))
    return np.stack(outs, 0)


def acc1(pred, tgt):
    return float((pred == tgt).mean())


def oracle(preds, tgt):
    return float(np.mean([(preds[:, i] == tgt[i]).any() for i in range(len(tgt))]))


def vote(preds, tgt):
    S = GRID * GRID
    n = preds.shape[1]
    ok = 0
    for i in range(n):
        c = np.bincount(preds[:, i], minlength=S)
        ok += int(np.argmax(c) == tgt[i])
    return ok / n


def params_of(width, inst_dim=32):
    m = P0Net(width=width, head="cls", num_slots=GRID * GRID, grid=GRID,
              num_labels=len(LABELS), inst_dim=inst_dim,
              in_res=IMG_SIZE, stages=3)
    return sum(p.numel() for p in m.parameters())


# ---------- Router：从屏幕推断 domain，把请求转给对应的专精者 ----------

class Router(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.bb = SpatialBackbone(width, IMG_SIZE, 3)
        self.head = nn.Linear(self.bb.out_channels, N_DOMAINS)

    def forward(self, x):
        f = self.bb(x).mean(dim=(2, 3))
        return self.head(f)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_router(width, tr, epochs, device, seed=0, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    r = Router(width).to(device)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    imgs = torch.from_numpy(tr["images"].astype(np.float32) / 255.0).to(device)
    dom = torch.from_numpy(tr["dom"]).to(device)
    n = imgs.shape[0]; spe = max(1, n // batch)
    r.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(r(imgs[idx]), dom[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return r


def router_acc(r, data, device, batch=256):
    r.eval()
    imgs = torch.from_numpy(data["images"].astype(np.float32) / 255.0).to(device)
    p = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch):
            p.append(r(imgs[i:i + batch]).argmax(-1).cpu().numpy())
    p = np.concatenate(p)
    return p, float((p == data["dom"]).mean())


def main():
    global GRID, N_DOMAINS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--widths", type=str, default="4,6,8,12,16,24,32")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--domains", type=int, default=6)
    args = ap.parse_args()
    GRID = args.grid
    N_DOMAINS = args.domains
    epochs = args.epochs if args.epochs else (10 if args.quick else 25)
    n_tr = args.n_train if args.n_train else (2000 if args.quick else 6000)
    n_te = args.n_test if args.n_test else (500 if args.quick else 1500)
    widths = [int(x) for x in args.widths.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perms = domain_perms(0)
    all_domains = list(range(N_DOMAINS))
    print(f"device={device} grid={GRID} S={GRID*GRID} domains={N_DOMAINS} "
          f"epochs={epochs} n_train={n_tr} n_test={n_te}")
    print(f"widths={widths}")

    te = build_data(n_te, 777, all_domains, perms)
    tgt = te["slots"]

    # ---- 1) 通才宽度扫描：找失败区间 ----
    print(f"\n{'width':>6}{'params':>10}{'gen acc':>10}{'per-domain acc':>34}")
    print("-" * 62)
    gen_rows = []
    for w in widths:
        tr = build_data(n_tr, 100, all_domains, perms)
        m = train(w, tr, epochs, device)
        preds = predict_all([m], te, device)
        a = acc1(preds[0], tgt)
        per_d = []
        for d in all_domains:
            mask = te["dom"] == d
            per_d.append(float((preds[0][mask] == tgt[mask]).mean()))
        gen_rows.append({"width": w, "params": params_of(w), "acc": a,
                         "per_domain": per_d})
        print(f"{w:>6}{params_of(w):>10,}{a:>10.4f}   " +
              " ".join(f"{v:.3f}" for v in per_d))

    # ---- 2) 参数匹配：population vs 通才 ----
    print(f"\n{'N':>3}{'w_mem':>7}{'w_gen':>7}{'pop+router':>12}{'gen':>12}"
          f"{'gen acc':>9}{'oracle':>9}{'vote':>9}{'routed':>9}{'router acc':>11}")
    print("-" * 92)
    match_rows = []
    for N in sorted(set([2, min(3, N_DOMAINS), N_DOMAINS])):
        for w in (6, 8, 12):
            p_mem = params_of(w)
            total = N * p_mem
            w_gen = min(widths, key=lambda W: abs(params_of(W) - total))
            if params_of(w_gen) > total * 1.6:
                continue
            spec_tr = [build_data(n_tr // max(1, N), 200 + d, [d], perms)
                       for d in range(N)]
            members = [train(w, spec_tr[i % N], epochs, device, seed=i)
                       for i in range(N)]
            preds = predict_all(members, te, device)
            o, v = oracle(preds, tgt), vote(preds, tgt)

            # router（训练在混合数据上，含 domain 标签）
            rt_tr = build_data(n_tr, 500, all_domains, perms)
            R = train_router(8, rt_tr, max(8, epochs // 2), device)
            rd, racc = router_acc(R, te, device)
            rd = np.minimum(rd, N - 1)      # 专精者数量可能少于 domain 数
            routed = float(np.mean([preds[rd[i], i] == tgt[i]
                                    for i in range(len(tgt))]))
            r_params = R.n_params()

            gt = next(r for r in gen_rows if r["width"] == w_gen)
            match_rows.append({"N": N, "w_mem": w, "w_gen": w_gen,
                               "pop_params": total + r_params,
                               "gen_params": params_of(w_gen),
                               "router_params": r_params,
                               "router_acc": racc,
                               "gen_acc": gt["acc"], "oracle": o, "vote": v,
                               "routed": routed})
            print(f"{N:>3}{w:>7}{w_gen:>7}{total + r_params:>12,}"
                  f"{params_of(w_gen):>12,}{gt['acc']:>9.4f}"
                  f"{o:>9.4f}{v:>9.4f}{routed:>9.4f}{racc:>11.4f}")

    # ---- 3) 逐 domain 对比（N=6 全专精 vs 通才）----
    print(f"\n逐 domain 准确率（N=6 专精者 vs 最优通才）")
    print(f"{'domain':>8}" + "".join(f"{'m'+str(i):>10}" for i in range(N_DOMAINS))
          + f"{'pop':>10}{'gen':>10}")
    print("-" * 88)
    spec_tr = [build_data(n_tr // N_DOMAINS, 300 + d, [d], perms)
               for d in range(N_DOMAINS)]
    members = [train(12, spec_tr[d], epochs, device, seed=d)
               for d in range(N_DOMAINS)]
    preds = predict_all(members, te, device)
    best_gen = max(gen_rows, key=lambda r: r["acc"])
    for d in range(N_DOMAINS):
        mask = te["dom"] == d
        row = [float((preds[i][mask] == tgt[mask]).mean()) for i in range(N_DOMAINS)]
        po = float(np.mean([(preds[:, i] == tgt[i]).any()
                            for i in np.where(mask)[0]]))
        print(f"{d:>8}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{po:>10.4f}{best_gen['per_domain'][d]:>10.4f}")

    out = {"device": str(device), "grid": GRID, "domains": N_DOMAINS,
           "epochs": epochs, "n_train": n_tr, "n_test": n_te,
           "widths": widths, "generalist": gen_rows, "matched": match_rows}
    Path("p4b_capacity_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p4b_capacity_results.json")


if __name__ == "__main__":
    main()
