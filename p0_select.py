"""
P0.5：把 oracle 结构换成廉价解析器，扫描「解析器质量 → H2 优势」。

三个系统在同一测试集上比较：
  reg         : 像素 → soft-argmax → (x,y)          无需解析器
  parser-cls  : 解析器候选 → 选一个 → 执行框中心     需要解析器
  (grid-cls)  : P0 已覆盖，此处不重复

parser-cls 的上限由解析器 recall 决定（目标没被检出就一定失败）。
关键问题：解析器质量退化到什么程度时，parser-cls 掉到 reg 的水平？

用法：
    python p0_select.py --quick
    python p0_select.py --epochs 25 --seeds 2
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

from p0_env import EnvCfg, make_sample, hit, instruction_onehot, LABELS
from p0_models import SpatialBackbone, P0Net
from p0_parser import cheap_parse, degrade, match_index, quality

CACHE = Path("p0_cache")
CACHE.mkdir(exist_ok=True)

N_MAX = 96
IMG_SIZE = 160
GRID = 6
FILL = 0.40
ENV_CFG = EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=GRID * GRID,
                 fill=FILL, jitter=4, color_by_label=True)

# 解析器质量档位（越往下越差）
LEVELS = [
    ("Q1-clean", {}),
    ("Q2-jit1", {"jit": 1}),
    ("Q3-jit2", {"jit": 2}),
    ("Q4-jit3", {"jit": 3}),
    ("Q5-drop15", {"jit": 2, "drop": 0.15}),
    ("Q6-drop30", {"jit": 2, "drop": 0.30}),
    ("Q7-drop30+spur", {"jit": 3, "drop": 0.30, "spurious": 2.0}),
    ("Q8-heavy", {"jit": 4, "drop": 0.45, "spurious": 3.0, "shrink": 0.15}),
]


# ---------- 数据集 ----------

def build_dataset(n: int, seed: int, level_kw: dict) -> dict:
    cfg = ENV_CFG
    rng = np.random.default_rng(seed)
    prng = np.random.default_rng(seed + 777)

    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    boxes = np.zeros((n, N_MAX, 4), dtype=np.float32)
    mask = np.zeros((n, N_MAX), dtype=bool)
    match = np.full(n, -1, dtype=np.int64)
    tbox = np.zeros((n, 4), dtype=np.int32)
    centers = np.zeros((n, 2), dtype=np.float32)

    for i in range(n):
        s = make_sample(rng, cfg)
        img_hwc = (s.image.transpose(1, 2, 0) * 255).astype(np.uint8)   # 给解析器
        imgs[i] = (s.image * 255).astype(np.uint8)                       # 给模型
        inst[i] = instruction_onehot(s.instruction, len(LABELS))
        tbox[i] = s.target_bbox
        centers[i] = np.array(s.target_xy, dtype=np.float32)

        dets = degrade(cheap_parse(img_hwc, img_size=IMG_SIZE), prng,
                       img_size=IMG_SIZE, **level_kw)[:N_MAX]
        for j, (x1, y1, x2, y2) in enumerate(dets):
            boxes[i, j] = (x1 / IMG_SIZE, y1 / IMG_SIZE, x2 / IMG_SIZE, y2 / IMG_SIZE)
            mask[i, j] = True
        match[i] = match_index(dets, s.target_bbox)

    return {"images": imgs, "inst": inst, "boxes": boxes, "mask": mask,
            "match": match, "tbox": tbox, "centers": centers}


def load_or_build(n, seed, level_kw, tag) -> dict:
    lv = "-".join(f"{k}{v}" for k, v in sorted(level_kw.items())) or "clean"
    key = f"g{GRID}_i{IMG_SIZE}_f{FILL}_{lv}"
    p = CACHE / f"sel_{tag}_{key}_{n}_{seed}.npz"
    if p.exists():
        d = np.load(p)
        return {k: d[k] for k in d.files}
    d = build_dataset(n, seed, level_kw)
    np.savez_compressed(p, **d)
    return d


def parser_quality_on(data) -> dict:
    cov, ious = [], []
    for i in range(data["tbox"].shape[0]):
        m = data["mask"][i]
        dets = [tuple(data["boxes"][i, j] * IMG_SIZE) for j in range(m.sum())]
        q = quality(dets, data["tbox"][i])
        cov.append(q["covered"]); ious.append(q["best_iou"])
    return {"recall": float(np.mean(cov)), "best_iou": float(np.mean(ious)),
            "n_det": float(data["mask"].sum(1).mean())}


# ---------- 选择器模型 ----------

class SelectorNet(nn.Module):
    """backbone → 指令 query → 对每个候选框打分。参数量与 P0Net 一致（同 backbone）。"""

    def __init__(self, width=32, num_labels=64, inst_dim=64,
                 in_res=IMG_SIZE, stages=3):
        super().__init__()
        self.backbone = SpatialBackbone(width, in_res, stages)
        C = self.backbone.out_channels
        self.inst_emb = nn.Embedding(num_labels, inst_dim)
        self.query = nn.Linear(inst_dim, C)
        self.C = C

    def pool(self, feat, boxes, mask):
        B, C, h, w = feat.shape
        N = boxes.shape[1]
        out = torch.zeros(B, N, C, dtype=feat.dtype, device=feat.device)
        scale = torch.tensor([w, h, w, h], dtype=feat.dtype, device=feat.device)
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            bx = boxes[b, idx] * scale
            rois = torch.cat([
                torch.full((idx.numel(), 1), float(b), dtype=feat.dtype, device=feat.device),
                bx], dim=1)
            p = roi_align(feat, rois, output_size=(1, 1),
                          spatial_scale=1.0, aligned=True)
            out[b, idx] = p.reshape(idx.numel(), C)
        return out

    def forward(self, images, inst, boxes, mask):
        feat = self.backbone(images)
        q = self.query(self.inst_emb(inst.argmax(-1)))
        pooled = self.pool(feat, boxes, mask)
        score = (pooled * q[:, None, :]).sum(-1) / math.sqrt(self.C)
        return score.masked_fill(~mask, -1e9)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------- 训练 / 评估 ----------

def to_t(x, device):
    if x.dtype == np.uint8:
        return torch.from_numpy(x).float().div_(255.0).to(device)
    return torch.from_numpy(x).to(device)


def train_selector(width, tr, te, epochs, device, seed=0, lr=1e-3, batch=64):
    torch.manual_seed(seed)
    m = SelectorNet(width=width).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    images = to_t(tr["images"], device); inst = to_t(tr["inst"], device)
    boxes = to_t(tr["boxes"], device); mask = torch.from_numpy(tr["mask"]).to(device)
    match = torch.from_numpy(tr["match"]).to(device)

    n = images.shape[0]; spe = max(1, n // batch)
    t0 = time.time(); m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            score = m(images[idx], inst[idx], boxes[idx], mask[idx])
            loss = F.cross_entropy(score, match[idx], ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step()

    # 评估：argmax → 执行该框中心 → 是否落在真实目标 bbox 内
    m.eval()
    hits, sel_ok = [], []
    te_n = te["images"].shape[0]
    with torch.no_grad():
        for i in range(0, te_n, 128):
            ti = to_t(te["images"][i:i + 128], device)
            ii = to_t(te["inst"][i:i + 128], device)
            bb = to_t(te["boxes"][i:i + 128], device)
            mk = torch.from_numpy(te["mask"][i:i + 128]).to(device)
            sc = m(ti, ii, bb, mk)
            pred = sc.argmax(-1).cpu().numpy()
            for k in range(pred.shape[0]):
                gi = i + k
                if not mk.cpu().numpy()[k, pred[k]]:
                    hits.append(False); sel_ok.append(False); continue
                bx = te["boxes"][gi, pred[k]]
                cx = (bx[0] + bx[2]) / 2 * IMG_SIZE
                cy = (bx[1] + bx[3]) / 2 * IMG_SIZE
                hits.append(bool(hit((cx / IMG_SIZE, cy / IMG_SIZE),
                                     te["tbox"][gi], ENV_CFG)))
                sel_ok.append(pred[k] == te["match"][gi])

    return {"params": m.n_params(), "hit_rate": float(np.mean(hits)),
            "sel_acc_on_covered": float(np.mean(sel_ok)),
            "train_s": round(time.time() - t0, 1)}


def train_reg(width, tr, te, epochs, device, seed=0, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    m = P0Net(width=width, head="reg", num_slots=GRID * GRID, grid=GRID,
              num_labels=len(LABELS), in_res=IMG_SIZE, stages=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    images = to_t(tr["images"], device); inst = to_t(tr["inst"], device)
    target = to_t(tr["centers"], device)
    n = images.shape[0]; spe = max(1, n // batch)
    t0 = time.time(); m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            o = m(images[idx], inst[idx])
            loss = F.mse_loss(o, target[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    hits = []
    with torch.no_grad():
        for i in range(0, te["images"].shape[0], 256):
            ti = to_t(te["images"][i:i + 256], device)
            ii = to_t(te["inst"][i:i + 256], device)
            o = m(ti, ii).cpu().numpy()
            for k in range(o.shape[0]):
                hits.append(bool(hit(o[k], te["tbox"][i + k], ENV_CFG)))
    return {"params": m.n_params(), "hit_rate": float(np.mean(hits)),
            "train_s": round(time.time() - t0, 1)}


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    epochs, seeds = (8, 1) if args.quick else (args.epochs, args.seeds)
    n_tr, n_te = (1200, 400) if args.quick else (4000, 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} grid={GRID} slots={GRID*GRID} width={args.width} "
          f"train={n_tr} test={n_te} epochs={epochs} seeds={seeds}")
    tag = "q" if args.quick else "m"

    # reg 基线（不依赖解析器）
    ref = load_or_build(n_tr, 0, {}, f"reg{tag}")
    ref_te = load_or_build(n_te, 999, {}, f"reg{tag}")
    regs = [train_reg(args.width, ref, ref_te, epochs, device, seed=s) for s in range(seeds)]
    reg_hit = float(np.mean([r["hit_rate"] for r in regs]))
    print(f"\n[reg baseline] hit={reg_hit:.4f} params={regs[0]['params']:,}\n")

    levels = LEVELS[:3] if args.quick else LEVELS
    rows = []
    for name, kw in levels:
        tr = load_or_build(n_tr, 0, kw, tag)
        te = load_or_build(n_te, 999, kw, tag)
        pq = parser_quality_on(te)
        ms = [train_selector(args.width, tr, te, epochs, device, seed=s)
              for s in range(seeds)]
        hit_m = float(np.mean([m["hit_rate"] for m in ms]))
        sel_m = float(np.mean([m["sel_acc_on_covered"] for m in ms]))
        rows.append({"level": name, **pq, "cls_hit": hit_m, "sel_acc": sel_m,
                     "params": ms[0]["params"], "kw": kw})
        print(f"{name:<16} recall={pq['recall']:.3f} IoU={pq['best_iou']:.3f} "
              f"n_det={pq['n_det']:.1f}  cls_hit={hit_m:.4f} "
              f"sel_acc={sel_m:.4f}  Δ(vs reg)={hit_m-reg_hit:+.4f}")
        del tr, te

    print(f"\n{'-'*78}")
    print(f"{'level':<16}{'recall':>8}{'IoU':>7}{'n_det':>7}{'cls_hit':>9}"
          f"{'reg':>9}{'Δ':>9}{'sel_acc':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['level']:<16}{r['recall']:>8.3f}{r['best_iou']:>7.3f}"
              f"{r['n_det']:>7.1f}{r['cls_hit']:>9.4f}{reg_hit:>9.4f}"
              f"{r['cls_hit']-reg_hit:>+9.4f}{r['sel_acc']:>9.4f}")
    print("-" * 78)

    out = {"device": str(device), "grid": GRID, "width": args.width,
           "n_train": n_tr, "n_test": n_te, "epochs": epochs, "seeds": seeds,
           "reg_hit": reg_hit, "reg_params": regs[0]["params"], "levels": rows}
    Path("p0_parser_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p0_parser_results.json")


if __name__ == "__main__":
    main()
