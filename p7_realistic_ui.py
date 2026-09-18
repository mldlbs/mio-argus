"""
P7：非网格、视觉复杂的 UI —— H2 能否迁移？

动机（REPORT.md §10 最大局限）
------------------------------
所有结论都建立在**网格状、纯色、无纹理**的合成 UI 上。
真实 UI 是：元素自由摆放、大小不一、有圆角/边框/阴影/渐变、会重叠、背景有纹理。

本实验去掉网格，做两因子分解：

  reg          像素 → (x,y)                 连续动作空间
  cls-oracle   真值框 → 选择 → 执行中心        纯动作空间效应（解析器完美）
  cls-parser   廉价 CV 解析器框 → 选择 → 执行  加上解析器代价

判据：
  - `cls-oracle > reg` → **动作空间优势可迁移**（与网格无关）
  - `cls-parser < cls-oracle` → 解析器代价；差距大小 = 真实 UI 的额外成本
  - 解析器 recall 是关键量（P0.5 已证 crossing 在 recall≈0.5）

用法:
    python p7_realistic_ui.py --quick
    python p7_realistic_ui.py
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision.ops import roi_align

from p0_env import PALETTE, LABEL_TO_ID, LABELS, instruction_onehot, _font
from p0_models import P0Net, SpatialBackbone

IMG_SIZE = 192
N_ELEM = 12
N_MAX = 64            # 候选上限（extreme 布局可达 42 个元素）
VOCAB = 48


# ---------- 自由摆放的元素 ----------

class El:
    __slots__ = ("box", "label", "color", "z")

    def __init__(self, box, label, color, z):
        self.box, self.label, self.color, self.z = box, label, color, z


def make_elements(rng, n=N_ELEM, min_gap=4, tries=200,
                  wrange=(34, 118), hrange=(20, 46)):
    """拒绝采样：自由位置 + 大小不一。wrange/hrange 控制**精度要求**。"""
    boxes, labels = [], []
    vocab = LABELS[:VOCAB]
    used = set()
    for _ in range(n):
        for _t in range(tries):
            w = int(rng.integers(wrange[0], wrange[1]))
            h = int(rng.integers(hrange[0], hrange[1]))
            x = int(rng.integers(2, IMG_SIZE - w - 2))
            y = int(rng.integers(2, IMG_SIZE - h - 2))
            box = (x, y, x + w, y + h)
            if all(not (x < b[2] + min_gap and b[0] - min_gap < box[2] and
                        y < b[3] + min_gap and b[1] - min_gap < box[3])
                   for b in boxes):
                boxes.append(box)
                break
        else:
            continue
        for _t in range(50):
            lab = vocab[int(rng.integers(0, len(vocab)))]
            if lab not in used:
                used.add(lab)
                break
        labels.append(lab)
    return boxes, labels


def render_ui(boxes, labels, rng, background="flat", rich=True):
    """自由摆放 + 圆角 + 阴影 + 渐变 + 文字。"""
    if background == "flat":
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (238, 240, 244))
    elif background == "texture":
        k = int(rng.integers(8, 20))
        small = rng.integers(120, 235, (k, k, 3), dtype=np.uint8)
        img = Image.fromarray(small, "RGB").resize((IMG_SIZE, IMG_SIZE),
                                                   Image.BICUBIC)
    else:  # natural
        src = Image.open("assets/nanogpt.jpg").convert("RGB")
        W, H = src.size
        cw = int(rng.integers(120, W // 2)); ch = int(cw * 0.8)
        x = int(rng.integers(0, max(1, W - cw))); y = int(rng.integers(0, max(1, H - ch)))
        img = src.crop((x, y, x + cw, y + ch)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    order = list(range(len(boxes)))
    rng.shuffle(order)
    font = _font(11)

    # 阴影层
    if rich:
        sh = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        for i in order:
            x1, y1, x2, y2 = boxes[i]
            ds.rounded_rectangle([x1 + 2, y1 + 3, x2 + 2, y2 + 3], radius=5,
                                 fill=(0, 0, 0, 70))
        sh = sh.filter(ImageFilter.GaussianBlur(2))
        img = Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB")

    d = ImageDraw.Draw(img)
    els = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        lab = labels[i]
        base = PALETTE[LABEL_TO_ID[lab] % len(PALETTE)]
        base = tuple(int(c) for c in base)
        if rich:
            # 竖直渐变
            for yy in range(y1, y2):
                t = (yy - y1) / max(1, y2 - y1)
                c = tuple(int(base[k] * (1 - 0.25 * t) + 255 * 0.25 * t * 0.35)
                          for k in range(3))
                d.line([(x1 + 1, yy), (x2 - 1, yy)], fill=c)
            d.rounded_rectangle([x1, y1, x2, y2], radius=5,
                                outline=(35, 35, 40), width=1)
        else:
            d.rectangle([x1, y1, x2, y2], fill=base, outline=(25, 25, 25))
        if (x2 - x1) >= 30 and (y2 - y1) >= 14:
            bb = d.textbbox((0, 0), lab, font=font)
            d.text((x1 + max(2, ((x2 - x1) - (bb[2] - bb[0])) // 2),
                    y1 + max(2, ((y2 - y1) - (bb[3] - bb[1])) // 2 - 2)),
                   lab, fill=(255, 255, 255), font=font)
        els.append(El((x1, y1, x2, y2), lab, base, len(els)))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), els


# ---------- 廉价 CV 解析器 ----------

def cheap_parse(chw, dist_thr=34, min_area=90, open_k=3):
    import cv2
    img = (chw.transpose(1, 2, 0) * 255).astype(np.uint8)
    # 背景估计：取四角中位数
    corners = np.concatenate([img[:6, :6].reshape(-1, 3), img[:6, -6:].reshape(-1, 3),
                              img[-6:, :6].reshape(-1, 3), img[-6:, -6:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    d = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    mask = (d > dist_thr).astype(np.uint8)
    if open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((open_k, open_k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 12 or h < 10:
            continue
        out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def center(box):
    return ((box[0] + box[2]) / 2 / IMG_SIZE, (box[1] + box[3]) / 2 / IMG_SIZE)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_in(box, tbox):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    return tbox[0] <= cx <= tbox[2] and tbox[1] <= cy <= tbox[3]


def match_idx(dets, tbox):
    cand = [i for i, b in enumerate(dets) if center_in(b, tbox)]
    return max(cand, key=lambda i: iou(dets[i], tbox)) if cand else -1


# ---------- 数据 ----------

def build_data(n, seed, background="flat", rich=True, use_parser=True,
               layout="sparse"):
    rng = np.random.default_rng(seed)
    if layout == "sparse":       # 少而大 → 低精度要求（典型 UI）
        ne, wr, hr = 12, (34, 118), (20, 46)
    elif layout == "medium":
        ne, wr, hr = 20, (26, 80), (16, 36)
    elif layout == "dense":
        ne, wr, hr = 30, (18, 56), (12, 26)
    else:                        # extreme: 多而小 → 高精度要求（拥挤 UI / 小图标）
        ne, wr, hr = 42, (13, 42), (10, 22)
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    xy = np.zeros((n, 2), dtype=np.float32)
    tbox = np.zeros((n, 4), dtype=np.int32)
    boxes = np.zeros((n, N_MAX, 4), dtype=np.float32)
    mask = np.zeros((n, N_MAX), dtype=bool)
    match = np.full(n, -1, dtype=np.int64)
    recall = []

    for i in range(n):
        b, labs = make_elements(rng, n=ne, wrange=wr, hrange=hr)
        chw, els = render_ui(b, labs, rng, background, rich)
        k = int(rng.integers(0, len(els)))
        t = els[k]
        imgs[i] = (chw * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {t.label}", len(LABELS))
        xy[i] = center(t.box)
        tbox[i] = t.box

        cands = cheap_parse(chw) if use_parser else [e.box for e in els]
        mi = match_idx(cands, t.box)
        # 只保留前 N_MAX 个候选：若真值候选被截断，该样本不可用（target 越界会
        # 触发 CUDA device-side assert —— 踩过这个坑）
        if mi >= N_MAX:
            mi = -1
        recall.append(mi >= 0)
        for j, bx in enumerate(cands[:N_MAX]):
            boxes[i, j] = (bx[0]/IMG_SIZE, bx[1]/IMG_SIZE,
                           bx[2]/IMG_SIZE, bx[3]/IMG_SIZE)
            mask[i, j] = True
        match[i] = mi

    return {"images": imgs, "inst": inst, "xy": xy, "tbox": tbox,
            "boxes": boxes, "mask": mask, "match": match,
            "recall": float(np.mean(recall))}


# ---------- 模型 ----------

class Selector(nn.Module):
    def __init__(self, width=32, inst_dim=48):
        super().__init__()
        self.bb = SpatialBackbone(width, IMG_SIZE, 3)
        C = self.bb.out_channels
        self.emb = nn.Embedding(len(LABELS), inst_dim)
        self.q = nn.Linear(inst_dim, C)
        self.C = C

    def pool(self, feat, boxes, mask):
        B, C, h, w = feat.shape
        N = boxes.shape[1]
        out = torch.zeros(B, N, C, dtype=feat.dtype, device=feat.device)
        sc = torch.tensor([w, h, w, h], dtype=feat.dtype, device=feat.device)
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            bx = boxes[b, idx] * sc
            rois = torch.cat([torch.full((idx.numel(), 1), float(b),
                                         dtype=feat.dtype, device=feat.device), bx], 1)
            out[b, idx] = roi_align(feat, rois, (1, 1), 1.0, aligned=True
                                    ).reshape(idx.numel(), C)
        return out

    def forward(self, images, inst, boxes, mask):
        feat = self.bb(images)
        q = self.q(self.emb(inst.argmax(-1)))
        pooled = self.pool(feat, boxes, mask)
        s = (pooled * q[:, None, :]).sum(-1) / math.sqrt(self.C)
        return s.masked_fill(~mask, -1e9)


def to_t(x, device):
    if x.dtype == np.uint8:
        return torch.from_numpy(x).float().div_(255.0).to(device)
    return torch.from_numpy(x).to(device)


def train_reg(width, tr, epochs, device, lr=3e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    m = P0Net(width=width, head="reg", num_slots=N_ELEM, grid=3,
              num_labels=len(LABELS), inst_dim=48, in_res=IMG_SIZE, stages=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    I = to_t(tr["images"], device); T = to_t(tr["inst"], device)
    Y = to_t(tr["xy"], device)
    n = I.shape[0]; spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k*batch:(k+1)*batch]
            loss = F.mse_loss(m(I[idx], T[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_sel(width, tr, epochs, device, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    m = Selector(width).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    I = to_t(tr["images"], device); T = to_t(tr["inst"], device)
    BO = to_t(tr["boxes"], device); MK = torch.from_numpy(tr["mask"]).to(device)
    Y = torch.from_numpy(tr["match"]).to(device)
    n = I.shape[0]; spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k*batch:(k+1)*batch]
            yb = Y[idx]
            # 整批都没有正样本时 cross_entropy(ignore_index) 会产生 NaN 并触发
            # CUDA device-side assert —— 必须跳过（extreme 布局下解析 recall 低，常见）
            if (yb == -1).all():
                continue
            loss = F.cross_entropy(m(I[idx], T[idx], BO[idx], MK[idx]), yb,
                                   ignore_index=-1)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def eval_reg(m, te, device):
    m.eval()
    I = to_t(te["images"], device); T = to_t(te["inst"], device)
    hits = []
    with torch.no_grad():
        for i in range(0, I.shape[0], 256):
            o = m(I[i:i+256], T[i:i+256]).cpu().numpy()
            for k in range(o.shape[0]):
                x = o[k, 0]*IMG_SIZE; y = o[k, 1]*IMG_SIZE
                b = te["tbox"][i+k]
                hits.append(b[0] <= x <= b[2] and b[1] <= y <= b[3])
    return float(np.mean(hits))


def eval_sel(m, te, device):
    m.eval()
    I = to_t(te["images"], device); T = to_t(te["inst"], device)
    BO = to_t(te["boxes"], device); MK = torch.from_numpy(te["mask"]).to(device)
    hits = []
    with torch.no_grad():
        for i in range(0, I.shape[0], 256):
            s = m(I[i:i+256], T[i:i+256], BO[i:i+256], MK[i:i+256])
            pred = s.argmax(-1).cpu().numpy()
            mk = te["mask"][i:i+256]
            for k in range(pred.shape[0]):
                gi = i + k
                if not mk[k, pred[k]]:
                    hits.append(False); continue
                bx = te["boxes"][gi, pred[k]] * IMG_SIZE
                cx, cy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2
                b = te["tbox"][gi]
                hits.append(b[0] <= cx <= b[2] and b[1] <= cy <= b[3])
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (12 if args.quick else 30)
    n_tr = args.n_train if args.n_train else (2000 if args.quick else 6000)
    n_te = args.n_test if args.n_test else (600 if args.quick else 1500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} IMG={IMG_SIZE} elements={N_ELEM} "
          f"n_train={n_tr} n_test={n_te} epochs={epochs} width={args.width}")

    if args.quick:
        combos = [("sparse", "flat", True), ("dense", "flat", True)]
    else:
        combos = [("sparse", "flat", True), ("medium", "flat", True),
                  ("dense", "flat", True), ("extreme", "flat", True),
                  ("extreme", "texture", True), ("extreme", "natural", True)]

    rows = []
    print(f"\n{'layout+bg':<16}{'elems':>7}{'parser recall':>14}{'reg':>9}"
          f"{'cls-oracle':>12}{'cls-parser':>12}{'space':>11}{'parser cost':>13}")
    print("-" * 95)
    for layout, bg, rich in combos:
        name = f"{layout}+{bg}"
        tr_o = build_data(n_tr, 100, bg, rich, False, layout)
        te_o = build_data(n_te, 900, bg, rich, False, layout)
        tr_p = build_data(n_tr, 100, bg, rich, True, layout)
        te_p = build_data(n_te, 900, bg, rich, True, layout)

        reg = train_reg(args.width, tr_o, epochs, device)
        sel_o = train_sel(args.width, tr_o, epochs, device)
        sel_p = train_sel(args.width, tr_p, epochs, device)

        r_reg = eval_reg(reg, te_o, device)
        r_or = eval_sel(sel_o, te_o, device)
        r_pa = eval_sel(sel_p, te_p, device)
        rows.append({"config": name, "layout": layout, "background": bg,
                     "parser_recall": te_p["recall"],
                     "reg": r_reg, "cls_oracle": r_or, "cls_parser": r_pa,
                     "space_effect": r_or - r_reg,
                     "parser_cost": r_or - r_pa})
        print(f"{name:<16}{te_p['recall']:>14.4f}{r_reg:>9.4f}"
              f"{r_or:>12.4f}{r_pa:>12.4f}{r_or - r_reg:>+11.4f}"
              f"{r_or - r_pa:>+13.4f}")

    print(f"\n{'='*95}")
    print("decomposition: space = cls-oracle - reg (pure action-space advantage)")
    print("               parser cost = cls-oracle - cls-parser (imperfect parser)")
    for r in rows:
        print(f"  {r['config']:<16} recall={r['parser_recall']:.3f}  "
              f"space={r['space_effect']:+.4f}  parser_cost={r['parser_cost']:+.4f}  "
              f"cls-parser vs reg={r['cls_parser']-r['reg']:+.4f}")
    print("=" * 95)

    out = {"device": str(device), "img": IMG_SIZE, "n_elem": N_ELEM,
           "n_train": n_tr, "n_test": n_te, "epochs": epochs,
           "width": args.width, "rows": rows}
    Path("p7_realistic_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p7_realistic_results.json")


if __name__ == "__main__":
    main()
