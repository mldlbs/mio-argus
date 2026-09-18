"""
P8：学习式 UI 检测器 —— 参数预算能否保住 H2？

背景
----
P0.5 埋下的风险：**"如果解析器本身要个大模型，那就只是把参数搬了个位置"**。
P7 证实朴素 CV 解析器在纹理背景上崩溃（recall 0.847 → 0.026）。

本实验用一个**小型学习式检测器**（CenterNet-lite）替换朴素解析器，回答：

  在**等总参数量**（检测器 + 选择器）下，
  学习式解析器 + 选择器 能否打败 坐标回归（reg）？

  - 若很小的检测器就能拿到高 recall → 结构化在小预算下复活
  - 若检测器必须很大 → 总参数超过 reg → H2 在真实 UI 上仍不划算

检测器（anchor-free，无 NMS 库依赖）
------------------------------------
  backbone → 特征图 (C,h,w)
  objectness : 1×1 conv → sigmoid     目标 = 元素中心的 Gaussian
  size       : 1×1 conv → 2 通道       目标 = 元素 (w,h)/IMG_SIZE
  decode     : max-pool 局部极大 + 阈值 + top-k → 框

用法:
    python p8_learned_parser.py --quick
    python p8_learned_parser.py
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p7_realistic_ui as P7
from p0_env import LABELS, instruction_onehot
from p0_models import SpatialBackbone
from torchvision.ops import roi_align

IMG_SIZE = P7.IMG_SIZE
N_MAX = P7.N_MAX


# ---------- 数据（含中心的 Gaussian 目标 + 尺寸目标）----------

def build_det_data(n, seed, background="flat", layout="tiny",
                   sigma=1.0, rich=True):
    rng = np.random.default_rng(seed)
    if layout == "tiny":
        ne, wr, hr = 42, (8, 26), (7, 14)
    else:
        ne, wr, hr = 30, (18, 56), (12, 26)
    res = IMG_SIZE // 8              # backbone stages=3 → /8
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    hm = np.zeros((n, res, res), dtype=np.float32)
    sz = np.zeros((n, 2, res, res), dtype=np.float32)
    xy = np.zeros((n, 2), dtype=np.float32)
    tbox = np.zeros((n, 4), dtype=np.int32)

    for i in range(n):
        b, labs = P7.make_elements(rng, n=ne, wrange=wr, hrange=hr)
        chw, els = P7.render_ui(b, labs, rng, background, rich)
        imgs[i] = (chw * 255).astype(np.uint8)
        k = int(rng.integers(0, len(els)))
        t = els[k]
        inst[i] = instruction_onehot(f"click {t.label}", len(LABELS))
        xy[i] = P7.center(t.box)
        tbox[i] = t.box

        gx = np.arange(res, dtype=np.float32)
        for e in els:
            x1, y1, x2, y2 = e.box
            cx = (x1 + x2) / 2 / IMG_SIZE * res
            cy = (y1 + y2) / 2 / IMG_SIZE * res
            d2 = (gx[None, :] - cx) ** 2 + (gx[:, None] - cy) ** 2
            g = np.exp(-d2 / (2 * sigma ** 2))
            hm[i] = np.maximum(hm[i], g.astype(np.float32))
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= ix < res and 0 <= iy < res:
                sz[i, 0, iy, ix] = (x2 - x1) / IMG_SIZE
                sz[i, 1, iy, ix] = (y2 - y1) / IMG_SIZE

    return {"images": imgs, "inst": inst, "hm": hm, "sz": sz,
            "xy": xy, "tbox": tbox}


# ---------- 检测器 ----------

class Detector(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.bb = SpatialBackbone(width, IMG_SIZE, 3)
        C = self.bb.out_channels
        self.obj = nn.Conv2d(C, 1, 1)
        self.size = nn.Conv2d(C, 2, 1)
        self.C = C

    def forward(self, x):
        f = self.bb(x)
        return self.obj(f), self.size(f)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_det(width, tr, epochs, device, lr=2e-3, batch=32, seed=0):
    torch.manual_seed(seed)
    m = Detector(width).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    I = P7.to_t(tr["images"], device)
    HM = torch.from_numpy(tr["hm"]).to(device)
    SZ = torch.from_numpy(tr["sz"]).to(device)
    n = I.shape[0]; spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k*batch:(k+1)*batch]
            o, s = m(I[idx])
            o = o.squeeze(1)
            tgt = HM[idx]
            # 正样本加权（中心少、背景多）
            pos = (tgt > 0.5).float()
            w = 1.0 + 20.0 * pos
            lo = F.binary_cross_entropy_with_logits(o, tgt, weight=w)
            mask = pos.unsqueeze(1)
            if mask.sum() > 0:
                ls = (F.l1_loss(s, SZ[idx], reduction="none") * mask).sum() / mask.sum()
            else:
                ls = torch.zeros((), device=device)
            loss = lo + 5.0 * ls
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def detect(m, chw, device, thr=0.25, topk=N_MAX):
    """返回框列表（像素坐标）。"""
    m.eval()
    x = torch.from_numpy(chw).unsqueeze(0).to(device)
    o, s = m(x)
    hm = torch.sigmoid(o)[0, 0]
    size = s[0]
    mx = F.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
    peaks = (hm >= mx - 1e-6) & (hm > thr)
    ys, xs = peaks.nonzero(as_tuple=True)
    if len(ys) == 0:
        return []
    sc = hm[ys, xs]
    order = torch.argsort(-sc)[:topk]
    res = hm.shape[0]
    out = []
    for i in order.tolist():
        y, x = int(ys[i]), int(xs[i])
        w = float(size[0, y, x]) * IMG_SIZE
        h = float(size[1, y, x]) * IMG_SIZE
        if w < 4 or h < 4:
            continue
        cx = (x + 0.5) / res * IMG_SIZE
        cy = (y + 0.5) / res * IMG_SIZE
        out.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return out


def eval_recall(m, data, device, thr=0.25):
    rec, ious = [], []
    for i in range(data["images"].shape[0]):
        chw = data["images"][i].astype(np.float32) / 255.0
        dets = detect(m, chw, device, thr)
        tb = tuple(data["tbox"][i])
        mi = P7.match_idx(dets, tb)
        rec.append(mi >= 0)
        ious.append(max((P7.iou(b, tb) for b in dets), default=0.0))
    return float(np.mean(rec)), float(np.mean(ious))


# ---------- 选择器（对检测出的候选打分）----------

class Sel(nn.Module):
    def __init__(self, width=16, inst_dim=48):
        super().__init__()
        self.bb = SpatialBackbone(width, IMG_SIZE, 3)
        C = self.bb.out_channels
        self.emb = nn.Embedding(len(LABELS), inst_dim)
        self.q = nn.Linear(inst_dim, C)
        self.C = C

    def forward(self, images, inst, boxes, mask):
        feat = self.bb(images)
        B, C, h, w = feat.shape
        N = boxes.shape[1]
        q = self.q(self.emb(inst.argmax(-1)))
        pooled = torch.zeros(B, N, C, dtype=feat.dtype, device=feat.device)
        sc = torch.tensor([w, h, w, h], dtype=feat.dtype, device=feat.device)
        for b in range(B):
            ii = mask[b].nonzero(as_tuple=True)[0]
            if ii.numel() == 0:
                continue
            bx = boxes[b, ii] * sc
            rois = torch.cat([torch.full((ii.numel(), 1), float(b),
                                         dtype=feat.dtype, device=feat.device), bx], 1)
            pooled[b, ii] = roi_align(feat, rois, (1, 1), 1.0, aligned=True
                                      ).reshape(ii.numel(), C)
        s = (pooled * q[:, None, :]).sum(-1) / math.sqrt(C)
        return s.masked_fill(~mask, -1e9)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_sel(width, tr_det, device, epochs, lr=1e-3, batch=32, seed=0):
    """在检测器产出的候选上训练选择器。"""
    torch.manual_seed(seed)
    m = Sel(width).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(tr_det)
    m.train()
    for ep in range(epochs):
        for i0 in range(0, n, batch):
            chunk = list(range(i0, min(i0 + batch, n)))
            b = len(chunk)
            I = torch.zeros(b, 3, IMG_SIZE, IMG_SIZE)
            BOX = torch.zeros(b, N_MAX, 4)
            MK = torch.zeros(b, N_MAX, dtype=torch.bool)
            Y = torch.full((b,), -1, dtype=torch.long)
            for j, rec in enumerate(chunk):
                r = tr_det[rec]
                I[j] = torch.from_numpy(r["img"].astype(np.float32) / 255.0)
                for k, bx in enumerate(r["dets"][:N_MAX]):
                    BOX[j, k, 0] = bx[0] / IMG_SIZE
                    BOX[j, k, 1] = bx[1] / IMG_SIZE
                    BOX[j, k, 2] = bx[2] / IMG_SIZE
                    BOX[j, k, 3] = bx[3] / IMG_SIZE
                    MK[j, k] = True
                Y[j] = r["match"]
            I, BOX, MK, Y = I.to(device), BOX.to(device), MK.to(device), Y.to(device)
            T = torch.from_numpy(np.stack([tr_det[r]["inst"] for r in chunk]))
            T = T.to(device)
            if (Y == -1).all():
                continue
            loss = F.cross_entropy(m(I, T, BOX, MK), Y, ignore_index=-1)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--det-widths", type=str, default=None)
    ap.add_argument("--bgs", type=str, default=None)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (12 if args.quick else 30)
    n_tr = args.n_train if args.n_train else (2000 if args.quick else 4000)
    n_te = args.n_test if args.n_test else (600 if args.quick else 1200)
    det_widths = ([int(x) for x in args.det_widths.split(",")] if args.det_widths
                  else ([8, 16] if args.quick else [4, 8, 16, 32]))
    bgs = (args.bgs.split(",") if args.bgs
           else (["flat", "texture"] if args.quick
                 else ["flat", "texture", "natural"]))
    sel_width = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} IMG={IMG_SIZE} sel_width={sel_width} "
          f"det_widths={det_widths} epochs={epochs} n_train={n_tr}")

    reg_params = sum(p.numel() for p in P7.P0Net(
        width=32, head="reg", num_slots=12, grid=3, num_labels=len(LABELS),
        inst_dim=48, in_res=IMG_SIZE, stages=3).parameters())
    print(f"参考：reg(width=32) 参数量 = {reg_params:,}\n")

    rows = []
    print(f"{'bg':<9}{'det_w':>6}{'det_params':>11}{'sel_params':>11}"
          f"{'total':>10}{'recall':>8}{'IoU':>7}{'cls':>8}{'reg':>8}"
          f"{'total<reg':>10}")
    print("-" * 92)
    for bg in bgs:
        tr = build_det_data(n_tr, 100, bg, "tiny", rich=True)
        te = build_det_data(n_te, 900, bg, "tiny", rich=True)
        # reg 基线（与 P7 同配置）
        reg = P7.train_reg(32, P7.build_data(n_tr, 100, bg, True, False, "tiny"),
                           epochs, device)
        r_reg = P7.eval_reg(reg, P7.build_data(n_te, 900, bg, True, False, "tiny"),
                            device)
        for dw in det_widths:
            det = train_det(dw, tr, epochs, device)
            rec, miou = eval_recall(det, te, device)

            # 用检测器产出候选，训练/评估选择器
            tr_det = []
            for i in range(n_tr):
                chw = tr["images"][i].astype(np.float32) / 255.0
                dets = detect(det, chw, device)
                tb = tuple(tr["tbox"][i])
                tr_det.append({"img": tr["images"][i], "inst": tr["inst"][i],
                               "dets": dets, "match": P7.match_idx(dets, tb)})
            sel = train_sel(sel_width, tr_det, device, epochs)

            hits = 0
            for i in range(n_te):
                chw = te["images"][i].astype(np.float32) / 255.0
                dets = detect(det, chw, device)[:N_MAX]
                if not dets:
                    continue
                I = torch.from_numpy(chw).unsqueeze(0).to(device)
                T = torch.from_numpy(te["inst"][i]).unsqueeze(0).to(device)
                BX = torch.tensor([[(b[0]/IMG_SIZE, b[1]/IMG_SIZE, b[2]/IMG_SIZE,
                                     b[3]/IMG_SIZE) for b in dets]],
                                  dtype=torch.float32, device=device)
                MK = torch.ones(1, len(dets), dtype=torch.bool, device=device)
                with torch.no_grad():
                    pred = int(sel(I, T, BX, MK).argmax(-1).item())
                bx = dets[pred]
                cx, cy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2
                tb = te["tbox"][i]
                hits += int(tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3])
            cls_acc = hits / n_te

            dp, sp = det.n_params(), sel.n_params()
            rows.append({"bg": bg, "det_w": dw, "det_params": dp,
                         "sel_params": sp, "total": dp + sp,
                         "recall": rec, "iou": miou, "cls": cls_acc,
                         "reg": r_reg})
            print(f"{bg:<9}{dw:>6}{dp:>11,}{sp:>11,}{dp+sp:>10,}"
                  f"{rec:>8.3f}{miou:>7.3f}{cls_acc:>8.4f}{r_reg:>8.4f}"
                  f"{(dp+sp < reg_params):>10}")

    print(f"\n{'='*92}")
    print("判据：total < reg_params 且 cls > reg  ->  结构化在真实 UI 上以小预算复活")
    for r in rows:
        ok = (r["total"] < reg_params) and (r["cls"] > r["reg"])
        print(f"  {r['bg']:<9} det_w={r['det_w']:<3} total={r['total']:>8,} "
              f"(reg={reg_params:,})  recall={r['recall']:.3f}  "
              f"cls={r['cls']:.4f} vs reg={r['reg']:.4f}  -> "
              f"{'OK' if ok else 'NO'}")
    print("=" * 92)

    out = {"device": str(device), "img": IMG_SIZE, "sel_width": sel_width,
           "reg_params": reg_params, "epochs": epochs, "n_train": n_tr,
           "n_test": n_te, "rows": rows}
    Path("p8_learned_parser_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p8_learned_parser_results.json")


if __name__ == "__main__":
    main()
