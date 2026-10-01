"""
P9：顺序感知 vs 单体计算 —— iso-FLOPs 边界刻画（路线 b）

动机（docs/ARCHITECTURE.md §3.4 / 上轮整体评估 §3）
------------------------------------------------
P8 证伪了"小学习检测器 + 选择器 能打赢坐标回归"（12/12 NO），
并自认唯一未排除的替代解释是**分辨率**。但"多看几眼"在 2025–2026
已是 GUI grounding 的主流做法（MEGA-GUI / ZoomClick / GUI-Lens / GUI-C²），
所以 P9 **不主张机制首创**，只做别人少做的一件事：

    在**等总算力（iso-FLOPs）**下，
    顺序自适应缩放 与 单体等算力模型，谁更好、边界在哪、何时更差。

三种策略（同一份训练数据、同一个检测器结构族）
----------------------------------------------
  oneshot        ：一次全图检测（基宽）
  mono-w         ：单体加宽到与 glimpse-T 等 MACs，一次前向（可并行）
  glimpse-T      ：1 次粗看 + T 次自适应裁剪缩放（顺序，不可并行）
  random-crop-T  ：同预算，但裁剪窗口随机（控制"主动选择"）

判据（路线 b：边界，而非"我们赢"）
----------------------------------
  - 每个预算 M 上给出 (mono_recall, glimpse_recall, random_recall, 实际 MACs)
  - 找 crossing：glimpse 在低预算占优、高预算被 mono 反超的第一个预算
  - 若 glimpse ≈ random → 增益来自分辨率而非主动；若 glimpse ≤ mono →
    顺序分配无净收益，P9 的机制主张被证伪

复现：
    python experiments/paper/p9_active_zoom.py --quick
    python experiments/paper/p9_active_zoom.py
"""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p7_realistic_ui as P7
import p8_learned_parser as P8
from p0_env import LABELS

IMG = P7.IMG_SIZE
N_MAX = P7.N_MAX
BG = "flat"


# ---------- 计算量核算：每样本 MACs（multiply-accumulate）----------

def count_macs(model, res=IMG):
    """遍历所有 Conv2d，按输出空间尺寸×核面积×输入通道 累加每样本 MACs。

    只统计卷积（占绝对主导），忽略 BN/ReLU/sigmoid。
    """
    total = 0
    handles = []

    def hook(mod, inp, out):
        nonlocal total
        total += (out.shape[1] * out.shape[2] * out.shape[3]
                  * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1])

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, res, res, device=dev))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return int(total)


def _self_check_macs():
    """对照手算：单层 3→8、k=3、s=1、p=1、输入 16×16 → 输出 16×16。"""
    m = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1))
    got = count_macs(m, res=16)
    want = 16 * 16 * 8 * (3 * 3 * 3)
    assert got == want, f"MAC counter 错误：got={got} want={want}"
    print(f"[self-check] count_macs 正确 (16x16x8x27 = {want:,})")


# ---------- 多尺度数据：全图 + 随机裁剪缩放（模拟"再看一眼"）----------

LAYOUTS = {
    "extreme": dict(ne=42, wr=(13, 42), hr=(10, 22)),
    "tiny":    dict(ne=42, wr=(8, 26),  hr=(7, 14)),
    "micro":   dict(ne=48, wr=(7, 20),  hr=(6, 12)),
}


def _crop_resize_np(chw, x0, y0, w):
    """从 float chw (3,H,W) 取 w×w 裁剪并双线性缩放到 IMG×IMG。"""
    sub = chw[:, y0:y0 + w, x0:x0 + w]
    if w == IMG:
        return sub
    t = torch.from_numpy(np.ascontiguousarray(sub)).unsqueeze(0)
    t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return t[0].numpy()


def build_zoom_data(n, seed, layout="tiny", bg=BG, sigma=1.0, full_prob=0.5):
    """返回 images(192) / hm / sz，元素以**裁剪后坐标**为监督。

    full_prob：整图（scale=1）样本的比例；其余为随机裁剪缩放样本。
    这样同一个检测器既能做粗看，也能做缩放后细看。
    """
    p = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    res = IMG // 8
    imgs = np.zeros((n, 3, IMG, IMG), dtype=np.uint8)
    hm = np.zeros((n, res, res), dtype=np.float32)
    sz = np.zeros((n, 2, res, res), dtype=np.float32)
    xy_full = np.zeros((n, 2), dtype=np.float32)
    tbox = np.zeros((n, 4), dtype=np.int32)

    g = np.arange(res, dtype=np.float32)

    for i in range(n):
        boxes, labs = P7.make_elements(rng, n=p["ne"], wrange=p["wr"], hrange=p["hr"])
        chw, els = P7.render_ui(boxes, labs, rng, bg, rich=True)
        k = int(rng.integers(0, len(els)))
        t = els[k]
        tbox[i] = t.box
        xy_full[i] = P7.center(t.box)

        cx = (t.box[0] + t.box[2]) / 2
        cy = (t.box[1] + t.box[3]) / 2

        if rng.random() < full_prob:
            w = IMG
            x0 = y0 = 0
        else:
            scale = float(rng.uniform(0.22, 0.72))
            w = max(32, int(IMG * scale))
            x0 = int(np.clip(cx - w / 2, 0, IMG - w))
            y0 = int(np.clip(cy - w / 2, 0, IMG - w))

        crop = _crop_resize_np(chw, x0, y0, w)
        imgs[i] = (np.clip(crop, 0, 1) * 255).astype(np.uint8)

        # 检测器是无条件的 objectness：必须把所有元素画进热力图，
        # 而不是只画目标（目标每样本随机，检测器无从得知是哪一个）。
        for e in els:
            ecx = ((e.box[0] + e.box[2]) / 2 - x0) / w
            ecy = ((e.box[1] + e.box[3]) / 2 - y0) / w
            if not (0.0 <= ecx <= 1.0 and 0.0 <= ecy <= 1.0):
                continue
            gx = ecx * res
            gy = ecy * res
            d2 = (g[None, :] - gx) ** 2 + (g[:, None] - gy) ** 2
            gg = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
            hm[i] = np.maximum(hm[i], gg)
            ix, iy = int(round(gx)), int(round(gy))
            if 0 <= ix < res and 0 <= iy < res:
                sz[i, 0, iy, ix] = (e.box[2] - e.box[0]) / w
                sz[i, 1, iy, ix] = (e.box[3] - e.box[1]) / w

    return {"images": imgs, "hm": hm, "sz": sz, "xy": xy_full, "tbox": tbox}


# ---------- 原生高分辨率：单一高分辨率前向（mono-R 臂）----------
#
# P7 的渲染管线把画布尺寸硬编码在模块全局 IMG_SIZE。用 _at_res 临时切换，
# 按 R/192 等比放大元素尺寸，从而在 R 上**原生**渲染（不是上采样）。
# 任务在归一化坐标下是尺度不变的，因此 @192 训练的全卷积检测器应可直接
# 迁移到 R；是否成立由实测 recall(R) 单调性判定。

@contextlib.contextmanager
def _at_res(R):
    old = P7.IMG_SIZE
    P7.IMG_SIZE = R
    try:
        yield
    finally:
        P7.IMG_SIZE = old


def build_data_at(n, seed, layout="tiny", R=IMG, bg=BG):
    """在 R×R 上原生渲染，返回 images 与 R 像素坐标下的 tbox。"""
    p = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    s = R / 192.0
    wr = (max(3, int(round(p["wr"][0] * s))), max(4, int(round(p["wr"][1] * s))))
    hr = (max(3, int(round(p["hr"][0] * s))), max(4, int(round(p["hr"][1] * s))))
    gap = max(2, int(round(4 * s)))
    imgs = np.zeros((n, 3, R, R), dtype=np.uint8)
    tbox = np.zeros((n, 4), dtype=np.int32)
    with _at_res(R):
        for i in range(n):
            boxes, labs = P7.make_elements(rng, n=p["ne"], min_gap=gap,
                                           wrange=wr, hrange=hr)
            chw, els = P7.render_ui(boxes, labs, rng, bg, rich=True)
            k = int(rng.integers(0, len(els)))
            t = els[k]
            imgs[i] = (np.clip(chw, 0, 1) * 255).astype(np.uint8)
            tbox[i] = t.box
    return {"images": imgs, "tbox": tbox}


def build_det_data_at(n, seed, layout, R, sigma=1.0, bg=BG):
    """在 R×R 原生渲染并生成 all-element 热力图/尺寸监督（用于 per-R 训练）。

    与 build_zoom_data(full_prob=1.0) 等价，只是分辨率换成 R；
    训练只用全图（§2.3：混入裁剪会破坏 BatchNorm）。
    """
    p = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    s = R / 192.0
    wr = (max(3, int(round(p["wr"][0] * s))), max(4, int(round(p["wr"][1] * s))))
    hr = (max(3, int(round(p["hr"][0] * s))), max(4, int(round(p["hr"][1] * s))))
    gap = max(2, int(round(4 * s)))
    res = R // 8
    # 关键：sigma 必须以 cell 为单位随 res 放大，保持目标在归一化坐标下的尺寸不变。
    # 否则 res 越大正样本越稀疏（R=576 时 sigma=1/72 的图），per-R 训练会欠拟合，
    # 被误读为"高分辨率本身无用"。
    sig = sigma * res / (IMG // 8)
    imgs = np.zeros((n, 3, R, R), dtype=np.uint8)
    hm = np.zeros((n, res, res), dtype=np.float32)
    sz = np.zeros((n, 2, res, res), dtype=np.float32)
    xy = np.zeros((n, 2), dtype=np.float32)
    tbox = np.zeros((n, 4), dtype=np.int32)
    g = np.arange(res, dtype=np.float32)

    with _at_res(R):
        for i in range(n):
            boxes, labs = P7.make_elements(rng, n=p["ne"], min_gap=gap,
                                           wrange=wr, hrange=hr)
            chw, els = P7.render_ui(boxes, labs, rng, bg, rich=True)
            imgs[i] = (np.clip(chw, 0, 1) * 255).astype(np.uint8)
            k = int(rng.integers(0, len(els)))
            t = els[k]
            tbox[i] = t.box
            xy[i] = ((t.box[0] + t.box[2]) / 2 / R, (t.box[1] + t.box[3]) / 2 / R)
            for e in els:
                cx = (e.box[0] + e.box[2]) / 2 / R * res
                cy = (e.box[1] + e.box[3]) / 2 / R * res
                d2 = (g[None, :] - cx) ** 2 + (g[:, None] - cy) ** 2
                gg = np.exp(-d2 / (2 * sig ** 2)).astype(np.float32)
                hm[i] = np.maximum(hm[i], gg)
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= ix < res and 0 <= iy < res:
                    sz[i, 0, iy, ix] = (e.box[2] - e.box[0]) / R
                    sz[i, 1, iy, ix] = (e.box[3] - e.box[1]) / R
    return {"images": imgs, "hm": hm, "sz": sz, "xy": xy, "tbox": tbox}


def _finetune_det(base, width, tr, epochs, device, lr=1e-3, batch=32, seed=0):
    """从 @192 权重热启动，在 R 上微调。

    从零 per-R 训练不收敛（size head 欠拟合、检测被尺寸过滤丢掉），
    但 @192 权重迁移已能用；热启动直接回答"高 R 崩溃是不是迁移假象"。
    """
    torch.manual_seed(seed)
    m = P8.Detector(width).to(device)
    m.load_state_dict(base.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    I = P7.to_t(tr["images"], device)
    HM = torch.from_numpy(tr["hm"]).to(device)
    SZ = torch.from_numpy(tr["sz"]).to(device)
    n = I.shape[0]
    spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k * batch:(k + 1) * batch]
            o, s = m(I[idx])
            o = o.squeeze(1)
            tgt = HM[idx]
            pos = (tgt > 0.5).float()
            w = 1.0 + 20.0 * pos
            lo = F.binary_cross_entropy_with_logits(o, tgt, weight=w)
            mask = pos.unsqueeze(1)
            if mask.sum() > 0:
                ls = (F.l1_loss(s, SZ[idx], reduction="none") * mask).sum() / mask.sum()
            else:
                ls = torch.zeros((), device=device)
            loss = lo + 5.0 * ls
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def detect_at(det, chw, device, img_size, thr=0.25, topk=N_MAX):
    """与 P8.detect 同逻辑，但显式接受分辨率（P8.detect 硬编码 192）。"""
    det.eval()
    x = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0).to(device)
    with torch.no_grad():
        o, s = det(x)
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
    min_side = 4.0 * img_size / 192.0
    out = []
    for i in order.tolist():
        yy, xx = int(ys[i]), int(xs[i])
        w = float(size[0, yy, xx]) * img_size
        h = float(size[1, yy, xx]) * img_size
        if w < min_side or h < min_side:
            continue
        cx = (xx + 0.5) / res * img_size
        cy = (yy + 0.5) / res * img_size
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def eval_recall_at(det, data, device, R, thr=0.25):
    n = data["images"].shape[0]
    hits = 0
    for i in range(n):
        chw = _chw(data["images"][i])
        dets = detect_at(det, chw, device, R, thr)
        tb = tuple(int(v) for v in data["tbox"][i])
        hits += int(P7.match_idx(dets, tb) >= 0)
    return hits / max(1, n)


# ---------- 推理策略 ----------

def _chw(img_u8):
    return img_u8.astype(np.float32) / 255.0


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def eval_glimpse(det, data, device, T, crop_frac=0.35, thr=0.25,
                 random_center=False, seed=0):
    """粗看 1 次 + 对最可信的 T 个候选区域各裁剪缩放再看 1 次。

    关键（实测发现，见 P9_RESULTS）：
      - 全卷积检测器**只用全图训练**即可泛化到缩放裁剪（oracle 裁剪 recall 0.95）；
        混入随机裁剪训练会 poison BN，oracle 裁剪 recall 掉到 0.0。
      - 因此不用"贪心重居中"（一旦粗看选错元素就回不来），
        而是把预算花在 top-T 粗候选上，合并所有映射回原图的检测。

    random_center=True 时改为随机窗口（控制"主动选择"）。
    """
    rng = np.random.default_rng(seed)
    n = data["images"].shape[0]
    w = max(24, int(IMG * crop_frac))
    hits = 0

    for i in range(n):
        chw = _chw(data["images"][i])
        tb = tuple(int(v) for v in data["tbox"][i])

        dets = P8.detect(det, chw, device, thr)   # 已按分数降序
        if P7.match_idx(dets, tb) >= 0:
            hits += 1
            continue

        if random_center:
            windows = [(int(rng.integers(0, IMG - w + 1)),
                        int(rng.integers(0, IMG - w + 1))) for _ in range(T)]
        else:
            windows = []
            for k in range(T):
                if k < len(dets):
                    cx, cy = _center(dets[k])
                    windows.append((int(np.clip(cx - w / 2, 0, IMG - w)),
                                    int(np.clip(cy - w / 2, 0, IMG - w))))
                else:
                    windows.append((int((IMG - w) / 2), int((IMG - w) / 2)))

        ok = False
        for (x0, y0) in windows:
            crop = _crop_resize_np(chw, x0, y0, w)
            d2 = P8.detect(det, crop, device, thr)
            if not d2:
                continue
            mapped = [(x0 + b[0] * w / IMG, y0 + b[1] * w / IMG,
                       x0 + b[2] * w / IMG, y0 + b[3] * w / IMG) for b in d2]
            if P7.match_idx(mapped, tb) >= 0:
                ok = True
                break
        hits += int(ok)
    return hits / max(1, n)


# ---------- 主流程 ----------

WIDTH_CANDIDATES = [8, 12, 16, 24, 32, 48, 64, 80, 96]


def _closest_width(target_macs, unit, device, cache):
    best, best_err = None, None
    for w in WIDTH_CANDIDATES:
        if w not in cache:
            cache[w] = count_macs(P8.Detector(width=w), IMG)
        err = abs(cache[w] - target_macs)
        if best_err is None or err < best_err:
            best, best_err = w, err
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--ts", type=str, default=None)
    ap.add_argument("--layouts", type=str, default=None)
    ap.add_argument("--crop-frac", type=float, default=0.35)
    ap.add_argument("--width", type=int, default=16, help="基检测器宽度")
    ap.add_argument("--mono-rs", type=str, default=None,
                    help="mono-R 的原生分辨率列表，如 192,288,384,576")
    ap.add_argument("--train-per-r", action="store_true",
                    help="从 @192 权重热启动，在每个 R 上微调检测器（排除迁移效应）")
    ap.add_argument("--ft-lr", type=float, default=3e-4,
                    help="per-R 微调学习率（1e-3 会发散）")
    args = ap.parse_args()

    _self_check_macs()
    if args.train_per_r:
        print("[WARN] --train-per-r 是实验性路径：per-R 复训当前不稳定"
              "（常见 0.0；见 P9_RESULTS §4.3），不要用它下结论。")

    epochs = args.epochs if args.epochs else (12 if args.quick else 30)
    n_tr = args.n_train if args.n_train else (1200 if args.quick else 3000)
    n_te = args.n_test if args.n_test else (400 if args.quick else 1000)
    ts = ([int(x) for x in args.ts.split(",")] if args.ts else [1, 2, 3])
    layouts = (args.layouts.split(",") if args.layouts
               else (["extreme", "tiny"] if args.quick
                     else ["extreme", "tiny", "micro"]))
    mono_rs = ([int(x) for x in args.mono_rs.split(",")] if args.mono_rs
               else [192, 288, 384, 576])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} IMG={IMG} base_width={args.width} epochs={epochs} "
          f"n_train={n_tr} n_test={n_te} crop_frac={args.crop_frac}")

    mac_cache = {}
    unit = count_macs(P8.Detector(width=args.width), IMG)
    print(f"base detector MACs/样本 = {unit:,}（1 个 compute unit）\n")

    rows = []
    bases = {}
    print(f"{'layout':<9}{'budget':>7}{'T':>3}{'mono_w':>7}{'mono_macs':>11}"
          f"{'mono_rec':>9}{'glimpse':>9}{'random':>9}{'winner':>9}")
    print("-" * 82)

    for layout in layouts:
        # 训练必须只用全图：混入随机裁剪会破坏 BatchNorm 统计，
        # 使 oracle 裁剪 recall 从 0.95 掉到 0.0（P9 实测，详见 results/paper/P9_RESULTS.md）。
        # 全卷积检测器只用全图训练即可泛化到缩放裁剪——这正是顺序缩放可行的前提。
        tr = build_zoom_data(n_tr, 100, layout, full_prob=1.0)
        te = build_zoom_data(n_te, 900, layout, full_prob=1.0)

        base = P8.train_det(args.width, tr, epochs, device)
        bases[layout] = base
        mono_cache = {}

        # T=0：单次全图（基宽），三种策略退化为同一个
        r0 = P8.eval_recall(base, te, device)[0]
        rows.append(dict(layout=layout, budget=unit, T=0,
                         mono_w=args.width, mono_macs=unit, mono_recall=r0,
                         glimpse_recall=r0, random_recall=r0, winner="tie"))
        print(f"{layout:<9}{unit:>7,}{0:>3}{args.width:>7}{unit:>11,}"
              f"{r0:>9.4f}{r0:>9.4f}{r0:>9.4f}{'tie':>9}")

        for T in ts:
            budget = (T + 1) * unit
            w = _closest_width(budget, unit, device, mac_cache)
            if w not in mono_cache:
                mono_cache[w] = P8.train_det(w, tr, epochs, device)
            mono = mono_cache[w]
            mono_macs = mac_cache[w]
            r_mono = P8.eval_recall(mono, te, device)[0]

            r_gl = eval_glimpse(base, te, device, T, args.crop_frac,
                                random_center=False, seed=7)
            r_rd = eval_glimpse(base, te, device, T, args.crop_frac,
                                random_center=True, seed=7)

            if abs(r_gl - r_mono) < 1e-9:
                winner = "tie"
            else:
                winner = "glimpse" if r_gl > r_mono else "mono"
            rows.append(dict(layout=layout, budget=budget, T=T, mono_w=w,
                             mono_macs=mono_macs, mono_recall=r_mono,
                             glimpse_recall=r_gl, random_recall=r_rd,
                             winner=winner))
            print(f"{layout:<9}{budget:>7,}{T:>3}{w:>7}{mono_macs:>11,}"
                  f"{r_mono:>9.4f}{r_gl:>9.4f}{r_rd:>9.4f}{winner:>9}")

    # ---- mono-R：原生 R 上单次前向（分辨率换算力）----
    print(f"\n--- mono-R（原生 R，单次前向）"
          f"{'；per-R=从@192 热启动微调' if args.train_per_r else '；@192 权重迁移'} ---")
    hdr = f"{'layout':<9}{'R':>6}{'mono_macs':>13}{'transfer':>10}"
    if args.train_per_r:
        hdr += f"{'per-R':>10}"
    hdr += f"{'glimpse3_macs':>15}{'glimpse3':>10}"
    print(hdr)
    for layout in layouts:
        base = bases[layout]
        sub = [r for r in rows if r["layout"] == layout and r["T"] == max(ts)]
        gl3_macs = sub[0]["budget"] if sub else 0
        gl3_rec = sub[0]["glimpse_recall"] if sub else None
        for R in mono_rs:
            if R == IMG:
                continue
            dR = build_data_at(n_te, 900, layout, R)
            mR = count_macs(base, R)
            r_t = eval_recall_at(base, dR, device, R)
            r_p = None
            if args.train_per_r:
                trR = build_det_data_at(n_tr, 100, layout, R)
                detR = _finetune_det(base, args.width, trR, epochs, device,
                                     lr=args.ft_lr)
                r_p = eval_recall_at(detR, dR, device, R)
            rec = r_p if r_p is not None else r_t
            rows.append(dict(layout=layout, budget=mR, T=None, kind="mono-R",
                             R=R, mono_w=None, mono_macs=mR, mono_recall=rec,
                             recall_transfer=r_t, recall_per_r=r_p,
                             glimpse_recall=None, random_recall=None,
                             winner="mono-R"))
            g3s = f"{gl3_rec:.4f}" if gl3_rec is not None else "-"
            line = f"{layout:<9}{R:>6}{mR:>13,}{r_t:>10.4f}"
            if args.train_per_r:
                line += f"{r_p:>10.4f}"
            line += f"{gl3_macs:>15,}{g3s:>10}"
            print(line)

    print(f"\n{'='*82}")
    print("判据（路线 b）：")
    print("  1) glimpse > mono            -> 顺序分配在该预算下有净收益")
    print("  2) glimpse > random-crop     -> 增益来自『主动选择看哪里』，非单纯上采样")
    print("  3) winner 随预算单调翻转处    -> 边界（glimpse 何时被等算力单体反超）")
    for layout in layouts:
        sub = [r for r in rows if r["layout"] == layout and r["T"] is not None]
        flips = [f"T={sub[i]['T']}" for i in range(1, len(sub))
                 if sub[i]["winner"] != sub[i - 1]["winner"]]
        wins = sum(1 for r in sub if r["winner"] == "glimpse")
        print(f"  {layout:<9} glimpse 胜 {wins}/{len(sub)}  翻转点: "
              f"{', '.join(flips) if flips else '无'}")

    # 等 MACs 下 mono-R 与 glimpse 的直接对比（路线 b 的核心）
    print("\n  [等算力对比] mono-R 是否反超 glimpse：")
    for layout in layouts:
        gl = [r for r in rows if r["layout"] == layout and r["T"] is not None]
        mr = [r for r in rows if r["layout"] == layout and r.get("kind") == "mono-R"]
        if not mr:
            continue
        for g in gl:
            near = min(mr, key=lambda r: abs(r["mono_macs"] - g["budget"]))
            verdict = ("mono-R" if near["mono_recall"] > g["glimpse_recall"]
                       else "glimpse")
            print(f"    {layout:<9} T={g['T']} MACs={g['budget']/1e6:>6.1f}M "
                  f"glimpse={g['glimpse_recall']:.4f}  "
                  f"mono-R@R={near['R']} ({near['mono_macs']/1e6:.1f}M)="
                  f"{near['mono_recall']:.4f}  -> {verdict}")
    print("=" * 82)

    out = {"device": str(device), "img": IMG, "base_width": args.width,
           "unit_macs": unit, "epochs": epochs, "n_train": n_tr,
           "n_test": n_te, "crop_frac": args.crop_frac,
           "width_macs": mac_cache, "rows": rows}
    Path("results/paper/p9_active_zoom_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 results/paper/p9_active_zoom_results.json")


if __name__ == "__main__":
    main()
