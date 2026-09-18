"""
P5 / H4：窄域低熵 —— 输入熵如何影响达到同等准确率所需的参数量。

H4 原表述
---------
> UI 输入的熵远低于自然图像，所以达到同等 CUA 能力所需参数更少。
> 预测：CUA 的 scaling curve 比通用 VLM 更早饱和。

受控实验设计
------------
**同一个任务、同一个输出空间、同一套元素渲染**，只改变**背景的熵**。
这是最干净的操作化：把"输入熵"从"任务难度"里分离出来。

背景熵级（元素与网格线在所有级别下完全相同，保持可辨识）：

| 级 | 背景 | 熵 |
|----|------|-----|
| E0 | 纯色 (246,247,249) | 最低（≈当前合成 UI） |
| E1 | 逐像素均匀噪声 | 高 |
| E2 | 低频平滑色块 | 中 |
| E3 | 自然图像随机裁剪（assets/nanogpt.jpg） | 高且结构化（最接近自然图像） |

判据：
  - 若高熵级在每个参数预算下都更差，且达到 0.90 所需参数显著更多
    → **H4 成立**（低熵输入确实更省参数）
  - 若各级曲线重合 → H4 不成立（熵不是参数需求的主因）

用法:
    python p5_h4_entropy.py --quick
    python p5_h4_entropy.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from p0_env import EnvCfg, Element, PALETTE, LABEL_TO_ID, LABELS, instruction_onehot, _font

GRID = 6
IMG_SIZE = 160
S = GRID * GRID
EPOCHS = 20
N_TR = 3000
N_TE = 1000
WIDTHS = [4, 6, 8, 12, 16, 24, 32]
LEVELS = ["E0-flat", "E1-noise", "E2-blobs", "E3-natural"]


def env_cfg(fill=0.40, jitter=4):
    return EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=S,
                  fill=fill, jitter=jitter, color_by_label=True)


# ---------- 背景 ----------

_NAT = None


def _natural_pool(n=64, seed=0):
    global _NAT
    if _NAT is None:
        im = Image.open("assets/nanogpt.jpg").convert("RGB")
        W, H = im.size
        rng = np.random.default_rng(seed)
        pool = []
        for _ in range(n):
            cw = int(rng.integers(80, W // 2))
            ch = int(cw * 0.75)
            x = int(rng.integers(0, max(1, W - cw)))
            y = int(rng.integers(0, max(1, H - ch)))
            pool.append(im.crop((x, y, x + cw, y + ch)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BICUBIC))
        _NAT = pool
    return _NAT


def make_bg(level, rng):
    if level == "E0-flat":
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (246, 247, 249))
    if level == "E1-noise":
        arr = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    if level == "E2-blobs":
        k = int(rng.integers(6, 14))
        small = rng.integers(0, 256, (k, k, 3), dtype=np.uint8)
        return Image.fromarray(small, "RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    pool = _natural_pool()
    return pool[int(rng.integers(0, len(pool)))]


# ---------- 渲染（元素与网格线在所有级别下一致）----------

def build_elements(labels, cfg, rng):
    els = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = cfg.slot_bbox(slot)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, IMG_SIZE - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, IMG_SIZE - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, IMG_SIZE - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, IMG_SIZE - 1))
        lab = labels[slot]
        color = tuple(int(c) for c in PALETTE[LABEL_TO_ID[lab] % len(PALETTE)])
        els.append(Element(slot, (ex1, ey1, ex2, ey2), lab, "button", color))
    return els


def render_level(labels, cfg, rng, level):
    img = make_bg(level, rng).copy()
    dr = ImageDraw.Draw(img)
    for slot in range(cfg.num_slots):
        x1, y1, x2, y2 = cfg.slot_bbox(slot)
        dr.rectangle([x1, y1, x2, y2], outline=(214, 218, 224))
    els = build_elements(labels, cfg, rng)
    font = _font(10)
    for e in els:
        x1, y1, x2, y2 = e.bbox
        dr.rectangle([x1, y1, x2, y2], fill=tuple(int(c) for c in e.color),
                     outline=(25, 25, 25))
        if (x2 - x1) >= 30 and (y2 - y1) >= 14:
            bb = dr.textbbox((0, 0), e.label, font=font)
            dr.text((x1 + max(0, ((x2 - x1) - (bb[2] - bb[0])) // 2), y1 + 2),
                    e.label, fill=(255, 255, 255), font=font)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), els


def build_data(n, seed, level):
    rng = np.random.default_rng(seed)
    cfg = env_cfg()
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    vocab = LABELS[:S]
    for i in range(n):
        idx = rng.choice(len(vocab), size=S, replace=False)
        labels = [vocab[int(j)] for j in idx]
        slot = int(rng.integers(0, S))
        img, _ = render_level(labels, cfg, rng, level)
        imgs[i] = (img * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {labels[slot]}", len(LABELS))
        slots[i] = slot
    return {"images": imgs, "inst": inst, "slots": slots,
            "xys": np.zeros((n, 2), dtype=np.float32),
            "bbox": np.zeros((n, 4), dtype=np.int32),
            "centers": np.zeros((n, S, 2), dtype=np.float32)}


# ---------- 模型 / 训练 ----------

def make_model(width, device):
    from p0_models import P0Net
    return P0Net(width=width, head="cls", num_slots=S, grid=GRID,
                 num_labels=len(LABELS), inst_dim=32,
                 in_res=IMG_SIZE, stages=3).to(device)


def train(width, tr, epochs, device, batch=64, lr=3e-3, seed=0):
    import torch.nn.functional as F
    torch.manual_seed(seed)
    m = make_model(width, device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    imgs = torch.from_numpy(tr["images"].astype(np.float32) / 255.0).to(device)
    inst = torch.from_numpy(tr["inst"]).to(device)
    tgt = torch.from_numpy(tr["slots"]).to(device)
    n = imgs.shape[0]
    spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k * batch:(k + 1) * batch]
            loss = F.cross_entropy(m(imgs[idx], inst[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def acc(m, data, device, batch=256):
    m.eval()
    imgs = torch.from_numpy(data["images"].astype(np.float32) / 255.0).to(device)
    inst = torch.from_numpy(data["inst"]).to(device)
    ok = 0
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch):
            p = m(imgs[i:i + batch], inst[i:i + batch]).argmax(-1).cpu().numpy()
            ok += int((p == data["slots"][i:i + batch]).sum())
    return ok / imgs.shape[0]


def params_of(width):
    return sum(p.numel() for p in make_model(width, "cpu").parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (10 if args.quick else EPOCHS)
    widths = [8, 16, 32] if args.quick else WIDTHS
    n_tr = 1200 if args.quick else N_TR
    n_te = 400 if args.quick else N_TE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} grid={GRID} S={S} epochs={epochs} "
          f"n_train={n_tr} n_test={n_te}")
    print(f"widths={widths}")
    print(f"levels={LEVELS}\n")

    tes = {lv: build_data(n_te, 999, lv) for lv in LEVELS}
    curves = {}
    print(f"{'level':<12}" + "".join(f"{'w'+str(w):>10}" for w in widths))
    print("-" * (12 + 10 * len(widths)))
    for lv in LEVELS:
        tr = build_data(n_tr, 100, lv)
        row = []
        for w in widths:
            m = train(w, tr, epochs, device)
            row.append(acc(m, tes[lv], device))
        curves[lv] = row
        print(f"{lv:<12}" + "".join(f"{v:>10.4f}" for v in row))

    # ---- 达到 0.90 所需参数（线性插值，log 参数轴）----
    def params_for(target, row):
        ps = [params_of(w) for w in widths]
        for i in range(len(widths) - 1):
            if row[i] < target <= row[i + 1]:
                # 在 log 参数上线性插值
                t = (target - row[i]) / (row[i + 1] - row[i])
                lp = np.log(ps[i]) + t * (np.log(ps[i + 1]) - np.log(ps[i]))
                return float(np.exp(lp))
        if row[0] >= target:
            return float(ps[0])
        return None

    print(f"\n达到目标准确率所需参数量（log 插值）")
    print(f"{'level':<12}{'p@0.70':>12}{'p@0.80':>12}{'p@0.90':>12}{'p@0.95':>12}")
    print("-" * 60)
    need = {}
    for lv in LEVELS:
        vals = [params_for(t, curves[lv]) for t in (0.70, 0.80, 0.90, 0.95)]
        need[lv] = vals
        print(f"{lv:<12}" + "".join(
            f"{int(v):>12,}" if v else f"{'—':>12}" for v in vals))

    # ---- 判定 ----
    lo, hi = LEVELS[0], LEVELS[-1]
    print(f"\n{'='*60}")
    print(f"低熵({lo}) vs 高熵({hi})：")
    for i in range(len(widths)):
        d = curves[lo][i] - curves[hi][i]
        print(f"  width={widths[i]:>3} (params {params_of(widths[i]):>8,})  "
              f"低熵={curves[lo][i]:.4f}  高熵={curves[hi][i]:.4f}  Δ={d:+.4f}")
    p90_lo, p90_hi = need[lo][2], need[hi][2]
    if p90_lo and p90_hi:
        ratio = p90_hi / p90_lo
        verdict = "H4 成立" if ratio > 1.5 else "H4 不成立（差距不显著）"
        print(f"\n达到 0.90 所需参数: 低熵={int(p90_lo):,}  高熵={int(p90_hi):,}  "
              f"ratio={ratio:.2f}×")
        print(f"判定: {verdict}")
    else:
        print("\n判定: 存在未达到 0.90 的曲线，无法直接比较所需参数；"
              "见上表逐宽度差距")
    print("=" * 60)

    out = {"device": str(device), "grid": GRID, "epochs": epochs,
           "n_train": n_tr, "n_test": n_te, "widths": widths,
           "params": {str(w): params_of(w) for w in widths},
           "curves": curves, "params_needed": need}
    Path("p5_h4_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p5_h4_results.json")


if __name__ == "__main__":
    main()
