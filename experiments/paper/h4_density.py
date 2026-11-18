"""H4 干扰密度实验 runner（新文件；不改 P5 / P3）。

协议：`results/paper/H4_GENERALIZATION_PREREG.md` v1.0
  6 条件 × 7 width × 3 seed = 126 次
  C0 flat/0  C1 flat/1  C2 flat/2  C3 flat/4  C4 noise/0  C5 natural/0

主指标：params_needed(0.80)（log 参数轴线性插值；未达到记 null）。

用法：
    python experiments/paper/h4_density.py --quick           # smoke（小样本）
    python experiments/paper/h4_density.py                   # 126 次全量
    python experiments/paper/h4_density.py --render-dir artifacts/h4_preview --only C0,C3
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from p0_env import (EnvCfg, LABELS, PALETTE, _font, instruction_onehot,
                    make_sample)
from p0_models import P0Net

IMG_SIZE = 160
GRID = 6
S = GRID * GRID
WIDTHS = [4, 6, 8, 12, 16, 24, 32]
SEEDS = [0, 1, 2]
N_TR = 3000
N_TE = 1000
EPOCHS = 20
BATCH = 64
LR = 3e-3
TRAIN_SEED = 100
TEST_SEED = 999
PROTO_VERSION = "h4-v1.0"

# (name, level, decoys_per_cell)
CONDS = [("C0", "flat", 0), ("C1", "flat", 1), ("C2", "flat", 2),
         ("C3", "flat", 4), ("C4", "noise", 0), ("C5", "natural", 0)]

_NAT = None


def cfg6() -> EnvCfg:
    return EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=S,
                  fill=0.40, jitter=4, color_by_label=True)


# ---------- 背景 ----------

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
    if level == "flat":
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), (246, 247, 249))
    if level == "noise":
        arr = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    if level == "blobs":  # 保留定义，本次不用（H4 只测 noise/natural）
        k = int(rng.integers(6, 14))
        small = rng.integers(0, 256, (k, k, 3), dtype=np.uint8)
        return Image.fromarray(small, "RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    pool = _natural_pool()
    return pool[int(rng.integers(0, len(pool)))]


# ---------- 干扰项（与真实元素同分布，仅颜色不同）----------

def gen_decoys(cfg, target_color, k, rng):
    """每格 k 个类元素矩形；排除目标色以避免不可解歧义。"""
    out = []
    palette_n = len(PALETTE)
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = cfg.slot_bbox(slot)
        w = (bx2 - bx1) // 2
        h = (by2 - by1) // 2
        idxs = rng.choice(4, size=k, replace=False) if k > 0 else []
        for j in idxs:
            sx = bx1 + (int(j) % 2) * w
            sy = by1 + (int(j) // 2) * h
            ew = max(5, int(w * 0.70) - int(rng.integers(0, 3)))
            eh = max(5, int(h * 0.70) - int(rng.integers(0, 3)))
            ex1 = sx + int(rng.integers(0, max(1, w - ew)))
            ey1 = sy + int(rng.integers(0, max(1, h - eh)))
            ex2 = min(ex1 + ew, IMG_SIZE - 1)
            ey2 = min(ey1 + eh, IMG_SIZE - 1)
            color = None
            for _ in range(8):
                c = PALETTE[int(rng.integers(0, palette_n))]
                if tuple(int(v) for v in c) != tuple(int(v) for v in target_color):
                    color = tuple(int(v) for v in c)
                    break
            if color is None:
                continue
            out.append(((ex1, ey1, ex2, ey2), color))
    return out


def render_cond(sample, level, k, rng, cfg):
    """背景 → 网格线 → 干扰项 → 真实元素（真实元素永远最上层）。"""
    img = make_bg(level, rng).copy()
    d = ImageDraw.Draw(img)
    for slot in range(cfg.num_slots):
        x1, y1, x2, y2 = cfg.slot_bbox(slot)
        d.rectangle([x1, y1, x2, y2], outline=(214, 218, 224))

    tgt = sample.elements[sample.target_slot]
    for (x1, y1, x2, y2), color in gen_decoys(cfg, tgt.color, k, rng):
        d.rectangle([x1, y1, x2, y2], fill=color, outline=(25, 25, 25))

    font = _font(10)
    for e in sample.elements:
        x1, y1, x2, y2 = e.bbox
        d.rectangle([x1, y1, x2, y2], fill=tuple(int(v) for v in e.color),
                    outline=(25, 25, 25))
        if (x2 - x1) >= 30 and (y2 - y1) >= 14:
            bb = d.textbbox((0, 0), e.label, font=font)
            d.text((x1 + max(0, ((x2 - x1) - (bb[2] - bb[0])) // 2), y1 + 2),
                   e.label, fill=(255, 255, 255), font=font)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


# ---------- 数据 ----------

def build_data(n, split_seed, cond_idx, cfg):
    level, k = CONDS[cond_idx][1], CONDS[cond_idx][2]
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    for i in range(n):
        brng = np.random.default_rng(split_seed * 100000 + i)
        s = make_sample(brng, cfg)
        drng = np.random.default_rng(split_seed * 100000 + i * 13 + cond_idx * 7 + 1)
        imgs[i] = (render_cond(s, level, k, drng, cfg) * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {s.target_label}", len(LABELS))
        slots[i] = s.target_slot
    return {"images": imgs, "inst": inst, "slots": slots}


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ---------- 模型 ----------

def make_model(width, device):
    return P0Net(width=width, head="cls", num_slots=S, grid=GRID,
                 num_labels=len(LABELS), inst_dim=64,
                 in_res=IMG_SIZE, stages=3).to(device)


def n_params(width):
    return sum(p.numel() for p in make_model(width, "cpu").parameters())


def train(width, tr, epochs, device, seed=0):
    torch.manual_seed(seed)
    m = make_model(width, device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    imgs = torch.from_numpy(tr["images"].astype(np.float32) / 255.0).to(device)
    inst = torch.from_numpy(tr["inst"]).to(device)
    tgt = torch.from_numpy(tr["slots"]).to(device)
    n = imgs.shape[0]
    spe = max(1, n // BATCH)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k * BATCH:(k + 1) * BATCH]
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


def params_needed(curve, target=0.80):
    """log 参数轴线性插值；未达到返回 (None, 'none')；w4 已达标返回 (p0, 'left')。"""
    ps = [n_params(w) for w in WIDTHS]
    for i in range(len(curve) - 1):
        if curve[i] < target <= curve[i + 1]:
            t = (target - curve[i]) / (curve[i + 1] - curve[i])
            lp = np.log(ps[i]) + t * (np.log(ps[i + 1]) - np.log(ps[i]))
            return float(np.exp(lp)), "interp"
    if curve[0] >= target:
        return float(ps[0]), "left"
    return None, "none"


def config_fp():
    payload = {
        "proto": PROTO_VERSION, "grid": GRID, "img": IMG_SIZE, "s": S,
        "widths": WIDTHS, "seeds": SEEDS, "n_tr": N_TR, "n_te": N_TE,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR,
        "train_seed": TRAIN_SEED, "test_seed": TEST_SEED,
        "head": "cls", "inst_dim": 64, "stages": 3, "in_res": IMG_SIZE,
        "conds": CONDS, "decoy_rule": "2x2 subcells, ew=max(5,int(w*0.70)-U{0..2}), exclude target color",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--runs", type=int, default=0, help="只跑前 N 个 (w,seed)")
    ap.add_argument("--only", default="", help="逗号分隔条件名，仅用于预览")
    ap.add_argument("--render-dir", default="")
    ap.add_argument("--out", default="results/paper/h4_density_runs.jsonl")
    ap.add_argument("--hashes", default="results/paper/h4_density_hashes.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tr = 300 if args.quick else N_TR
    n_te = 200 if args.quick else N_TE
    epochs = 2 if args.quick else EPOCHS
    cfg = cfg6()

    conds = [c for c in CONDS if (not args.only or c[0] in args.only.split(","))]
    idxs = [CONDS.index(c) for c in conds]

    t0 = time.time()
    tr, te, meta = {}, {}, {}
    for ci in idxs:
        tr[ci] = build_data(n_tr, TRAIN_SEED, ci, cfg)
        te[ci] = build_data(n_te, TEST_SEED, ci, cfg)
        m = {"name": CONDS[ci][0], "level": CONDS[ci][1], "decoys": CONDS[ci][2],
             "img_hash_train": digest(tr[ci]["images"]),
             "img_hash_test": digest(te[ci]["images"]),
             "base_hash_train": digest(tr[ci]["inst"], tr[ci]["slots"]),
             "base_hash_test": digest(te[ci]["inst"], te[ci]["slots"])}
        meta[ci] = m
        print(f"[data] {m['name']} decoys={m['decoys']} "
              f"img={m['img_hash_train']}/{m['img_hash_test']} "
              f"base={m['base_hash_train']}/{m['base_hash_test']} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # 配对自检：同 split 下 6 个条件的 base_hash 必须完全一致
    for split, d in (("train", "base_hash_train"), ("test", "base_hash_test")):
        hs = {meta[ci][d] for ci in idxs}
        print(f"[check] {split} base_hash 唯一数 = {len(hs)} "
              f"{'OK（任务流配对）' if len(hs) == 1 else 'MISMATCH'}", flush=True)

    if args.render_dir:
        Path(args.render_dir).mkdir(parents=True, exist_ok=True)
        for ci in idxs:
            for j in range(2):
                a = (te[ci]["images"][j].transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(a).save(Path(args.render_dir) /
                                        f"{CONDS[ci][0]}_sample{j}.png")
        print(f"[render] 预览图已写入 {args.render_dir}", flush=True)

    fp = config_fp()
    print(f"[cfg] config_fp={fp} device={device} n_tr={n_tr} epochs={epochs}", flush=True)

    hashes = {"config_fp": fp, "proto": PROTO_VERSION,
              "device": str(device),
              "torch": torch.__version__, "numpy": np.__version__,
              "run": {"n_tr": n_tr, "n_te": n_te, "epochs": epochs},
              "conds": meta}
    Path(args.hashes).parent.mkdir(parents=True, exist_ok=True)
    Path(args.hashes).write_text(json.dumps(hashes, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    pairs = [(wi, w, s) for wi, w in enumerate(WIDTHS) for s in SEEDS]
    if args.runs:
        pairs = pairs[:args.runs]

    order = []
    for wi, w, s in pairs:
        rot = (wi + s) % len(conds)
        order.append((wi, w, s, conds[rot:] + conds[:rot]))
    order = [(wi, w, s, c) for (wi, w, s, cs) in order for c in cs]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not args.quick and not args.runs and Path(args.out).exists():
        Path(args.out).unlink()

    done = 0
    for wi, w, s, c in order:
        ci = CONDS.index(c)
        t1 = time.time()
        m = train(w, tr[ci], epochs, device, seed=s)
        a = acc(m, te[ci], device)
        row = {"cond": c[0], "level": c[1], "decoys": c[2], "width": w,
               "seed": s, "acc": round(float(a), 4), "params": n_params(w),
               "config_fp": fp,
               "base_hash": meta[ci]["base_hash_train"],
               "img_hash": meta[ci]["img_hash_test"],
               "seconds": round(time.time() - t1, 1),
               "n_tr": n_tr, "epochs": epochs}
        with Path(args.out).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        done += 1
        print(f"[{done}/{len(order)}] {c[0]} w{w} s{s} acc={row['acc']:.4f} "
              f"({row['seconds']}s)", flush=True)

    print(f"\n完成 {done} 个 run，用时 {time.time() - t0:.1f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
