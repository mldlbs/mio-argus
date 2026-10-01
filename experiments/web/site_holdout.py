"""
site_holdout：站点级 held-out —— 覆盖不足的第一刀（受控版本）

任务（两通道都可解）：
  页面 2×2 放 4 个色块；给一个 **目标 RGB 线索**，指出目标在哪个象限。
  - 像素通道：截图 + 线索 → 象限（必须自己"看"出哪个块颜色匹配）
  - 结构通道：4 个元素的 [cx,cy,w,h,r,g,b] + 线索 → 象限（DOM 特权，免费框+色）

站点 = **主题/调色板/布局/字体** 的独立风格；评测：
  - in-domain：训练风格上的留出页
  - OOD     ：**未见风格**

核心信号：像素通道的 in→OOD 掉幅 >> 结构通道（结构把"风格差异"这件事直接抹掉了）。

用法:
    python experiments/web/site_holdout.py --quick
    python experiments/web/site_holdout.py
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
IMG = 192
VIEW = {"width": 384, "height": 288}
VEL = 4                                    # 每页元素数

STYLES = {
    "light_blue": dict(bg="#f2f5fb", pal=["#2b6cff", "#4d8cff", "#123f9e", "#7fb0ff"],
                       radius=6, font="Arial", jitter=40, pad=14),
    "dark_warm":  dict(bg="#221f1c", pal=["#e8833a", "#f2b134", "#b84b1f", "#ffd27f"],
                       radius=16, font="Georgia", jitter=70, pad=26),
    "light_green": dict(bg="#f4fbf4", pal=["#1f9d55", "#4fd18a", "#0b6b39", "#a8e6c3"],
                        radius=999, font="Verdana", jitter=20, pad=8),
    "dark_purple": dict(bg="#1a1630", pal=["#8b5cf6", "#c4b5fd", "#4c1d95", "#ddd6fe"],
                        radius=10, font="Tahoma", jitter=55, pad=20),   # OOD
}


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def make_page(style, seed):
    rng = np.random.default_rng(seed)
    s = STYLES[style]
    cidx = rng.choice(len(s["pal"]), size=VEL, replace=False)
    target = int(rng.integers(0, VEL))
    cue = hex2rgb(s["pal"][cidx[target]])
    cells, boxes = [], []
    cell_w, cell_h = (VIEW["width"] - 2 * s["pad"]) // 2, (VIEW["height"] - 2 * s["pad"]) // 2
    for q in range(VEL):
        r, c = divmod(q, 2)
        bw = int(rng.integers(cell_w * 0.45, cell_w * 0.8))
        bh = int(rng.integers(cell_h * 0.4, cell_h * 0.8))
        ox = int(rng.integers(0, s["jitter"]))
        oy = int(rng.integers(0, s["jitter"]))
        x = s["pad"] + c * cell_w + (cell_w - bw) // 2 + ox - s["jitter"] // 2
        y = s["pad"] + r * cell_h + (cell_h - bh) // 2 + oy - s["jitter"] // 2
        x = max(0, min(VIEW["width"] - bw, x)); y = max(0, min(VIEW["height"] - bh, y))
        color = s["pal"][cidx[q]]
        cells.append(f"<div style='position:absolute;left:{x}px;top:{y}px;"
                     f"width:{bw}px;height:{bh}px;background:{color};"
                     f"border-radius:{s['radius']}px'></div>")
        boxes.append((x, y, x + bw, y + bh, hex2rgb(color)))
    html = (f"<!doctype html><html><head><meta charset='utf-8'></head>"
            f"<body style='margin:0;background:{s['bg']};"
            f"font-family:{s['font']}'>{''.join(cells)}</body></html>")
    return html, target, cue, boxes


def quadrant(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (0 if cx < VIEW["width"] / 2 else 1) + (0 if cy < VIEW["height"] / 2 else 2)


def collect(style, seeds, headless=True):
    rows = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=headless)
        pg = b.new_page(viewport=VIEW, device_scale_factor=1)
        for sd in seeds:
            html, target, cue, boxes = make_page(style, sd)
            pg.set_content(html)
            shot = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
            rows.append(dict(img=shot, cue=cue, cls=quadrant(boxes[target]),
                             struct=[v for bd in boxes for v in
                                     (bd[0] / VIEW["width"], bd[1] / VIEW["height"],
                                      bd[2] / VIEW["width"], bd[3] / VIEW["height"],
                                      bd[4][0], bd[4][1], bd[4][2])],
                             style=style))
        b.close()
    return rows


class PixelNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU(),
            nn.Conv2d(w * 2, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU())
        self.dim = w * 2
        self.qc = nn.Linear(3, self.dim)

    def forward(self, x, cue):
        f = self.conv(x)
        q = self.qc(cue)
        s = (f * q[..., None, None]).sum(1)
        return F.adaptive_avg_pool2d(s.unsqueeze(1), (2, 2)).flatten(1)


class StructNet(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(VEL * 7 + 3, w), nn.ReLU(),
                                 nn.Linear(w, w), nn.ReLU(), nn.Linear(w, VEL))

    def forward(self, x):
        return self.net(x)


def prep(rows, kind, device):
    if kind == "pixel":
        X = torch.from_numpy(np.stack([np.asarray(
            Image.fromarray(r["img"]).resize((IMG, IMG), Image.BILINEAR))
            for r in rows])).permute(0, 3, 1, 2).contiguous().float().div_(255).to(device)
        C = torch.tensor([r["cue"] for r in rows], device=device).float()
    else:
        X = torch.tensor([r["struct"] for r in rows], device=device).float()
        C = torch.tensor([r["cue"] for r in rows], device=device).float()
    Y = torch.tensor([r["cls"] for r in rows], device=device)
    return X, C, Y


def run(kind, tr, te, ood, device, epochs, seed=0):
    torch.manual_seed(seed)
    m = (PixelNet() if kind == "pixel" else StructNet()).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    if kind == "pixel":
        X, Cc, Y = prep(tr, kind, device)
        def fwd(Xb, Cb): return m(Xb, Cb)
    else:
        X, Cc, Y = prep(tr, kind, device)
        def fwd(Xb, Cb): return m(torch.cat([Xb, Cb], -1))
    n = len(X); B = 64; m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            loss = F.cross_entropy(fwd(X[idx], Cc[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(rows):
        m.eval()
        Xd, Cd, Yd = prep(rows, kind, device)
        return float((fwd(Xd, Cd).argmax(-1) == Yd).float().mean())
    return acc(te), acc(ood)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    n_tr = args.n_train if args.n_train else (400 if args.quick else 1200)
    epochs = args.epochs if args.epochs else (12 if args.quick else 25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    styles = list(STYLES)
    ood_style = styles[-1]
    in_styles = styles[:-1]
    print(f"device={device} in={in_styles} OOD={ood_style} n_tr/style={n_tr}")

    tr, te = [], []
    for i, st in enumerate(in_styles):
        tr += collect(st, range(i * 100000, i * 100000 + n_tr))
        te += collect(st, range(i * 700000, i * 700000 + 200))
    ood = collect(ood_style, range(900000, 900000 + 400))
    print(f"samples: train={len(tr)} in_test={len(te)} ood={len(ood)}")

    out = {}
    for kind in ("pixel", "struct"):
        a_in, a_ood = run(kind, tr, te, ood, device, epochs)
        out[kind] = dict(in_domain=a_in, ood=a_ood, drop=a_in - a_ood)
        print(f"  [{kind:<6}] in-domain={a_in:.4f} OOD={a_ood:.4f} "
              f"drop={a_in - a_ood:+.4f}")

    print(f"\n判据：像素 in→OOD 掉幅应远大于结构")
    print(f"  pixel drop={out['pixel']['drop']:+.3f}  "
          f"struct drop={out['struct']['drop']:+.3f}")
    Path("results/web/site_holdout_results.json").write_text(
        json.dumps({"in_styles": in_styles, "ood_style": ood_style,
                    "n": dict(train=len(tr), in_test=len(te), ood=len(ood)),
                    "results": out}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 results/web/site_holdout_results.json")


if __name__ == "__main__":
    main()
