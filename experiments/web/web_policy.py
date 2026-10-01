"""
web_policy：headless 训练 → headed 评测的迁移差

任务：GRID×GRID 网格里每格放一个"符号"，给线索 id，策略指出目标在哪个格。
  - 颜色任务：符号=纯色块，线索=颜色 id     （几何/颜色）
  - 字形任务：符号=小号 unicode 字形，线索=字形 id（依赖字形光栅化，差异大）
网格越密、字号越小，越接近策略能力边界，越能暴露渲染差异造成的迁移损失。

指标：headless_acc / headed_acc / **预测翻转率**（同一页面两种渲染下预测不同的比例）。
翻转率在高准确率下依然敏感。

用法:
    python experiments/web/web_policy.py --quick
    python experiments/web/web_policy.py
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

VIEW = {"width": 288, "height": 288}
GRID = 3
N = GRID * GRID
FS = 18                                    # 小字号 → 渲染差异占比大
COLORS = ["#d33", "#3a7", "#37c", "#da3", "#83c", "#2aa",
          "#a55", "#666", "#b94", "#4a6", "#55a", "#999"]
GLYPHS = ["★", "●", "■", "▶", "◆", "▲", "✚", "✦", "✖"]


def make_page(task, seed):
    rng = np.random.default_rng(seed)
    if task == "color":
        sym = rng.choice(len(COLORS), size=N, replace=False)
    else:
        sym = rng.permutation(len(GLYPHS))
    target = int(rng.integers(0, N))
    cue = int(sym[target])
    cells = []
    for q in range(N):
        if task == "color":
            cells.append(f"<div style='background:{COLORS[sym[q]]};"
                         f"border-radius:4px'></div>")
        else:
            cells.append(f"<div style='display:flex;align-items:center;"
                         f"justify-content:center;font-size:{FS}px;color:#222'>"
                         f"{GLYPHS[sym[q]]}</div>")
    grid = (f"display:grid;grid-template-columns:repeat({GRID},1fr);"
            f"grid-template-rows:repeat({GRID},1fr);gap:3px;"
            f"width:{VIEW['width']}px;height:{VIEW['height']}px;"
            f"box-sizing:border-box;padding:6px;background:#f4f5f7")
    html = (f"<!doctype html><html><head><meta charset='utf-8'></head>"
            f"<body style='margin:0'><div style='{grid}'>{''.join(cells)}</div>"
            f"</body></html>")
    return html, target, cue


class Net(nn.Module):
    """cue × 空间点积 → 每格一个分数。"""
    def __init__(self, ncue=12, w=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU(),
            nn.Conv2d(w * 2, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU())
        self.dim = w * 2
        self.cue = nn.Embedding(ncue, self.dim)
        self.q = nn.Linear(self.dim, self.dim)

    def forward(self, x, cue):
        f = self.conv(x)
        q = self.q(self.cue(cue))
        s = (f * q[..., None, None]).sum(1)
        return F.adaptive_avg_pool2d(s.unsqueeze(1), (GRID, GRID)).flatten(1)


def render(pw, task, seeds, headless):
    b = pw.chromium.launch(channel="chrome", headless=headless)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    imgs, cues, tgts = [], [], []
    for s in seeds:
        html, t, cue = make_page(task, s)
        pg.set_content(html)
        img = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        imgs.append(img); tgts.append(t); cues.append(cue)
    b.close()
    return np.stack(imgs), np.array(cues), np.array(tgts)


def _prep(imgs, cues, tgts, device):
    X = (torch.from_numpy(np.ascontiguousarray(imgs))
         .permute(0, 3, 1, 2).contiguous().float().div_(255.0).to(device))
    C = torch.from_numpy(np.asarray(cues)).long().to(device)
    Y = torch.from_numpy(np.asarray(tgts)).long().to(device)
    assert int(Y.min()) >= 0 and int(Y.max()) < N
    return X, C, Y


def train(imgs, cues, tgts, device, epochs, aug=False):
    torch.manual_seed(0)
    m = Net().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    X, C, Y = _prep(imgs, cues, tgts, device)
    n, _, H, Wd = X.shape
    B = 64
    m.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            Xb = X[idx]
            bs = Xb.shape[0]
            if aug:                                   # 抗渲染差异：平移/亮度/噪声
                Xb = Xb * (1.0 + 0.12 * torch.randn(bs, 1, 1, 1, device=device))
                Xb = Xb + 0.03 * torch.randn_like(Xb)
                p = 3
                Xp = F.pad(Xb, (p, p, p, p), mode="replicate")
                ox = int(torch.randint(0, 2 * p + 1, (1,)).item())
                oy = int(torch.randint(0, 2 * p + 1, (1,)).item())
                Xb = Xp[..., oy:oy + H, ox:ox + Wd]
            loss = F.cross_entropy(m(Xb, C[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def preds(m, imgs, cues, tgts, device):
    m.eval()
    X, C, Y = _prep(imgs, cues, tgts, device)
    p = m(X, C).argmax(-1)
    return p, float((p == Y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--aug", action="store_true", help="训练加抗渲染增强")
    args = ap.parse_args()
    tasks = args.tasks.split(",") if args.tasks else ["color", "glyph"]
    n_tr = args.n_train if args.n_train else (1500 if args.quick else 3000)
    n_te = 400 if args.quick else 800
    ep = args.epochs if args.epochs else (15 if args.quick else 25)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} grid={GRID}x{GRID} fs={FS} n_tr={n_tr} "
          f"n_te={n_te} epochs={ep}")
    tr = list(range(n_tr)); te = list(range(100000, 100000 + n_te))

    out = {}
    with sync_playwright() as pw:
        for task in tasks:
            tri, trc, trt = render(pw, task, tr, headless=True)
            hi, hc, ht = render(pw, task, te, headless=True)
            Hi, _, _ = render(pw, task, te, headless=False)
            m = train(tri, trc, trt, device, ep, aug=args.aug)
            ph, ah = preds(m, hi, hc, ht, device)
            pH, aH = preds(m, Hi, hc, ht, device)
            flip = float((ph != pH).float().mean())
            print(f"  [{task}] headless_acc={ah:.4f} headed_acc={aH:.4f} "
                  f"gap={aH-ah:+.4f} flip={flip:.4f}")
            out[task] = dict(headless=ah, headed=aH, gap=aH - ah, flip=flip)

    Path("results/web/web_policy_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 results/web/web_policy_results.json")


if __name__ == "__main__":
    main()
