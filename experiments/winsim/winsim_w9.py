"""
WinSim W9：更强检测头 —— 0.26 是方法极限还是实现极限？

W7/W8 的检测器是 stride 4、448×296 输入。W9 试 stride 2 高分辨率头。
但先修一个**与 P9 同源的真 bug**：尺寸监督只在中心 cell 非零，而 objectness
正样本是整片高斯 → size head 被压向 0（~3.6px）→ 检测被 min-size 全部滤掉。
修法：把尺寸标签铺满高斯正区域（与 objectness 对齐）。

对照：
  D0  = stride 4（W7.Det），修复前后
  D1  = stride 2 高分辨率头（Det2），修复后
判据：若 D1（或修复后的 D0）明显抬手 → 0.26 是**实现极限**；否则更像**方法极限**。

用法:
    python experiments/winsim/winsim_w9.py
"""
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from playwright.sync_api import sync_playwright

import winsim_w5 as W5
import winsim_w6 as W6
import winsim_w7 as W7
import winsim_w8 as W8

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW
IW, IH = W7.IW, W7.IH


class Det2(nn.Module):
    """/4 backbone + 上采样回 /2 的高分辨率头。"""
    def __init__(self, w=32):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(w, w * 2, 3, 2, 1),
                                nn.BatchNorm2d(w * 2), nn.ReLU())
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                nn.Conv2d(w * 2, w * 2, 3, 1, 1),
                                nn.BatchNorm2d(w * 2), nn.ReLU())
        self.obj = nn.Conv2d(w * 2, 1, 1)
        self.size = nn.Conv2d(w * 2, 2, 1)

    def forward(self, x):
        f = self.up(self.b2(self.b1(x)))
        return self.obj(f), self.size(f)


def heat_fixed(boxes, stride, gh, gw, sigma):
    """尺寸标签铺满高斯正区域（修复 P9 同源 bug）。"""
    hm = np.zeros((gh, gw), np.float32)
    sz = np.zeros((2, gh, gw), np.float32)
    g = np.arange(gw, dtype=np.float32)
    gy = np.arange(gh, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        d2 = (g[None, :] - cx) ** 2 + (gy[:, None] - cy) ** 2
        gg = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        reg = gg > 0.5
        sz[0][reg] = (x2 - x1) / IW
        sz[1][reg] = (y2 - y1) / IH
        hm = np.maximum(hm, gg)
    return hm, sz


@torch.no_grad()
def decode(m, img_u8, stride, device, thr=0.2, topk=W5.MAXC):
    x = (torch.from_numpy(np.ascontiguousarray(img_u8))
         .permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device))
    o, s = m(x)
    hm = torch.sigmoid(o)[0, 0]
    mx = F.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
    peaks = (hm >= mx - 1e-6) & (hm > thr)
    ys, xs = peaks.nonzero(as_tuple=True)
    if len(ys) == 0:
        return []
    order = torch.argsort(-hm[ys, xs])[:topk]
    out = []
    for i in order.tolist():
        y, x = int(ys[i]), int(xs[i])
        w = float(s[0, 0, y, x]) * IW
        h = float(s[0, 1, y, x]) * IH
        if w < 6 or h < 6:
            continue
        cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def train(model, stride, gh, gw, sigma, samples, device, epochs=60):
    torch.manual_seed(0)
    m = model.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    B = 16
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(samples))
        for i in range(0, len(samples), B):
            idx = perm[i:i + B]
            imgs = torch.from_numpy(np.stack([samples[j]["img"] for j in idx]))
            imgs = imgs.permute(0, 3, 1, 2).float().div_(255).to(device)
            HM = torch.zeros(len(idx), gh, gw, device=device)
            SZ = torch.zeros(len(idx), 2, gh, gw, device=device)
            for r, j in enumerate(idx):
                hm, sz = heat_fixed(samples[j]["boxes"], stride, gh, gw, sigma)
                HM[r] = torch.from_numpy(hm).to(device)
                SZ[r] = torch.from_numpy(sz).to(device)
            o, s = m(imgs)
            o = o.squeeze(1)
            pos = (HM > 0.5).float()
            lo = F.binary_cross_entropy_with_logits(o, HM, weight=1 + 20 * pos)
            msk = pos.unsqueeze(1)
            ls = (F.l1_loss(s, SZ, reduction="none") * msk).sum() / msk.sum().clamp_min(1)
            loss = lo + ls
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def recall_of(det, stride, samples, device):
    hit = 0
    for s in samples:
        dets = decode(det, s["img"], stride, device)
        if any(W8.center_in(b, s["boxes"][s["target"]]) for b in dets):
            hit += 1
    return hit / max(1, len(samples))


@torch.no_grad()
def e2e(det, stride, sel, samples, device):
    det.eval(); sel.eval()
    hit = 0
    N = W5.MAXC
    for s in samples:
        dets = decode(det, s["img"], stride, device)
        if not dets:
            continue
        n = min(N, len(dets))
        crops = torch.zeros(1, N, 3, W6.CH, W6.CW, device=device)
        box = torch.zeros(1, N, 4, device=device)
        mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        for c in range(n):
            x1, y1, x2, y2 = dets[c]
            sub = s["img"][int(max(0, y1)):int(min(IH, y2)),
                           int(max(0, x1)):int(min(IW, x2))]
            if sub.size == 0:
                continue
            cr = np.asarray(Image.fromarray(sub).resize((W6.CW, W6.CH), Image.BILINEAR))
            crops[0, c] = torch.from_numpy(cr).permute(2, 0, 1).float().div_(255).to(device)
            box[0, c] = torch.tensor([x1 / IW, y1 / IH, x2 / IW, y2 / IH], device=device)
            mask[0, c] = True
        role = torch.zeros(1, N, dtype=torch.long, device=device)
        name = torch.zeros(1, N, dtype=torch.long, device=device)
        typ = torch.tensor([W5.TTYPE.get(s["type"], len(W5.TTYPE))], device=device)
        tnm = torch.tensor([W5.nid(s["tname"])], device=device)
        ctx = torch.tensor([s["ctx"]], dtype=torch.float, device=device)
        pick = int(sel(role, name, box, mask, typ, tnm, ctx, crops)[0].argmax())
        if mask[0, pick] and W8.center_in(dets[pick], s["boxes"][s["target"]]):
            hit += 1
    return hit / max(1, len(samples))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    insts = {
        "in_domain": W5.make_instances("in_domain", 160, rng),
        "ood_theme": W5.make_instances("ood_theme", 90, rng),
        "ood_app":   W5.make_instances("ood_app", 90, rng),
    }
    data = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW, device_scale_factor=1)
        pg.goto(URL)
        for k, ins in insts.items():
            S = []
            for inst in ins:
                S += W8.collect(pg, inst)
            data[k] = S
            print(f"  {k}: {len(S)} samples")
        b.close()

    tr = data["in_domain"]
    trS = tr[:int(len(tr) * 0.8)]
    sel = W8.train_selector(trS, device)

    g4h, g4w = IH // 4, IW // 4
    g2h, g2w = IH // 2, IW // 2
    d0 = train(W7.Det(w=32), 4, g4h, g4w, 2.0, trS, device)
    d1 = train(Det2(w=32), 2, g2h, g2w, 3.0, trS, device)

    rows = {}
    for k, v in data.items():
        rows[k] = dict(
            d0_recall_fixed=recall_of(d0, 4, v, device),
            d1_recall=recall_of(d1, 2, v, device),
            d1_e2e=e2e(d1, 2, sel, v, device))
    print(f"\n{'split':<12}{'D0 rec(fix)':>13}{'D1 rec':>9}{'D1 e2e':>9}")
    print("-" * 44)
    for k, r in rows.items():
        print(f"{k:<12}{r['d0_recall_fixed']:>13.3f}"
              f"{r['d1_recall']:>9.3f}{r['d1_e2e']:>9.3f}")
    print("-" * 44)
    print("对照（未修复的 W7/W8）：D0 recall in 0.307；D0 e2e in 0.260")

    imp_rec = rows["in_domain"]["d1_recall"] - 0.307
    imp_e2e = rows["in_domain"]["d1_e2e"] - 0.260
    print(f"\nin-domain：D1 recall 提升 {imp_rec:+.3f}，D1 e2e 提升 {imp_e2e:+.3f}")
    verdict = ("实现极限（更强头/修监督明显抬手）"
               if (imp_rec > 0.15 or imp_e2e > 0.15) else
               "更像方法极限（更强头也抬不动）")
    print(f"W9 判断：{verdict}")

    (OUT / "winsim_w9_results.json").write_text(json.dumps(
        {"rows": rows, "baseline": {"d0_recall": 0.307, "d0_e2e": 0.260},
         "improvement": {"recall": imp_rec, "e2e": imp_e2e},
         "verdict": verdict}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w9_results.json")


if __name__ == "__main__":
    main()
