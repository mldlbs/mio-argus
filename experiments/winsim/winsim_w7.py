"""
WinSim W7：端到端 —— 自己从像素里检测元素（去掉"完美候选框"）

前情：W6 的像素通道是**给定候选框**只做选择（被 W6/W5 优惠了）。
真实 CUA 必须自己**检测**元素。本实验：
  1. 训练一个小检测器（CenterNet-lite）：截图 → 元素框
  2. 度量它对**目标元素**的召回（center-in-box），对照：
       结构通道（uia 白送框） = 100%
       给定框的像素选择（W6）  = in-domain 100%
  3. 三个 split：in-domain / OOD-theme / OOD-app

预期（与 P7/P8 一致）：检测是瓶颈——recall 明显低于 100%，且随主题/应用外移进一步下降。

用法:
    python experiments/winsim/winsim_w7.py
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

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW
IW, IH = 448, 296                       # 检测输入尺寸
STRIDE = 4
GW, GH = IW // STRIDE, IH // STRIDE     # 112 x 74
MAXC = W5.MAXC


def downsample(img):
    return np.asarray(Image.fromarray(img).resize((IW, IH), Image.BILINEAR))


def collect(pg, inst):
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    out = []
    for role, name in inst["seq"]:
        els = pg.evaluate("WinSim.elements()")
        ti = W5.find(els, role, name)
        if ti < 0 or not els:
            break
        img = downsample(np.asarray(
            Image.open(io.BytesIO(pg.screenshot())).convert("RGB")))
        boxes = np.array([[e["x"] * IW / 900, e["y"] * IH / 600,
                           (e["x"] + e["w"]) * IW / 900,
                           (e["y"] + e["h"]) * IH / 600] for e in els], np.float32)
        target = boxes[ti]
        out.append(dict(img=img, boxes=boxes, target=target, n=len(boxes)))
        e = els[ti]
        pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")
    return out


def heat_targets(boxes, sigma=2.0):
    hm = np.zeros((GH, GW), np.float32)
    sz = np.zeros((2, GH, GW), np.float32)
    g = np.arange(GW, dtype=np.float32)
    gy = np.arange(GH, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        d2 = (g[None, :] - cx) ** 2 + (gy[:, None] - cy) ** 2
        gg = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        hm = np.maximum(hm, gg)
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < GW and 0 <= iy < GH:
            sz[0, iy, ix] = (x2 - x1) / IW
            sz[1, iy, ix] = (y2 - y1) / IH
    return hm, sz


class Det(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.BatchNorm2d(w * 2), nn.ReLU(),
            nn.Conv2d(w * 2, w * 2, 3, 1, 1), nn.BatchNorm2d(w * 2), nn.ReLU())
        self.obj = nn.Conv2d(w * 2, 1, 1)
        self.size = nn.Conv2d(w * 2, 2, 1)

    def forward(self, x):
        f = self.net(x)
        return self.obj(f), self.size(f)


@torch.no_grad()
def decode(m, img_u8, device, thr=0.2, topk=MAXC):
    x = (torch.from_numpy(np.ascontiguousarray(img_u8))
         .permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device))
    o, s = m(x)
    hm = torch.sigmoid(o)[0, 0]
    mx = F.max_pool2d(hm[None, None], 3, 1, 1)[0, 0]
    peaks = (hm >= mx - 1e-6) & (hm > thr)
    ys, xs = peaks.nonzero(as_tuple=True)
    if len(ys) == 0:
        return []
    sc = hm[ys, xs]
    order = torch.argsort(-sc)[:topk]
    out = []
    for i in order.tolist():
        y, x = int(ys[i]), int(xs[i])
        w = float(s[0, 0, y, x]) * IW
        h = float(s[0, 1, y, x]) * IH
        if w < 6 or h < 6:
            continue
        cx, cy = (x + 0.5) * STRIDE, (y + 0.5) * STRIDE
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def center_in(box, t):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return t[0] <= cx <= t[2] and t[1] <= cy <= t[3]


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
                S += collect(pg, inst)
            data[k] = S
            print(f"  {k}: {len(S)} samples")
        b.close()

    # 训练集 = in_domain 的前 80%
    tr = data["in_domain"]
    cut = int(len(tr) * 0.8)
    trS, valS = tr[:cut], tr[cut:]

    torch.manual_seed(0)
    m = Det(w=32).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    B = 32
    for ep in range(60):
        perm = np.random.default_rng(ep).permutation(len(trS))
        for i in range(0, len(trS), B):
            idx = perm[i:i + B]
            imgs = torch.from_numpy(np.stack([trS[j]["img"] for j in idx]))
            imgs = imgs.permute(0, 3, 1, 2).float().div_(255).to(device)
            HM = torch.zeros(len(idx), GH, GW, device=device)
            SZ = torch.zeros(len(idx), 2, GH, GW, device=device)
            for r, j in enumerate(idx):
                hm, sz = heat_targets(trS[j]["boxes"])
                HM[r] = torch.from_numpy(hm).to(device)
                SZ[r] = torch.from_numpy(sz).to(device)
            o, s = m(imgs)
            o = o.squeeze(1)
            pos = (HM > 0.5).float()
            wgt = 1.0 + 20.0 * pos
            lo = F.binary_cross_entropy_with_logits(o, HM, weight=wgt)
            msk = pos.unsqueeze(1)
            ls = ((F.l1_loss(s, SZ, reduction="none") * msk).sum() / msk.sum().clamp_min(1))
            loss = lo + 1.0 * ls
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    def recall(ss):
        hit = 0
        for r in ss:
            dets = decode(m, r["img"], device)
            if any(center_in(b, r["target"]) for b in dets):
                hit += 1
        return hit / max(1, len(ss))

    res = {k: recall(v) for k, v in data.items()}
    print(f"\n{'split':<12}{'detector recall':>17}")
    print("-" * 30)
    for k, v in res.items():
        print(f"{k:<12}{v:>17.3f}")
    print("-" * 30)
    print(f"对照：结构通道(uia 白送框)=1.000；给定框的像素选择(W6) in-domain=1.000")
    print(f"端到端 theme drop={res['in_domain']-res['ood_theme']:+.3f}  "
          f"app drop={res['in_domain']-res['ood_app']:+.3f}")
    ok = res["in_domain"] < 0.95 and res["in_domain"] > res["ood_theme"] - 0.05
    print(f"\nW7 判据：检测 recall 明显低于 1（结构恢复是瓶颈）-> {ok}")
    print(f"W7 {'通过' if ok else '未通过'}")
    (OUT / "winsim_w7_results.json").write_text(json.dumps(
        {"detector_recall": res, "n": {k: len(v) for k, v in data.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w7_results.json")


if __name__ == "__main__":
    main()
