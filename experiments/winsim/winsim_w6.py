"""
WinSim W6：像素通道 vs 结构通道 —— 同一沙箱内的 held-out 对照

在 W5 的基础上加**像素通道**：对每个候选元素，从其 bbox 裁剪图像 → 小 CNN → 特征，
再结合任务/状态特征打分。与结构通道（role/name/bbox）在同一批样本上对比。

split：in-domain / OOD-theme(d) / OOD-app(run)
预期：结构对主题不变（drop≈0）；像素对主题掉分（与 site_holdout 一致）。

用法:
    python experiments/winsim/winsim_w6.py
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
MAXC = W5.MAXC
CH, CW = 32, 48                      # 裁剪尺寸


def crop(img, e):
    x, y = int(e["x"]), int(e["y"])
    w, h = max(1, int(e["w"])), max(1, int(e["h"]))
    H0, W0 = img.shape[:2]
    x, y = max(0, x), max(0, y)
    x2, y2 = min(W0, x + w), min(H0, y + h)
    if x2 <= x or y2 <= y:
        return np.zeros((CH, CW, 3), np.uint8)
    sub = img[y:y2, x:x2]
    return np.asarray(Image.fromarray(sub).resize((CW, CH), Image.BILINEAR))


def collect(pg, inst):
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    samples = []
    for role, name in inst["seq"]:
        els = pg.evaluate("WinSim.elements()")
        if len(els) > MAXC:
            els = els[:MAXC]
        ti = W5.find(els, role, name)
        if ti < 0:
            break
        st = pg.evaluate("WinSim.truth()")
        act_app = 4
        for w in st["windows"]:
            if w["id"] == st["active"]:
                act_app = W5.APPID.get(w["appId"], 4)
        ctx = [act_app, len(st["path"]), int(bool(st["selected"])), int(bool(st["clipboard"]))]
        img = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        crops = np.zeros((MAXC, CH, CW, 3), np.uint8)
        for j, e in enumerate(els):
            crops[j] = crop(img, e)
        samples.append(dict(els=els, type=inst["type"], tname=inst["tname"],
                            target=ti, ctx=ctx, crops=crops))
        e = els[ti]
        pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")
    return samples


def common(samples):
    B = len(samples)
    role = torch.zeros(B, MAXC, dtype=torch.long)
    name = torch.zeros(B, MAXC, dtype=torch.long)
    box = torch.zeros(B, MAXC, 4)
    mask = torch.zeros(B, MAXC, dtype=torch.bool)
    typ = torch.zeros(B, dtype=torch.long)
    tnm = torch.zeros(B, dtype=torch.long)
    ctx = torch.zeros(B, 4)
    tgt = torch.zeros(B, dtype=torch.long)
    crops = torch.zeros(B, MAXC, 3, CH, CW)
    for i, s in enumerate(samples):
        for j, e in enumerate(s["els"]):
            role[i, j] = W5.ROLES.get(e["role"], 0)
            name[i, j] = W5.nid(e["name"])
            box[i, j] = torch.tensor([e["x"] / 900, e["y"] / 600,
                                      e["w"] / 900, e["h"] / 600])
            mask[i, j] = True
            crops[i, j] = torch.from_numpy(s["crops"][j]).permute(2, 0, 1).float() / 255.0
        typ[i] = W5.TTYPE.get(s["type"], len(W5.TTYPE))
        tnm[i] = W5.nid(s["tname"])
        ctx[i] = torch.tensor(s["ctx"], dtype=torch.float)
        tgt[i] = s["target"]
    return role, name, box, mask, typ, tnm, ctx, crops, tgt


class PixelSel(nn.Module):
    def __init__(self, d=64, dc=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, dc, 3, 2, 1), nn.BatchNorm2d(dc), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.typ = nn.Embedding(len(W5.TTYPE) + 1, d)
        self.tnm = nn.Embedding(len(W5.TRAIN_NAMES) + 1, d)
        self.ctx = nn.Linear(4, d)
        self.proj = nn.Linear(dc * 4, d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, role, name, box, mask, typ, tnm, ctx, crops):
        B, N = role.shape
        cf = self.proj(self.cnn(crops.reshape(B * N, 3, CH, CW)).reshape(B, N, -1))
        t = (self.typ(typ) + self.tnm(tnm) + self.ctx(ctx))[:, None, :]
        x = torch.cat([cf, t.expand_as(cf), cf * t], -1)
        return self.mlp(x).squeeze(-1).masked_fill(~mask, -1e9)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    insts = {
        "in_domain": W5.make_instances("in_domain", 200, rng),
        "ood_theme": W5.make_instances("ood_theme", 110, rng),
        "ood_app":   W5.make_instances("ood_app", 110, rng),
    }
    raw = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW, device_scale_factor=1)
        pg.goto(URL)
        for k, ins in insts.items():
            S = []
            for inst in ins:
                S += collect(pg, inst)
            raw[k] = S
            print(f"  {k}: {len(S)} samples")
        b.close()

    cache = {k: common_tensors(v, device) for k, v in raw.items()}
    tr = cache["in_domain"]
    cut = tr[0].shape[0] - 80
    tr = tuple(t[:cut] for t in tr)
    val = tuple(t[cut:] for t in cache["in_domain"])
    splits = {"in_domain": val, "ood_theme": cache["ood_theme"],
              "ood_app": cache["ood_app"]}

    # struct 复用 W5 的 Sel（输入 7 个）
    def train_struct():
        torch.manual_seed(0)
        m = W5.Sel().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(30):
            perm = torch.randperm(tr[0].shape[0], device=device)
            for i in range(0, tr[0].shape[0], 64):
                idx = perm[i:i + 64]
                loss = F.cross_entropy(m(*[t[idx] for t in tr[:7]]), tr[8][idx])
                opt.zero_grad(); loss.backward(); opt.step()
        @torch.no_grad()
        def acc(T):
            return float((m(*T[:7]).argmax(-1) == T[8]).float().mean())
        return {k: acc(v) for k, v in splits.items()}
    def train_pixel():
        torch.manual_seed(0)
        m = PixelSel().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(30):
            perm = torch.randperm(tr[0].shape[0], device=device)
            for i in range(0, tr[0].shape[0], 64):
                idx = perm[i:i + 64]
                loss = F.cross_entropy(m(*[t[idx] for t in tr[:8]]), tr[8][idx])
                opt.zero_grad(); loss.backward(); opt.step()
        @torch.no_grad()
        def acc(T):
            return float((m(*T[:8]).argmax(-1) == T[8]).float().mean())
        return {k: acc(v) for k, v in splits.items()}

    s = train_struct(); p = train_pixel()
    print(f"\n{'split':<12}{'struct':>9}{'pixel':>9}")
    print("-" * 30)
    for k in ("in_domain", "ood_theme", "ood_app"):
        print(f"{k:<12}{s[k]:>9.3f}{p[k]:>9.3f}")
    print("-" * 30)
    print(f"struct theme drop={s['in_domain']-s['ood_theme']:+.3f}  app drop={s['in_domain']-s['ood_app']:+.3f}")
    print(f"pixel  theme drop={p['in_domain']-p['ood_theme']:+.3f}  app drop={p['in_domain']-p['ood_app']:+.3f}")
    ok = (s["in_domain"] - s["ood_theme"]) < 0.05 and (p["in_domain"] - p["ood_theme"]) > 0.10
    print(f"\nW6 判据：结构主题几乎不变 且 像素主题明显掉 -> {ok}")
    print(f"W6 {'通过' if ok else '未通过'}")

    (OUT / "winsim_w6_results.json").write_text(json.dumps(
        {"struct": s, "pixel": p}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w6_results.json")


def common_tensors(samples, device):
    return tuple(t.to(device) for t in common(samples))


if __name__ == "__main__":
    main()
