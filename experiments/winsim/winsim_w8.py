"""
WinSim W8：端到端联合 —— 检测器 + 选择器 → 真实动作

W7 只量了检测 recall。这里把整条像素链路接起来：
  截图 → 检测器出候选框 → 选择器在候选里选一个 → 点它的中心
度量：next-action **命中率**（点中的框中心落在 oracle 目标元素内）。
对照：结构通道（uia）≈ 1.0；给定框的像素选择（W6）in-domain = 1.0。
split：in-domain / OOD-theme / OOD-app

用法:
    python experiments/winsim/winsim_w8.py
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

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW


def collect(pg, inst):
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    out = []
    for role, name in inst["seq"]:
        els = pg.evaluate("WinSim.elements()")
        ti = W5.find(els, role, name)
        if ti < 0 or not els:
            break
        raw = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        img = np.asarray(Image.fromarray(raw).resize((W7.IW, W7.IH), Image.BILINEAR))
        sx, sy = W7.IW / 900, W7.IH / 600
        boxes = np.array([[e["x"] * sx, e["y"] * sy, (e["x"] + e["w"]) * sx,
                           (e["y"] + e["h"]) * sy] for e in els], np.float32)
        crops = np.zeros((len(els), W6.CH, W6.CW, 3), np.uint8)
        for j, e in enumerate(els):
            crops[j] = W6.crop(raw, e)
        st = pg.evaluate("WinSim.truth()")
        act = 4
        for w in st["windows"]:
            if w["id"] == st["active"]:
                act = W5.APPID.get(w["appId"], 4)
        ctx = [act, len(st["path"]), int(bool(st["selected"])), int(bool(st["clipboard"]))]
        out.append(dict(img=img, boxes=boxes, crops=crops, target=ti,
                        type=inst["type"], tname=inst["tname"], ctx=ctx))
        e = els[ti]
        pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")
    return out


def train_detector(samples, device, epochs=60, w=32):
    torch.manual_seed(0)
    m = W7.Det(w=w).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    B = 32
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(samples))
        for i in range(0, len(samples), B):
            idx = perm[i:i + B]
            imgs = torch.from_numpy(np.stack([samples[j]["img"] for j in idx]))
            imgs = imgs.permute(0, 3, 1, 2).float().div_(255).to(device)
            HM = torch.zeros(len(idx), W7.GH, W7.GW, device=device)
            SZ = torch.zeros(len(idx), 2, W7.GH, W7.GW, device=device)
            for r, j in enumerate(idx):
                hm, sz = W7.heat_targets(samples[j]["boxes"])
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


def train_selector(samples, device, epochs=30):
    torch.manual_seed(0)
    m = W6.PixelSel().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    N = W5.MAXC
    B = 64
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(samples))
        for i in range(0, len(samples), B):
            idx = perm[i:i + B]
            Bn = len(idx)
            role = torch.zeros(Bn, N, dtype=torch.long, device=device)
            name = torch.zeros(Bn, N, dtype=torch.long, device=device)
            box = torch.zeros(Bn, N, 4, device=device)
            mask = torch.zeros(Bn, N, dtype=torch.bool, device=device)
            crops = torch.zeros(Bn, N, 3, W6.CH, W6.CW, device=device)
            typ = torch.zeros(Bn, dtype=torch.long, device=device)
            tnm = torch.zeros(Bn, dtype=torch.long, device=device)
            ctx = torch.zeros(Bn, 4, device=device)
            tgt = torch.zeros(Bn, dtype=torch.long, device=device)
            for r, j in enumerate(idx):
                s = samples[j]
                for c in range(min(N, len(s["boxes"]))):
                    role[r, c] = 0
                    name[r, c] = 0
                    box[r, c] = torch.tensor(s["boxes"][c] / [W7.IW, W7.IH, W7.IW, W7.IH])
                    mask[r, c] = True
                    crops[r, c] = torch.from_numpy(s["crops"][c]).permute(2, 0, 1).float() / 255.0
                typ[r] = W5.TTYPE.get(s["type"], len(W5.TTYPE))
                tnm[r] = W5.nid(s["tname"])
                ctx[r] = torch.tensor(s["ctx"], dtype=torch.float)
                tgt[r] = s["target"]
            loss = F.cross_entropy(m(role, name, box, mask, typ, tnm, ctx, crops), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def center_in(box, t):
    return (t[0] <= (box[0] + box[2]) / 2 <= t[2]
            and t[1] <= (box[1] + box[3]) / 2 <= t[3])


@torch.no_grad()
def evaluate(det, sel, samples, device):
    det.eval(); sel.eval()
    hit = 0
    N = W5.MAXC
    for s in samples:
        dets = W7.decode(det, s["img"], device)
        if not dets:
            continue
        n = min(N, len(dets))
        crops = torch.zeros(1, N, 3, W6.CH, W6.CW, device=device)
        box = torch.zeros(1, N, 4, device=device)
        mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        for c in range(n):
            x1, y1, x2, y2 = dets[c]
            sub = s["img"][int(max(0, y1)):int(min(W7.IH, y2)),
                           int(max(0, x1)):int(min(W7.IW, x2))]
            if sub.size == 0:
                continue
            cr = np.asarray(Image.fromarray(sub).resize((W6.CW, W6.CH), Image.BILINEAR))
            crops[0, c] = torch.from_numpy(cr).permute(2, 0, 1).float().div_(255).to(device)
            box[0, c] = torch.tensor([x1 / W7.IW, y1 / W7.IH, x2 / W7.IW, y2 / W7.IH],
                                     device=device)
            mask[0, c] = True
        role = torch.zeros(1, N, dtype=torch.long, device=device)
        name = torch.zeros(1, N, dtype=torch.long, device=device)
        typ = torch.tensor([W5.TTYPE.get(s["type"], len(W5.TTYPE))], device=device)
        tnm = torch.tensor([W5.nid(s["tname"])], device=device)
        ctx = torch.tensor([s["ctx"]], dtype=torch.float, device=device)
        logits = sel(role, name, box, mask, typ, tnm, ctx, crops)[0]
        pick = int(logits.argmax())
        if mask[0, pick] and center_in(dets[pick], s["boxes"][s["target"]]):
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
                S += collect(pg, inst)
            data[k] = S
            print(f"  {k}: {len(S)} samples")
        b.close()

    tr = data["in_domain"]
    cut = int(len(tr) * 0.8)
    trS, valS = tr[:cut], tr[cut:]
    det = train_detector(trS, device)
    sel = train_selector(trS, device)

    res = {}
    for k, v in data.items():
        res[k] = evaluate(det, sel, v, device)
        print(f"  [{k}] end-to-end hit = {res[k]:.3f}")

    print(f"\n对照：结构通道(uia) ≈ 1.000；给定框像素选择(W6) in-domain = 1.000；"
          f"W7 纯检测 recall in-domain = 0.307")
    print(f"端到端 theme drop={res['in_domain']-res['ood_theme']:+.3f}  "
          f"app drop={res['in_domain']-res['ood_app']:+.3f}")
    ok = res["in_domain"] < 0.6
    print(f"\nW8 判据：端到端命中率明显低于结构通道 -> {ok}")
    print(f"W8 {'通过' if ok else '未通过'}")
    (OUT / "winsim_w8_results.json").write_text(json.dumps(
        {"end_to_end_hit": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w8_results.json")


if __name__ == "__main__":
    main()
