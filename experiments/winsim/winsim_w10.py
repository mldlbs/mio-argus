"""
WinSim W10：检测+选择联合 —— 消除训练/推理分布不匹配

W8/W9 的选择器用 **oracle 框**训练，推理时却面对**检测框**（分布不匹配）。
W10：先用检测器产出候选，再把选择器**直接训在检测框上**（两阶段标准做法），
      必要时再联合微调一轮。

对照：
  W9：oracle框选择器 + stride2 检测器 → in e2e 0.290
  W10：检测框选择器 + 同一检测器
判据：若明显抬手 → W9 的 0.29 部分是分布不匹配（实现问题）；
      若不动 → 方法极限更硬。

用法:
    python experiments/winsim/winsim_w10.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from playwright.sync_api import sync_playwright

import winsim_w5 as W5
import winsim_w6 as W6
import winsim_w7 as W7
import winsim_w8 as W8
import winsim_w9 as W9

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW
IW, IH = W7.IW, W7.IH


def crops_of(img, boxes):
    out = np.zeros((len(boxes), W6.CH, W6.CW, 3), np.uint8)
    for c, (x1, y1, x2, y2) in enumerate(boxes):
        sub = img[int(max(0, y1)):int(min(IH, y2)), int(max(0, x1)):int(min(IW, x2))]
        if sub.size == 0:
            continue
        out[c] = np.asarray(Image.fromarray(sub).resize((W6.CW, W6.CH), Image.BILINEAR))
    return out


def build_proposals(det, samples, device):
    """用冻结检测器产出候选；只保留"含目标中心"的样本，正例=该框。"""
    det.eval()
    props = []
    for s in samples:
        dets = W9.decode(det, s["img"], 2, device)
        if not dets:
            continue
        ti = -1
        for j, b in enumerate(dets):
            if W8.center_in(b, s["boxes"][s["target"]]):
                ti = j
                break
        if ti < 0:
            continue
        props.append(dict(crops=crops_of(s["img"], dets), boxes=np.array(dets, np.float32),
                          type=s["type"], tname=s["tname"], ctx=s["ctx"], target=ti))
    return props


def train_selector_props(props, device, epochs=40):
    torch.manual_seed(0)
    m = W6.PixelSel().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    N = W5.MAXC
    B = 32
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(props))
        for i in range(0, len(props), B):
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
            ok = True
            for r, j in enumerate(idx):
                p = props[j]
                n = min(N, len(p["boxes"]))
                if p["target"] >= n:          # 目标框被 top-k 截断
                    ok = False
                    break
                for c in range(n):
                    box[r, c] = torch.tensor(p["boxes"][c] / [IW, IH, IW, IH])
                    mask[r, c] = True
                    crops[r, c] = torch.from_numpy(p["crops"][c]).permute(2, 0, 1).float() / 255.0
                typ[r] = W5.TTYPE.get(p["type"], len(W5.TTYPE))
                tnm[r] = W5.nid(p["tname"])
                ctx[r] = torch.tensor(p["ctx"], dtype=torch.float)
                tgt[r] = p["target"]
            if not ok:
                continue
            loss = F.cross_entropy(m(role, name, box, mask, typ, tnm, ctx, crops), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


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
    selection_rate = 0.0

    print("训练检测器（stride2 + 修尺寸监督）...")
    det = W9.train(W9.Det2(w=32), 2, IH // 2, IW // 2, 3.0, trS, device)

    print("构建检测框候选（冻结检测器）...")
    props = build_proposals(det, trS, device)
    selection_rate = len(props) / len(trS)
    print(f"  可用样本 {len(props)}/{len(trS)}（目标被检测到的比例）")

    print("训练选择器（在检测框上）...")
    sel_props = train_selector_props(props, device)
    print("对照：训练选择器（在 oracle 框上）...")
    sel_oracle = W8.train_selector(trS, device)

    rows = {}
    print(f"\n{'split':<12}{'oracle-box sel':>17}{'detector-box sel':>18}")
    print("-" * 48)
    for k, v in data.items():
        e_o = W9.e2e(det, 2, sel_oracle, v, device)
        e_p = W9.e2e(det, 2, sel_props, v, device)
        rows[k] = dict(oracle_box_sel=e_o, detector_box_sel=e_p)
        print(f"{k:<12}{e_o:>17.3f}{e_p:>18.3f}")
    print("-" * 48)
    imp = rows["in_domain"]["detector_box_sel"] - rows["in_domain"]["oracle_box_sel"]
    print(f"in-domain 提升 = {imp:+.3f}")
    verdict = ("分布不匹配是主因（联合后明显抬手）" if imp > 0.08
               else "方法极限更硬（联合也抬不动）")
    print(f"W10 判断：{verdict}")

    (OUT / "winsim_w10_results.json").write_text(json.dumps(
        {"rows": rows, "proposal_selection_rate": selection_rate,
         "improvement_in_domain": imp, "verdict": verdict},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w10_results.json")


if __name__ == "__main__":
    main()
