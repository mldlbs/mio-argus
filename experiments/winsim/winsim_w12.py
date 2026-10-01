"""
WinSim W12：混合臂 —— 结构可用时用结构，不可用时回退像素

新增 canvas 应用：控件画在 <canvas> 上 → **不进 elements()**，只能靠像素。
任务族：DOM（move/delete/toggle）与 canvas（点画布上的目标色块）。

三条臂，同一批样本：
  struct-only ：只用 elements()（canvas 步没有候选 → 必失手）
  pixel-only  ：检测器 + 选择器（全像素端到端）
  hybrid      ：目标名出现在 elements() → 走 struct；否则 → 走 pixel
指标：next-action 命中率（按族分解 + 总体）

用法:
    python experiments/winsim/winsim_w12.py
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
import winsim_w10 as W10

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW
IW, IH = W7.IW, W7.IH
MAXC = W5.MAXC
SX, SY = IW / 900, IH / 600
ROLES = W5.ROLES


def make_instances(family, n, rng):
    out = []
    for i in range(n):
        theme = str(rng.choice(list("abc")))
        if family == "dom":
            t = str(rng.choice(["move", "delete", "toggle"]))
            f = (str(rng.choice(["notes.md", "report.txt"])) if t in ("move", "delete")
                 else str(rng.choice(["Wi-Fi", "Bluetooth"])))
            out.append(dict(family="dom", type=t, tname=f, theme=theme, seed=1000 + i))
        else:
            c = str(rng.choice(["red", "green", "blue"]))
            out.append(dict(family="canvas", type="canvas", tname=c,
                            theme=theme, seed=1000 + i))
    return out


def seq_for(inst):
    f, t = inst["tname"], inst["type"]
    if inst["family"] == "canvas":
        return [("click", "icon", "Canvas"), ("click", "canvas", f)]
    if t == "move":
        return [("click", "icon", "Explorer"), ("click", "file", f),
                ("click", "button", "Cut"), ("click", "dir", "Documents"),
                ("click", "button", "Paste")]
    if t == "delete":
        return [("click", "icon", "Explorer"), ("click", "file", f),
                ("click", "button", "Delete")]
    return [("click", "icon", "Settings"), ("click", "switch", f)]


def find(els, role, name):
    for i, e in enumerate(els):
        if e["role"] == role and e["name"] == name:
            return i
    return -1


def center_in(box, t):
    return (t[0] <= (box[0] + box[2]) / 2 <= t[2]
            and t[1] <= (box[1] + box[3]) / 2 <= t[3])


def collect(pg, inst):
    import io as _io
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    out = []
    for op, role, name in seq_for(inst):
        els = pg.evaluate("WinSim.elements()")
        st = pg.evaluate("WinSim.truth()")
        if role == "canvas":                      # 目标在 canvas 上：不是元素
            cvi = next((i for i, e in enumerate(els)
                        if e["role"] == "canvas-surface"), -1)
            if cvi < 0:
                break
            cv = els[cvi]
            shape = next((s for s in (st["canvas"] or []) if s["name"] == name), None)
            if shape is None:
                break
            tb = [cv["x"] + shape["x"], cv["y"] + shape["y"],
                  cv["x"] + shape["x"] + shape["w"], cv["y"] + shape["y"] + shape["h"]]
            els_target = -1
            click_xy = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        else:
            ti = find(els, role, name)
            if ti < 0:
                break
            e0 = els[ti]
            tb = [e0["x"], e0["y"], e0["x"] + e0["w"], e0["y"] + e0["h"]]
            els_target = ti
            click_xy = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)

        img = np.asarray(Image.fromarray(np.asarray(
            Image.open(_io.BytesIO(pg.screenshot())).convert("RGB"))
        ).resize((IW, IH), Image.BILINEAR))
        db = [[e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]] for e in els]
        if st["canvas"]:                          # 形状是画布局部坐标 → 加画布偏移
            cv = els[next(i for i, e in enumerate(els)
                          if e["role"] == "canvas-surface")]
            db += [[cv["x"] + s["x"], cv["y"] + s["y"],
                    cv["x"] + s["x"] + s["w"], cv["y"] + s["y"] + s["h"]]
                   for s in st["canvas"]]
        db = np.array(db, np.float32) * [SX, SY, SX, SY]
        tb_s = np.array(tb, np.float32) * [SX, SY, SX, SY]
        tbi = next((j for j, b in enumerate(db) if center_in(b, tb_s)), -1)
        act = 4
        for w in st["windows"]:
            if w["id"] == st["active"]:
                act = W5.APPID.get(w["appId"], 4)
        ctx = [act, len(st["path"]), int(bool(st["selected"])),
               int(bool(st["clipboard"])), int(bool(st["renaming"]))]
        out.append(dict(img=img, det_boxes=db, det_target=tbi, els=els,
                        els_target=els_target, tname=inst["tname"], type=inst["type"],
                        family=inst["family"], theme=inst["theme"], ctx=ctx,
                        step_name=name, tbox=np.array(tb, np.float32)))
        pg.evaluate(f"WinSim.click({click_xy[0]},{click_xy[1]})")
    return out


def train_struct(samples, device, epochs=30):
    """W5.Sel：候选=elements()，目标=oracle 点击的元素。"""
    torch.manual_seed(0)
    m = W5.Sel().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    B = 64
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(samples))
        for i in range(0, len(samples), B):
            idx = perm[i:i + B]
            Bn = len(idx)
            role = torch.zeros(Bn, MAXC, dtype=torch.long, device=device)
            name = torch.zeros(Bn, MAXC, dtype=torch.long, device=device)
            box = torch.zeros(Bn, MAXC, 4, device=device)
            mask = torch.zeros(Bn, MAXC, dtype=torch.bool, device=device)
            typ = torch.zeros(Bn, dtype=torch.long, device=device)
            tnm = torch.zeros(Bn, dtype=torch.long, device=device)
            ctx = torch.zeros(Bn, 4, device=device)
            tgt = torch.zeros(Bn, dtype=torch.long, device=device)
            for r, j in enumerate(idx):
                s = samples[j]
                for c, e in enumerate(s["els"][:MAXC]):
                    role[r, c] = ROLES.get(e["role"], 0)
                    name[r, c] = W5.nid(e["name"])
                    box[r, c] = torch.tensor([e["x"] / 900, e["y"] / 600,
                                              e["w"] / 900, e["h"] / 600])
                    mask[r, c] = True
                typ[r] = W5.TTYPE.get(s["type"], len(W5.TTYPE))
                tnm[r] = W5.nid(s["tname"])
                ctx[r] = torch.tensor(s["ctx"][:4], dtype=torch.float)
                tgt[r] = s["els_target"] if s["els_target"] < MAXC else 0
            loss = F.cross_entropy(m(role, name, box, mask, typ, tnm, ctx), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def struct_pred(m, s, device):
    role = torch.zeros(1, MAXC, dtype=torch.long, device=device)
    name = torch.zeros(1, MAXC, dtype=torch.long, device=device)
    box = torch.zeros(1, MAXC, 4, device=device)
    mask = torch.zeros(1, MAXC, dtype=torch.bool, device=device)
    for c, e in enumerate(s["els"][:MAXC]):
        role[0, c] = ROLES.get(e["role"], 0)
        name[0, c] = W5.nid(e["name"])
        box[0, c] = torch.tensor([e["x"] / 900, e["y"] / 600,
                                  e["w"] / 900, e["h"] / 600])
        mask[0, c] = True
    typ = torch.tensor([W5.TTYPE.get(s["type"], len(W5.TTYPE))], device=device)
    tnm = torch.tensor([W5.nid(s["tname"])], device=device)
    ctx = torch.tensor([s["ctx"][:4]], dtype=torch.float, device=device)
    j = int(m(role, name, box, mask, typ, tnm, ctx)[0].argmax())
    if j >= len(s["els"]):
        return None
    e = s["els"][j]
    return [e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    insts = {"dom": make_instances("dom", 130, rng),
             "canvas": make_instances("canvas", 70, rng)}
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

    dom_tr = data["dom"][:int(len(data["dom"]) * 0.8)]
    mixed_tr = dom_tr + data["canvas"][:int(len(data["canvas"]) * 0.8)]
    mixed_te = (data["dom"][int(len(data["dom"]) * 0.8):]
                + data["canvas"][int(len(data["canvas"]) * 0.8):])

    print("训练 struct（训在将被路由到 struct 的步上）/ detector / pixel 选择器...")
    struct_tr = [s for s in mixed_tr
                 if s["step_name"] in [e["name"] for e in s["els"]]]
    m_struct = train_struct(struct_tr, device)
    det_tr = [{"img": s["img"], "boxes": s["det_boxes"]} for s in mixed_tr]
    det = W9.train(W9.Det2(w=32), 2, IH // 2, IW // 2, 3.0, det_tr, device)
    prop_in = [{"img": s["img"], "boxes": s["det_boxes"], "target": s["det_target"],
                "type": s["type"], "tname": s["tname"], "ctx": s["ctx"][:4]}
               for s in mixed_tr if s["det_target"] >= 0]
    props = W10.build_proposals(det, prop_in, device)
    m_pixel = W10.train_selector_props(props, device)

    rows = {"struct": {}, "pixel": {}, "hybrid": {}}
    for fam in ("dom", "canvas"):
        ss = [s for s in mixed_te if s["family"] == fam]
        for arm in rows:
            hit = 0
            for s in ss:
                tb = s["tbox"]
                if arm == "struct":
                    ok = False
                    pb = struct_pred(m_struct, s, device)
                elif arm == "pixel":
                    pb = pixel_pred(det, m_pixel, s, device)
                else:
                    if s["step_name"] in [e["name"] for e in s["els"]]:
                        pb = struct_pred(m_struct, s, device)
                    else:
                        pb = pixel_pred(det, m_pixel, s, device)
                if pb is not None and center_in(pb, tb):
                    hit += 1
            rows[arm][fam] = hit / max(1, len(ss))

    print(f"\n{'arm':<9}{'dom':>8}{'canvas':>9}{'overall':>9}")
    print("-" * 35)
    for arm, r in rows.items():
        overall = np.mean(list(r.values()))
        print(f"{arm:<9}{r['dom']:>8.3f}{r['canvas']:>9.3f}{overall:>9.3f}")
    print("-" * 35)
    print("说明：canvas 控件不进 elements() → struct 无候选（必失手）；"
          "hybrid 按'目标名是否在 elements() 里'路由")

    (OUT / "winsim_w12_results.json").write_text(json.dumps(
        {k: v for k, v in rows.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 winsim_w12_results.json")


@torch.no_grad()
def pixel_pred(det, sel, s, device):
    dets = W9.decode(det, s["img"], 2, device)
    if not dets:
        return None
    N = MAXC
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
    ctx = torch.tensor([s["ctx"][:4]], dtype=torch.float, device=device)
    pick = int(sel(role, name, box, mask, typ, tnm, ctx, crops)[0].argmax())
    if not mask[0, pick]:
        return None
    bx = dets[pick]
    return [bx[0] / SX, bx[1] / SY, bx[2] / SX, bx[3] / SY]


if __name__ == "__main__":
    main()
