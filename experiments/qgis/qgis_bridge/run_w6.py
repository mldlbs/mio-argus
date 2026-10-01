"""
QGIS W6：把 QGIS "训练好" —— 状态生成 + 三条臂训练 + 两种 held-out

状态轴：窗口尺寸(最大化/1200x800) × Python 控制台(关/开) × 表达式对话框(关/开)
        × 随机 canvas 视图 × 随机图层可见性
样本   ：(整窗截图, 候选控件列表, 指令=目标控件名) → 目标下标
三条臂 ：struct（控件名文本 + bbox）/ pixel（候选裁剪 → CNN）/ hybrid
held-out：① 未见**控件名** ② 未见**UI 状态**

用法:
    python experiments/qgis/qgis_bridge/run_w6.py                 # 复用 w6_data.npz 只训练
    python experiments/qgis/qgis_bridge/run_w6.py --recollect     # 重新采集（需 QGIS 在跑）
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

B = 1024
MAXC = 16
CH, CW = 32, 48
DATA = HERE / "w6_data.npz"


def bag(s, out):
    t = (s or "").lower().strip()
    if not t:
        return out
    if len(t) < 3:
        out[zlib.crc32(t.encode()) % B] += 1
        return out
    for i in range(len(t) - 2):
        out[zlib.crc32(t[i:i+3].encode()) % B] += 1
    return out


def truth():
    return call({"op": "truth"})["truth"]


def controls():
    st = call({"op": "structure"})
    out = []
    for a in st["actions"]:
        r = a.get("rect") or [0, 0, 0, 0]
        if a["text"] and r[2] >= 8 and r[3] >= 8:
            out.append(dict(cls="QAction", name=a["text"], rect=list(r)))
    for w in st["widgets"]:
        r = w.get("rect") or [0, 0, 0, 0]
        if w["text"] and r[2] >= 8 and r[3] >= 8:
            out.append(dict(cls=w["class"], name=w["text"], rect=list(r)))
    best = {}
    for c in out:
        k = (c["cls"], c["name"])
        if k not in best or c["rect"][2]*c["rect"][3] > best[k]["rect"][2]*best[k]["rect"][3]:
            best[k] = c
    return list(best.values())


STATES = [dict(size=s, console=c, dialog=d)
          for s in ("max", "1200x800") for c in (False, True) for d in (False, True)]


def apply_state(st):
    call({"op": "win", "maximize": st["size"] == "max"})
    if st["size"] != "max":
        call({"op": "win", "w": 1200, "h": 800})
    t = truth()
    if bool(t.get("console_visible")) != st["console"]:
        call({"op": "act", "kind": "action", "text": "Python Console"})
    if st["dialog"]:
        call({"op": "activate_layer", "name": "cities"})
        call({"op": "act", "kind": "action", "text": "Select Features by Expression…"})
    else:
        call({"op": "close_dialogs"})
    time.sleep(0.25)


def collect_all(n_per_state, rng):
    rows = []
    for si, st in enumerate(STATES):
        call({"op": "reset"})
        apply_state(st)
        win = call({"op": "win"})
        ox, oy = win["rect"][0], win["rect"][1]
        for k in range(n_per_state):
            t = truth()
            e = t["layers"][0]["extent"]
            fx = float(rng.uniform(0.2, 0.8)); fy = float(rng.uniform(0.2, 0.8))
            call({"op": "set_extent", "box": [e[0], e[1],
                  e[0] + (e[2]-e[0])*fx, e[1] + (e[3]-e[1])*fy]})
            if rng.random() < 0.3:
                call({"op": "layer_tree_click", "name": "cities"})
            shot = call({"op": "shot", "path": str(HERE / "_w6_tmp.png")})
            img = np.asarray(Image.open(shot["path"]).convert("RGB"))
            cs = controls()
            if len(cs) < MAXC + 1:
                continue
            ti = int(rng.integers(0, len(cs)))
            others = [i for i in range(len(cs)) if i != ti]
            rng.shuffle(others)
            idxs = [ti] + others[:MAXC-1]
            rng.shuffle(idxs)
            tgt = idxs.index(ti)
            crops = np.zeros((MAXC, CH, CW, 3), np.uint8)
            names, boxes = [], []
            for j, ci in enumerate(idxs):
                c = cs[ci]
                x, y, w, h = c["rect"]
                x0, y0 = max(0, x-ox), max(0, y-oy)
                x1 = min(img.shape[1], x-ox+w); y1 = min(img.shape[0], y-oy+h)
                if x1-x0 >= 4 and y1-y0 >= 4:
                    crops[j] = np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((CW, CH)))
                names.append(c["name"])
                boxes.append([x/2560, y/1440, w/2560, h/1440])
            rows.append(dict(crops=crops, names=names, boxes=boxes,
                             target=tgt, instr=cs[ti]["name"], state=si))
            if (k+1) % 20 == 0:
                print(f"  state {si} ({st['size']}, console={st['console']}, "
                      f"dialog={st['dialog']}): {k+1}/{n_per_state}", flush=True)
        call({"op": "close_dialogs"})
    return rows


def featurize(samples):
    n = len(samples)
    txt = torch.zeros(n, MAXC, B)
    box = torch.zeros(n, MAXC, 4)
    mask = torch.zeros(n, MAXC, dtype=torch.bool)
    instr = torch.zeros(n, B)
    tgt = torch.zeros(n, dtype=torch.long)
    crops = torch.zeros(n, MAXC, 3, CH, CW)
    for i, s in enumerate(samples):
        for j in range(MAXC):
            bag(s["names"][j], txt[i, j].numpy())
            box[i, j] = torch.tensor(s["boxes"][j])
            mask[i, j] = True
            crops[i, j] = torch.from_numpy(s["crops"][j]).permute(2, 0, 1).float()/255.0
        bag(s["instr"], instr[i].numpy())
        tgt[i] = s["target"]
    return txt, box, mask, instr, crops, tgt


class Struct(nn.Module):
    def __init__(s, d=96):
        super().__init__()
        s.tx = nn.Sequential(nn.Linear(B, d), nn.ReLU(), nn.Linear(d, d))
        s.bx = nn.Linear(4, d)
        s.d = d

    def forward(s, txt, box, mask, instr, crops):
        c = s.tx(txt) + s.bx(box)
        q = s.tx(instr)
        return ((c*q[:, None, :]).sum(-1)/s.d**0.5).masked_fill(~mask, -1e9)


class Pixel(nn.Module):
    def __init__(s, d=96, dc=64):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                              nn.Conv2d(16, dc, 3, 2, 1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten())
        s.proj = nn.Linear(dc, d)
        s.tx = nn.Sequential(nn.Linear(B, d), nn.ReLU(), nn.Linear(d, d))
        s.d = d

    def forward(s, txt, box, mask, instr, crops):
        n, c = crops.shape[0], crops.shape[1]
        H, W = crops.shape[3], crops.shape[4]
        f = s.proj(s.cnn(crops.reshape(n*c, 3, H, W))).reshape(n, c, -1)
        q = s.tx(instr)
        return ((f*q[:, None, :]).sum(-1)/s.d**0.5).masked_fill(~mask, -1e9)


class Hybrid(nn.Module):
    """结构可用（候选有名字）→ 用 struct 分支；否则用 pixel 分支。"""
    def __init__(s, d=96):
        super().__init__()
        s.st = Struct(d); s.px = Pixel(d)

    def forward(s, txt, box, mask, instr, crops):
        return s.st(txt, box, mask, instr, crops) + s.px(txt, box, mask, instr, crops)


def run_training(rows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names_all = sorted({s["instr"] for s in rows})
    cut = int(len(names_all) * 0.75)
    seen_names = set(names_all[:cut])
    seen_states = set(range(len(STATES) // 2))
    tr = [s for s in rows if s["instr"] in seen_names and s["state"] in seen_states]
    te_id = [s for s in rows if s["instr"] in seen_names and s["state"] not in seen_states]
    te_name = [s for s in rows if s["instr"] not in seen_names]
    print(f"train={len(tr)}  held-out 状态={len(te_id)}  held-out 控件名={len(te_name)}")

    def train(model, samples, epochs=30):
        T = tuple(t.to(device) for t in featurize(samples))
        m = model.to(device)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(epochs):
            p = torch.randperm(len(samples), device=device)
            for i in range(0, len(samples), 32):
                idx = p[i:i+32]
                loss = F.cross_entropy(m(*[t[idx] for t in T[:5]]), T[5][idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return m

    @torch.no_grad()
    def acc(m, samples):
        if not samples:
            return float("nan")
        T = tuple(t.to(device) for t in featurize(samples))
        m.eval()
        return float((m(*T[:5]).argmax(-1) == T[5]).float().mean())

    res = {}
    print(f"\n{'arm':<8}{'in-domain':>11}{'heldout_state':>15}{'heldout_name':>14}")
    print("-" * 48)
    for arm, Model in (("struct", Struct), ("pixel", Pixel), ("hybrid", Hybrid)):
        m = train(Model(), tr)
        res[arm] = dict(in_domain=acc(m, tr), heldout_state=acc(m, te_id),
                        heldout_name=acc(m, te_name))
        print(f"{arm:<8}{res[arm]['in_domain']:>11.3f}"
              f"{res[arm]['heldout_state']:>15.3f}{res[arm]['heldout_name']:>14.3f}")
    print("-" * 48)
    print(f"chance = 1/{MAXC} = {1/MAXC:.3f}")

    (HERE / "w6_results.json").write_text(json.dumps(
        {"n": len(rows), "states": len(STATES),
         "splits": dict(train=len(tr), heldout_state=len(te_id), heldout_name=len(te_name)),
         "chance": 1/MAXC, "results": res}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w6_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-state", type=int, default=60)
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    if DATA.exists() and not args.recollect:
        rows = list(np.load(DATA, allow_pickle=True)["rows"])
        print(f"复用 {DATA.name}: {len(rows)} 样本")
    else:
        rows = collect_all(args.n_per_state, rng)
        np.savez_compressed(DATA, rows=np.array(rows, dtype=object), allow_pickle=True)
        print(f"样本总数 = {len(rows)}；已写出 {DATA}")
    run_training(rows)


if __name__ == "__main__":
    main()
