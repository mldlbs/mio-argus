"""
QGIS W12：把 actor 换成真模型 —— V 的价值取决于 actor 的通道

W9–W11 的 actor 是注入噪声的硬币（每步 q=0.5），W11 推出的分水岭是 q*≈0.53。
本轮训练**真的 actor**（在真 UI 上从真实候选按钮里挑），测它的每步 q_hat，
再判：对这样的 actor，V（W10/W11 的截图+重试）到底划不划算。

三步骤（都是真工具栏动作，真值可判；指令是任务级短语，不是动作名）：
  1) "把地图缩放到当前选中要素" → 正确 = Zoom to Selection
  2) "把地图缩放到图层的完整范围" → 正确 = Zoom Full
  3) "打开 Python 控制台"        → 正确 = Python Console
每步候选 = 1 个正确 + 15 个从**真 UI 里现抽的**干扰项（真执行，错就真的错）。

两条通道的 actor（沿用 W6 结构）：
  struct = 候选的（动作名 bag-of-trigrams + bbox）；pixel = 候选裁剪 → CNN
评测：in-domain / **未见 UI 状态**（未见的窗口尺寸×对话框组合）→ q_hat
闭环：live 感知（截图 + structure）→ 挑 → 真执行；no-V vs full-V（V 用 actor 自己的
错误分布训练，含截图成本），指标 = 每步成功率与正确步/分钟。

注意：真值按**窗口尺寸**分别标定——"缩放到选择"的 extent 依赖画布宽高比。

用法: python experiments/qgis/qgis_bridge/run_w12.py
"""
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
SCENE = "regions"
DATA = HERE / "w12_actor_data.npz"
VMODEL = HERE / "w12_v.pt"
N_PER_STATE = 24
MAX_RETRY = 3
BUDGET = 150.0
SIZES = ("max", "1000x750")

STEPS = [
    dict(instr="把地图缩放到当前选中要素", correct="Zoom to Selection", kind="zoom"),
    dict(instr="把地图缩放到图层的完整范围", correct="Zoom Full", kind="zoom"),
    dict(instr="打开 Python 控制台", correct="Python Console", kind="console"),
]
EXCLUDE = {"Zoom to Layer(s)", "Pan Map to Selection"}
SEL_EXPR = '"area" > 400'      # 只选 North（单个小区域）：与"全范围"视野差足够大
STATES = [(s, d) for s in SIZES for d in (False, True)]

REF = {"FULL_X0": {}, "SEL_X0": {}, "FULL_BOX": None}


def truth():
    return call({"op": "truth"})["truth"]


def extent_x0():
    return truth()["extent"][0]


def console_visible():
    return bool(truth().get("console_visible"))


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


SAFE_POOL = [
    "Zoom In", "Zoom Out", "Zoom Full", "Zoom to Selection", "Zoom Last", "Zoom Next",
    "Zoom to Native Resolution (100%)", "Pan Map", "Pan Map to Selection",
    "Identify Features", "Show Map Tips", "Python Console", "Toggle Editing",
    "Refresh", "Undo", "Redo", "Expand All", "Collapse All",
    "Show/Hide Labels and Diagrams", "Show Unplaced Labels",
    "Temporal Controller", "Add Polygon Feature", "Save Layer Edits", "Current Edits",
    "Copy Features", "Cut Features", "Paste Features", "Delete Selected",
]


def pool():
    """候选 = 真 UI 里存在且 enabled 的**安全动作集**（不会弹模态框）。
    这是环境设计选择：任务定义在不含模态对话框的工具栏子集上。"""
    st = call({"op": "structure"})
    have = {}
    for a in st["actions"]:
        t = (a.get("text") or "").replace("&", "").strip()
        r = a.get("rect") or [0, 0, 0, 0]
        if t in SAFE_POOL and r[2] >= 8 and r[3] >= 8:
            have[t] = dict(name=t, rect=list(r))
    return [have[t] for t in SAFE_POOL if t in have]


def shot_img():
    p = HERE / "_w12_tmp.png"
    call({"op": "shot", "path": str(p)})
    return np.asarray(Image.open(p).convert("RGB"))


def crop_of(img, rect):
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1 = min(img.shape[1], x+w); y1 = min(img.shape[0], y+h)
    if x1-x0 < 4 or y1-y0 < 4:
        return np.zeros((CH, CW, 3), np.uint8)
    return np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((CW, CH)))


def set_size(size):
    if size == "max":
        call({"op": "win", "maximize": True})
    else:
        w, h = size.split("x")
        call({"op": "win", "w": int(w), "h": int(h)})


def setup_state(st, rng, random_extent=True):
    size, dialog = st
    set_size(size)
    call({"op": "reset", "scene": SCENE})
    if console_visible():                      # 控制台会改变画布尺寸 -> 必须关
        call({"op": "act", "kind": "action", "text": "Python Console"})
    call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
    time.sleep(0.2)
    if random_extent:
        e = REF["FULL_BOX"]
        fx = float(rng.uniform(0.3, 0.9)); fy = float(rng.uniform(0.3, 0.9))
        call({"op": "set_extent", "box": [e[0], e[1],
              e[0] + (e[2]-e[0])*fx, e[1] + (e[3]-e[1])*fy]})
    if dialog:
        call({"op": "activate_layer", "name": SCENE})
        call({"op": "act", "kind": "action", "text": "Select Features by Expression…"})
    else:
        call({"op": "close_dialogs"})
    time.sleep(0.2)


def build_candidates(rng, correct, img):
    cs = pool()
    names = [c["name"] for c in cs]
    if correct not in names:
        return None
    ti = names.index(correct)
    others = [i for i in range(len(cs)) if i != ti]
    rng.shuffle(others)
    idxs = [ti] + others[:MAXC-1]
    rng.shuffle(idxs)
    tgt = idxs.index(ti)
    crops = np.zeros((MAXC, CH, CW, 3), np.uint8)
    names_out, boxes = [], []
    for j, ci in enumerate(idxs):
        c = cs[ci]
        crops[j] = crop_of(img, c["rect"])
        names_out.append(c["name"])
        boxes.append([c["rect"][0]/2560, c["rect"][1]/1440,
                      c["rect"][2]/2560, c["rect"][3]/1440])
    return dict(crops=crops, names=names_out, boxes=boxes, target=tgt)


def _priv_upto(k):
    for j in range(k + 1):
        call({"op": "act", "kind": "action", "text": STEPS[j]["correct"]})
        time.sleep(0.15)


def collect_actor(n_per_state, rng):
    rows = []
    for si, st in enumerate(STATES):
        for step_i, step in enumerate(STEPS):
            for k in range(n_per_state):
                setup_state(st, rng)
                if step_i > 0:
                    _priv_upto(step_i - 1)
                    time.sleep(0.15)
                img = shot_img()
                cand = build_candidates(rng, step["correct"], img)
                if cand is None:
                    print("  !! correct action missing from pool:", step["correct"])
                    continue
                cand.update(instr=step["instr"], state=si, step=step_i)
                rows.append(cand)
            print(f"  state{si} step{step_i} ({st[0]}, dialog={st[1]}): {n_per_state} ok", flush=True)
    return rows


def featurize(rows):
    n = len(rows)
    txt = torch.zeros(n, MAXC, B); box = torch.zeros(n, MAXC, 4)
    mask = torch.zeros(n, MAXC, dtype=torch.bool)
    instr = torch.zeros(n, B); tgt = torch.zeros(n, dtype=torch.long)
    crops = torch.zeros(n, MAXC, 3, CH, CW)
    for i, s in enumerate(rows):
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
        s.t = nn.Sequential(nn.Linear(B, 256), nn.ReLU(), nn.Linear(256, d))
        s.b = nn.Sequential(nn.Linear(4, d), nn.ReLU())
        s.i = nn.Sequential(nn.Linear(B, 256), nn.ReLU(), nn.Linear(256, d))
        s.s = nn.Bilinear(d, d, 1); s.si = nn.Bilinear(d, d, 1)

    def forward(s, txt, box, mask, instr, crops):
        t = s.t(txt); b = s.b(box); ii = s.i(instr)
        sc = s.s(t, b).squeeze(-1) + s.si(t, ii[:, None, :].expand_as(t)).squeeze(-1)
        return sc.masked_fill(~mask, -1e9)


class Pixel(nn.Module):
    def __init__(s, d=96):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, d))
        s.i = nn.Sequential(nn.Linear(B, 256), nn.ReLU(), nn.Linear(256, d))
        s.s = nn.Bilinear(d, d, 1)

    def forward(s, txt, box, mask, instr, crops):
        n, m = crops.shape[0], crops.shape[1]
        c = s.c(crops.view(n*m, 3, CH, CW)).view(n, m, -1)
        ii = s.i(instr)
        sc = s.s(c, ii[:, None, :].expand_as(c)).squeeze(-1)
        return sc.masked_fill(~mask, -1e9)


def eval_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(*X).argmax(-1)
    return float((pred == y).float().mean())


def train_arm(cls, X, y, epochs=40):
    n = len(y)
    torch.manual_seed(0)
    m = cls()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i+32]
            loss = F.cross_entropy(m(*[x[idx] for x in X]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


class VNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(6, 16, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
    def forward(s, x):
        return s.f(x)


def vis_step(img):
    return np.asarray(Image.fromarray(img).resize((256, 128)))


def train_v(pairs, epochs=40):
    data = [np.concatenate([b, a], -1) for b, a, _ in pairs]
    y = np.array([lab for _, _, lab in pairs])
    X = torch.tensor(np.stack(data)).permute(0, 3, 1, 2).float()/255.0
    yt = torch.tensor(y)
    perm = torch.randperm(len(y)); cut = int(len(y)*0.75)
    tr = perm[:cut]
    torch.manual_seed(0)
    m = VNet(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 32):
            idx = p[i:i+32]
            loss = F.cross_entropy(m(X[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = F.softmax(m(X), -1)[:, 1].numpy()
    order = np.argsort(prob); ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob)+1)
    pos, neg = y == 1, y == 0
    auc = float((ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum())) \
        if pos.sum() and neg.sum() else float("nan")
    print(f"  V：AUC={auc:.3f} (n={len(y)}, pos={int(pos.sum())})")
    return m


def pick(model, cand, arm):
    row = dict(names=cand["names"], boxes=cand["boxes"], crops=cand["crops"],
               instr=cand["instr"], target=cand["target"])
    X = featurize([row])[:5]
    model.eval()
    with torch.no_grad():
        return int(model(*X).argmax(-1)[0])


def execute(name):
    r = call({"op": "act", "kind": "action", "text": name})
    return r.get("ok", False)


def step_truth(step, size):
    """真值 = 当前 extent 既在正确参照的容差内，又比错误参照更近。
    （单独用"更近"会被随机预置的 extent 骗到：随机子范围天然更接近全范围。）"""
    if step["kind"] == "zoom":
        x0 = extent_x0()
        sel, full = REF["SEL_X0"][size], REF["FULL_X0"][size]
        if step["correct"] == "Zoom to Selection":
            target, other = sel, full
        else:
            target, other = full, sel
        return abs(x0 - target) < 1.5 and abs(x0 - target) < abs(x0 - other)
    return console_visible()


def live_episode(actor, arm, mode, rng, vnets):
    size, dialog = STATES[int(rng.integers(0, len(STATES)))]
    setup_state((size, dialog), rng)
    ok_steps = []
    for step in STEPS:
        got = False
        for _ in range(1 + (MAX_RETRY if mode == "v" else 0)):
            img = shot_img()
            cand = build_candidates(rng, step["correct"], img)
            if cand is None:
                break
            cand["instr"] = step["instr"]
            chosen = cand["names"][pick(actor, cand, arm)]
            before = vis_step(img)
            execute(chosen)
            time.sleep(0.25)
            after = vis_step(shot_img())
            if mode == "nov":
                got = step_truth(step, size)
                break
            if step["kind"] == "zoom":
                m, thr = vnets[step["correct"]]
                x = torch.tensor(np.concatenate([before, after], -1))[None]
                x = x.permute(0, 3, 1, 2).float()/255.0
                with torch.no_grad():
                    p = float(F.softmax(m(x), -1)[0, 1])
                if p > thr:
                    got = step_truth(step, size)
                    break
            else:
                if step_truth(step, size):
                    got = True
                    break
        ok_steps.append(bool(got))
    return ok_steps


def main():
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})

    print("标定真值参照（按窗口尺寸；控制台必须先关，否则画布尺寸不同）……")
    for size in SIZES:
        set_size(size)
        call({"op": "reset", "scene": SCENE})
        if console_visible():
            call({"op": "act", "kind": "action", "text": "Python Console"})
        call({"op": "close_dialogs"})
        if REF["FULL_BOX"] is None:
            REF["FULL_BOX"] = list(truth()["layers"][0]["extent"])
        REF["FULL_X0"][size] = extent_x0()
        call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
        time.sleep(0.2)
        call({"op": "zoom_to_selection"})
        time.sleep(0.4)
        REF["SEL_X0"][size] = extent_x0()
        print(f"  {size:>9}: 全范围 x0={REF['FULL_X0'][size]:.2f}  "
              f"缩放到选择 x0={REF['SEL_X0'][size]:.2f}  "
              f"margin={abs(REF['SEL_X0'][size]-REF['FULL_X0'][size]):.2f}")

    if DATA.exists():
        z = np.load(DATA, allow_pickle=True)
        rows = list(z["rows"])
        print(f"载入已有 actor 数据：{len(rows)} 行")
    else:
        print("采集 actor 数据……")
        rows = collect_actor(N_PER_STATE, rng)
        np.savez_compressed(DATA, rows=np.array(rows, dtype=object))
    print(f"样本 {len(rows)}（{len(STATES)} 状态 × 3 步 × {N_PER_STATE}）")

    X = featurize(rows)
    y = X[5]
    seen_states = {0, 2}
    tr = [i for i, r in enumerate(rows) if r["state"] in seen_states]
    te = [i for i, r in enumerate(rows) if r["state"] not in seen_states]
    Xtr = [x[tr] for x in X[:5]]; ytr = y[tr]
    Xte = [x[te] for x in X[:5]]; yte = y[te]
    arms = {}
    print(f"\n{'arm':>7}{'in-domain':>11}{'未见UI状态':>12}   每步（未见状态）")
    for arm, cls in (("struct", Struct), ("pixel", Pixel)):
        m = train_arm(cls, Xtr, ytr)
        a_id = eval_acc(m, Xtr, ytr); a_te = eval_acc(m, Xte, yte)
        per = []
        for si in range(3):
            idx = [k for k, i in enumerate(te) if rows[i]["step"] == si]
            per.append(eval_acc(m, [x[idx] for x in Xte], yte[idx]) if idx else float("nan"))
        arms[arm] = dict(model=m, in_domain=a_id, heldout=a_te, per_step=per)
        print(f"{arm:>7}{a_id:>11.3f}{a_te:>12.3f}   " +
              " ".join(f"{p:.2f}" for p in per))

    q_struct = arms["struct"]["heldout"]; q_pixel = arms["pixel"]["heldout"]
    qstar = 0.526
    print(f"\nW11 的分水岭 q* = {qstar:.3f}")
    for nm, q in (("struct", q_struct), ("pixel", q_pixel)):
        print(f"  {nm:>6} q_hat={q:.3f} -> " +
              ("q > q*: V 不划算" if q > qstar else "q < q*: V 划算"))

    if VMODEL.exists():
        vnets = torch.load(VMODEL, weights_only=False)
        print("载入已有 V")
    else:
        print("\n采集 V 训练数据（live actor + 50% 注入错误动作，覆盖失败分布）……")
        pairs = {"Zoom to Selection": [], "Zoom Full": []}
        for arm in ("struct", "pixel"):
            for i in range(50):
                size, dialog = STATES[int(rng.integers(0, 4))]
                setup_state((size, dialog), rng)
                for step in STEPS[:2]:
                    img = shot_img()
                    cand = build_candidates(rng, step["correct"], img)
                    if cand is None:
                        continue
                    cand["instr"] = step["instr"]
                    j = pick(arms[arm]["model"], cand, arm)
                    if rng.random() < 0.5:                 # 注入错误动作
                        j = int(rng.integers(0, MAXC))
                    chosen = cand["names"][j]
                    b = vis_step(img)
                    execute(chosen); time.sleep(0.25)
                    a = vis_step(shot_img())
                    pairs[step["correct"]].append((b, a, int(step_truth(step, size))))
        vnets = {}
        for k, v in pairs.items():
            print(f"  {k}: {len(v)} 对")
            vnets[k] = (train_v(v), 0.5)
        torch.save(vnets, VMODEL)

    print(f"\nlive 闭环（预算 {BUDGET:.0f}s/格）")
    print(f"{'arm':>7}{'mode':>5}{'每步成功率':>12}{'整集全对':>10}{'正确步/分钟':>12}{'集数':>7}")
    out = {}
    for arm in ("struct", "pixel"):
        for mode in ("nov", "v"):
            t0 = time.time(); n = 0; ok_all = 0; ok_steps = 0; tot_steps = 0
            while time.time() - t0 < BUDGET:
                n += 1
                res = live_episode(arms[arm]["model"], arm, mode, rng, vnets)
                ok_steps += sum(res); tot_steps += len(res)
                ok_all += int(all(res))
            el = time.time() - t0
            row = dict(steps=tot_steps, step_rate=ok_steps/tot_steps, ep_all=ok_all/n,
                       steps_per_min=ok_steps/(el/60), n=n)
            out[f"{arm}-{mode}"] = row
            print(f"{arm:>7}{mode:>5}{row['step_rate']:>12.3f}{row['ep_all']:>10.3f}"
                  f"{row['steps_per_min']:>12.2f}{n:>7}")

    (HERE / "w12_results.json").write_text(json.dumps(
        {"ref": REF, "q_hat": {"struct": q_struct, "pixel": q_pixel}, "q_star": qstar,
         "arms": {a: dict(in_domain=arms[a]["in_domain"], heldout=arms[a]["heldout"],
                          per_step=arms[a]["per_step"]) for a in arms},
         "live": out, "n_rows": len(rows), "budget": BUDGET},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w12_results.json")


if __name__ == "__main__":
    main()
