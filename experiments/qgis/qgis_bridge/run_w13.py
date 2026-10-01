"""
QGIS W13：换成真实可得的感知通道（Windows UIA），并修正被污染的 pixel 臂

起因（rect 溯源审计）：
  · 桥 structure() 的动作 rect 与真实按钮位置不符
    （"Zoom Full" 桥=504,53 vs UIA=359,49；按桥的 rect 裁出来是"选择工具"图标）。
  · 于是 W12/W6 的 **pixel 臂**是在错误裁剪上训练的——它量的是"记住固定错位的补丁"，
    不是"认按钮"；W12 里它在未见状态掉到 0.48 正是这个病灶。
  · 而 Windows UIA（pywinauto）给出**正确**的名字+rect（已用截图与真点击验证）。

本轮：
  1) 感知与动作都走 **UIA**（真实 agent 可得；不依赖进程内注入的桥）；
  2) 候选池 = 桥与 UIA 共有的安全动作（17 个），rect **一律用 UIA 的**；
  3) 三步链（全部 UIA 可见、真值可判）：Zoom to Selection → Zoom Full → Zoom Last
     （真值 x0：SEL → FULL → SEL，Zoom Last 回到上一步的视野）；
  4) 两条执行路径：UIA `click_input`（真鼠标）vs 桥 `act`（特权），各自计成本；
  5) 重训 struct / pixel 两臂 → 给出**干净**的 pixel 数字。

用法: python experiments/qgis/qgis_bridge/run_w13.py
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
from pywinauto import Desktop  # noqa: E402

B = 1024
MAXC = 16
CH, CW = 32, 48
SCENE = "regions"
DATA = HERE / "w13_actor_data.npz"
N_PER_STATE = 16
SIZES = ("max", "1000x750")
SEL_EXPR = '"area" > 400'

TARGETS = ["Zoom to Selection", "Zoom Full", "Zoom Last"]
# 桥与 UIA 共有的安全动作（不弹模态框）
SAFE = ["Zoom Full", "Zoom to Selection", "Zoom Last", "Zoom Next",
        "Zoom to Native Resolution (100%)", "Pan Map to Selection", "Refresh",
        "Undo", "Redo", "Expand All", "Collapse All", "Copy Features",
        "Cut Features", "Paste Features", "Delete Selected", "Save Layer Edits",
        "Current Edits"]
STEPS = [
    dict(instr="把地图缩放到当前选中要素", correct="Zoom to Selection"),
    dict(instr="把地图缩放到图层的完整范围", correct="Zoom Full"),
    dict(instr="回到上一次的视野", correct="Zoom Last"),
]
STATES = [(s, d) for s in SIZES for d in (False, True)]
REF = {"FULL_X0": {}, "SEL_X0": {}, "FULL_BOX": None}


def truth():
    return call({"op": "truth"})["truth"]


def extent_x0():
    return truth()["extent"][0]


def console_visible():
    return bool(truth().get("console_visible"))


def win_origin():
    r = call({"op": "win"})["rect"]
    return r[0], r[1]


def uia_window():
    d = Desktop(backend="uia")
    for w in d.windows():
        try:
            if "QGIS" in (w.window_text() or ""):
                return w
        except Exception:
            continue
    return None


def pool_uia(qg, ox, oy, t_enum=None):
    """UIA 通道：名字 + **窗口内相对坐标**（屏幕坐标减窗口原点）。"""
    t0 = time.time()
    out = {}
    for b in qg.descendants(control_type="Button"):
        try:
            n = (b.window_text() or "").strip()
            r = b.rectangle()
            en = bool(b.is_enabled())
        except Exception:
            continue
        if n in SAFE and n not in out and r.width() >= 8 and r.height() >= 8:
            out[n] = dict(name=n, rect=[r.left-ox, r.top-oy, r.width(), r.height()],
                          enabled=en)
    el = time.time() - t0
    if t_enum is not None:
        t_enum.append(el)
    return list(out.values())


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


def shot_img():
    p = HERE / "_w13_tmp.png"
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
    if console_visible():
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


def build_candidates(rng, correct, img, cands):
    names = [c["name"] for c in cands]
    if correct not in names:
        return None
    ti = names.index(correct)
    others = [i for i in range(len(cands)) if i != ti]
    rng.shuffle(others)
    idxs = [ti] + others[:MAXC-1]
    rng.shuffle(idxs)
    crops = np.zeros((MAXC, CH, CW, 3), np.uint8)
    nm, boxes = [], []
    for j, ci in enumerate(idxs):
        c = cands[ci]
        crops[j] = crop_of(img, c["rect"])
        nm.append(c["name"])
        boxes.append([c["rect"][0]/2560, c["rect"][1]/1440,
                      c["rect"][2]/2560, c["rect"][3]/1440])
    return dict(crops=crops, names=nm, boxes=boxes, target=idxs.index(ti))


def _priv_upto(k):
    for j in range(k + 1):
        call({"op": "act", "kind": "action", "text": STEPS[j]["correct"]})
        time.sleep(0.15)


def collect(n_per_state, rng):
    rows = []
    qg = uia_window()
    for si, st in enumerate(STATES):
        for step_i, step in enumerate(STEPS):
            for k in range(n_per_state):
                setup_state(st, rng)
                if step_i > 0:
                    _priv_upto(step_i - 1)
                    time.sleep(0.15)
                ox, oy = win_origin()
                cands = pool_uia(qg, ox, oy)
                img = shot_img()
                cand = build_candidates(rng, step["correct"], img, cands)
                if cand is None:
                    print("  !! target missing from UIA pool:", step["correct"])
                    continue
                cand.update(instr=step["instr"], state=si, step=step_i, size=st[0])
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
        return float((model(*X).argmax(-1) == y).float().mean())


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


def pick(model, cand):
    row = dict(names=cand["names"], boxes=cand["boxes"], crops=cand["crops"],
               instr=cand["instr"], target=cand["target"])
    X = featurize([row])[:5]
    model.eval()
    with torch.no_grad():
        return int(model(*X).argmax(-1)[0])


def execute_uia(qg, name, cost):
    t0 = time.time()
    for b in qg.descendants(control_type="Button"):
        try:
            if (b.window_text() or "").strip() == name:
                b.click_input()
                cost.append(time.time() - t0)
                return True
        except Exception:
            continue
    return False


def execute_qt(name, cost):
    t0 = time.time()
    r = call({"op": "act", "kind": "action", "text": name})
    cost.append(time.time() - t0)
    return r.get("ok", False)


def step_truth(step, size):
    x0 = extent_x0()
    sel, full = REF["SEL_X0"][size], REF["FULL_X0"][size]
    if step["correct"] == "Zoom Full":
        target, other = full, sel
    else:                       # Zoom to Selection / Zoom Last 都要求回到 SEL 视野
        target, other = sel, full
    return abs(x0 - target) < 1.5 and abs(x0 - target) < abs(x0 - other)


def live_episode(actor, exec_mode, rng, costs, enum_cost):
    size, dialog = STATES[int(rng.integers(0, len(STATES)))]
    setup_state((size, dialog), rng)
    ok_steps = []
    for step in STEPS:
        img = shot_img()
        ox, oy = win_origin()
        qg = uia_window()
        cands = pool_uia(qg, ox, oy, enum_cost)
        cand = build_candidates(rng, step["correct"], img, cands)
        if cand is None:
            ok_steps.append(False)
            continue
        cand["instr"] = step["instr"]
        chosen = cand["names"][pick(actor, cand)]
        if exec_mode == "uia":
            execute_uia(qg, chosen, costs)
        else:
            execute_qt(chosen, costs)
        time.sleep(0.3)
        ok_steps.append(bool(step_truth(step, size)))
    return ok_steps


def main():
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})

    print("标定真值参照（窗口尺寸 × 控制台关闭）……")
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
        print(f"  {size:>9}: FULL x0={REF['FULL_X0'][size]:.2f}  SEL x0={REF['SEL_X0'][size]:.2f}"
              f"  margin={abs(REF['SEL_X0'][size]-REF['FULL_X0'][size]):.2f}")

    # UIA 覆盖审计
    ox, oy = win_origin()
    qg = uia_window()
    enum_cost = []
    cands = pool_uia(qg, ox, oy, enum_cost)
    print(f"\nUIA 通道：候选 {len(cands)}/{len(SAFE)}（SAFE 白名单），"
          f"枚举 {enum_cost[-1]*1000:.0f} ms；缺失={[t for t in SAFE if t not in [c['name'] for c in cands]]}")

    if DATA.exists():
        z = np.load(DATA, allow_pickle=True)
        rows = list(z["rows"])
        print(f"载入已有数据：{len(rows)} 行")
    else:
        print("采集 actor 数据（UIA rect 通道）……")
        rows = collect(N_PER_STATE, rng)
        np.savez_compressed(DATA, rows=np.array(rows, dtype=object))
    print(f"样本 {len(rows)}（{len(STATES)} 状态 × 3 步 × {N_PER_STATE}）")

    X = featurize(rows)
    y = X[5]
    seen = {0, 2}
    tr = [i for i, r in enumerate(rows) if r["state"] in seen]
    te = [i for i, r in enumerate(rows) if r["state"] not in seen]
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
        print(f"{arm:>7}{a_id:>11.3f}{a_te:>12.3f}   " + " ".join(f"{p:.2f}" for p in per))

    print(f"\nlive 闭环（150 s/格；UIA 真点击 vs 桥特权触发）")
    print(f"{'arm':>7}{'exec':>6}{'每步成功率':>12}{'整集全对':>10}{'正确步/分钟':>12}{'动作ms':>9}{'集数':>7}")
    out = {}
    for arm in ("struct", "pixel"):
        for exec_mode in ("uia", "qt"):
            t0 = time.time(); n = 0; ok_all = 0; ok_steps = 0; tot = 0
            costs = []
            while time.time() - t0 < 150.0:
                n += 1
                res = live_episode(arms[arm]["model"], exec_mode, rng, costs, enum_cost)
                ok_steps += sum(res); tot += len(res); ok_all += int(all(res))
            el = time.time() - t0
            row = dict(steps=tot, step_rate=ok_steps/tot, ep_all=ok_all/n,
                       steps_per_min=ok_steps/(el/60), n=n,
                       exec_ms=float(np.median(costs))*1000 if costs else None,
                       uia_enum_ms=float(np.median(enum_cost))*1000)
            out[f"{arm}-{exec_mode}"] = row
            print(f"{arm:>7}{exec_mode:>6}{row['step_rate']:>12.3f}{row['ep_all']:>10.3f}"
                  f"{row['steps_per_min']:>12.2f}{row['exec_ms']:>9.0f}{n:>7}")

    (HERE / "w13_results.json").write_text(json.dumps(
        {"ref": REF, "targets": TARGETS, "safe": SAFE,
         "uia_enum_ms": float(np.median(enum_cost))*1000, "n_rows": len(rows),
         "arms": {a: dict(in_domain=arms[a]["in_domain"], heldout=arms[a]["heldout"],
                          per_step=arms[a]["per_step"]) for a in arms},
         "live": out}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w13_results.json")


if __name__ == "__main__":
    main()
