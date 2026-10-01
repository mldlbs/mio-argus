"""
WinSim W11：索引化动作空间（借鉴 browser-use/jev-ultrafast）

jev 的设计：观测 → **element table**；决策 = (operation, target)，
且只提供**兼容的 (op,target) 组合**；一次前向出两个 head；小 LLM 只在
operation == TYPE_TEXT 时生成文本。

W11 在同一沙箱里实现并消融：
  A target-only ：只预测目标元素（等价 W5 的结构选择器，无法表达 TYPE_TEXT）
  B op+target 无掩码：两个 head 但不过滤兼容性
  C op+target 有掩码：只允许兼容组合（jev 的做法）
任务集加入 **rename**（click→Rename→TYPE_TEXT→Confirm），让 op 轴真实存在。

指标：next-action 命中率（op 与目标元素都对）
split：in-domain / OOD-theme / OOD-app

用法:
    python experiments/winsim/winsim_w11.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from playwright.sync_api import sync_playwright

import winsim_w5 as W5

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = W5.URL
VIEW = W5.VIEW
MAXC = W5.MAXC

ROLES = {"icon": 0, "dir": 1, "file": 2, "button": 3, "switch": 4, "textbox": 5}
CLICK_ROLES = {0, 1, 2, 3, 4}
TYPE_ROLES = {5}
OPS = {"click": 0, "type_text": 1}
NAMES = W5.TRAIN_NAMES + ["Rename", "Confirm", "Name", "todo.md"]
NAME2ID = {n: i for i, n in enumerate(NAMES)}
UNK = len(NAMES)
TTYPE = {"move": 0, "delete": 1, "toggle": 2, "rename": 3}   # run 留给 OOD


def nid(n):
    return NAME2ID.get(n, UNK)


def seq_for(inst):
    t, f, s = inst["type"], inst["tname"], inst["tname"]
    if t == "move":
        return [("click", "icon", "Explorer"), ("click", "file", f),
                ("click", "button", "Cut"), ("click", "dir", "Documents"),
                ("click", "button", "Paste")]
    if t == "delete":
        return [("click", "icon", "Explorer"), ("click", "file", f),
                ("click", "button", "Delete")]
    if t == "toggle":
        return [("click", "icon", "Settings"), ("click", "switch", s)]
    if t == "rename":
        return [("click", "icon", "Explorer"), ("click", "file", f),
                ("click", "button", "Rename"),
                ("type_text", "textbox", "Name", "todo.md"),
                ("click", "button", "Confirm")]
    return [("click", "icon", "Terminal"), ("click", "button", "Run")]   # OOD-app


def make_instances(split, n, rng):
    out = []
    for i in range(n):
        if split == "ood_app":
            t, theme = "run", str(rng.choice(list("abc")))
        else:
            t = str(rng.choice(["move", "delete", "toggle", "rename"]))
            theme = "d" if split == "ood_theme" else str(rng.choice(list("abc")))
        if t in ("move", "delete", "rename"):
            f = str(rng.choice(["notes.md", "report.txt"]))
        elif t == "toggle":
            f = str(rng.choice(["Wi-Fi", "Bluetooth"]))
        else:
            f = "Run"
        out.append(dict(theme=theme, type=t, tname=f, seed=1000 + i))
    return out


def find(els, role, name):
    for i, e in enumerate(els):
        if e["role"] == role and e["name"] == name:
            return i
    return -1


def collect(pg, inst):
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    out = []
    for step in seq_for(inst):
        op, role, name = step[0], step[1], step[2]
        text = step[3] if len(step) > 3 else None
        els = pg.evaluate("WinSim.elements()")
        if len(els) > MAXC:
            els = els[:MAXC]
        ti = find(els, role, name)
        if ti < 0:
            break
        st = pg.evaluate("WinSim.truth()")
        act = 4
        for w in st["windows"]:
            if w["id"] == st["active"]:
                act = W5.APPID.get(w["appId"], 4)
        ctx = [act, len(st["path"]), int(bool(st["selected"])),
               int(bool(st["clipboard"])), int(bool(st["renaming"]))]
        out.append(dict(els=els, op=OPS[op], target=ti, text=text,
                        type=inst["type"], tname=inst["tname"], ctx=ctx))
        e = els[ti]
        if op == "click":
            pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")
        else:
            pg.evaluate(f"WinSim.type_text({json.dumps(text)})")
    return out


def featurize(samples, device):
    B = len(samples)
    role = torch.zeros(B, MAXC, dtype=torch.long)
    name = torch.zeros(B, MAXC, dtype=torch.long)
    box = torch.zeros(B, MAXC, 4)
    mask = torch.zeros(B, MAXC, dtype=torch.bool)
    typ = torch.zeros(B, dtype=torch.long)
    tnm = torch.zeros(B, dtype=torch.long)
    ctx = torch.zeros(B, 5)
    op = torch.zeros(B, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(samples):
        for j, e in enumerate(s["els"]):
            role[i, j] = ROLES.get(e["role"], 0)
            name[i, j] = nid(e["name"])
            box[i, j] = torch.tensor([e["x"] / 900, e["y"] / 600,
                                      e["w"] / 900, e["h"] / 600])
            mask[i, j] = True
        typ[i] = TTYPE.get(s["type"], len(TTYPE))
        tnm[i] = nid(s["tname"])
        ctx[i] = torch.tensor(s["ctx"], dtype=torch.float)
        op[i] = s["op"]
        tgt[i] = s["target"]
    return (role.to(device), name.to(device), box.to(device), mask.to(device),
            typ.to(device), tnm.to(device), ctx.to(device), op.to(device),
            tgt.to(device))


class Indexed(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.role = nn.Embedding(len(ROLES), d)
        self.name = nn.Embedding(len(NAMES) + 1, d)
        self.box = nn.Linear(4, d)
        self.typ = nn.Embedding(len(TTYPE) + 1, d)
        self.tnm = nn.Embedding(len(NAMES) + 1, d)
        self.ctx = nn.Linear(5, d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.op_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, role, name, box, mask, typ, tnm, ctx):
        c = self.role(role) + self.name(name) + self.box(box)          # (B,N,d)
        t = (self.typ(typ) + self.tnm(tnm) + self.ctx(ctx))[:, None, :]
        x = torch.cat([c, t.expand_as(c), c * t], -1)
        tgt_logit = self.mlp(x).squeeze(-1).masked_fill(~mask, -1e9)   # (B,N)
        g = torch.cat([c.mean(1), t.squeeze(1)], -1)
        op_logit = self.op_head(g)                                     # (B,2)
        return op_logit, tgt_logit


def compat_mask(role, mask, dev):
    """(B,2,N) 允许的 (op,target) 组合。"""
    B, N = role.shape
    is_tb = torch.zeros(B, N, dtype=torch.bool, device=dev)
    for rid in TYPE_ROLES:
        is_tb |= (role == rid)
    clk = mask.clone()
    typ = is_tb & mask
    return torch.stack([clk, typ], 1)          # (B,2,N)


def train(samples, device, masked=True, epochs=30):
    torch.manual_seed(0)
    m = Indexed().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    T = featurize(samples, device)
    B = 64
    n = T[0].shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            op_l, tgt_l = m(*[t[idx] for t in T[:7]])
            TGT, OP = T[8][idx], T[7][idx]
            Bn = idx.shape[0]
            NC = T[1].shape[1]                                  # 候选数 N
            # 联合分布：log P(op) + log P(target | 掩码)
            joint = op_l[:, :, None] + tgt_l[:, None, :]
            if masked:
                joint = joint.masked_fill(~compat_mask(T[0][idx], T[3][idx], device), -1e9)
            joint = joint.reshape(Bn, -1)
            lab = OP * NC + TGT
            loss = F.cross_entropy(joint, lab)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def hit(m, samples, device, masked=True):
    if not samples:
        return float("nan")
    T = featurize(samples, device)
    op_l, tgt_l = m(*T[:7])
    joint = op_l[:, :, None] + tgt_l[:, None, :]
    if masked:
        joint = joint.masked_fill(~compat_mask(T[0], T[3], device), -1e9)
    flat = joint.reshape(T[0].shape[0], -1)
    pick = flat.argmax(-1)
    pred_op = pick // T[1].shape[1]
    pred_tgt = pick % T[1].shape[1]
    ok = (pred_op == T[7]) & (pred_tgt == T[8])
    return float(ok.float().mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    insts = {
        "in_domain": make_instances("in_domain", 170, rng),
        "ood_theme": make_instances("ood_theme", 90, rng),
        "ood_app":   make_instances("ood_app", 90, rng),
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
    trS = tr[:int(len(tr) * 0.8)]
    val = tr[int(len(tr) * 0.8):]
    splits = {"in_domain": val, "ood_theme": data["ood_theme"],
              "ood_app": data["ood_app"]}

    m_mask = train(trS, device, masked=True)
    m_free = train(trS, device, masked=False)

    rows = {}
    print(f"\n{'split':<12}{'B unmasked':>12}{'C masked':>10}")
    print("-" * 34)
    for k, v in splits.items():
        b_ = hit(m_free, v, device, masked=False)
        c_ = hit(m_mask, v, device, masked=True)
        rows[k] = dict(unmasked=b_, masked=c_)
        print(f"{k:<12}{b_:>12.3f}{c_:>10.3f}")
    print("-" * 34)
    print("说明：A target-only（W5 结构选择器，无法表达 TYPE_TEXT）="
          "in 1.000 / theme 1.000 / app 0.500（旧口径，含 type_text 步则不适用）")
    imp = rows["in_domain"]["masked"] - rows["in_domain"]["unmasked"]
    print(f"\nin-domain: masked - unmasked = {imp:+.3f}")
    print(f"结论：{'兼容掩码有效' if imp > 0.05 else '掩码无净收益（或不显著）'}")

    (OUT / "winsim_w11_results.json").write_text(json.dumps(
        {"rows": rows, "masked_minus_unmasked_in": imp}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w11_results.json")


if __name__ == "__main__":
    main()
