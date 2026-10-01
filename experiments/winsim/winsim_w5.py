"""
WinSim W5：学习型 next-action 选择器 + 应用级/主题级 held-out

数据集：让 oracle 在 WinSim 上跑任务，记录每一步的
        候选元素(role/name/bbox) + 任务特征 → 该点哪个（模仿学习）
选择器：结构通道（uia）特征打分，softmax over 候选
评测：
  in-domain  ：任务 {move,delete,toggle}，主题 {a,b,c}
  OOD-theme  ：同样任务，主题 d（未见）
  OOD-app    ：终端任务 run（未见应用/未见动作名）

预期：结构特征对主题近似不变（OOD-theme ≈ in-domain）；
      但对**未见应用/任务**不泛化（OOD-app 掉）。

用法:
    python experiments/winsim/winsim_w5.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}
MAXC = 20

ROLES = {"icon": 0, "dir": 1, "file": 2, "button": 3, "switch": 4}
TTYPE = {"move": 0, "delete": 1, "toggle": 2}          # run 留给 OOD（→ UNK）
TRAIN_NAMES = ["Explorer", "Settings", "notes.md", "report.txt",
               "Documents", "Pictures", "Downloads",
               "Cut", "Paste", "Delete", "Up", "Wi-Fi", "Bluetooth", "Brightness"]
UNK = len(TRAIN_NAMES)
NAME2ID = {n: i for i, n in enumerate(TRAIN_NAMES)}


def nid(name):
    return NAME2ID.get(name, UNK)


def make_instances(split, n, rng):
    out = []
    for i in range(n):
        if split == "in_domain":
            t = rng.choice(["move", "delete", "toggle"])
            theme = str(rng.choice(list("abc")))
        elif split == "ood_theme":
            t = rng.choice(["move", "delete", "toggle"])
            theme = "d"
        else:  # ood_app
            t = "run"; theme = str(rng.choice(list("abc")))
        if t == "move":
            f = str(rng.choice(["notes.md", "report.txt"]))
            seq = [("icon", "Explorer"), ("file", f), ("button", "Cut"),
                   ("dir", "Documents"), ("button", "Paste")]; tgt = f
        elif t == "delete":
            f = str(rng.choice(["notes.md", "report.txt"]))
            seq = [("icon", "Explorer"), ("file", f), ("button", "Delete")]; tgt = f
        elif t == "toggle":
            s = str(rng.choice(["Wi-Fi", "Bluetooth"]))
            seq = [("icon", "Settings"), ("switch", s)]; tgt = s
        else:
            seq = [("icon", "Terminal"), ("button", "Run")]; tgt = "Run"
        out.append(dict(theme=theme, seq=seq, type=t, tname=tgt, seed=1000 + i))
    return out


def find(els, role, name):
    for i, e in enumerate(els):
        if e["role"] == role and e["name"] == name:
            return i
    return -1


APPID = {"explorer": 0, "settings": 1, "terminal": 2, "notepad": 3}


def collect(pw, inst):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    pg.evaluate(f"WinSim.reset({inst['seed']})")
    pg.evaluate(f"WinSim.setTheme('{inst['theme']}')")
    samples = []
    for role, name in inst["seq"]:
        els = pg.evaluate("WinSim.elements()")
        if len(els) > MAXC:
            els = els[:MAXC]
        ti = find(els, role, name)
        if ti < 0:
            break
        st = pg.evaluate("WinSim.truth()")
        act_app = 4
        for w in st["windows"]:
            if w["id"] == st["active"]:
                act_app = APPID.get(w["appId"], 4)
        ctx = [act_app, len(st["path"]), int(bool(st["selected"])),
               int(bool(st["clipboard"]))]
        samples.append(dict(els=els, type=inst["type"], tname=inst["tname"],
                            target=ti, ctx=ctx))
        e = els[ti]
        pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")
    b.close()
    return samples


def featurize(samples, device):
    B = len(samples)
    role = torch.zeros(B, MAXC, dtype=torch.long)
    name = torch.zeros(B, MAXC, dtype=torch.long)
    box = torch.zeros(B, MAXC, 4)
    mask = torch.zeros(B, MAXC, dtype=torch.bool)
    typ = torch.zeros(B, dtype=torch.long)
    tnm = torch.zeros(B, dtype=torch.long)
    ctx = torch.zeros(B, 4)
    tgt = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(samples):
        for j, e in enumerate(s["els"]):
            role[i, j] = ROLES.get(e["role"], 0)
            name[i, j] = nid(e["name"])
            box[i, j] = torch.tensor([e["x"] / 900, e["y"] / 600,
                                      e["w"] / 900, e["h"] / 600])
            mask[i, j] = True
        typ[i] = TTYPE.get(s["type"], len(TTYPE))          # run→UNK
        tnm[i] = nid(s["tname"])
        ctx[i] = torch.tensor(s["ctx"], dtype=torch.float)
        tgt[i] = s["target"]
    return (role.to(device), name.to(device), box.to(device), mask.to(device),
            typ.to(device), tnm.to(device), ctx.to(device), tgt.to(device))


class Sel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.role = nn.Embedding(len(ROLES), d)
        self.name = nn.Embedding(len(TRAIN_NAMES) + 1, d)
        self.box = nn.Linear(4, d)
        self.typ = nn.Embedding(len(TTYPE) + 1, d)
        self.tnm = nn.Embedding(len(TRAIN_NAMES) + 1, d)
        self.ctx = nn.Linear(4, d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, role, name, box, mask, typ, tnm, ctx):
        c = self.role(role) + self.name(name) + self.box(box)      # (B,N,d)
        t = (self.typ(typ) + self.tnm(tnm) + self.ctx(ctx))[:, None, :]
        x = torch.cat([c, t.expand_as(c), c * t], -1)
        return self.mlp(x).squeeze(-1).masked_fill(~mask, -1e9)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    splits = {
        "in_domain": make_instances("in_domain", 220, rng),
        "ood_theme": make_instances("ood_theme", 120, rng),
        "ood_app":   make_instances("ood_app", 120, rng),
    }
    data = {}
    with sync_playwright() as pw:
        for k, insts in splits.items():
            S = []
            for inst in insts:
                S += collect(pw, inst)
            data[k] = S
            print(f"  {k}: {len(S)} samples")
    data["in_domain_val"] = data["in_domain"][-80:]
    data["in_domain_tr"] = data["in_domain"][:-80]

    torch.manual_seed(0)
    m = Sel().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    tr = featurize(data["in_domain_tr"], device)
    B = 64; n = tr[0].shape[0]
    for ep in range(30):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            logits = m(*[t[idx] for t in tr[:7]])
            loss = F.cross_entropy(logits, tr[7][idx])
            opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(ss):
        if not ss:
            return float("nan")
        T = featurize(ss, device)
        return float((m(*T[:7]).argmax(-1) == T[7]).float().mean())

    res = {k: acc(data[k]) for k in ("in_domain_val", "ood_theme", "ood_app")}
    chance = float(np.mean([1.0 / min(len(s["els"]), MAXC) for s in data["in_domain_val"]]))

    print(f"\n{'split':<16}{'acc':>8}")
    print("-" * 24)
    for k, v in res.items():
        print(f"{k:<16}{v:>8.4f}")
    print(f"{'chance':<16}{chance:>8.4f}")
    print(f"\nW5：结构选择器 in-domain {res['in_domain_val']:.3f} | "
          f"OOD-theme {res['ood_theme']:.3f} (drop {res['in_domain_val']-res['ood_theme']:+.3f}) | "
          f"OOD-app {res['ood_app']:.3f} (drop {res['in_domain_val']-res['ood_app']:+.3f})")
    passes = (res["in_domain_val"] > 0.8 and
              res["in_domain_val"] - res["ood_app"] > 0.2)
    print(f"W5 {'通过' if passes else '未通过'}"
          "（判据：in-domain 高 且 OOD-app 明显掉）")

    (OUT / "winsim_w5_results.json").write_text(json.dumps(
        {"n": {k: len(v) for k, v in data.items()}, "acc": res,
         "chance": chance}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w5_results.json")


if __name__ == "__main__":
    main()
