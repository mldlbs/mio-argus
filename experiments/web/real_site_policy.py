"""
real_site_policy：扩数据版本（v2）—— 学习型策略到底行不行？

上一版（356 样本 / 候选上限 64）结论：学习型策略打不过关键词，且站点级泛化差（0.716→0.184）。
但那是小数据 + 弱编码 + 小候选上限。本版把三个限制都放开：
  - 站点 32 个（24 训练 / 8 个未见站测试）
  - 候选上限 64 → 300
  - 文本编码：char-trigram 多热(2048) → Linear→ReLU→Linear（加非线性）
特征**按批即时计算**，不预存稠密 bag。

评测：名字匹配（训练站 / held-out 站）+ 四类规范任务零样本（对照关键词基线）。

用法:
    python experiments/web/real_site_policy.py
"""
import json
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from playwright.sync_api import sync_playwright

from real_site_agent import SNAPSHOT_JS_V2, TASKS, oracle_pick, keyword_pick

SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Operating_system",
    "mdn": "https://developer.mozilla.org/en-US/",
    "python": "https://www.python.org/",
    "pypi": "https://pypi.org/",
    "nodejs": "https://nodejs.org/en",
    "django": "https://www.djangoproject.com/",
    "numpy": "https://numpy.org/",
    "jquery": "https://jquery.com/",
    "flask": "https://flask.palletsprojects.com/",
    "fastapi": "https://fastapi.tiangolo.com/",
    "pandas": "https://pandas.pydata.org/",
    "pytorch": "https://pytorch.org/",
    "docker": "https://www.docker.com/",
    "kubernetes": "https://kubernetes.io/",
    "redis": "https://redis.io/",
    "postgres": "https://www.postgresql.org/",
    "sqlite": "https://www.sqlite.org/",
    "git": "https://git-scm.com/",
    "nginx": "https://nginx.org/",
    "vue": "https://vuejs.org/",
    "svelte": "https://svelte.dev/",
    "react": "https://react.dev/",
    "astro": "https://astro.build/",
    "deno": "https://deno.com/",
    # ↓ 8 个从未参与训练
    "github": "https://github.com/",
    "hn": "https://news.ycombinator.com/",
    "rust": "https://www.rust-lang.org/",
    "go": "https://go.dev/",
    "mozilla": "https://www.mozilla.org/",
    "arxiv": "https://arxiv.org/",
    "atom": "https://atom.io/",
    "huggingface": "https://huggingface.co/",
}
TEST_SITES = ["github", "hn", "rust", "go", "mozilla", "arxiv", "atom", "huggingface"]

B = 2048
MAXC = 300
ROLES = {"": 0, "button": 1, "link": 2, "searchbox": 3, "combobox": 4,
         "tab": 5, "textbox": 6, "navigation": 7}


def bag(text, out):
    s = text.lower().strip()
    if not s:
        return out
    if len(s) < 3:
        out[zlib.crc32(s.encode()) % B] += 1.0
        return out
    for i in range(len(s) - 2):
        out[zlib.crc32(s[i:i + 3].encode()) % B] += 1.0
    return out


def collect(pw):
    data = {}
    b = pw.chromium.launch(channel="chrome", headless=True)
    for name, url in SITES.items():
        pg = b.new_page(viewport={"width": 1000, "height": 700})
        try:
            pg.goto(url, wait_until="load", timeout=25000)
            pg.wait_for_timeout(1200)
            tbl = [e for e in pg.evaluate(SNAPSHOT_JS_V2)
                   if e["name"] and e["w"] >= 6 and e["h"] >= 6]
            tbl = sorted(tbl, key=lambda e: -(e["w"] * e["h"]))[:MAXC]
            data[name] = tbl
            print(f"  [{name:<12}] named={len(tbl)}")
        except Exception as e:
            print(f"  [{name:<12}] FAILED {type(e).__name__}")
            data[name] = []
        pg.close()
    b.close()
    return data


def make_samples(data, sites):
    out = []
    for s in sites:
        tbl = data.get(s, [])
        if len(tbl) < 3:
            continue
        for j, e in enumerate(tbl):
            out.append(dict(names=[x["name"] for x in tbl],
                            roles=[ROLES.get(x["role"], 0) for x in tbl],
                            boxes=[[x["x"] / 1000, x["y"] / 700,
                                    x["w"] / 1000, x["h"] / 700] for x in tbl],
                            instr=e["name"], target=j, site=s, tbl=tbl))
    return out


def tensorize(samples, device):
    Bn = len(samples)
    N = MAXC
    txt = torch.zeros(Bn, N, B, device=device)
    role = torch.zeros(Bn, N, dtype=torch.long, device=device)
    box = torch.zeros(Bn, N, 4, device=device)
    mask = torch.zeros(Bn, N, dtype=torch.bool, device=device)
    instr = torch.zeros(Bn, B, device=device)
    tgt = torch.zeros(Bn, dtype=torch.long, device=device)
    for i, s in enumerate(samples):
        n = min(N, len(s["names"]))
        buf = np.zeros((n, B), np.float32)
        for k in range(n):
            bag(s["names"][k], buf[k])
        buf /= np.maximum(buf.sum(1, keepdims=True), 1e-6)
        txt[i, :n] = torch.from_numpy(buf)
        role[i, :n] = torch.tensor(s["roles"][:n])
        box[i, :n] = torch.tensor(s["boxes"][:n], dtype=torch.float)
        mask[i, :n] = True
        ib = np.zeros(B, np.float32)
        bag(s["instr"], ib)
        ssum = ib.sum()
        if ssum > 0:
            ib /= ssum
        instr[i] = torch.from_numpy(ib)
        tgt[i] = min(s["target"], n - 1)
    return txt, role, box, instr, mask, tgt


class Policy(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.tx = nn.Sequential(nn.Linear(B, d), nn.ReLU(), nn.Linear(d, d))
        self.role = nn.Embedding(8, d)
        self.box = nn.Linear(4, d)
        self.scale = d ** 0.5

    def forward(self, txt, role, box, instr, mask):
        e = self.tx(txt) + self.role(role) + self.box(box)
        q = self.tx(instr)
        return ((e * q[:, None, :]).sum(-1) / self.scale).masked_fill(~mask, -1e9)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with sync_playwright() as pw:
        data = collect(pw)
    tr_sites = [s for s in SITES if s not in TEST_SITES]
    tr = make_samples(data, tr_sites)
    te = make_samples(data, TEST_SITES)
    print(f"\ntrain samples={len(tr)}  held-out samples={len(te)}  "
          f"train sites={len(tr_sites)}  test sites={len(TEST_SITES)}")

    torch.manual_seed(0)
    m = Policy().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    BATCH = 32
    n_param = sum(p.numel() for p in m.parameters())
    print(f"model params={n_param:,}")
    for ep in range(40):
        perm = np.random.default_rng(ep).permutation(len(tr))
        for i in range(0, len(tr), BATCH):
            idx = perm[i:i + BATCH]
            T = tensorize([tr[j] for j in idx], device)
            loss = F.cross_entropy(m(*T[:5]), T[5])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}: loss={loss.item():.3f}")

    @torch.no_grad()
    def name_acc(samples):
        m.eval()
        hit = 0
        for i in range(0, len(samples), BATCH):
            T = tensorize(samples[i:i + BATCH], device)
            hit += int((m(*T[:5]).argmax(-1) == T[5]).sum())
        return hit / max(1, len(samples))

    a_tr, a_te = name_acc(tr), name_acc(te)
    print(f"\n名字匹配：train {a_tr:.3f} | **held-out 站点 {a_te:.3f}**")

    print(f"\n{'task':<8}{'learned':>10}{'keyword':>10}{'n':>5}")
    print("-" * 34)
    res = {}
    for task in TASKS:
        hit = kw_hit = tot = 0
        for s in TEST_SITES:
            tbl = data.get(s, [])
            if len(tbl) < 3:
                continue
            oi = oracle_pick(tbl, task)
            if oi < 0 or oi >= len(tbl):
                continue
            smp = [dict(names=[x["name"] for x in tbl],
                        roles=[ROLES.get(x["role"], 0) for x in tbl],
                        boxes=[[x["x"] / 1000, x["y"] / 700,
                                x["w"] / 1000, x["h"] / 700] for x in tbl],
                        instr=task, target=oi, site=s, tbl=tbl)]
            with torch.no_grad():
                T = tensorize(smp, device)
                pred = int(m(*T[:5]).argmax(-1))
            hit += int(pred == oi)
            kw_hit += int(keyword_pick(tbl, task) == oi)
            tot += 1
        if tot:
            res[task] = dict(learned=hit / tot, keyword=kw_hit / tot, n=tot)
            print(f"{task:<8}{hit/tot:>10.3f}{kw_hit/tot:>10.3f}{tot:>5}")
    if res:
        n = sum(v["n"] for v in res.values())
        L = sum(v["learned"] * v["n"] for v in res.values()) / n
        K = sum(v["keyword"] * v["n"] for v in res.values()) / n
        print(f"\n合计：learned {L:.3f} vs keyword {K:.3f}")

    Path("results/web/real_site_policy_results.json").write_text(json.dumps(
        {"name_match": {"train": a_tr, "heldout_sites": a_te},
         "canonical": res, "params": n_param,
         "n_train": len(tr), "n_test": len(te),
         "test_sites": TEST_SITES}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 results/web/real_site_policy_results.json")


if __name__ == "__main__":
    main()
