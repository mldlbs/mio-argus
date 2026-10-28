"""
real_site_scale：成百站点扩数据（REAL_SITE §5 新高优）

目标：把 §3.6 的"学习型 vs 关键词 ≈ 持平（未翻案）"从 n=8 held-out 站
扩到 ~120 站，用**站级 bootstrap CI** 收窄 Δ 的区间；并带上 §7 的教训：
oracle 用**可见性过滤版**（隐藏容器/文档范围/头部区域/search 必须 input），
同时报告量具自身的不确定性（oracle 覆盖率与不可得原因）。

两个阶段：
  Phase 1 爬站：headless 加载 → v3 快照 → 逐 (站,任务) 求 v2 oracle（+状态分类）
  Phase 2 策略：可见命名元素建样本 → 站级划分 → 训练小模型（与 §3.6 同架构）
              → held-out **名字匹配** 与 **规范任务**（learned vs keyword）+ 站级 CI

用法: python experiments/web/real_site_scale.py [--phase1-only] [--limit N]
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from playwright.sync_api import sync_playwright
from real_site_agent import TASKS, keyword_pick
from real_site_reach import snapshot_v3

OUT = Path("results/web/real_site_scale_results.json")
B = 2048
MAXC = 300
BATCH = 64
EPOCHS = 30
ROLES = {"": 0, "button": 1, "link": 2, "searchbox": 3, "combobox": 4,
         "textbox": 5, "tab": 6, "menuitem": 7}

SITE_LIST = [
    # 新闻/媒体
    "https://www.bbc.com/", "https://edition.cnn.com/", "https://www.theguardian.com/",
    "https://www.nytimes.com/", "https://www.reuters.com/", "https://apnews.com/",
    "https://www.aljazeera.com/", "https://www.npr.org/", "https://www.dw.com/",
    "https://www.france24.com/",
    # 文档/参考
    "https://docs.python.org/3/", "https://developer.mozilla.org/en-US/",
    "https://devdocs.io/", "https://kubernetes.io/", "https://docs.docker.com/",
    "https://nodejs.org/en", "https://go.dev/", "https://www.rust-lang.org/",
    "https://www.ruby-lang.org/en/", "https://www.php.net/", "https://www.perl.org/",
    "https://www.java.com/en/", "https://spring.io/", "https://www.djangoproject.com/",
    "https://flask.palletsprojects.com/", "https://fastapi.tiangolo.com/",
    "https://numpy.org/", "https://pandas.pydata.org/", "https://pytorch.org/",
    "https://www.tensorflow.org/", "https://scikit-learn.org/", "https://matplotlib.org/",
    "https://jupyter.org/", "https://www.anaconda.com/",
    # 开发/社区
    "https://github.com/", "https://gitlab.com/", "https://stackoverflow.com/",
    "https://superuser.com/", "https://askubuntu.com/", "https://www.npmjs.com/",
    "https://pypi.org/", "https://crates.io/", "https://packagist.org/",
    "https://hub.docker.com/", "https://news.ycombinator.com/", "https://lobste.rs/",
    "https://dev.to/", "https://medium.com/", "https://hashnode.com/",
    "https://www.quora.com/", "https://mastodon.social/", "https://bsky.app/",
    # 电商/服务
    "https://www.ebay.com/", "https://www.etsy.com/", "https://www.aliexpress.com/",
    "https://www.shopify.com/", "https://www.walmart.com/", "https://www.target.com/",
    "https://www.bestbuy.com/", "https://www.booking.com/", "https://www.airbnb.com/",
    "https://www.tripadvisor.com/",
    # 政府/教育
    "https://www.usa.gov/", "https://www.whitehouse.gov/", "https://www.nasa.gov/",
    "https://www.noaa.gov/", "https://www.cdc.gov/", "https://www.nih.gov/",
    "https://www.mit.edu/", "https://www.stanford.edu/", "https://www.berkeley.edu/",
    "https://www.harvard.edu/", "https://www.ox.ac.uk/", "https://www.cam.ac.uk/",
    # 公司
    "https://www.microsoft.com/", "https://www.apple.com/", "https://www.google.com/",
    "https://about.meta.com/", "https://www.netflix.com/", "https://www.spotify.com/",
    "https://www.adobe.com/", "https://www.salesforce.com/", "https://www.oracle.com/",
    "https://www.ibm.com/", "https://www.intel.com/", "https://www.nvidia.com/",
    "https://www.amd.com/", "https://www.samsung.com/", "https://www.sony.com/",
    "https://www.siemens.com/", "https://www.bosch.com/", "https://www.toyota.com/",
    "https://www.bmw.com/", "https://www.vercel.com/", "https://www.cloudflare.com/",
    "https://www.digitalocean.com/", "https://www.heroku.com/", "https://www.mongodb.com/",
    "https://redis.io/", "https://www.postgresql.org/", "https://www.mysql.com/",
    "https://www.elastic.co/", "https://grafana.com/", "https://www.jetbrains.com/",
    # 工具/杂
    "https://www.wikipedia.org/", "https://www.wikimedia.org/", "https://archive.org/",
    "https://www.openstreetmap.org/", "https://weather.com/", "https://www.accuweather.com/",
    "https://www.timeanddate.com/", "https://www.worldometers.info/",
    "https://ourworldindata.org/", "https://www.khanacademy.org/", "https://www.coursera.org/",
    "https://www.edx.org/", "https://www.udemy.com/", "https://www.duolingo.com/",
    "https://www.imdb.com/", "https://www.rottentomatoes.com/", "https://www.goodreads.com/",
    "https://www.espn.com/", "https://www.nationalgeographic.com/", "https://www.smithsonianmag.com/",
    "https://www.britannica.com/", "https://www.nature.com/", "https://www.science.org/",
    "https://arxiv.org/", "https://www.semanticscholar.org/", "https://scholar.google.com/",
    "https://www.producthunt.com/", "https://news.google.com/", "https://www.bing.com/",
    "https://duckduckgo.com/", "https://www.ecosia.org/", "https://www.startpage.com/",
]

HIDDEN_JS = r"""
(rid) => {
  const el = document.querySelector('[data-reach-id="' + rid + '"]');
  if (!el) return null;
  const r = el.getBoundingClientRect();
  let hidden = false, zero = null, footer = false;
  let n = el;
  while (n && n !== document.documentElement) {
    const nr = n.getBoundingClientRect();
    const st = getComputedStyle(n);
    if (n.tagName === 'FOOTER') footer = true;
    if ((st.overflow === 'hidden' || st.overflow === 'clip' || st.clipPath !== 'none')
        && (nr.height < 2 || nr.width < 2)) { hidden = true; zero = n.tagName; break; }
    n = n.parentElement;
  }
  return {x: r.x, y: r.y, w: r.width, h: r.height, hidden, zero, footer,
          doc: document.documentElement.scrollHeight,
          vw: innerWidth, vh: innerHeight, tag: el.tagName.toLowerCase(),
          role: el.getAttribute('role') || ''};
}
"""


def oracle_v2(pg, v3, task):
    """可见性过滤版 oracle。返回 (entry 或 None, 原因)。"""
    pat = TASKS[task]["pattern"]
    reasons = Counter()
    for e in v3:
        blob = f"{e['name']} {e['role']}"
        if task == "logo":
            if not e.get("is_logo"):
                continue
        else:
            if not pat.search(blob):
                continue
            # search 优先 input/searchbox，但保留"名字命中即可"的回退（否则覆盖崩）
        if task == "search" and not (e["tag"] in ("input", "textarea")
                                     or e["role"] in ("searchbox", "combobox")):
            reasons["prefer-input"] += 1
            # 不 continue：作为回退候选继续体检
        c = pg.evaluate(HIDDEN_JS, e["rid"])
        if c is None:
            continue
        if c["hidden"]:
            reasons["hidden-container"] += 1
            continue
        if c["y"] < -2 or c["y"] > c["doc"]:
            reasons["out-of-doc"] += 1
            continue
        if task in ("home", "logo") and (c["y"] > 200 or c["footer"]):
            reasons["footer-or-deep"] += 1
            continue
        state = ("in_view" if (0 <= c["y"] and c["y"] + c["h"] <= c["vh"]
                               and 0 <= c["x"] and c["x"] + c["w"] <= c["vw"])
                 else "needs_scroll")
        return dict(rid=e["rid"], name=e["name"], tag=e["tag"], role=e["role"],
                    is_logo=bool(e.get("is_logo")), rect=[round(c["x"]), round(c["y"]),
                    round(c["w"]), round(c["h"])], state=state), ""
    why = ",".join(f"{k}x{v}" for k, v in reasons.items()) if reasons else "no-candidate"
    return None, why


def crawl(limit=None, save_every=10):
    sites = SITE_LIST[:limit] if limit else SITE_LIST
    rows = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True,
                               args=["--no-proxy-server"])
        for i, url in enumerate(sites):
            t0 = time.time()
            row = dict(url=url, ok=False)
            pg = b.new_page(viewport={"width": 1000, "height": 700})
            try:
                pg.goto(url, wait_until="load", timeout=12000)
                pg.wait_for_timeout(1200)
                v3, _, snap_ms = snapshot_v3(pg)
                vis = []
                for e in v3:
                    c = pg.evaluate(HIDDEN_JS, e["rid"])
                    if c and not c["hidden"] and c["y"] >= -2 and c["y"] <= c["doc"]:
                        vis.append(dict(name=e["name"], tag=e["tag"], role=e["role"],
                                        is_logo=bool(e.get("is_logo")),
                                        x=c["x"], y=c["y"], w=c["w"], h=c["h"]))
                row.update(ok=True, n_all=len(v3), n_vis=len(vis), snap_ms=snap_ms,
                           load_ms=(time.time() - t0) * 1000 - snap_ms,
                           table=vis, tasks={})
                for task in TASKS:
                    o, why = oracle_v2(pg, v3, task)
                    row["tasks"][task] = dict(ok=bool(o), why=why,
                                              oracle=o)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:70]}"
            pg.close()
            rows.append(row)
            if (i + 1) % save_every == 0 or i + 1 == len(sites):
                Path("results/web").mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[{i+1}/{len(sites)}] {url[:44]:44} "
                  f"{'ok' if row.get('ok') else 'FAIL'} "
                  f"vis={row.get('n_vis', '-')} "
                  f"tasks={sum(1 for t in row.get('tasks', {}).values() if t['ok'])}/4 "
                  f"({time.time()-t0:.1f}s)", flush=True)
        b.close()
    return rows


def bag(text, out):
    import zlib
    t = (text or "").lower().strip()
    if not t:
        return out
    if len(t) < 3:
        out[zlib.crc32(t.encode()) % B] += 1
        return out
    for i in range(len(t) - 2):
        out[zlib.crc32(t[i:i+3].encode()) % B] += 1
    return out


def make_samples(rows, sites):
    out = []
    for r in rows:
        if not r.get("ok") or r["url"] not in sites:
            continue
        tbl = sorted(r["table"], key=lambda e: -(e["w"] * e["h"]))[:MAXC]
        named = [e for e in tbl if e["name"]]
        if len(named) < 3:
            continue
        for j, e in enumerate(named):
            out.append(dict(names=[x["name"] for x in named],
                            roles=[ROLES.get(x["role"], 0) for x in named],
                            boxes=[[x["x"] / 1000, x["y"] / 700, x["w"] / 1000, x["h"] / 700]
                                   for x in named],
                            instr=e["name"], target=j, site=r["url"], tbl=named))
    return out


def tensorize(samples):
    n = len(samples)
    txt = torch.zeros(n, MAXC, B)
    role = torch.zeros(n, MAXC, dtype=torch.long)
    box = torch.zeros(n, MAXC, 4)
    mask = torch.zeros(n, MAXC, dtype=torch.bool)
    instr = torch.zeros(n, B)
    tgt = torch.zeros(n, dtype=torch.long)
    for i, s in enumerate(samples):
        k = min(MAXC, len(s["names"]))
        buf = np.zeros((k, B), np.float32)
        for j in range(k):
            bag(s["names"][j], buf[j])
        buf /= np.maximum(buf.sum(1, keepdims=True), 1e-6)
        txt[i, :k] = torch.from_numpy(buf)
        role[i, :k] = torch.tensor(s["roles"][:k])
        box[i, :k] = torch.tensor(s["boxes"][:k], dtype=torch.float)
        mask[i, :k] = True
        ib = np.zeros(B, np.float32)
        bag(s["instr"], ib)
        ss = ib.sum()
        if ss > 0:
            ib /= ss
        instr[i] = torch.from_numpy(ib)
        tgt[i] = min(s["target"], k - 1)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = crawl(limit=args.limit)
    ok = [r for r in rows if r.get("ok")]
    print(f"\nPhase1：加载成功 {len(ok)}/{len(rows)}")
    tc = Counter(t["why"] for r in ok for t in r["tasks"].values() if not t["ok"])
    okc = sum(1 for r in ok for t in r["tasks"].values() if t["ok"])
    print(f"  v2 oracle 可得 {okc}/{len(ok)*4}；不可得原因 {dict(tc)}")
    if args.phase1_only:
        return

    # Phase 2：站级划分
    urls = sorted({r["url"] for r in ok})
    rng = np.random.default_rng(0)
    rng.shuffle(urls)
    cut = int(len(urls) * 0.7)
    tr_urls, te_urls = set(urls[:cut]), set(urls[cut:])
    tr = make_samples(rows, tr_urls)
    te = make_samples(rows, te_urls)
    print(f"Phase2：站点 {len(tr_urls)} 训练 / {len(te_urls)} held-out；"
          f"样本 train={len(tr)} test={len(te)}")
    if not tr or not te:
        print("样本不足，退出"); return
    torch.manual_seed(0)
    Xtr = tensorize(tr)
    Xte = tensorize(te)
    model = Policy()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n = len(tr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            logits = model(*[x[idx] for x in Xtr[:5]])
            loss = nn.functional.cross_entropy(logits, Xtr[5][idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}: loss {tot/n:.4f}", flush=True)

    def name_acc(samples, X):
        model.eval()
        with torch.no_grad():
            pred = model(*X[:5]).argmax(-1)
        return float((pred == X[5]).float().mean())

    acc_tr = name_acc(tr, Xtr)
    acc_te = name_acc(te, Xte)
    print(f"名字匹配：train {acc_tr:.3f} / held-out {acc_te:.3f}")

    # 规范任务：keyword vs learned（都用同一可见表；oracle 来自 v3 的 v2 过滤）
    task_rows = []
    task_by_site = {}
    for r in ok:
        if r["url"] not in te_urls:
            continue
        tbl = r["table"]
        have = {t: r["tasks"][t]["oracle"] for t in TASKS
                if r["tasks"][t]["ok"] and (r["tasks"][t]["oracle"] or {}).get("name")}
        if not have:
            continue
        task_by_site[r["url"]] = have
    # keyword 通道
    for url, have in task_by_site.items():
        r = next(x for x in ok if x["url"] == url)
        tbl = r["table"]
        for task, o in have.items():
            ki = keyword_pick(tbl, task)
            task_rows.append(dict(url=url, task=task, arm="keyword",
                                  ok=bool(ki >= 0 and tbl[ki]["name"] == o["name"]
                                          and tbl[ki]["tag"] == o["tag"])))
    # learned 通道
    idx_by_site = {}
    for i, s in enumerate(te):
        idx_by_site.setdefault(s["site"], []).append(i)
    model.eval()
    with torch.no_grad():
        preds = model(*Xte[:5]).argmax(-1).numpy()
    for url, idxs in idx_by_site.items():
        have = task_by_site.get(url, {})
        for task, o in have.items():
            # 找出该站样本里 instr == oracle 名字的那些，取其在候选表里的位置
            hit = False
            for i in idxs:
                s = te[i]
                if s["instr"] != o["name"]:
                    continue
                p = int(preds[i])
                if p < len(s["tbl"]) and s["tbl"][p]["name"] == o["name"] \
                        and s["tbl"][p]["tag"] == o["tag"]:
                    hit = True
                break
            task_rows.append(dict(url=url, task=task, arm="learned", ok=hit))

    # 站级 bootstrap CI
    def rate(rows_sub):
        return sum(1 for r in rows_sub if r["ok"]) / max(1, len(rows_sub))

    urls_te = sorted({r["url"] for r in task_rows})
    boots = []
    rng2 = np.random.default_rng(1)
    for _ in range(2000):
        pick = rng2.choice(len(urls_te), size=len(urls_te), replace=True)
        chosen = {urls_te[i] for i in pick}
        sub = [r for r in task_rows if r["url"] in chosen]
        k = rate([r for r in sub if r["arm"] == "keyword"])
        l = rate([r for r in sub if r["arm"] == "learned"])
        boots.append(l - k)
    boots = np.array(boots)
    kw_all = rate([r for r in task_rows if r["arm"] == "keyword"])
    lr_all = rate([r for r in task_rows if r["arm"] == "learned"])
    print(f"\n规范任务（held-out {len(urls_te)} 站，n={len(task_rows)//2} 任务）：")
    print(f"  keyword {kw_all:.3f}  learned {lr_all:.3f}")
    print(f"  Δ(learned-keyword) = {lr_all-kw_all:+.3f} "
          f"[95% CI {np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}]")
    per_task = {}
    for task in TASKS:
        rs = [r for r in task_rows if r["task"] == task]
        per_task[task] = dict(n=len(rs)//2,
                              keyword=rate([r for r in rs if r["arm"] == "keyword"]),
                              learned=rate([r for r in rs if r["arm"] == "learned"]))
        print(f"    {task:7} n={per_task[task]['n']:3} "
              f"kw={per_task[task]['keyword']:.3f} lr={per_task[task]['learned']:.3f}")

    out = json.loads(OUT.read_text(encoding="utf-8"))
    summary = dict(sites_loaded=len(ok), sites_total=len(rows),
                   oracle_ok=okc, oracle_slots=len(ok) * 4, oracle_missing=dict(tc),
                   train_sites=len(tr_urls), test_sites=len(te_urls),
                   samples_train=len(tr), samples_test=len(te),
                   name_acc_train=acc_tr, name_acc_heldout=acc_te,
                   task_keyword=kw_all, task_learned=lr_all,
                   delta=lr_all - kw_all,
                   delta_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   per_task=per_task)
    Path("results/web/real_site_scale_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 results/web/real_site_scale_results.json（逐站）"
          "与 real_site_scale_summary.json（汇总）")


if __name__ == "__main__":
    main()
