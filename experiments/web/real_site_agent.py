"""
real_site_agent：把 WinSim 的结构通道搬到真实站点

WinSim 的 struct 臂在合成桌面里是 1.000。真实站点呢？
本实验用同一个配方：**element table（可见可交互元素 + bbox）+ 关键词策略**，
在真实站点上量：
  - table 规模（候选数）与快照耗时（"one call per snapshot" 的成本轴）
  - 三类典型目标（search / login / home）是否**在 table 里**
  - 关键词策略能否选中（与规则 oracle 对比）

对照/说明：像素通道在这类页面上要自己检测（WinSim W7–W10：端到端 0.79），
而结构通道只要 DOM 里有名字就能白拿 → 这就是本实验要量的"结构可用性"。

用法:
    python experiments/web/real_site_agent.py
"""
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Operating_system",
    "mdn": "https://developer.mozilla.org/en-US/",
    "python": "https://www.python.org/",
    "pypi": "https://pypi.org/",
    "github": "https://github.com/",
    "hn": "https://news.ycombinator.com/",
}

SNAPSHOT_JS = r"""
() => {
  const sel = 'a,button,input,select,textarea,[role=button],[role=link],[role=searchbox],[role=combobox],[role=tab]';
  const out = [];
  document.querySelectorAll(sel).forEach(e => {
    const r = e.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return;
    const st = getComputedStyle(e);
    if (st.visibility === 'hidden' || st.display === 'none') return;
    if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) return;
    const name = (e.getAttribute('aria-label') || e.getAttribute('placeholder') ||
                  e.innerText || e.value || e.title || '').trim().slice(0, 60);
    const href = e.getAttribute('href') || '';
    const is_logo = e.tagName === 'A' && (href === '/' || href === location.origin + '/')
                    && (!!e.querySelector('img,svg') || r.y < 120);
    out.push({tag: e.tagName.toLowerCase(), role: e.getAttribute('role') || '',
              name, is_logo, href,
              x: r.x, y: r.y, w: r.width, h: r.height});  });
  return out;
}
"""

SNAPSHOT_JS_V2 = r"""
() => {
  const out = [], seen = new Set();
  function nameOf(e){
    const g = a => e.getAttribute ? e.getAttribute(a) : null;
    const cands = [g('aria-label'), g('placeholder'), g('title'), g('name')];
    const img = e.querySelector && e.querySelector('img[alt]');
    if (img) cands.push(img.getAttribute('alt'));
    if (g('aria-labelledby')) {
      const t = document.getElementById(g('aria-labelledby'));
      if (t) cands.push(t.innerText);
    }
    if (e.labels && e.labels.length) cands.push(e.labels[0].innerText);
    cands.push(e.value, e.innerText, e.textContent);
    for (const c of cands) if (c && String(c).trim()) return String(c).trim().slice(0, 60);
    return '';
  }
  function walk(root, depth){
    if (depth > 3 || !root.querySelectorAll) return;
    let nodes; try { nodes = root.querySelectorAll(
      'a,button,input,select,textarea,[role],[tabindex]'); } catch(e){ return; }
    nodes.forEach(e => {
      if (seen.has(e)) return; seen.add(e);
      const r = e.getBoundingClientRect();
      if (r.width < 4 || r.height < 4) return;
      const st = getComputedStyle(e);
      if (st.display === 'none' || st.visibility === 'hidden' ||
          parseFloat(st.opacity) < 0.05) return;
      if (e.getAttribute('aria-hidden') === 'true') return;
      const name = nameOf(e);
      const href = e.getAttribute('href') || '';
      const is_logo = e.tagName === 'A' &&
        (href === '/' || href === location.origin + '/') &&
        (!!e.querySelector('img,svg') || r.y < 120);
      out.push({tag: e.tagName.toLowerCase(), role: e.getAttribute('role') || '',
                name, is_logo, href, x: r.x, y: r.y, w: r.width, h: r.height});
      if (e.shadowRoot) walk(e.shadowRoot, depth + 1);
    });
    root.querySelectorAll('*').forEach(e => { if (e.shadowRoot) walk(e.shadowRoot, depth + 1); });
    root.querySelectorAll('iframe').forEach(f => {
      try { if (f.contentDocument) walk(f.contentDocument, depth + 1); } catch(e){}
    });
  }
  walk(document, 0);
  return out;
}
"""

TASKS = {
    "search": dict(pattern=re.compile(r"search", re.I),
                   kw=["search", "find", "query"]),
    "login":  dict(pattern=re.compile(r"log ?in|sign ?in|login|signin", re.I),
                   kw=["login", "log", "in", "sign"]),
    "home":   dict(pattern=re.compile(r"\bhome\b|main page", re.I),
                   kw=["home"]),
    "logo":   dict(pattern=re.compile(r""), kw=["logo", "home", "back"]),
}


def oracle_pick(table, task):
    """规则 oracle：按名字/角色匹配目标类型。"""
    cfg = TASKS[task]
    for i, e in enumerate(table):
        blob = f"{e['name']} {e['role']}"
        if task == "logo":
            if e.get("is_logo"):
                return i
            continue
        if task == "search":
            if e["tag"] in ("input", "textarea") and cfg["pattern"].search(blob):
                return i
            if e["role"] in ("searchbox", "combobox") and cfg["pattern"].search(blob):
                return i
        if cfg["pattern"].search(blob):
            return i
    return -1


def keyword_pick(table, task):
    """关键词策略（无学习）：名字里命中任务关键词最多者。"""
    kw = TASKS[task]["kw"]
    best, bs = -1, 0.0
    for i, e in enumerate(table):
        n = e["name"].lower()
        if not n:
            continue
        s = sum(1.0 for k in kw if k in n)
        if task == "search" and e["tag"] in ("input", "textarea"):
            s += 0.5
        if s > bs:
            best, bs = i, s
    return best if bs > 0 else -1


def main():
    rows = []
    extractors = {"v1_15line": SNAPSHOT_JS, "v2_shadow_iframe": SNAPSHOT_JS_V2}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, url in SITES.items():
            pg = b.new_page(viewport={"width": 1000, "height": 700})
            try:
                pg.goto(url, wait_until="load", timeout=30000)
                pg.wait_for_timeout(1500)
                for ex_name, ex_js in extractors.items():
                    t0 = time.perf_counter()
                    table = pg.evaluate(ex_js)
                    snap_ms = (time.perf_counter() - t0) * 1000
                    for task in TASKS:
                        oi = oracle_pick(table, task)
                        ki = keyword_pick(table, task)
                        rows.append(dict(site=name, extractor=ex_name, task=task,
                                         n=len(table), snap_ms=snap_ms,
                                         found=oi >= 0, hit=bool(oi >= 0 and ki == oi)))
                    print(f"[{name:<9}] {ex_name:<17} table={len(table):>4} "
                          f"snap={snap_ms:6.1f}ms found="
                          f"{sum(oracle_pick(table,t)>=0 for t in TASKS)}/{len(TASKS)}")
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            pg.close()
        b.close()

    print(f"\n{'task':<8}{'v1 found':>10}{'v1 hit':>8}{'v2 found':>10}{'v2 hit':>8}")
    print("-" * 46)
    summary = {}
    for task in TASKS:
        line = f"{task:<8}"
        for ex in extractors:
            rs = [r for r in rows if r["task"] == task and r["extractor"] == ex]
            f = sum(r["found"] for r in rs) / max(1, len(rs))
            h = sum(r["hit"] for r in rs) / max(1, len(rs))
            summary[f"{ex}|{task}"] = dict(found=f, hit=h)
            line += f"{f:>10.3f}{h:>8.3f}"
        print(line)
    print("-" * 46)
    for ex in extractors:
        sn = [r["snap_ms"] for r in rows if r["extractor"] == ex]
        tb = [r["n"] for r in rows if r["extractor"] == ex]
        print(f"{ex}: median snap {sorted(sn)[len(sn)//2]:.1f} ms, "
              f"median table {sorted(tb)[len(tb)//2]}")
    print("对照：WinSim 结构通道 1.000；像素端到端 0.790")

    Path("results/web/real_site_agent_results.json").write_text(json.dumps(
        {"rows": rows, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 results/web/real_site_agent_results.json")


if __name__ == "__main__":
    main()
