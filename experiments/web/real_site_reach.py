"""
real_site_reach：提取器 v3（跨 frame 闭合）+ 可点性漏斗

REAL_SITE §5 高优第 2 项：v2 只走页面内 JS——同源 iframe 靠 contentDocument
（跨域为 null，同源策略硬墙）、弹窗/遮挡未判。本实验补两块：
  1. **v3 = 逐 Playwright frame 注入**：跨域 iframe 可从 frame tree 进入
     （frame.evaluate 在各 frame 上下文执行），坐标按 frameElement 链累加到顶层视口；
  2. **found → reachable 漏斗**：结构通道说"找到了"不等于"点得到"——
     对关键词策略选中的目标做 elementFromPoint 命中测试（在目标自己的 frame 内）
     与真实点击（Playwright actionability，遮挡/超时即失败），逐级归因。

指标（site × task）：
  - found_v2 / found_v3（规则 oracle，对齐 real_site_agent 的口径）
  - n_v2 / n_v3、快照耗时（v3 为全部 frame 之和）
  - 漏斗：pick（关键词选中）→ hit（elementFromPoint 命中目标）→ click_ok
  - 失败归因：no_pick / occluded（记遮挡者 tag.class.id）/ click_timeout / ok
  - 跨域贡献：v3 新增条目的 frame_depth>0 计数；顶层遮挡 overlay 标记

用法:
    python experiments/web/real_site_reach.py --quick
    python experiments/web/real_site_reach.py
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Error as PWError

from real_site_agent import (SITES, SNAPSHOT_JS_V2, TASKS, oracle_pick,
                             keyword_pick)

EXTRA_SITES = {
    "yt_embed": "https://www.youtube.com/embed/jNQXAC9IVRw",
}

# 受控跨域场景：真实站的四类目标大概率不在跨域 frame 里，
# 用允许被嵌的 OSM embed 构造 v2 必盲、v3 可进的跨域 iframe。
HARNESS_HTML = (
    '<body style="margin:0">'
    '<iframe title="osm" '
    'src="https://www.openstreetmap.org/export/embed.html'
    '?bbox=116.0%2C39.8%2C116.6%2C40.1&layer=mapnik" '
    'style="width:100%;height:640px;border:0"></iframe></body>'
)

SNAPSHOT_JS_V3 = r"""
(prefix) => {
  const out = [], seen = new Set();
  let seq = 0;
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
      const rid = prefix + '_' + (seq++);
      try { e.setAttribute('data-reach-id', rid); } catch(err){}
      out.push({rid, tag: e.tagName.toLowerCase(), role: e.getAttribute('role') || '',
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

HIT_JS = r"""
([x, y, rid]) => {
  if (x < 0 || y < 0 || x > innerWidth || y > innerHeight)
    return {hit: false, occluder: 'offscreen'};
  const el = document.elementFromPoint(x, y);
  if (!el) return {hit: false, occluder: 'none'};
  if (el.closest('[data-reach-id="' + rid + '"]')) return {hit: true, occluder: ''};
  const other = el.closest('[data-reach-id]');
  if (other)
    return {hit: false, occluder: 'ui-element:' + other.getAttribute('data-reach-id')};
  const tag = el.tagName.toLowerCase();
  if (el === document.body || el === document.documentElement)
    return {hit: false, occluder: 'page-background'};
  return {hit: false, occluder: tag + '.' + String(el.className || '').split(' ')[0]
                       + (el.id ? '#' + el.id : '')};
}
"""

OVERLAY_JS = r"""
() => {
  const el = document.elementFromPoint(innerWidth / 2, innerHeight / 2);
  if (!el) return null;
  const st = getComputedStyle(el);
  const fixed = st.position === 'fixed' || st.position === 'sticky';
  const big = el.getBoundingClientRect().width * el.getBoundingClientRect().height
              > innerWidth * innerHeight * 0.6;
  return {tag: el.tagName.toLowerCase(), id: el.id || '',
          cls: String(el.className || '').split(' ')[0],
          fullscreen: !!(fixed && big)};
}
"""


def snapshot_v3(page):
    """逐 frame 注入 v3；返回 (entries, frames_meta, total_ms)。"""
    t0 = time.perf_counter()
    entries, frames_meta = [], []
    for fi, frame in enumerate(page.frames):
        try:
            els = frame.evaluate(SNAPSHOT_JS_V3, f"f{fi}")
        except Exception:
            continue
        try:
            fe = frame.frame_element
            box = fe.bounding_box() if fe else None
        except Exception:
            box = None
        ox = box["x"] if box else 0.0
        oy = box["y"] if box else 0.0
        host = ""
        try:
            host = frame.url.split("/")[2] if "://" in frame.url else frame.url[:40]
        except Exception:
            host = ""
        frames_meta.append(dict(idx=fi, host=host, n=len(els),
                                cross_origin=(fi > 0 and host != "" and
                                              host not in page.url)))
        for e in els:
            e["px"] = e["x"] + ox
            e["py"] = e["y"] + oy
            e["frame_idx"] = fi
            e["frame_depth"] = 0 if fi == 0 else 1
            e["frame_host"] = host
            entries.append(e)
    return entries, frames_meta, (time.perf_counter() - t0) * 1000


def flat_table(entries):
    """去掉坐标键差异，喂给线内 oracle_pick/keyword_pick（它们只用 name/role/tag/is_logo）。"""
    return [dict(tag=e["tag"], role=e["role"], name=e["name"],
                 is_logo=e["is_logo"], href=e.get("href", ""))
            for e in entries]


def prepare_target(entry, frames):
    """滚动进视口并重取 frame 局部坐标（否则 elementFromPoint 出视口恒 miss）。"""
    frame = frames[entry["frame_idx"]]
    rid = entry["rid"]
    try:
        frame.locator(f'[data-reach-id="{rid}"]').scroll_into_view_if_needed(
            timeout=2000)
        rect = frame.evaluate(
            "rid => {const el = document.querySelector('[data-reach-id=\"' + rid + '\"]');"
            " if (!el) return null; const r = el.getBoundingClientRect();"
            " return {x: r.x, y: r.y, w: r.width, h: r.height};}", rid)
        if rect:
            entry = dict(entry, x=rect["x"], y=rect["y"],
                         w=rect["w"], h=rect["h"])
        entry["scrolled"] = True
    except Exception:
        entry["scrolled"] = False
    return entry


def hit_test(entry, frames):
    frame = frames[entry["frame_idx"]]
    x = entry["x"] + entry["w"] / 2
    y = entry["y"] + entry["h"] / 2
    try:
        res = frame.evaluate(HIT_JS, [x, y, entry["rid"]])
    except Exception as e:
        return dict(hit=False, occluder=f"eval-error:{type(e).__name__}",
                    is_target=False)
    return dict(hit=bool(res.get("hit")),
                occluder=res.get("occluder") or "",
                is_target=bool(res.get("hit")))


def attempt_click(entry, frames):
    frame = frames[entry["frame_idx"]]
    loc = frame.locator(f'[data-reach-id="{entry["rid"]}"]')
    try:
        loc.click(timeout=2000)
        return dict(click_ok=True, click_err="")
    except PWError as e:
        msg = str(e).split("\n")[0][:80]
        return dict(click_ok=False, click_err=msg)


def run_site(b, name, url, do_click=True):
    rec = dict(site=name, url=url, load_ok=False, v2=None, v3=None,
               frames=[], overlay=None, funnel=[], click=[])
    # —— 漏斗装载 1：一次加载同时取 v2 / v3 ——
    pg = b.new_page(viewport={"width": 1000, "height": 700})
    try:
        if url:
            pg.goto(url, wait_until="load", timeout=30000)
        else:
            pg.set_content(HARNESS_HTML, wait_until="load", timeout=30000)
        pg.wait_for_timeout(1500)
        rec["load_ok"] = True
        rec["overlay"] = pg.evaluate(OVERLAY_JS)

        t0 = time.perf_counter()
        v2 = pg.evaluate(SNAPSHOT_JS_V2)
        ms2 = (time.perf_counter() - t0) * 1000
        v3, frames_meta, ms3 = snapshot_v3(pg)
        rec["v2"] = dict(n=len(v2), snap_ms=ms2)
        rec["v3"] = dict(n=len(v3), snap_ms=ms3)
        rec["frames"] = frames_meta

        v3_new = [e for e in v3 if e["frame_depth"] > 0]
        rec["v3_cross"] = len(v3_new)

        # 表级可达率采样：初始视口内前 100 个元素，不做滚动的命中测试
        sample = [e for e in v3
                  if 0 <= e["px"] < 1000 and 0 <= e["py"] < 700][:100]
        if sample:
            h = sum(hit_test(e, pg.frames)["hit"] for e in sample)
            rec["reach_sample"] = dict(n=len(sample), hit=h)

        tab2, tab3 = flat_table(v2), flat_table(v3)
        for task in TASKS:
            oi2, oi3 = oracle_pick(tab2, task), oracle_pick(tab3, task)
            ki3 = keyword_pick(tab3, task)
            row = dict(task=task, found_v2=oi2 >= 0, found_v3=oi3 >= 0,
                       pick=ki3 >= 0)
            if ki3 >= 0:
                e3 = prepare_target(v3[ki3], pg.frames)
                ht = hit_test(e3, pg.frames)
                row.update(ht)
                row["in_view"] = 0 <= e3["px"] < 1000 and 0 <= e3["py"] < 700
                row["scrolled"] = e3.get("scrolled", False)
            else:
                row.update(hit=False, occluder="", is_target=False,
                           in_view=False)
            rec["funnel"].append(row)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    pg.close()

    # —— 点击阶段：每 (site, task) 独立装载，点关键词选中的目标 ——
    if do_click and rec["load_ok"] and url:
        for task in TASKS:
            crow = dict(task=task, attempted=False)
            pg = b.new_page(viewport={"width": 1000, "height": 700})
            try:
                pg.goto(url, wait_until="load", timeout=30000)
                pg.wait_for_timeout(1200)
                v3, _, _ = snapshot_v3(pg)
                tab3 = flat_table(v3)
                ki = keyword_pick(tab3, task)
                if ki < 0:
                    crow["reason"] = "no_pick"
                else:
                    e3 = prepare_target(v3[ki], pg.frames)
                    ht = hit_test(e3, pg.frames)
                    before = pg.url
                    ck = attempt_click(e3, pg.frames)
                    crow.update(attempted=True, **ck)
                    crow["hit"] = ht["hit"]
                    crow["occluder"] = ht["occluder"]
                    crow["url_changed"] = pg.url != before
                    crow["reason"] = ("ok" if ck["click_ok"]
                                      else "click_timeout")
            except Exception as e:
                crow["reason"] = (f"error:{type(e).__name__}:"
                                  f"{str(e)[:100]}")
            rec["click"].append(crow)
            pg.close()
    return rec


def summarize(records):
    print(f"\n{'site':<10}{'load':>5}{'n_v2':>6}{'n_v3':>6}{'cross':>6}"
          f"{'ovl':>5}  funnel(pick/hit/click per task)")
    print("-" * 88)
    for r in records:
        if not r["load_ok"]:
            print(f"{r['site']:<10} FAIL {r.get('error','')[:60]}")
            continue
        fn = {f["task"]: f for f in r["funnel"]}
        ck = {c["task"]: c for c in r["click"]}
        cells = []
        for t in TASKS:
            f, c = fn.get(t, {}), ck.get(t, {})
            p = "1" if f.get("pick") else "0"
            h = "1" if f.get("hit") else "0"
            k = "1" if c.get("click_ok") else ("-" if not c.get("attempted") else "0")
            cells.append(f"{t[:4]}:{p}{h}{k}")
        ovl = "Y" if (r["overlay"] or {}).get("fullscreen") else "."
        print(f"{r['site']:<10}{'ok':>5}{r['v2']['n']:>6}{r['v3']['n']:>6}"
              f"{r.get('v3_cross',0):>6}{ovl:>5}  " + " ".join(cells))

    # 全局漏斗
    loaded = [r for r in records if r["load_ok"]]
    n_tasks = len(TASKS)
    def rate(xs): return sum(xs) / max(1, len(xs))
    fv2 = rate([f["found_v2"] for r in loaded for f in r["funnel"]])
    fv3 = rate([f["found_v3"] for r in loaded for f in r["funnel"]])
    fpick = rate([f["pick"] for r in loaded for f in r["funnel"]])
    fhit = rate([f["hit"] for r in loaded for f in r["funnel"]])
    clicks = [c for r in records for c in r.get("click", [])]
    catt = [c for c in clicks if c.get("attempted")]
    cok = rate([c["click_ok"] for c in catt]) if catt else float("nan")
    print(f"\n全局漏斗（{len(loaded)} 站 × {n_tasks} 任务）：")
    print(f"  found: v2 {fv2:.3f} → v3 {fv3:.3f}  (Δ {fv3-fv2:+.3f})")
    print(f"  pick(关键词): {fpick:.3f} → hit(命中测试): {fhit:.3f} "
          f"→ click_ok: {cok:.3f}  (n_click={len(catt)})")
    reasons = {}
    for c in clicks:
        reasons[c.get("reason", "?")] = reasons.get(c.get("reason", "?"), 0) + 1
    print(f"  点击归因: {reasons}")
    occ = [f["occluder"] for r in loaded for f in r["funnel"]
           if f["pick"] and not f["hit"] and f.get("occluder")]
    print(f"  hit 失败的遮挡者: {occ[:6]}{' ...' if len(occ)>6 else ''}")
    cross = sum(r.get("v3_cross", 0) for r in loaded)
    print(f"  v3 跨 frame 新增条目合计: {cross}")
    rs = [(r["reach_sample"]["hit"], r["reach_sample"]["n"])
          for r in loaded if r.get("reach_sample")]
    if rs:
        h, n = sum(x[0] for x in rs), sum(x[1] for x in rs)
        print(f"  表级可达率（视口内未遮挡，抽样）: {h}/{n} = "
              f"{h/max(1,n):.3f}")
    ms2 = sorted(r["v2"]["snap_ms"] for r in loaded)
    ms3 = sorted(r["v3"]["snap_ms"] for r in loaded)
    print(f"  快照耗时中位: v2 {ms2[len(ms2)//2]:.1f} ms / "
          f"v3 {ms3[len(ms3)//2]:.1f} ms")
    return dict(found_v2=fv2, found_v3=fv3, pick=fpick, hit=fhit,
                click_ok=(None if catt == [] else cok), n_click=len(catt),
                click_reasons=reasons, cross_entries=cross,
                snap_ms_v2=ms2[len(ms2)//2], snap_ms_v3=ms3[len(ms3)//2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-click", action="store_true")
    args = ap.parse_args()
    sites = dict(SITES)
    if not args.quick:
        sites.update(EXTRA_SITES)
        sites["harness_xf"] = None
    if args.quick:
        sites = dict(list(sites.items())[:3])

    records = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, url in sites.items():
            records.append(run_site(b, name, url, do_click=not args.no_click))
            r = records[-1]
            print(f"[{name}] load={'ok' if r['load_ok'] else 'FAIL'} "
                  f"v2={r['v2']['n'] if r['v2'] else '-'} "
                  f"v3={r['v3']['n'] if r['v3'] else '-'}")
        b.close()

    summary = summarize(records)
    Path("results/web/real_site_reach_results.json").write_text(
        json.dumps(dict(records=records, summary=summary),
                    indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/web/real_site_reach_results.json")


if __name__ == "__main__":
    main()
