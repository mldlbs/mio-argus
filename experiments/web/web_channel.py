"""
web_channel：观测通道 —— DOM/结构 vs 像素

真实 CUA 只有像素；DOM/a11y 是**特权通道**，能免费给出元素边界框。
本实验量化这个特权通道对 grounding 的"补贴"，并对照两种真实退化：
  - 真实截图的熵（自然背景/文字/图片）——P5 说结构化干扰才是吃参数的东西
  - 真实站点的多样性——不同站点同一个语义目标长什么样

三部分：
  1. real_grounding：真实站点（维基百科）上，DOM 通道 vs 像素通道定位同名元素
  2. diversity：K 个真实站点上"看起来像搜索框/按钮"的像素统计与外观差异
  3. 报告结构通道的补贴（DOM bbox 与真实渲染 bbox 的一致性）

用法:
    python experiments/web/web_channel.py --quick
    python experiments/web/web_channel.py
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

TMP = Path("_webch"); TMP.mkdir(exist_ok=True)

# 宽泛、站点无关的"类元素"探测（模拟 CUA 要自己找出的可交互元素）
CAND = ("a,button,input,select,textarea,summary,[role=button],"
        "[role=search],[role=link],h1,h2")

SITES = {
    "wikipedia": "https://en.wikipedia.org/wiki/Operating_system",
    "mdn": "https://developer.mozilla.org/en-US/",
    "python": "https://www.python.org/",
    "pypi": "https://pypi.org/",
    "github": "https://github.com/",
    "hn": "https://news.ycombinator.com/",
}
QUERY = ("search", "searchbox", "searchinput")


def _dom_boxes(pg):
    return pg.eval_on_selector_all(
        CAND, "els=>els.map(e=>{const r=e.getBoundingClientRect();"
              "return [e.tagName,e.getAttribute('role')||'',"
              "(e.getAttribute('aria-label')||''),"
              "(e.getAttribute('placeholder')||''),"
              "(e.id||''),(e.className||'').toString(),"
              "r.x,r.y,r.width,r.height]})")


def _find_search(boxes):
    """在 DOM 描述里模糊匹配搜索类元素（这才是 DOM 通道的"特权"）。"""
    for b in boxes:
        blob = " ".join(str(x) for x in b[:6]).lower()
        if any(q in blob for q in QUERY):
            return b
    return None


def region_stats(img, box):
    x, y, w, h = [int(v) for v in box[6:10]]
    x, y = max(0, x), max(0, y)
    crop = img[y:y + max(1, h), x:x + max(1, w)]
    if crop.size == 0:
        return None, None
    gray = crop.mean(-1)
    return crop, dict(std=float(gray.std()), grad=float(
        np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()),
        mean=float(gray.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sites = dict(list(SITES.items())[: (3 if args.quick else len(SITES))])
    summaries, patch_stats, found = {}, [], {}

    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, url in sites.items():
            pg = b.new_page(viewport={"width": 900, "height": 620})
            try:
                pg.goto(url, wait_until="domcontentloaded", timeout=25000)
                pg.wait_for_timeout(800)
                shot = np.asarray(Image.open(io.BytesIO(pg.screenshot(  # noqa
                    full_page=False))).convert("RGB"))
                boxes = _dom_boxes(pg)
                sb = _find_search(boxes)
                # 与真实渲染框一致性：DOM bbox 是否落在截图中（特权补贴）
                in_view = [bd for bd in boxes if 0 <= bd[6] < 900 and 0 <= bd[7] < 620
                           and bd[8] > 0 and bd[9] > 0]
                found[name] = bool(sb)
                if sb is not None:
                    crop, st = region_stats(shot, sb)
                    if st:
                        patch_stats.append(st)
                # 整屏像素熵 + 元素密度
                g = shot.mean(-1)
                hist = np.histogram(g, bins=32, range=(0, 255))[0]
                p = hist / hist.sum()
                ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
                summaries[name] = dict(
                    elements_total=len(boxes), elements_in_view=len(in_view),
                    search_found=bool(sb), search_box=(sb[6:10] if sb else None),
                    screen_entropy=ent, screen_std=float(g.std()),
                    max_width=max((bd[8] for bd in in_view), default=0),
                    min_click_area=min((bd[8] * bd[9] for bd in in_view
                                        if bd[8] * bd[9] > 0), default=0))
                print(f"[{name}] elems={len(boxes)} in_view={len(in_view)} "
                      f"search={bool(sb)} entropy={ent:.2f} "
                      f"min_click_area={summaries[name]['min_click_area']:.0f}")
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            pg.close()
        b.close()

    # DOM 通道的补贴：search 元素在 DOM 描述中可被文本匹配到的比例
    hit = sum(found.values()); tot = len(found)
    print(f"\nDOM 通道补贴：{hit}/{tot} 站点的搜索元素可被 DOM 文本匹配定位 "
          f"（像素通道必须自己找）")

    # 站点多样性：同一"搜索框"外观差异
    if patch_stats:
        stds = [p["std"] for p in patch_stats]
        grads = [p["grad"] for p in patch_stats]
        means = [p["mean"] for p in patch_stats]
        div = dict(n=len(patch_stats), std_spread=float(np.std(stds)),
                   grad_spread=float(np.std(grads)),
                   mean_spread=float(np.std(means)),
                   std_range=(float(min(stds)), float(max(stds))))
        print(f"跨站点『搜索框』外观：n={div['n']} "
              f"std_spread={div['std_spread']:.2f} mean_spread={div['mean_spread']:.2f} "
              f"grad_spread={div['grad_spread']:.2f}")
    else:
        div = {}

    out = {"sites": summaries, "dom_search_found": found,
           "search_patch_diversity": div}
    Path("results/web/web_channel_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/web/web_channel_results.json")


if __name__ == "__main__":
    main()
