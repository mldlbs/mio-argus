"""
web_gap：headless 与 headed 的渲染差异有多大，会怎样影响训练？

参考系 = **headed**（真实用户看到的），对照 = 若干 headless 配置。
测两类量：
  A. 渲染发散：逐像素差、>阈值像素占比、布局 bbox 漂移、文本度量漂移
  B. 训练敏感性（--policy）：在 headless 上训练一个像素策略，评测 headless vs headed

用法:
    python experiments/web/web_gap.py --quick
    python experiments/web/web_gap.py --policy
"""
import argparse
import io
import json
import pathlib
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

VIEW = {"width": 480, "height": 360}
TMP = Path("_webgap")
TMP.mkdir(exist_ok=True)

MODES = {
    "headed":            dict(headless=False, args=[]),
    "headless":          dict(headless=True,  args=[]),
    "headless_nogpu":    dict(headless=True,  args=["--disable-gpu"]),
    "headless_grayscale": dict(headless=True, args=["--disable-lcd-text"]),
}

TEXT_PROBE = [("Arial", 11, "Save file"), ("Times New Roman", 13, "Settings"),
              ("monospace", 12, "commit 0x1f"),
              ("system-ui", 14, "Open Recent Item")]


def pages():
    P = {}
    P["text"] = """<div style="font:12px Arial">The quick brown fox jumps over 0123456789</div>
      <div style="font:11px 'Times New Roman';margin-top:6px">Serif <b>bold</b> <i>italic</i> Settings</div>
      <div style="font:12px monospace;margin-top:6px">commit 0x1f3a  merged</div>"""
    P["gradient"] = """<div data-id="g1" style="width:140px;height:40px;
      background:linear-gradient(135deg,#c33,#119);border-radius:8px"></div>
      <div data-id="g2" style="width:110px;height:34px;margin-top:8px;
      background:linear-gradient(#eee,#888);border-radius:4px"></div>"""
    P["border"] = """<div data-id="b1" style="width:160px;height:50px;border:1px solid #333;border-radius:10px"></div>
      <div data-id="b2" style="width:120px;height:30px;margin-top:6px;border:2px dashed #777"></div>"""
    P["svg"] = """<svg width="60" height="60" data-id="s1" data-svg="1">
      <circle cx="30" cy="30" r="20" fill="none" stroke="#26c" stroke-width="2"/>
      <path d="M15 30 L45 30 M30 15 L30 45" stroke="#c33" stroke-width="2"/></svg>"""
    P["canvas"] = """<canvas data-id="c1" width="150" height="90" style="border:1px solid #999"></canvas>
      <script>const x=document.querySelector('canvas').getContext('2d');
      const g=x.createLinearGradient(0,0,150,90);g.addColorStop(0,'#4af');g.addColorStop(1,'#f47');
      x.fillStyle=g;x.fillRect(0,0,150,90);x.font='13px Arial';x.fillStyle='#000';x.fillText('Chart 42',12,50);</script>"""
    P["emoji"] = """<div style="font-size:22px" data-id="e1">📁 🔒 ⚙️ ✅ ❌ ⭐</div>
      <div style="font-size:14px;margin-top:6px" data-id="e2">Search… • 日本語 • Ω≈ç√</div>"""
    P["shadow"] = """<div data-id="h1" style="width:140px;height:44px;background:#fff;
      box-shadow:0 2px 10px rgba(0,0,0,.35);border-radius:8px"></div>
      <div data-id="h2" style="width:100px;height:34px;margin-top:10px;background:#eef;
      box-shadow:inset 0 1px 3px rgba(0,0,0,.4)"></div>"""
    P["transform"] = """<div data-id="t1" style="width:90px;height:40px;background:#3a7;
      transform:rotate(18deg) scale(1.05)"></div>
      <div data-id="t2" style="width:80px;height:30px;margin-top:14px;background:#73a;
      transform:skewX(-12deg)"></div>"""
    return {k: f"""<!doctype html><html><head><meta charset="utf-8">
      <style>body{{margin:12px;font-family:Arial;background:#f7f8fa}}</style></head>
      <body>{v}</body></html>""" for k, v in P.items()}


def capture():
    pgs = pages()
    for k, html in pgs.items():
        (TMP / f"{k}.html").write_text(html, encoding="utf-8")
    out = {}
    with sync_playwright() as pw:
        for mode, cfg in MODES.items():
            b = pw.chromium.launch(channel="chrome", headless=cfg["headless"],
                                   args=cfg["args"])
            for k in pgs:
                pg = b.new_page(viewport=VIEW, device_scale_factor=1)
                pg.goto("file:///" + str((TMP / f"{k}.html").resolve()).replace("\\", "/"))
                shot = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
                boxes = pg.eval_on_selector_all(
                    "[data-id]", "els=>els.map(e=>{const r=e.getBoundingClientRect();"
                                 "return [e.dataset.id,r.x,r.y,r.width,r.height]})")
                if k == "text":
                    tm = pg.evaluate("""(probe)=>{const c=document.createElement('canvas');
                      const x=c.getContext('2d');return probe.map(([f,s,t])=>{
                      x.font=`${s}px ${f}`;return x.measureText(t).width;})}""",
                                     TEXT_PROBE)
                else:
                    tm = []
                out[(mode, k)] = dict(shot=shot, boxes=boxes, tm=tm)
                pg.close()
            b.close()
    return out


def diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(mean=float(d.mean()), frac_gt8=float((d.max(-1) > 8).mean()),
                max=int(d.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    data = capture()
    keys = sorted({k for (_, k) in data})
    ref = "headed"

    print(f"{'page':<11}{'mode':<20}{'mean':>7}{'>8占比':>8}{'max':>5}{'bboxΔpx':>10}")
    print("-" * 62)
    rows = []
    for k in keys:
        rb = {bx[0]: bx[1:] for bx in data[(ref, k)]["boxes"]}
        for mode in MODES:
            if mode == ref:
                continue
            d = diff(data[(ref, k)]["shot"], data[(mode, k)]["shot"])
            mx = 0.0
            for bx in data[(mode, k)]["boxes"]:
                if bx[0] in rb:
                    mx = max(mx, max(abs(bx[1 + i] - rb[bx[0]][i]) for i in range(4)))
            tm_ref = data[(ref, k)]["tm"]
            tm_cmp = data[(mode, k)]["tm"]
            tm_d = max((abs(a - b) for a, b in zip(tm_ref, tm_cmp)), default=0.0)
            print(f"{k:<11}{mode:<20}{d['mean']:>7.2f}{d['frac_gt8']:>8.3f}"
                  f"{d['max']:>5}{mx:>10.2f}"
                  + (f"  textΔ={tm_d:.2f}" if tm_d else ""))
            rows.append(dict(page=k, mode=mode, **d, bbox_max_delta=mx,
                             text_delta=tm_d))

    out = {"viewport": VIEW, "ref": ref, "rows": rows}
    Path("results/web/web_gap_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 results/web/web_gap_results.json")


if __name__ == "__main__":
    main()
