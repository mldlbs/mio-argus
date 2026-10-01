"""
web_action_gap：动作语义的差别 —— element.click vs 真坐标点击

真实 CUA 只有坐标（截图 → 点 x,y），没有 DOM，所以它必须自己处理：
元素会移动、会被遮挡、位置会随 hover 变化、会持续动画。
Playwright 的 `locator.click()` 是**特权通道**（会自动等待元素稳定、检测遮挡），
会让这些难点消失——训练时用它，等于把真实难点藏起来。

三种动作：
  A  locator.click(sel)                     特权：元素感知
  B  mouse.click(cx, cy)                    真实：坐标（用点击前测得的中心）
  C  mouse.move→wait→重测中心→mouse.click   真实+稳健：hover 后重定位再点

指标：每种 widget 上"预期 handler 是否触发"。
用法:
    python experiments/web/web_action_gap.py
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

WIDGETS = {
    "simple": """
      <button id="t" onclick="window.fired=true"
        style="width:120px;height:40px">OK</button>""",
    "shifting": """
      <button id="t" onmouseover="this.style.transform='translateX(70px)'"
        onclick="window.fired=true"
        style="width:120px;height:40px;margin-left:10px">Move on hover</button>""",
    "overlay": """
      <div style="position:relative;width:120px;height:40px">
        <button id="t" onclick="window.fired=true"
          style="width:120px;height:40px">Covered</button>
        <div style="position:absolute;inset:0;background:rgba(0,0,0,0.02)"></div>
      </div>""",
    "tiny": """
      <button id="t" onclick="window.fired=true"
        style="width:6px;height:6px;padding:0"></button>""",
    "animating": """
      <style>@keyframes mv{from{transform:translateX(0)}to{transform:translateX(90px)}}
        #t{animation:mv .6s infinite alternate}</style>
      <button id="t" onclick="window.fired=true"
        style="width:110px;height:40px">Animating</button>""",
    "scroll_off": """
      <div style="height:1000px;background:linear-gradient(#eef,#eef)"></div>
      <button id="t" onclick="window.fired=true"
        style="width:120px;height:40px">Far below</button>""",
}
PAGE = ("<!doctype html><html><head><meta charset='utf-8'></head>"
        "<body style='margin:20px'>{w}</body></html>")


def center(pg, sel):
    return pg.eval_on_selector(
        sel, "e=>{const r=e.getBoundingClientRect();"
             "return [r.x+r.width/2, r.y+r.height/2]}")


def attempt(pg, sel, mode):
    pg.evaluate("()=>{window.fired=false}")
    try:
        if mode == "A":
            pg.click(sel, timeout=2500)
        elif mode == "B":
            cx, cy = center(pg, sel)
            pg.mouse.click(cx, cy)
        elif mode == "C":
            cx, cy = center(pg, sel)
            pg.mouse.move(cx, cy)
            pg.wait_for_timeout(180)
            cx2, cy2 = center(pg, sel)      # hover 后重定位
            pg.mouse.click(cx2, cy2)
    except Exception as e:
        return False, type(e).__name__
    pg.wait_for_timeout(60)
    return bool(pg.evaluate("()=>window.fired")), ""


def main():
    modes = ["A", "B", "C"]
    rows = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, w in WIDGETS.items():
            pg = b.new_page(viewport={"width": 320, "height": 200})
            pg.set_content(PAGE.format(w=w))
            res = {}
            for m in modes:
                ok, err = attempt(pg, "#t", m)
                res[m] = ok
                if not ok and err:
                    res[m + "_err"] = err
            pg.close()
            rows[name] = res
        b.close()

    print(f"{'widget':<12}{'A locator':>12}{'B coord':>10}{'C hover+reloc':>16}")
    print("-" * 50)
    for name, r in rows.items():
        print(f"{name:<12}{str(r['A']):>12}{str(r['B']):>10}{str(r['C']):>16}"
              + (f"   A_err={r.get('A_err')}" if r.get("A_err") else ""))
    print("\n结论：A（元素感知）与 B/C（坐标）不一致的格子 = 特权通道藏起来的难点")
    Path("results/web/web_action_gap_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 results/web/web_action_gap_results.json")


if __name__ == "__main__":
    main()
