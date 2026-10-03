"""
QGIS W27：暗色科技风主题地图（视觉重构）——"智慧城市数字孪生"视觉层级

同一份真实数据（W24）上的**深色指挥中心**视觉：
  暗色画布 → 基础层弱化（省界深藏青 / 道路暗蓝 / 机场·港口灰蓝）→
  核心层 cities 用内置 **Inferno** 渐进色带（黑→紫→橙，冷端到暖端的风险渐变，
  避免纯红满屏）→ 制图（满幅地图 + HTML 亮色标题 + 图例亮色字体）→ 300 dpi PNG。

链条（全真实 UI；机械在 scn_style.py，制图在 map_gui.py）：
  样式段：canvas bg #0A1220 / provinces #16283F / roads #2C4A6E /
          airports #4A6B8A / ports #556B8C / cities graduated+Inferno(Quantile 5)
  制图段：new_layout → map(0,0,297,210 满幅) + Set to Map Canvas Extent →
          HTML 标题"中国城市人口分级 · 数字孪生主题图" → legend →
          图例字体 #DCE9F7 → Export as Image → PNG → close

真值：cities renderer（graduated/attr/method/ranges）+ 布局项（HTML 标题）+
      canvas bg（#0a1220）+ PNG 检查（>50KB、**平均亮度<140 暗色**、跨度>100）。

用法: python experiments/qgis/qgis_bridge/run_w27.py [--trials 3]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
REAL = ROOT / "artifacts" / "qgis_real"
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from map_gui import (add_item, close_designer,  # noqa: E402
                     designer as find_designer, dismiss_prompts, export_image,
                     new_layout, set_extent_canvas, setup_view)
from scn_style import (graduated_with_ramp, legend_style,  # noqa: E402
                       set_canvas_bg, set_layer_color)

import pyautogui  # noqa: E402

OUT = REAL / "out"
RES = HERE / "w27_results.json"
QGIS_BAT = r"D:\QGIS\bin\qgis-ltr.bat"
STARTUP = str(HERE / "_startup.py")

BG = "#0A1220"
PALETTE = {"provinces": "#16283F", "roads": "#2C4A6E",
           "airports": "#4A6B8A", "ports": "#556B8C"}
RAMP = "Inferno"
TITLE = "中国城市人口分级 · 数字孪生主题图"
TITLE_HTML = f'<span style="color:#E8F1FF; font-size:26pt">{TITLE}</span>'
FONT = "#DCE9F7"
BG_LEGEND = "#0F1E33"

RECT_MAP = (0, 0, 297, 210)
RECT_TITLE = (10, 8, 160, 16)
RECT_LEGEND = (230, 26, 60, 92)
CLICK = {"Add Map": (800, 500), "Add Label": (700, 250), "Add Legend": (1300, 320)}


def ensure_qgis(timeout=50.0):
    """QGIS 可能因 Qt 延迟析构在关设计器后几秒内崩溃 → 探测并自动重启。"""
    try:
        call({"op": "ping"}, timeout=3)
        return False
    except Exception:
        pass
    env = dict(os.environ, PYQGIS_STARTUP=STARTUP)
    subprocess.Popen(["cmd", "/c", QGIS_BAT], env=env,
                     creationflags=0x00000008)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2.5)
        try:
            call({"op": "ping"}, timeout=3)
            time.sleep(2.0)
            call({"op": "win", "maximize": True})
            return True
        except Exception:
            continue
    return False


def reset_env():
    ensure_qgis()
    call({"op": "close_dialogs"})
    pyautogui.moveTo(1280, 700)
    pyautogui.press("esc")
    time.sleep(0.3)
    d = find_designer(timeout=1.0)
    if d is not None:
        try:
            close_designer(d)
        except Exception:
            pass
        time.sleep(2.0)          # 让 Qt 完成布局项析构（否则可能异步崩溃）
        ensure_qgis()
    dismiss_prompts()
    call({"op": "reset", "scene": "real"})
    time.sleep(0.6)


def _activate_item_props(d):
    for t in d.descendants(control_type="TabItem"):
        try:
            if (t.window_text() or "").strip() == "Item Properties":
                t.click_input()
                time.sleep(0.5)
                return True
        except Exception:
            continue
    return False


def set_html_title(d):
    # 等 Item Properties 切到新 Label（Edit 显示 'Lorem ipsum'）→ 设 HTML 文本
    t0 = time.time()
    done = False
    while time.time() - t0 < 8.0 and not done:
        for c in d.descendants(control_type="Edit"):
            try:
                if (c.window_text() or "").strip() == "Lorem ipsum":
                    c.click_input()
                    time.sleep(0.2)
                    pyautogui.hotkey("ctrl", "a")
                    time.sleep(0.15)
                    c.set_edit_text(TITLE_HTML)
                    time.sleep(0.4)
                    done = True
                    break
            except Exception:
                continue
        if not done:
            time.sleep(0.4)
    if not done:
        return False
    # 勾选 Render as HTML（按名字找，不做位置过滤——面板可能仍在屏外）
    t0 = time.time()
    while time.time() - t0 < 4.0:
        for c in d.descendants(control_type="CheckBox"):
            try:
                if (c.window_text() or "").strip() == "Render as HTML":
                    try:
                        if not c.get_toggle_state():
                            c.toggle()
                    except Exception:
                        c.invoke()
                    time.sleep(0.4)
                    return True
            except Exception:
                continue
        time.sleep(0.3)
    return False


def run_chain(out_p, steps):
    # 样式段
    t0 = time.time()
    ok_bg = set_canvas_bg(BG)
    steps.append({"name": "canvas_bg", "ok": bool(ok_bg),
                  "ms": round((time.time() - t0) * 1000, 1),
                  "st": {"hex": BG}})
    if not ok_bg:
        return False, "style_bg"

    for layer, hexv in PALETTE.items():
        t0 = time.time()
        ok = set_layer_color(layer, hexv)
        steps.append({"name": f"color_{layer}", "ok": bool(ok),
                      "ms": round((time.time() - t0) * 1000, 1),
                      "st": {"hex": hexv}})
        if not ok:
            return False, f"style_{layer}"

    t0 = time.time()
    ok, st = graduated_with_ramp("cities", "pop_max", RAMP, classes=5)
    steps.append({"name": "cities_inferno", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, "style_cities"

    # 制图段
    t0 = time.time()
    d, st = new_layout()
    steps.append({"name": "new_layout", "ok": d is not None,
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if d is None:
        return False, "layout"
    setup_view(d)

    t0 = time.time()
    okm, stm = add_item(d, "Add Map", CLICK["Add Map"], RECT_MAP)
    ext = set_extent_canvas(d) if okm else False
    steps.append({"name": "add_map", "ok": okm and ext,
                  "ms": round((time.time() - t0) * 1000, 1),
                  "st": {**stm, "extent": ext}})
    if not (okm and ext):
        return False, "map"

    t0 = time.time()
    okl, stl = add_item(d, "Add Label", CLICK["Add Label"], RECT_TITLE)
    txt = False
    if okl:
        _activate_item_props(d)
        txt = set_html_title(d)
    steps.append({"name": "add_title", "ok": okl and txt,
                  "ms": round((time.time() - t0) * 1000, 1),
                  "st": {**stl, "html": txt}})
    if not (okl and txt):
        return False, "title"

    t0 = time.time()
    okg, stg = add_item(d, "Add Legend", CLICK["Add Legend"], RECT_LEGEND)
    fnt, stf = False, {}
    if okg:
        _activate_item_props(d)
        fnt, stf = legend_style(BG_LEGEND, FONT)
    steps.append({"name": "add_legend", "ok": okg and fnt,
                  "ms": round((time.time() - t0) * 1000, 1),
                  "st": {**stg, "style": stf}})
    if not (okg and fnt):
        return False, "legend"

    t0 = time.time()
    okx, stx = export_image(d, out_p)
    steps.append({"name": "export", "ok": bool(okx),
                  "ms": round((time.time() - t0) * 1000, 1), "st": stx})
    if not okx:
        return False, "export"

    t0 = time.time()
    cl = close_designer(d)
    steps.append({"name": "close", "ok": bool(cl),
                  "ms": round((time.time() - t0) * 1000, 1), "st": {}})
    return True, None


def truth_theme(out_p):
    t = call({"op": "truth"}) or {}
    truth = t.get("truth") or {}
    rend = next((l.get("renderer") for l in truth.get("layers", [])
                 if l["name"] == "cities"), None)
    rend_ok = bool(rend) and rend.get("type") == "graduatedSymbol" \
        and rend.get("attr") == "pop_max" and rend.get("method") == "color" \
        and rend.get("ranges") == 5
    bg = truth.get("bg")
    lays = (call({"op": "layouts"}) or {}).get("layouts") or []
    lay = lays[0] if lays else None
    items = (lay or {}).get("items") or []
    types = [i.get("type") for i in items]
    label_ok = any(i.get("type") == "QgsLayoutItemLabel"
                   and TITLE in (i.get("text") or "") for i in items)
    det = {"renderer": rend, "render_ok": rend_ok, "bg": bg,
           "n_layouts": len(lays), "items": types, "label_ok": label_ok,
           "legend": "QgsLayoutItemLegend" in types,
           "map": "QgsLayoutItemMap" in types}
    png = Path(out_p)
    if not png.exists():
        return False, {**det, "err": "no png"}
    sz = png.stat().st_size
    mean_luma = None
    spread = 0
    try:
        from PIL import Image
        im = Image.open(png)
        g = im.convert("L")
        step = max(1, (im.size[0] * im.size[1]) // 6000)
        vals = list(g.getdata())[::step]
        mean_luma = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        det.update({"png_bytes": sz, "png_size": list(im.size),
                    "mean_luma": round(mean_luma, 1),
                    "luma_spread": int(spread)})
    except Exception as e:
        det.update({"err": f"png check: {e}"})
    dark_ok = mean_luma is not None and mean_luma < 140
    ok = rend_ok and (bg == "#0a1220") and bool(lay) and label_ok \
        and det["legend"] and det["map"] and sz > 50000 and spread > 100 and dark_ok
    return ok, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ensure_qgis()
    try:
        call({"op": "win", "maximize": True})
    except Exception:
        pass
    rows = []
    print("== w27 暗色科技风主题地图（数字孪生视觉）==")
    for i in range(args.trials):
        out_p = OUT / f"w27_t{i}.png"
        reset_env()
        steps = []
        t0 = time.time()
        try:
            ok, fail = run_chain(out_p, steps)
        except Exception as e:
            ok, fail = False, "chain_exception"
            steps.append({"name": "chain", "ok": False, "err": str(e)})
        if not ok:  # 失败也清理设计器，避免残留状态影响下一试（并避免脏关闭崩溃）
            try:
                dd = find_designer(timeout=1.0)
                if dd is not None:
                    dismiss_prompts()
                    close_designer(dd)
                    dismiss_prompts()
                    time.sleep(2.0)
            except Exception:
                pass
            ensure_qgis()
        det = None
        if ok:
            try:
                ok_t, det = truth_theme(out_p)
            except Exception as e:
                ok_t, det = False, {"err": str(e)}
            if not ok_t:
                ok, fail = False, "truth"
        total = round(time.time() - t0, 1)
        row = {"trial": i + 1, "ok": bool(ok), "fail": fail,
               "total_s": total, "steps": steps, "truth": det}
        rows.append(row)
        mark = "PASS" if ok else f"FAIL({fail})"
        seg = " ".join(f"{s['name']}={s.get('ms', 0) / 1000:.1f}s"
                       for s in steps if "ms" in s)
        print(f"  t{i + 1}: {mark} {total}s  {seg}")

    succ = [r for r in rows if r["ok"]]
    fails = {}
    for r in rows:
        if not r["ok"]:
            fails[r["fail"] or "unknown"] = fails.get(r["fail"] or "unknown", 0) + 1
    summary = {
        "trials": len(rows), "pass": len(succ),
        "rate": round(len(succ) / len(rows), 3) if rows else None,
        "total_s_median": round(statistics.median([r["total_s"] for r in succ]), 1)
        if succ else None,
        "fails": fails,
    }
    print(f"w27: {summary['pass']}/{summary['trials']}"
          + (f"  中位 {summary['total_s_median']}s" if succ else "")
          + (f"  失败 {fails}" if fails else ""))
    RES.write_text(json.dumps({"meta": {"trials": args.trials, "bg": BG,
                                        "palette": PALETTE, "ramp": RAMP,
                                        "title": TITLE},
                               "rows": rows, "summary": summary},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w27_results.json")


if __name__ == "__main__":
    main()
