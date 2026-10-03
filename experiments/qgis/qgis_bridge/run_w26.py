"""
QGIS W26：主题表达族 —— 同一批数据的多种专题地图（分类 / 分级设色 / 比例符号 / 热力图）

"根据业务指标把空间对象用颜色、符号、大小、密度表达出来"：同一份 W24 真实数据，
四种渲染机制各出一张主题地图（复用 W25 制图模板：地图 + 图例 + 标题 → 300 dpi PNG）。

  v1 分类图    roads 按 type        → categorizedSymbol（W23 机械）
  v2 分级设色  cities 按 pop_max    → graduatedSymbol（method=Color, Quantile 5 级）
  v3 比例符号  cities 按 pop_max    → graduatedSymbol（method=Size,  Quantile 5 级）
  v4 热力图    cities 点密度        → heatmapRenderer（radius 10）

真值：renderer 段（type / attr / method / ranges，bridge truth）+ 布局项（layouts op）
+ PNG 文件/像素检查；失败分层与分段计时沿用 W23–W25。
交互机械：scn_gui.categorized_via_gui + scn_theme.graduated/heatmap + map_gui 布局链。

用法: python experiments/qgis/qgis_bridge/run_w26.py [--trials 3] [--only v1,v2]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
REAL = ROOT / "artifacts" / "qgis_real"
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from scn_gui import categorized_via_gui, truth_layers  # noqa: E402
from scn_theme import graduated_via_gui, heatmap_via_gui  # noqa: E402
from run_w23 import classify_alg  # noqa: E402
from map_gui import (add_item, close_designer,  # noqa: E402
                     designer as find_designer, dismiss_prompts, export_image,
                     new_layout, set_extent_canvas, set_label_text, setup_view)

import pyautogui  # noqa: E402

TRUTH = json.loads((REAL / "real_truth.json").read_text(encoding="utf-8"))
OUT = REAL / "out"
RES = HERE / "w26_results.json"

RECT_MAP = (8, 26, 195, 176)
RECT_TITLE = (8, 6, 150, 14)
RECT_LEGEND = (232, 26, 57, 90)
CLICK = {"Add Map": (800, 500), "Add Label": (700, 250), "Add Legend": (1300, 320)}


def _cats4(r):
    return sorted(c for c in (r.get("cats") or []) if c is not None)


THEMES = {
    "v1": {
        "layer": "roads", "title": "中国公路等级分类专题图",
        "apply": lambda: categorized_via_gui("roads", "type"),
        "check": lambda r: bool(r) and r.get("type") == "categorizedSymbol"
        and r.get("attr") == "type"
        and _cats4(r) == sorted(TRUTH["r1_types"]),
    },
    "v2": {
        "layer": "cities", "title": "中国城市人口分级设色专题图",
        "apply": lambda: graduated_via_gui("cities", "pop_max",
                                           method="color", mode="quantile", classes=5),
        "check": lambda r: bool(r) and r.get("type") == "graduatedSymbol"
        and r.get("attr") == "pop_max" and r.get("method") == "color"
        and r.get("ranges") == 5,
    },
    "v3": {
        "layer": "cities", "title": "中国城市人口比例符号专题图",
        "apply": lambda: graduated_via_gui("cities", "pop_max",
                                           method="size", mode="quantile", classes=5),
        "check": lambda r: bool(r) and r.get("type") == "graduatedSymbol"
        and r.get("attr") == "pop_max" and r.get("method") == "size"
        and r.get("ranges") == 5,
    },
    "v4": {
        "layer": "cities", "title": "中国城市分布热力图",
        "apply": lambda: heatmap_via_gui("cities", radius=10),
        "check": lambda r: bool(r) and r.get("type") == "heatmapRenderer",
    },
}


def reset_env():
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
        time.sleep(0.8)
    dismiss_prompts()
    call({"op": "reset", "scene": "real"})
    time.sleep(0.6)


def run_chain(tid, out_p, steps):
    th = THEMES[tid]
    t0 = time.time()
    ok, st = th["apply"]()
    steps.append({"name": "renderer", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, classify_alg(st)

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
    txt = set_label_text(d, th["title"]) if okl else False
    steps.append({"name": "add_label", "ok": okl and txt,
                  "ms": round((time.time() - t0) * 1000, 1),
                  "st": {**stl, "text": txt}})
    if not (okl and txt):
        return False, "label"

    t0 = time.time()
    okg, stg = add_item(d, "Add Legend", CLICK["Add Legend"], RECT_LEGEND)
    steps.append({"name": "add_legend", "ok": okg,
                  "ms": round((time.time() - t0) * 1000, 1), "st": stg})
    if not okg:
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


def truth_theme(tid, out_p):
    th = THEMES[tid]
    rend = next((l.get("renderer") for l in truth_layers()
                 if l["name"] == th["layer"]), None)
    rend_ok = th["check"](rend)
    lays = (call({"op": "layouts"}) or {}).get("layouts") or []
    lay = lays[0] if lays else None
    items = (lay or {}).get("items") or []
    types = [i.get("type") for i in items]
    label_ok = any(i.get("type") == "QgsLayoutItemLabel"
                   and th["title"] in (i.get("text") or "") for i in items)
    det = {"renderer": rend, "render_ok": rend_ok, "n_layouts": len(lays),
           "items": types, "label_ok": label_ok,
           "legend": "QgsLayoutItemLegend" in types,
           "map": "QgsLayoutItemMap" in types}
    png = Path(out_p)
    if not png.exists():
        return False, {**det, "err": "no png"}
    sz = png.stat().st_size
    spread = 0
    try:
        from PIL import Image
        im = Image.open(png)
        g = im.convert("L")
        step = max(1, (im.size[0] * im.size[1]) // 4000)
        vals = list(g.getdata())[::step]
        spread = max(vals) - min(vals)
        det.update({"png_bytes": sz, "png_size": list(im.size),
                    "luma_spread": int(spread)})
    except Exception as e:
        det.update({"err": f"png check: {e}"})
    ok = rend_ok and bool(lay) and label_ok and det["legend"] and det["map"] \
        and sz > 50000 and spread > 100
    return ok, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    tids = [t for t in THEMES if (not only or t in only)]
    if not tids:
        print("没有匹配的主题:", args.only)
        return
    OUT.mkdir(parents=True, exist_ok=True)

    call({"op": "win", "maximize": True})
    rows = []
    for tid in tids:
        print(f"\n== {tid} {THEMES[tid]['title']} ==")
        for i in range(args.trials):
            out_p = OUT / f"w26_{tid}_t{i}.png"
            reset_env()
            steps = []
            t0 = time.time()
            try:
                ok, fail = run_chain(tid, out_p, steps)
            except Exception as e:
                ok, fail = False, "chain_exception"
                steps.append({"name": "chain", "ok": False, "err": str(e)})
            det = None
            if ok:
                try:
                    ok_t, det = truth_theme(tid, out_p)
                except Exception as e:
                    ok_t, det = False, {"err": str(e)}
                if not ok_t:
                    ok, fail = False, "truth"
            total = round(time.time() - t0, 1)
            row = {"theme": tid, "trial": i + 1, "ok": bool(ok), "fail": fail,
                   "total_s": total, "steps": steps, "truth": det}
            rows.append(row)
            mark = "PASS" if ok else f"FAIL({fail})"
            seg = " ".join(f"{s['name']}={s.get('ms', 0) / 1000:.1f}s"
                           for s in steps if "ms" in s)
            print(f"  t{i + 1}: {mark} {total}s  {seg}")

    summary = {}
    for tid in tids:
        rs = [r for r in rows if r["theme"] == tid]
        succ = [r for r in rs if r["ok"]]
        fails = {}
        for r in rs:
            if not r["ok"]:
                fails[r["fail"] or "unknown"] = fails.get(r["fail"] or "unknown", 0) + 1
        summary[tid] = {
            "trials": len(rs), "pass": len(succ),
            "rate": round(len(succ) / len(rs), 3) if rs else None,
            "total_s_median": round(statistics.median(
                [r["total_s"] for r in succ]), 1) if succ else None,
            "fails": fails,
        }
        s = summary[tid]
        print(f"{tid}: {s['pass']}/{s['trials']}"
              + (f"  中位 {s['total_s_median']}s" if s["total_s_median"] else "")
              + (f"  失败 {fails}" if fails else ""))

    out = {"meta": {"trials": args.trials, "only": only,
                    "titles": {t: THEMES[t]["title"] for t in tids}},
           "rows": rows, "summary": summary}
    RES.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w26_results.json")


if __name__ == "__main__":
    main()
