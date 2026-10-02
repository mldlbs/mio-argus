"""
QGIS W25：复杂示例 —— 真实地理数据的主题地图（分类设色 → Print Layout → 300dpi PNG）

W23/W24 的链止于"数据产物"（GeoJSON）；本轮继续往上爬一层到**制图产物**：
真实数据 → 按属性分类设色（主题）→ 布局（地图项 + 图例 + 标题，数值定位）→
导出 300 dpi PNG。全链真实 UI（map_gui.py 沉淀机械）：

  1 categorized    roads 按 type 分类渲染（W23 机械）
  2 new_layout     Project → New Print Layout…
  3 add_map        单击 → 数值 rect (8,26,195,176) → Set to Map Canvas Extent
  4 add_label      单击 → (8,6,150,14) → 文本"中国公路等级专题图"
  5 add_legend     单击 → (232,26,57,90)
  6 export_image   Export as Image… → 保存路径（剪贴板）→ [覆盖确认] → 选项保存
  7 close_designer 丢弃未保存变更

真值（特权通道 bridge layouts op + 文件检查）：
  布局含 Label(文本匹配)/Legend/Map 三项；PNG 存在、>50KB、亮度有跨度（非空白）。

用法: python experiments/qgis/qgis_bridge/run_w25.py [--trials 3]
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
from scn_gui import categorized_via_gui  # noqa: E402
from run_w23 import classify_alg  # noqa: E402
from map_gui import (TITLE_TEXT, add_item, close_designer,  # noqa: E402
                     designer as find_designer, dismiss_prompts, export_image,
                     new_layout, set_extent_canvas, set_label_text, setup_view)

import pyautogui  # noqa: E402

OUT = REAL / "out"
RES = HERE / "w25_results.json"

RECT_MAP = (8, 26, 195, 176)
RECT_TITLE = (8, 6, 150, 14)
RECT_LEGEND = (232, 26, 57, 90)
CLICK = {"Add Map": (800, 500), "Add Label": (700, 250), "Add Legend": (1300, 320)}


def reset_env():
    call({"op": "close_dialogs"})
    pyautogui.moveTo(1280, 700)              # 防角落 fail-safe / 归位
    pyautogui.press("esc")
    time.sleep(0.3)
    d = find_designer(timeout=1.0)          # 失败试跑可能留下设计器 → 先关
    if d is not None:
        try:
            close_designer(d)
        except Exception:
            pass
        time.sleep(0.8)
    dismiss_prompts()                       # 残留保存提示（丢丢弃）
    call({"op": "reset", "scene": "real"})
    time.sleep(0.6)


def run_chain(out_p, steps):
    t0 = time.time()
    ok, st = categorized_via_gui("roads", "type")
    steps.append({"name": "categorized", "ok": bool(ok),
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
    txt = set_label_text(d, TITLE_TEXT) if okl else False
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


def truth_theme(out_p):
    lays = (call({"op": "layouts"}) or {}).get("layouts") or []
    lay = lays[0] if lays else None
    items = (lay or {}).get("items") or []
    types = [i.get("type") for i in items]
    label_ok = any(i.get("type") == "QgsLayoutItemLabel"
                   and TITLE_TEXT in (i.get("text") or "") for i in items)
    det = {"n_layouts": len(lays), "items": types, "label_ok": label_ok,
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
    ok = bool(lay) and label_ok and det["legend"] and det["map"] \
        and sz > 50000 and spread > 100
    return ok, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    call({"op": "win", "maximize": True})
    rows = []
    print("== w25 主题地图（分类 → 布局 → 图例/标题 → 300dpi PNG）==")
    for i in range(args.trials):
        out_p = OUT / f"w25_t{i}.png"
        reset_env()
        steps = []
        t0 = time.time()
        try:
            ok, fail = run_chain(out_p, steps)
        except Exception as e:
            ok, fail = False, "chain_exception"
            steps.append({"name": "chain", "ok": False, "err": str(e)})
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
        "step_ms_median": {
            s["name"]: round(statistics.median(
                [st["ms"] for r in succ for st in r["steps"] if st["name"] == s["name"]]), 1)
            for s in succ[0]["steps"]} if succ else None,
        "fails": fails,
    }
    print(f"w25: {summary['pass']}/{summary['trials']}"
          + (f"  中位 {summary['total_s_median']}s" if succ else "")
          + (f"  失败 {fails}" if fails else ""))
    RES.write_text(json.dumps({"meta": {"trials": args.trials,
                                        "title": TITLE_TEXT,
                                        "rects": {"map": RECT_MAP,
                                                  "title": RECT_TITLE,
                                                  "legend": RECT_LEGEND}},
                               "rows": rows, "summary": summary},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w25_results.json")


if __name__ == "__main__":
    main()
