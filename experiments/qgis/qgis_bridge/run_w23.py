"""
QGIS W23：场景化任务集第一批（6 个矢量场景）执行基线

W16–W19 的单点链（选择/导出）与 W23 新增构建块（scn_gui：缓冲/提取/分类渲染）
合成为 6 个**具体 GIS 业务场景**，每场景 N 次独立试跑，产出分段计时与失败分层。
本轮只立"执行基线"（机械可靠性 + 分段耗时），场景指令文本后续用于训练采集。

场景（数据 `data/scenario/`，真值 `data/scenario_truth.json`）：
  s1  专题渲染  roads 按 class 分类（4 类）
  s2  选择导出  筛 class=primary 的 6 条道路 → 导出选中要素
  s3  缓冲      hospitals 5 点 × 150 km 缓冲 → 导出
  s4  组合      缓冲 + Extract by location（被覆盖的 regions）→ 导出
  s5  空间查询  pois 落在 regions 内 → 导出
  s6  空间查询  floods 落在 regions 内 → 导出

失败分层：open（对话框没出）/ param（参数没选上）/ run（算法没跑成）/
          select（表达式选择失败）/ export_menu（菜单导航）/ export_dialog（导出对话框）/
          truth（跑完了但结果对不上）

用法: python experiments/qgis/qgis_bridge/run_w23.py [--trials 3] [--only s1,s2]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w16 import export_via_gui, features_of  # noqa: E402
from run_w18 import select_via_gui  # noqa: E402
from scn_gui import (buffer_via_gui, categorized_via_gui,  # noqa: E402
                     extract_via_gui, resolve_layer, truth_layers)

import pyautogui  # noqa: E402

TRUTH = json.loads((HERE / "data" / "scenario_truth.json").read_text(encoding="utf-8"))
OUT = HERE / "data"
RES = HERE / "w23_results.json"

INSTRUCTIONS = {
    "s1": "把 roads 图层按 class 字段做分类专题图（Categorized，4 类），"
          "让不同等级道路显示为不同符号",
    "s2": "从 roads 里筛出 class=primary 的主干道（6 条），把选中要素导出为 GeoJSON",
    "s3": "给 hospitals 的 5 个医院点各建 150 km 服务半径缓冲区，输出缓冲面并导出",
    "s4": "找出被医院 150 km 服务区覆盖到的行政区（regions），结果导出",
    "s5": "把落在行政区范围内的 POI 提取出来，导出清单",
    "s6": "把位于行政区内的内涝积水点提取出来，导出清单",
}

LAYERS = {
    "s1": ["roads"],
    "s2": ["roads"],
    "s3": ["hospitals"],
    "s4": ["hospitals", "regions"],
    "s5": ["pois", "regions"],
    "s6": ["floods", "regions"],
}


# ---------------------------------------------------------------- helpers ---
def reset_env():
    call({"op": "close_dialogs"})
    pyautogui.press("esc")
    time.sleep(0.4)
    call({"op": "reset", "scene": "scenario"})
    time.sleep(0.6)


def classify_alg(st):
    err = str(st.get("err") or "")
    if err in ("alg not found", "dialog not found", "no Run window") \
            or st.get("open") in ("alg not found", "dialog not found"):
        return "open"
    if err in ("no layer item", "no properties dialog"):
        return "open"
    for k in ("pick", "dist", "pick_in", "pick_ref", "rend", "val"):
        if st.get(k) is False:
            return "param"
    if st.get("ok") is False:
        return "run"
    return "chain"


def classify_export(note):
    n = str(note or "")
    if "menu" in n or "Export" in n or n.startswith("no Save"):
        return "export_menu"
    if "no window" in n or "no layer item" in n:
        return "export_env"
    return "export_dialog"


def do_export(steps, layer, out_p, selected_only=False, src_layer=None):
    actual = resolve_layer(layer) or layer
    p = Path(out_p)
    if p.exists():
        p.unlink()
    t0 = time.time()
    ok, st, note = export_via_gui(p, selected_only, layer=actual)
    steps.append({"name": "export", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1),
                  "note": note, "layer": actual, "st": st})
    if not ok:
        return False, classify_export(note), sorted(features_of(p))
    return True, None, sorted(features_of(p))


def truth_export(names, want_key):
    want = sorted(TRUTH[want_key])
    return names == want, {"n": len(names), "want": len(want)}


# ------------------------------------------------------------------ chains ---
def chain_s1(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = categorized_via_gui("roads", TRUTH["s1_field"])
    steps.append({"name": "categorized", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    return True, steps, None


def truth_s1():
    rend = next((l.get("renderer") for l in truth_layers()
                 if l["name"] == "roads"), None)
    cats = sorted(c for c in ((rend or {}).get("cats") or []) if c is not None)
    want = sorted(TRUTH["s1_classes"])
    ok = bool(rend) and rend.get("type") == "categorizedSymbol" \
        and rend.get("attr") == TRUTH["s1_field"] and cats == want
    return ok, {"renderer": rend, "cats": cats}


def chain_s2(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = select_via_gui('"class" = \'primary\'', expect_n=6,
                            run_mode="invoke", layer="roads")
    steps.append({"name": "select", "ok": bool(ok),
                  "ms": round(st.get("total_ms", 0), 1), "st": st})
    if not ok:
        return False, steps, "select"
    ok2, fail, names = do_export(steps, "roads", out_p, selected_only=True)
    if not ok2:
        return False, steps, fail
    return True, steps, None


def truth_s2(out_p):
    return truth_export(sorted(features_of(out_p)), "s2_primary")


def chain_s3(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = buffer_via_gui("hospitals", TRUTH["buffer_r"])
    steps.append({"name": "buffer", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    ok2, fail, names = do_export(steps, "Buffered", out_p)
    if not ok2:
        return False, steps, fail
    return True, steps, None


def truth_s3(out_p):
    return truth_export(sorted(features_of(out_p)), "s3_hospitals")


def chain_s4(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = buffer_via_gui("hospitals", TRUTH["buffer_r"])
    steps.append({"name": "buffer", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    actual_buf = resolve_layer("Buffered") or "Buffered"
    t1 = time.time()
    ok2, ste = extract_via_gui("regions", actual_buf)
    steps.append({"name": "extract", "ok": bool(ok2),
                  "ms": round((time.time() - t1) * 1000, 1), "st": ste})
    if not ok2:
        return False, steps, classify_alg(ste)
    ok3, fail, names = do_export(steps, "Extracted (location)", out_p)
    if not ok3:
        return False, steps, fail
    return True, steps, None


def truth_s4(out_p):
    return truth_export(sorted(features_of(out_p)), "s4_covered")


def make_chain_extract(src):
    def chain(i, out_p):
        steps = []
        t0 = time.time()
        ok, st = extract_via_gui(src, "regions")
        steps.append({"name": "extract", "ok": bool(ok),
                      "ms": round((time.time() - t0) * 1000, 1), "st": st})
        if not ok:
            return False, steps, classify_alg(st)
        ok2, fail, names = do_export(steps, "Extracted (location)", out_p)
        if not ok2:
            return False, steps, fail
        return True, steps, None
    return chain


SCENARIOS = {
    "s1": (chain_s1, truth_s1),
    "s2": (chain_s2, truth_s2),
    "s3": (chain_s3, truth_s3),
    "s4": (chain_s4, truth_s4),
    "s5": (make_chain_extract("pois"),
           lambda p: truth_export(sorted(features_of(p)), "s5_pois_inside")),
    "s6": (make_chain_extract("floods"),
           lambda p: truth_export(sorted(features_of(p)), "s6_floods_inside")),
}


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    sids = [s for s in SCENARIOS if (not only or s in only)]
    if not sids:
        print("没有匹配的场景:", args.only)
        return

    call({"op": "win", "maximize": True})
    rows = []
    for sid in sids:
        chain, truth_fn = SCENARIOS[sid]
        print(f"\n== {sid} {INSTRUCTIONS[sid][:30]}… ==")
        for i in range(args.trials):
            out_p = OUT / f"w23_{sid}_t{i}.geojson"
            reset_env()
            t0 = time.time()
            try:
                ok, steps, fail = chain(i, out_p)
            except Exception as e:
                ok, steps, fail = False, [], "chain_exception"
                steps.append({"name": "chain", "ok": False, "err": str(e)})
            detail = None
            if ok:
                try:
                    ok_t, detail = (truth_fn(out_p) if sid != "s1"
                                    else truth_fn())
                except Exception as e:
                    ok_t, detail = False, {"err": str(e)}
                if not ok_t:
                    ok, fail = False, "truth"
            total = round(time.time() - t0, 1)
            row = {"scenario": sid, "trial": i + 1, "ok": bool(ok),
                   "fail": fail, "total_s": total, "steps": steps,
                   "truth": detail}
            rows.append(row)
            mark = "PASS" if ok else f"FAIL({fail})"
            step_ms = " ".join(f"{s['name']}={s.get('ms', 0) / 1000:.1f}s"
                               for s in steps if "ms" in s)
            print(f"  t{i + 1}: {mark} {total}s  {step_ms}")

    # 汇总
    summary = {}
    for sid in sids:
        rs = [r for r in rows if r["scenario"] == sid]
        succ = [r for r in rs if r["ok"]]
        fails = {}
        for r in rs:
            if not r["ok"]:
                fails[r["fail"] or "unknown"] = fails.get(r["fail"] or "unknown", 0) + 1
        summary[sid] = {
            "trials": len(rs), "pass": len(succ),
            "rate": round(len(succ) / len(rs), 3) if rs else None,
            "total_s_median": round(statistics.median(
                [r["total_s"] for r in succ]), 1) if succ else None,
            "fails": fails,
        }
        s = summary[sid]
        print(f"{sid}: {s['pass']}/{s['trials']}"
              + (f"  中位 {s['total_s_median']}s" if s["total_s_median"] else "")
              + (f"  失败 {fails}" if fails else ""))

    out = {"meta": {"trials": args.trials, "only": only,
                    "instructions": INSTRUCTIONS, "layers": LAYERS},
           "rows": rows, "summary": summary}
    RES.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w23_results.json")


if __name__ == "__main__":
    main()
