"""
QGIS W24：真实地理数据场景集（Natural Earth 中国 ROI）执行基线

W23 的 6 场景跑在程序化"纸面几何"上；本轮同型场景换**真实地理数据**：
Natural Earth v5.1.2（公有领域）→ 中国 ROI，由 fetch_real_data.py 切片并投影
EPSG:3857（与 W23 同 CRS/米制缓冲语义）。真值在取数时由 shapely（GEOS）
一次算好 → `artifacts/qgis_real/real_truth.json`。

场景（数据 `artifacts/qgis_real/`，gitignored；rerun 前先 fetch_real_data.py）：
  r1 专题渲染  roads 11967 条按 type 分类（4 类）
  r2 选择导出  人口 > 1000 万的城市（8 个，bbox 内含周边国家）
  r3 缓冲      airports 37 个 × 300 km → 导出
  r4 组合      Buffer + Extract：被机场服务圈覆盖的省份（34/42）
  r5 空间查询  区域内城市 ∩ 省级行政区（397/1047）
  r6 空间查询  区域内港口 ∩ 省级行政区（15/87）

与 W23 相同的失败分层与分段计时；结果 → w24_results.json（就地）。

用法: python experiments/qgis/qgis_bridge/run_w24.py [--trials 3] [--only r1,r2]
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
from run_w16 import export_via_gui, features_of  # noqa: E402
from run_w18 import select_via_gui  # noqa: E402
from run_w23 import classify_alg, classify_export, do_export  # noqa: E402
from scn_gui import (buffer_via_gui, categorized_via_gui,  # noqa: E402
                     extract_via_gui, resolve_layer, truth_layers)

import pyautogui  # noqa: E402

TRUTH = json.loads((REAL / "real_truth.json").read_text(encoding="utf-8"))
OUT = REAL / "out"
RES = HERE / "w24_results.json"

INSTRUCTIONS = {
    "r1": "把 roads 图层按 type 道路等级做分类专题图",
    "r2": "筛出人口超过 1000 万的大城市，把选中要素导出为 GeoJSON",
    "r3": "给 airports 的每个机场建 300 km 服务半径缓冲区，输出并导出",
    "r4": "找出被机场 300 km 服务圈覆盖到的省份（provinces），结果导出",
    "r5": "找出区域内落在中国省级行政区内的城市（cities），结果导出",
    "r6": "找出区域内位于省级行政区内的港口（ports），结果导出",
}

LAYERS = {
    "r1": ["roads"],
    "r2": ["cities"],
    "r3": ["airports"],
    "r4": ["airports", "provinces"],
    "r5": ["cities", "provinces"],
    "r6": ["ports", "provinces"],
}


def reset_env():
    call({"op": "close_dialogs"})
    pyautogui.press("esc")
    time.sleep(0.4)
    call({"op": "reset", "scene": "real"})
    time.sleep(0.6)


def truth_names(out_p, key):
    got = set(features_of(out_p))
    want = set(TRUTH[key])
    return got == want, {"n": len(got), "want": len(want)}


# ------------------------------------------------------------------ chains ---
def chain_r1(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = categorized_via_gui("roads", "type")
    steps.append({"name": "categorized", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    return True, steps, None


def truth_r1():
    rend = next((l.get("renderer") for l in truth_layers()
                 if l["name"] == "roads"), None)
    cats = sorted(c for c in ((rend or {}).get("cats") or []) if c is not None)
    ok = bool(rend) and rend.get("type") == "categorizedSymbol" \
        and rend.get("attr") == "type" and cats == sorted(TRUTH["r1_types"])
    return ok, {"renderer": rend, "cats": cats}


def chain_r2(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = select_via_gui('"pop_max" > 10000000', expect_n=len(TRUTH["r2_megacities"]),
                            run_mode="invoke", layer="cities")
    steps.append({"name": "select", "ok": bool(ok),
                  "ms": round(st.get("total_ms", 0), 1), "st": st})
    if not ok:
        return False, steps, "select"
    ok2, fail, names = do_export(steps, "cities", out_p, selected_only=True)
    if not ok2:
        return False, steps, fail
    return True, steps, None


def chain_r3(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = buffer_via_gui("airports", TRUTH["buffer_m"],
                            expect_n=len(TRUTH["r3_airports"]))
    steps.append({"name": "buffer", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    ok2, fail, names = do_export(steps, "Buffered", out_p)
    if not ok2:
        return False, steps, fail
    return True, steps, None


def chain_r4(i, out_p):
    steps = []
    t0 = time.time()
    ok, st = buffer_via_gui("airports", TRUTH["buffer_m"],
                            expect_n=len(TRUTH["r3_airports"]))
    steps.append({"name": "buffer", "ok": bool(ok),
                  "ms": round((time.time() - t0) * 1000, 1), "st": st})
    if not ok:
        return False, steps, classify_alg(st)
    actual_buf = resolve_layer("Buffered") or "Buffered"
    t1 = time.time()
    ok2, ste = extract_via_gui("provinces", actual_buf)
    steps.append({"name": "extract", "ok": bool(ok2),
                  "ms": round((time.time() - t1) * 1000, 1), "st": ste})
    if not ok2:
        return False, steps, classify_alg(ste)
    ok3, fail, names = do_export(steps, "Extracted (location)", out_p)
    if not ok3:
        return False, steps, fail
    return True, steps, None


def make_chain_extract(src, ref):
    def chain(i, out_p):
        steps = []
        t0 = time.time()
        ok, st = extract_via_gui(src, ref)
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
    "r1": (chain_r1, truth_r1),
    "r2": (chain_r2, lambda p: truth_names(p, "r2_megacities")),
    "r3": (chain_r3, lambda p: truth_names(p, "r3_airports")),
    "r4": (chain_r4, lambda p: truth_names(p, "r4_provinces")),
    "r5": (make_chain_extract("cities", "provinces"),
           lambda p: truth_names(p, "r5_cities")),
    "r6": (make_chain_extract("ports", "provinces"),
           lambda p: truth_names(p, "r6_ports")),
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
    OUT.mkdir(parents=True, exist_ok=True)

    call({"op": "win", "maximize": True})
    rows = []
    for sid in sids:
        chain, truth_fn = SCENARIOS[sid]
        print(f"\n== {sid} {INSTRUCTIONS[sid][:30]}… ==")
        for i in range(args.trials):
            out_p = OUT / f"{sid}_t{i}.geojson"
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
                    ok_t, detail = (truth_fn(out_p) if sid != "r1"
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
                    "dataset": "natural-earth-v5.1.2-china-roi",
                    "truth_counts": TRUTH["counts"],
                    "instructions": INSTRUCTIONS, "layers": LAYERS},
           "rows": rows, "summary": summary}
    RES.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w24_results.json")


if __name__ == "__main__":
    main()
