"""
QGIS W15：像素 → 结构 —— 用 OCR 读出菜单里的快捷键，并改用快捷键执行

W14 结论：快捷键在 UIA 里**不可发现**（AcceleratorKey 空、HelpText 空），
只有菜单里画出来的 "Ctrl+J" 是像素。本轮把这条路走通：

  1) 全菜单扫描（UIA 开菜单）+ **OCR 快捷键列** → (动作 → 快捷键) 表；
     用桥的**特权 shortcuts** 做真值校验（只用于验证，不参与 agent）。
  2) 按快捷键模式过滤 OCR 噪声（必须含 '+' 或为 F 键/单词）。
  3) **用快捷键执行**若干动作（pyautogui），真值判定 + 计时。
  4) live 链（60 s，纯快捷键）：Zoom to Selection → Zoom Full → Zoom to Selection。

用法: python experiments/qgis/qgis_bridge/run_w15.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402
from PIL import Image  # noqa: E402
from rapidocr_onnxruntime import RapidOCR  # noqa: E402

SCENE = "regions"
SEL_EXPR = '"area" > 400'
SHORTCUT_RE = re.compile(r"^(F\d{1,2}|(Ctrl|Alt|Shift|Meta)(\+(Ctrl|Alt|Shift|Meta))*\+.{1,4})$",
                         re.IGNORECASE)


def truth():
    return call({"op": "truth"})["truth"]


def extent_x0():
    return truth()["extent"][0]


def qgis():
    d = Desktop(backend="uia")
    for w in d.windows():
        try:
            if "QGIS" in (w.window_text() or ""):
                return w
        except Exception:
            continue
    return None


def menu_items(qg):
    out = []
    for m in qg.descendants(control_type="MenuItem"):
        try:
            t = (m.window_text() or "").strip()
            r = m.rectangle()
        except Exception:
            continue
        if t and r.width() > 0 and m.parent().element_info.control_type in ("MenuBar",):
            out.append(t)
    return out


def open_menu(qg, name):
    for m in qg.descendants(control_type="MenuItem"):
        try:
            if (m.window_text() or "").strip() == name:
                m.click_input()
                return True
        except Exception:
            continue
    return False


def visible_new_items(qg, base):
    out = []
    for x in qg.descendants(control_type="MenuItem"):
        try:
            t = (x.window_text() or "").strip()
            r = x.rectangle()
            if t and t not in base and r.width() > 10 and x.is_visible():
                out.append((t, (r.left, r.top, r.width(), r.height())))
        except Exception:
            continue
    return out


def norm(s):
    return re.sub(r"\s+", "", s or "")


def parse_shortcut(s):
    """'Ctrl+Shift+F' -> ['ctrl','shift','f']；'F5' -> ['f5']。"""
    parts = [p.strip().lower() for p in norm(s).split("+") if p.strip()]
    if not parts:
        return None
    if parts[-1] == "":
        parts[-1] = "+"
    return parts


def scan_table(qg, ocr, tops):
    """全菜单扫描 + OCR 快捷键列。返回 {(name): (menu, shortcut_ocr)} 与统计。"""
    table, stats = {}, dict(items=0, ocr_nonempty=0, pattern_ok=0, menus=0, t=0.0)
    t0 = time.time()
    for top in tops:
        pyautogui.press("esc"); time.sleep(0.2)
        qg.set_focus(); time.sleep(0.2)
        base = {(x.window_text() or "") for x in qg.descendants(control_type="MenuItem")}
        if not open_menu(qg, top):
            continue
        time.sleep(0.8)
        items = visible_new_items(qg, base)
        if not items:
            print(f"  [warn] 菜单 {top} 未展开")
            pyautogui.press("esc")
            continue
        stats["menus"] += 1
        img = pyautogui.screenshot()
        for t, r in items:
            stats["items"] += 1
            x, y, w, h = r
            crop = img.crop((x+w-184, y, x+w-4, y+h)).resize((180*4, h*4), Image.LANCZOS)
            rr = ocr(np.asarray(crop))
            got = " ".join(z[1] for z in rr[0]) if rr and rr[0] else ""
            got = norm(got)
            if got:
                stats["ocr_nonempty"] += 1
            if got and SHORTCUT_RE.match(got):
                stats["pattern_ok"] += 1
                table.setdefault(t, (top, got))
        pyautogui.press("esc")
        time.sleep(0.1)
    stats["t"] = time.time() - t0
    return table, stats


def press_shortcut(keys):
    t0 = time.time()
    pyautogui.hotkey(*keys)
    return time.time() - t0


STEPS = [
    dict(instr="把地图缩放到当前选中要素", name="Zoom to Selection"),
    dict(instr="把地图缩放到图层的完整范围", name="Zoom Full"),
    dict(instr="回到当前选中要素的视野", name="Zoom to Selection"),
]


def main():
    call({"op": "win", "maximize": True})
    call({"op": "reset", "scene": SCENE})
    if truth().get("console_visible"):
        call({"op": "act", "kind": "action", "text": "Python Console"})
    time.sleep(0.3)

    # 参照（Zoom Full 必须走 act：桥没有 zoom_full 这个 op）
    call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
    time.sleep(0.3)
    call({"op": "zoom_to_selection"}); time.sleep(0.4)
    sel_x0 = extent_x0()
    call({"op": "act", "kind": "action", "text": "Zoom Full"}); time.sleep(0.5)
    full_x0 = extent_x0()
    print(f"参照 extent_x0：SEL={sel_x0:.2f}  FULL={full_x0:.2f}")

    def truth_zoom(name):
        x0 = extent_x0()
        tgt, other = (sel_x0, full_x0) if name == "Zoom to Selection" else (full_x0, sel_x0)
        return abs(x0 - tgt) < 1.5 and abs(x0 - tgt) < abs(x0 - other)

    qg = qgis()
    tops = [t for t in menu_items(qg) if len(t) < 20 and not t.startswith("系统")]
    ocr = RapidOCR()
    cache = HERE / "w15_table.json"
    if cache.exists():
        table = {k: tuple(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
        stats = dict(items=-1, ocr_nonempty=-1, pattern_ok=len(table), menus=-1, t=0.0)
        print(f"载入已缓存的快捷键表 {len(table)} 条（跳过 {len(tops)} 个菜单的扫描+OCR）")
    else:
        print(f"顶层菜单 {len(tops)}；开始扫描+OCR……")
        table, stats = scan_table(qg, ocr, tops)
        cache.write_text(json.dumps({k: list(v) for k, v in table.items()},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  扫描 {stats['menus']} 个菜单 / {stats['items']} 个条目，用时 {stats['t']:.1f}s")
    print(f"  OCR 非空 {stats['ocr_nonempty']}，符合快捷键模式 {stats['pattern_ok']} → 表 {len(table)} 条")

    gt = call({"op": "shortcuts"})["shortcuts"]
    hit = miss = fp = 0
    misses = []
    for t, (mnu, sc) in table.items():
        g = norm(gt.get(t, ""))
        if g and g == sc:
            hit += 1
        elif g:
            misses.append((t, g, sc)); miss += 1
        elif not g:
            fp += 1
    # 真值里有、我们表里没有的（含无快捷键的项）
    gt_with_sc = {k for k, v in gt.items() if norm(v)}
    print(f"\n表校验（对特权真值）：命中 {hit}，错读 {miss}，本无快捷键/表有(假阳) {fp}")
    for t, g, sc in misses[:8]:
        print(f"   错读 {t[:30]:30} 真值={g:12} OCR={sc}")
    covered = [k for k in gt_with_sc if k in table]
    print(f"  真值中有快捷键的动作 {len(gt_with_sc)}；其中被我们通过菜单+OCR发现 {len(covered)}")

    # 快捷键执行
    print(f"\n快捷键执行（各 3 次，真值判定）")
    exec_rows = {}
    for step in STEPS[:2] + [dict(instr="切到图层全范围", name="Zoom Full")]:
        if step["name"] not in table:
            print(f"  {step['name']}: 表里没有（未发现）"); continue
        keys = parse_shortcut(table[step["name"]][1])
        oks, ts = 0, []
        for _ in range(3):
            qg.set_focus(); time.sleep(0.15)
            dt = press_shortcut(keys)
            ts.append(dt)
            time.sleep(0.35)
            oks += int(truth_zoom(step["name"]))
        exec_rows[step["name"]] = dict(keys=keys, ms=float(np.median(ts))*1000, ok=f"{oks}/3")
        print(f"  {step['name']:22} {str(keys):22} {np.median(ts)*1000:6.0f} ms  {oks}/3")

    # live 链（纯快捷键）
    print(f"\nlive 链（60 s，纯快捷键；每步真值）")
    t0 = time.time(); n = 0; ok_steps = tot = ok_all = 0
    while time.time() - t0 < 60:
        n += 1
        call({"op": "reset", "scene": SCENE})
        call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
        qg.set_focus()
        res = []
        for step in STEPS:
            keys = parse_shortcut(table[step["name"]][1])
            press_shortcut(keys); time.sleep(0.3)
            res.append(truth_zoom(step["name"]))
        ok_steps += sum(res); tot += len(res); ok_all += int(all(res))
    el = time.time() - t0
    print(f"  每步 {ok_steps/tot:.3f}  整集 {ok_all/n:.3f}  正确步/分钟 {ok_steps/(el/60):.2f}  集数 {n}")

    (HERE / "w15_results.json").write_text(json.dumps({
        "table": {k: v[1] for k, v in table.items()},
        "table_size": len(table), "scan_stats": stats,
        "verify": dict(hit=hit, miss=miss, fp=fp, gt_with_sc=len(gt_with_sc), covered=len(covered)),
        "exec": exec_rows, "ref": dict(sel_x0=sel_x0, full_x0=full_x0),
        "live": {"step_rate": ok_steps/tot, "ep_all": ok_all/n,
                 "steps_per_min": ok_steps/(el/60), "n": n},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w15_results.json")


if __name__ == "__main__":
    main()
