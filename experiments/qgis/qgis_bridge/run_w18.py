"""
QGIS W18：零特权调用的完整工作流 —— 选择（Processing 工具箱）+ 导出（右键菜单）

W16/W17 的导出是纯 GUI 的，但"选择"一段仍走特权桥。本轮把选择也搬上 GUI：

    Processing 工具箱（搜索 "Select by Expression"）→ 双击算法
    → 算法对话框：Input layer + Expression 填 '"area" > 300' + 点 Run
    → 图层选中 2 个要素（真值判定）
    → 右键图层 → Export → Save Selected Features As…（W16 路径）
    → 磁盘出 GeoJSON（真值：恰好 2 个要素）

只有环境重置与真值判定用桥；**选择与导出全部走真实 UI**（UIA + 真鼠标/键盘）。

用法: python experiments/qgis/qgis_bridge/run_w18.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w16 import export_via_gui, features_of  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402

SCENE = "regions"
EXPR = '"area" > 300'
N_EXP = 2
OUT = HERE / "data" / "gui_sel.geojson"


def R(x):
    r = x.rectangle()
    return (r.left, r.top, r.width(), r.height())


def acquire():
    for _ in range(12):
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                if "QGIS" in (w.window_text() or "") and len(w.descendants(control_type="Button")) > 5:
                    return w
            except Exception:
                pass
        time.sleep(0.35)
    return None


def find_child(qg, title_kw, ctype="Window"):
    for x in qg.descendants():
        try:
            if x.element_info.control_type == ctype and title_kw in (x.window_text() or ""):
                return x
        except Exception:
            continue
    return None


def selected_count(layer=SCENE):
    t = call({"op": "truth"})["truth"]
    for l in t.get("layers", []):
        if l["name"] == layer:
            return l["selected"]
    return None


def select_via_gui(expr, timeout=30, expect_n=N_EXP, run_mode="invoke", layer=SCENE):
    """Processing → Select by expression。返回 (ok, stages)。expect_n = 期望选中的要素数。"""
    st = {}
    t0 = time.time()
    qg = acquire()
    if qg is None:
        return False, st
    tb = find_child(qg, "Processing Toolbox")
    if tb is None:                              # 没开就开（Ctrl+Alt+T，来自 W15 的 OCR 表）
        qg.set_focus(); time.sleep(0.2)
        pyautogui.hotkey("ctrl", "alt", "t")
        for _ in range(20):
            time.sleep(0.3)
            qg = acquire()
            tb = find_child(qg, "Processing Toolbox") if qg else None
            if tb is not None:
                break
    if tb is None:
        return False, st
    st["open_ms"] = (time.time() - t0) * 1000

    # 搜索算法
    t1 = time.time()
    ed = [x for x in tb.descendants(control_type="Edit")]
    ed = [x for x in ed if R(x)[2] > 150][0]
    ed.set_edit_text("Select by Expression")
    time.sleep(1.8)
    qg = acquire(); tb = find_child(qg, "Processing Toolbox")
    tgt = None
    for x in tb.descendants():
        try:
            if (x.window_text() or "").strip() == "Select by expression":
                tgt = x
        except Exception:
            pass
    if tgt is None:
        return False, st
    # UIA 双击在多窗/失焦时不稳（W23 实测：时灵时不灵）→ 像素双击兜底，再轮询
    try:
        tgt.double_click_input()
    except Exception:
        pass
    dlg = None
    for k in range(6):
        time.sleep(0.5)
        qg = acquire()
        dlg = find_child(qg, "Select by expression") if qg else None
        if dlg is not None:
            break
    if dlg is None:
        try:
            r = tgt.rectangle()
            pyautogui.doubleClick(r.left + r.width() // 2, r.top + r.height() // 2)
        except Exception:
            pass
    st["search_ms"] = (time.time() - t1) * 1000

    # 等算法对话框
    t2 = time.time()
    dlg = None
    for _ in range(30):
        time.sleep(0.3)
        qg = acquire()
        dlg = find_child(qg, "Select by expression") if qg else None
        if dlg is not None and len(dlg.descendants(control_type="ComboBox")) >= 3:
            break
    if dlg is None:
        return False, st
    st["dialog_ms"] = (time.time() - t2) * 1000

    # 填参数：Input layer（若空）+ Expression + Run
    t3 = time.time()
    qg = acquire(); dlg = find_child(qg, "Select by expression")
    combos = sorted([x for x in dlg.descendants(control_type="ComboBox")], key=lambda x: R(x)[1])
    if len(combos) >= 1:
        # 永远显式重选：reset 会重建图层，旧的选择记的是旧 layer id（W16 同型陷阱）；
        # # 真坑（W23）：下拉弹层是独立顶层窗，主窗子树里搜不到 ListItem，旧代码静默落空
        picked = False
        try:
            combos[0].click_input(); time.sleep(0.6)
            all_items = []
            for w in Desktop(backend="uia").windows(visible_only=True):
                try:
                    all_items += list(w.descendants(control_type="ListItem"))
                except Exception:
                    pass
            for it in all_items:
                try:
                    txt = (it.window_text() or "").strip()
                    if txt == layer or txt.startswith(layer + " "):
                        it.click_input(); picked = True; time.sleep(0.5); break
                except Exception:
                    continue
            if not picked:
                pyautogui.press("esc")
        except Exception:
            pass
        if not picked:
            try:
                combos[0].select(layer)
            except Exception:
                pass
    edits = [x for x in dlg.descendants(control_type="Edit") if R(x)[2] > 120]
    if not edits:
        return False, st
    e = edits[0]
    ok_expr = False
    for _ in range(3):
        try:
            e.set_edit_text(expr)
        except Exception:
            e.click_input(); pyautogui.hotkey("ctrl", "a")
            for ch in expr:
                pyautogui.write(ch); time.sleep(0.03)
        time.sleep(0.3)
        try:
            if expr.replace(" ", "") in (e.get_value() or "").replace(" ", ""):
                ok_expr = True; break
        except Exception:
            ok_expr = True; break
    if not ok_expr:
        return False, st
    run = None
    for b in dlg.descendants(control_type="Button"):
        try:
            if (b.window_text() or "").strip() == "Run":
                run = b
        except Exception:
            pass
    if run is None:
        return False, st
    # run_mode：invoke = UIA Invoke（低层已修）；verify = 真鼠标点击 + 验证/重试；
    #           blind = 真鼠标点击一次、不验证（W22 对照臂）
    invoked = False
    if run_mode in ("invoke", "verify"):
        if run_mode == "invoke":
            try:
                run.invoke()
                invoked = True
            except Exception:
                try:
                    dlg.set_focus(); time.sleep(0.3)
                    run.click_input()
                except Exception:
                    pass
        else:                                   # verify：故意用未聚焦的鼠标点击（复刻 W18 的坏通道）
            try:
                run.click_input()
            except Exception:
                pass
    else:                                       # blind：点一下就往下走
        try:
            run.click_input()
        except Exception:
            pass
        time.sleep(2.0)
    st["run_ms"] = (time.time() - t3) * 1000
    st["invoked"] = invoked

    # 等选择生效；没生效就再触发一次（最多 3 次）——blind 不重试
    t4 = time.time()
    ok = False
    attempts = 1 if run_mode == "blind" else 3
    for attempt in range(attempts):
        for _ in range(int(4 / 0.3)):
            time.sleep(0.3)
            if selected_count(layer) == expect_n:
                ok = True
                break
        if ok or run_mode == "blind":
            break
        try:
            qg = acquire(); d2 = find_child(qg, "Select by expression")
            if d2 is not None:
                b = [x for x in d2.descendants(control_type="Button")
                     if (x.window_text() or "").strip() == "Run"]
                if b:
                    try:
                        b[0].invoke()
                    except Exception:
                        d2.set_focus(); time.sleep(0.2); b[0].click_input()
        except Exception:
            pass
    st["wait_ms"] = (time.time() - t4) * 1000
    # 关掉算法对话框（Close 按钮）
    try:
        qg = acquire(); dlg = find_child(qg, "Select by expression")
        btns = sorted([b for b in dlg.descendants(control_type="Button")
                       if (b.window_text() or "").strip() in ("Close", "关闭", "确定")],
                      key=lambda b: R(b)[0])
        if btns:
            btns[0].click_input(); time.sleep(0.5)
    except Exception:
        pass
    st["total_ms"] = (time.time() - t0) * 1000
    return ok, st


def select_via_gui_retry(expr, attempts=2, expect_n=N_EXP, run_mode="invoke"):
    """裸跑 + 验证失败就重开一次对话框（关掉再走一遍）。返回 (ok, st, first_try_ok, tries)。"""
    first = None
    st_all = None
    for k in range(attempts):
        ok, st = select_via_gui(expr, expect_n=expect_n, run_mode=run_mode)
        if first is None:
            first = bool(ok)
        st_all = st
        if ok:
            return True, st, first, k + 1
        # 收尾：把可能残留的算法对话框关掉
        try:
            qg = acquire()
            dlg = find_child(qg, "Select by expression") if qg else None
            if dlg is not None:
                btns = sorted([b for b in dlg.descendants(control_type="Button")
                               if (b.window_text() or "").strip() in ("Close", "关闭", "Cancel", "确定")],
                              key=lambda b: R(b)[0])
                if btns:
                    btns[0].click_input(); time.sleep(0.6)
        except Exception:
            pass
    return False, st_all, first, attempts


def main():
    call({"op": "win", "maximize": True})
    rows = []
    print(f"完整工作流：Processing 选择 + GUI 导出 → {OUT.name}（真值 {N_EXP} 个要素）\n")
    print(f"{'trial':>6}{'选择':>6}{'导出':>6}{'要素':>5}{'选择ms':>9}{'导出ms':>9}{'合计ms':>9}{'裸成功':>7}{'重试':>5}  备注")
    for i in range(4):
        call({"op": "reset", "scene": SCENE})          # 环境重置（特权）
        time.sleep(0.3)
        t0 = time.time()
        ok_sel, st_sel, first, tries = select_via_gui_retry(EXPR)
        if OUT.exists():
            OUT.unlink()
        ok_exp, st_exp, note = export_via_gui(OUT, selected_only=True)
        names = features_of(OUT)
        good = bool(ok_sel and ok_exp and len(names) == N_EXP)
        total = (time.time() - t0) * 1000
        rows.append(dict(ok=good, sel_ms=st_sel.get("total_ms", 0) if st_sel else 0,
                         exp_ms=st_exp.get("total_ms", 0), n=len(names),
                         sel=bool(ok_sel), exp=bool(ok_exp), note=note,
                         first=first, tries=tries))
        print(f"{i+1:>6}{str(ok_sel):>6}{str(ok_exp):>6}{len(names):>5}"
              f"{(st_sel.get('total_ms',0) if st_sel else 0):>9.0f}{st_exp.get('total_ms',0):>9.0f}"
              f"{total:>9.0f}{str(first):>7}{tries:>5}  {note or ''}")
    okn = sum(r["ok"] for r in rows)
    raw = sum(1 for r in rows if r["first"])
    print(f"\n端到端 {okn}/{len(rows)} 全对（**裸首试 {raw}/{len(rows)}**）；"
          f"选择段中位 {np.median([r['sel_ms'] for r in rows if r['sel']]):.0f} ms，"
          f"导出段中位 {np.median([r['exp_ms'] for r in rows if r['exp']]):.0f} ms")
    (HERE / "w18_results.json").write_text(json.dumps(
        {"rows": rows, "success": f"{okn}/{len(rows)}", "first_try": f"{raw}/{len(rows)}",
         "expr": EXPR}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w18_results.json")


if __name__ == "__main__":
    main()
