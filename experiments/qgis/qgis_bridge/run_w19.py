"""
QGIS W19：根因 W18 的偶发失败（Processing 选择段 ~75%）

W18 的现象：约 1/4 概率选择段完全无效果，重开对话框也救不回。
本轮加仪表把失败签名抓出来：
  · Run 前：Input layer 的 selected_text / Expression 的 value / Run 是否 enabled
  · Run 后：算法对话框的进度-错误标签文本（lblProgress）+ 选择计数（每 0.5 s 采样 5 s）
  · 失败时留全屏截图，并列出当时仍打开的 "Select by expression" 窗口数

用法: python experiments/qgis/qgis_bridge/run_w19.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w18 import acquire, find_child, selected_count, R, SCENE, EXPR, N_EXP  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402

TRIALS = 6


def label_text(dlg):
    """算法对话框底部的进度/错误标签。"""
    best = ""
    for x in dlg.descendants(control_type="Text"):
        try:
            aid = getattr(x.element_info, "automation_id", "") or ""
            if "lblProgr" in aid or "lblLog" in aid:
                t = (x.window_text() or "").strip()
                if t:
                    best = t
        except Exception:
            continue
    return best


def open_algo_dialog():
    """工具箱 → 搜索 → 双击算法 → 返回 (dlg, stages)。"""
    st = {}
    t0 = time.time()
    qg = acquire()
    if qg is None:
        return None, st
    tb = find_child(qg, "Processing Toolbox")
    if tb is None:
        qg.set_focus(); time.sleep(0.2)
        pyautogui.hotkey("ctrl", "alt", "t")
        for _ in range(20):
            time.sleep(0.3)
            qg = acquire()
            tb = find_child(qg, "Processing Toolbox") if qg else None
            if tb is not None:
                break
    if tb is None:
        return None, st
    ed = [x for x in tb.descendants(control_type="Edit") if R(x)[2] > 150]
    if not ed:
        return None, st
    ed[0].set_edit_text("Select by Expression")
    time.sleep(1.8)
    qg = acquire(); tb = find_child(qg, "Processing Toolbox")
    tgt = None
    for x in tb.descendants():
        try:
            if (x.window_text() or "").strip() == "Select by expression":
                tgt = x
        except Exception:
            continue
    if tgt is None:
        return None, st
    tgt.double_click_input()
    st["search_ms"] = (time.time() - t0) * 1000
    dlg = None
    for _ in range(40):
        time.sleep(0.3)
        qg = acquire()
        dlg = find_child(qg, "Select by expression") if qg else None
        if dlg is not None and len(dlg.descendants(control_type="ComboBox")) >= 3:
            break
    st["open_ms"] = (time.time() - t0) * 1000
    return dlg, st


def fill_and_run(dlg, log):
    qg = acquire(); dlg = find_child(qg, "Select by expression")
    combos = sorted([x for x in dlg.descendants(control_type="ComboBox")], key=lambda x: R(x)[1])
    # 输入图层：显式重选
    picked = False
    try:
        combos[0].click_input(); time.sleep(0.6)
        for it in qg.descendants(control_type="ListItem"):
            try:
                if (it.window_text() or "").strip() == SCENE:
                    it.click_input(); picked = True; time.sleep(0.5); break
            except Exception:
                continue
        if not picked:
            pyautogui.press("esc")
    except Exception:
        pass
    if not picked:
        try:
            combos[0].select(SCENE); picked = True
        except Exception:
            pass
    try:
        cur_layer = (combos[0].selected_text() or "").strip()
    except Exception:
        cur_layer = "?"
    # 表达式
    edits = [x for x in dlg.descendants(control_type="Edit") if R(x)[2] > 120]
    e = edits[0]
    ok_expr = False
    for _ in range(3):
        try:
            e.set_edit_text(EXPR)
        except Exception:
            e.click_input(); pyautogui.hotkey("ctrl", "a")
            for ch in EXPR:
                pyautogui.write(ch); time.sleep(0.03)
        time.sleep(0.3)
        try:
            val = (e.get_value() or "")
        except Exception:
            val = "?"
        if EXPR.replace(" ", "") in val.replace(" ", ""):
            ok_expr = True
            break
    run = None
    for b in dlg.descendants(control_type="Button"):
        try:
            if (b.window_text() or "").strip() == "Run":
                run = b
        except Exception:
            continue
    run_en = None
    try:
        run_en = bool(run.is_enabled())
    except Exception:
        pass
    log.update(layer=picked, layer_text=cur_layer, expr_ok=ok_expr,
               run_enabled=run_en, pre_label=label_text(dlg))
    if run is not None:
        run.click_input()
    # 采样
    samples = []
    for k in range(10):
        time.sleep(0.5)
        qg = acquire(); d2 = find_child(qg, "Select by expression")
        lab = label_text(d2) if d2 is not None else "<dialog closed>"
        samples.append(dict(t=(k + 1) * 0.5, n=selected_count(), label=lab[:90]))
        if selected_count() == N_EXP:
            break
    log["samples"] = samples
    log["final_n"] = selected_count()
    return log


def main():
    call({"op": "win", "maximize": True})
    rows = []
    for i in range(TRIALS):
        call({"op": "reset", "scene": SCENE})
        time.sleep(0.35)
        # 残留对话框计数
        qg = acquire()
        leftovers = len([x for x in qg.descendants() if x.element_info.control_type == "Window"
                         and "Select by expression" in (x.window_text() or "")])
        dlg, st = open_algo_dialog()
        if dlg is None:
            rows.append(dict(trial=i + 1, opened=False, leftovers=leftovers))
            print(f"trial {i+1}: 对话框没开（残留 {leftovers}）")
            continue
        log = dict(trial=i + 1, opened=True, leftovers=leftovers)
        fill_and_run(dlg, log)
        ok = log.get("final_n") == N_EXP
        rows.append(dict(ok=ok, **log))
        print(f"trial {i+1}: 成功={ok} 图层选择={log.get('layer')} (text={log.get('layer_text')!r}) "
              f"表达式ok={log.get('expr_ok')} Run可用={log.get('run_enabled')} "
              f"最终选中={log.get('final_n')}")
        print(f"   Run前标签={log.get('pre_label')!r}")
        for s in log.get("samples", [])[:6]:
            print(f"     t={s['t']:.1f}s n={s['n']} label={s['label']!r}")
        if not ok:
            pyautogui.screenshot().save(str(HERE / f"_w19_fail_{i+1}.png"))
        # 收尾：关对话框
        try:
            qg = acquire(); d2 = find_child(qg, "Select by expression")
            if d2 is not None:
                btns = sorted([b for b in d2.descendants(control_type="Button")
                               if (b.window_text() or "").strip() in ("Close", "关闭", "确定", "Cancel")],
                              key=lambda b: R(b)[0])
                if btns:
                    btns[0].click_input(); time.sleep(0.5)
        except Exception:
            pass
    okn = sum(1 for r in rows if r.get("ok"))
    print(f"\n{okn}/{TRIALS} 成功")
    (HERE / "w19_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w19_results.json")


if __name__ == "__main__":
    main()
