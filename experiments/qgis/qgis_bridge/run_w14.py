"""
QGIS W14：补覆盖 —— UIA 菜单通道能补上工具栏的缺口吗？快捷键呢？

W13 发现：真实 UIA 通道下工具栏只有 17/28（61%）覆盖，"Python Console" 等根本不暴露。
本轮测两条真实退路：
  1) **菜单**：UIA 能否枚举菜单项（名字+rect）→ 覆盖面提升多少？执行成本多少？
  2) **快捷键**：UIA 是否暴露快捷键字符串（AcceleratorKey）？——若否，agent 只能靠 OCR 或经验。
然后：用"只有菜单能到达"的动作搭一条 live 链（Zoom In / Identify Features / Python Console），
每步真值可判（scale 变小 / mapTool 变化 / console_visible）。

用法: python experiments/qgis/qgis_bridge/run_w14.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402

UIA_NAME, UIA_ACCEL, UIA_ACCESS = 30005, 30006, 30007
SAFE = ["Zoom Full", "Zoom to Selection", "Zoom Last", "Zoom Next",
        "Zoom to Native Resolution (100%)", "Pan Map to Selection", "Refresh",
        "Undo", "Redo", "Expand All", "Collapse All", "Copy Features",
        "Cut Features", "Paste Features", "Delete Selected", "Save Layer Edits",
        "Current Edits",
        # 工具栏里 UIA 看不到的（W13 实测缺失）
        "Zoom In", "Zoom Out", "Pan Map", "Identify Features", "Show Map Tips",
        "Python Console", "Toggle Editing", "Show/Hide Labels and Diagrams",
        "Show Unplaced Labels", "Temporal Controller", "Add Polygon Feature"]
TOOLBAR_VISIBLE = set()


def truth():
    return call({"op": "truth"})["truth"]


def qgis(backend="uia"):
    d = Desktop(backend=backend)
    for w in d.windows():
        try:
            if "QGIS" in (w.window_text() or ""):
                return w
        except Exception:
            continue
    return None


def menu_names(qg):
    out = []
    for m in qg.descendants(control_type="MenuItem"):
        try:
            t = (m.window_text() or "").strip()
            r = m.rectangle()
        except Exception:
            continue
        if t and r.width() > 0:
            out.append(t)
    return out


def open_menu(qg, name):
    for m in qg.descendants(control_type="MenuItem"):
        try:
            if (m.window_text() or "").strip() == name:
                m.click_input()
                return m
        except Exception:
            continue
    return None


def items_of_open_menu(qg, base):
    out = []
    for x in qg.descendants(control_type="MenuItem"):
        try:
            t = (x.window_text() or "").strip()
            r = x.rectangle()
            vis = bool(x.is_visible())
        except Exception:
            continue
        if t and t not in base and r.width() > 10 and vis:
            out.append((t, [r.left, r.top, r.width(), r.height()]))
    return out


def toolbar_uia(qg):
    """UIA 工具栏按钮（W13 口径）。"""
    from pywinauto.keyboard import SendKeys  # noqa
    out = {}
    for b in qg.descendants(control_type="Button"):
        try:
            n = (b.window_text() or "").strip()
            r = b.rectangle()
            if n and r.width() >= 8 and r.height() >= 8 and n not in out:
                out[n] = [r.left, r.top, r.width(), r.height()]
        except Exception:
            continue
    return out


def scan_menus(qg, tops, accel_probe=True):
    """逐个打开顶层菜单，抓一级条目；返回 {name: (menu, rect)} 与成本。"""
    found = {}
    costs = []
    accel_info = []
    for top in tops:
        if top.startswith("系统"):
            continue
        base = set(menu_names(qg))
        qg.set_focus()
        t0 = time.time()
        m = open_menu(qg, top)
        if m is None:
            continue
        time.sleep(0.45)
        items = items_of_open_menu(qg, base)
        costs.append(time.time() - t0)
        if accel_probe and items:
            for it in qg.descendants(control_type="MenuItem"):
                if (it.window_text() or "").strip() == items[0][0]:
                    try:
                        accel_info.append(str(it.element_info.element.GetCurrentPropertyValue(UIA_ACCEL)))
                    except Exception:
                        accel_info.append("ERR")
                    break
        for n, r in items:
            if n not in found:
                found[n] = (top, r)
        pyautogui.press("esc")
        time.sleep(0.15)
    return found, costs, accel_info


def click_menu_item(qg, top, item):
    base = set(menu_names(qg))
    qg.set_focus()
    t0 = time.time()
    open_menu(qg, top)
    time.sleep(0.45)
    for n, r in items_of_open_menu(qg, base):
        if n == item:
            pyautogui.click(r[0] + r[2]//2, r[1] + r[3]//2)
            el = time.time() - t0
            time.sleep(0.35)
            return True, el
    pyautogui.press("esc")
    return False, time.time() - t0


def wait_truth(field, pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if pred(truth()):
                return True
        except Exception:
            pass
        time.sleep(0.08)
    return False


STEPS = [
    dict(instr="切换到放大工具", correct="Zoom In", menu="View",
         check=lambda t, ref: t.get("tool") == "Zoom In"),
    dict(instr="切换到识别要素工具", correct="Identify Features", menu="View",
         check=lambda t, ref: t.get("tool") == "Identify Features"),
    dict(instr="打开 Python 控制台", correct="Python Console", menu="Plugins",
         check=lambda t, ref: bool(t.get("console_visible"))),
]


def main():
    call({"op": "win", "maximize": True})
    call({"op": "reset", "scene": SCENE}) if False else None
    call({"op": "reset", "scene": "regions"})
    qg = qgis()
    if qg is None:
        print("QGIS 窗口未找到"); return

    # 1) 工具栏覆盖（W13 口径）
    tb = toolbar_uia(qg)
    tb_hit = [t for t in SAFE if t in tb]
    print(f"UIA 工具栏覆盖: {len(tb_hit)}/{len(SAFE)}")

    # 2) 菜单扫描
    tops = [t for t in menu_names(qg) if len(t) < 20 and not t.startswith("系统")]
    print("顶层菜单:", tops)
    t0 = time.time()
    found, costs, accel = scan_menus(qg, tops)
    el = time.time() - t0
    menu_hit = [t for t in SAFE if t in found]
    print(f"\n菜单扫描：{len(found)} 个一级条目，用时 {el:.1f}s，平均每菜单 {np.median(costs)*1000:.0f} ms")
    print(f"  带 AccelKey 的样本: {accel[:5]}（空='' 表示 Qt 桥不暴露快捷键）")
    only_menu = [t for t in menu_hit if t not in tb]
    neither = [t for t in SAFE if t not in tb and t not in found]
    print(f"  菜单覆盖: {len(menu_hit)}/{len(SAFE)}；工具栏没有但菜单有的: {only_menu}")
    print(f"  两条通道都到不了的: {neither}")
    print(f"  合并覆盖: {len(set(tb_hit) | set(menu_hit))}/{len(SAFE)}")

    # 3) 菜单执行的成本与成功率（缺口动作）
    print(f"\n菜单执行（缺口动作，各 3 次）")
    print(f"{'action':>22}{'打开+点击 ms':>14}{'成功':>6}   真值")
    exec_rows = {}
    for step in STEPS:
        if step["correct"] not in found:
            print(f"{step['correct']:>22}   —— 菜单里也没有")
            continue
        top = found[step["correct"]][0]
        oks, ts = 0, []
        for i in range(3):
            ref = truth()
            ok, dt = click_menu_item(qg, top, step["correct"])
            ts.append(dt)
            if ok and wait_truth(step["correct"], lambda t: step["check"](t, ref)):
                oks += 1
            # 复位：控制台/工具类动作切回来
            if step["correct"] == "Python Console" and truth().get("console_visible"):
                click_menu_item(qg, top, step["correct"])
        exec_rows[step["correct"]] = dict(top=top, ms=float(np.median(ts))*1000, ok=f"{oks}/3")
        print(f"{step['correct']:>22}{np.median(ts)*1000:>14.0f}{oks:>4}/3   {step['instr']}")

    # 4) live 链（只用"菜单才能到达"的动作）
    print(f"\nlive 链（60 s，菜单通道；每步真值判定）")
    t0 = time.time(); n = 0; ok_steps = 0; tot = 0; ok_all = 0
    while time.time() - t0 < 60:
        n += 1
        call({"op": "reset", "scene": "regions"})
        if truth().get("console_visible"):          # 环境重置（特权）
            call({"op": "act", "kind": "action", "text": "Python Console"})
            time.sleep(0.2)
        res = []
        for step in STEPS:
            ref = truth()
            ok, _ = click_menu_item(qg, found[step["correct"]][0], step["correct"])
            res.append(bool(ok and wait_truth(step["correct"], lambda t: step["check"](t, ref))))
        ok_steps += sum(res); tot += len(res); ok_all += int(all(res))
    el = time.time() - t0
    print(f"  每步成功率 {ok_steps/tot:.3f}  整集全对 {ok_all/n:.3f}  正确步/分钟 {ok_steps/(el/60):.2f}  集数 {n}")

    (HERE / "w14_results.json").write_text(json.dumps({
        "toolbar_coverage": {"hit": len(tb_hit), "n": len(SAFE), "names": tb_hit},
        "menu_scan": {"items": len(found), "seconds": el,
                      "ms_per_menu": float(np.median(costs))*1000},
        "menu_coverage": {"hit": len(menu_hit), "only_menu": only_menu, "neither": neither,
                          "union": len(set(tb_hit) | set(menu_hit))},
        "accel_samples": accel[:8],
        "menu_exec": exec_rows,
        "live": {"step_rate": ok_steps/tot, "ep_all": ok_all/n,
                 "steps_per_min": ok_steps/(el/60), "n": n},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w14_results.json")


if __name__ == "__main__":
    main()
