"""
QGIS W26 主题表达族：同一批数据的多种专题表达（分类 / 分级设色 / 比例符号 / 热力图）

W23 已有 categorized_via_gui（分类图）；本模块补另外三种渲染机制：

  graduated_via_gui  Symbology → Graduated → Value=字段 / Method=Color|Size
                     （Color=分级设色；Size=比例符号）/ Mode=Quantile|... →
                     Classify → 确定
  heatmap_via_gui    Symbology → Heatmap →（可选 Radius）→ 确定

控件定位经验（W26 探路，全部实机验证）：
  · 渲染器 combo 是对话框里最上方的 ComboBox（dialog 顶部）
  · Value / Method / Mode 用"标签 Text + 同行 ComboBox"定位（不依赖序号——
    W26 实测按序号会在 Method 切换后被重排的控件骗到）
  · 切换顺序：渲染器 → Value → Method → Mode → Classify → 确定；每步后重取对话框
  · 已探明的枚举值：Method ∈ {color, size}；Mode ∈ {equal, quantile, jenks, std, pretty}

真值由 bridge truth() 的 renderer 段给出（categorizedSymbol / graduatedSymbol
含 attr·method·mode·ranges / heatmapRenderer 含 radius）。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w18 import acquire, R  # noqa: E402
from scn_gui import btn, find_props, pick_combo  # noqa: E402

import pyautogui  # noqa: E402


def renderer_combo(dlg):
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    return combos[0] if combos else None


def combo_by_label(dlg, label):
    tr = None
    for t in dlg.descendants(control_type="Text"):
        try:
            if (t.window_text() or "").strip() == label:
                tr = R(t)
                break
        except Exception:
            continue
    if tr is None:
        return None
    cands = []
    for c in dlg.descendants(control_type="ComboBox"):
        try:
            rr = R(c)
            if abs(rr[1] - tr[1]) < 8 and rr[0] > tr[0]:
                cands.append((rr[0], c))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])
    return cands[0][1] if cands else None


def spinner_by_label(dlg, label):
    tr = None
    for t in dlg.descendants(control_type="Text"):
        try:
            if (t.window_text() or "").strip() == label:
                tr = R(t)
                break
        except Exception:
            continue
    if tr is None:
        return None
    cands = []
    for s in dlg.descendants(control_type="Spinner"):
        try:
            rr = R(s)
            if abs(rr[1] - tr[1]) < 10 and rr[0] > tr[0]:
                cands.append((rr[0], s))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])
    return cands[0][1] if cands else None


def _set_spinner(dlg, label, val):
    s = spinner_by_label(dlg, label)
    if s is None:
        return False
    target = s
    try:
        eds = s.descendants(control_type="Edit")
        if eds:
            target = eds[0]
    except Exception:
        pass
    try:
        target.click_input()
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        target.set_edit_text(str(val))
        pyautogui.press("tab")
        time.sleep(0.3)
        return True
    except Exception:
        return False


def _open_props(layer):
    from run_w16 import find_layer_item
    qg = acquire()
    item = find_layer_item(qg, layer=layer)
    if item is None:
        return None
    r = item.rectangle()
    try:
        item.double_click_input()
    except Exception:
        pass
    for _ in range(20):
        time.sleep(0.4)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg:
            return dlg
    pyautogui.doubleClick(r.left + r.width() // 2, r.top + r.height() // 2)
    for _ in range(20):
        time.sleep(0.4)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg:
            return dlg
    return None


def _confirm(dlg):
    ok = btn(dlg, "确定")
    if ok is None:
        return False
    try:
        ok.invoke()
    except Exception:
        ok.click_input()
    time.sleep(0.8)
    return True


def graduated_via_gui(layer, field, method="color", mode="quantile", classes=None):
    """分级渲染：method=color 分级设色 / size 比例符号。返回 (ok, st)。"""
    st = {}
    t0 = time.time()
    dlg = _open_props(layer)
    if dlg is None:
        return False, {"err": "no properties dialog"}
    st["rend"] = pick_combo(renderer_combo(dlg), "graduated")
    if not st["rend"]:
        return False, st
    time.sleep(0.8)

    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    st["field"] = pick_combo(combo_by_label(dlg, "Value"), field)

    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    st["method"] = pick_combo(combo_by_label(dlg, "Method"), method)

    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    st["mode"] = pick_combo(combo_by_label(dlg, "Mode"), mode)

    if classes:
        qg = acquire(); dlg = find_props(qg, layer)
        if dlg is not None:
            st["classes"] = _set_spinner(dlg, "Classes", classes)

    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    cb = btn(dlg, "Classify")
    if cb is None:
        return False, st
    try:
        cb.invoke()
    except Exception:
        cb.click_input()
    time.sleep(1.2)

    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    st["confirmed"] = _confirm(dlg)
    st["total_s"] = round(time.time() - t0, 1)
    return bool(st.get("field")) and bool(st.get("confirmed")), st


def heatmap_via_gui(layer, radius=None):
    """热力图渲染。返回 (ok, st)。"""
    st = {}
    t0 = time.time()
    dlg = _open_props(layer)
    if dlg is None:
        return False, {"err": "no properties dialog"}
    st["rend"] = pick_combo(renderer_combo(dlg), "heatmap")
    if not st["rend"]:
        return False, st
    time.sleep(0.8)
    if radius:
        qg = acquire(); dlg = find_props(qg, layer)
        if dlg is not None:
            st["radius"] = _set_spinner(dlg, "Radius", radius)
    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    st["confirmed"] = _confirm(dlg)
    st["total_s"] = round(time.time() - t0, 1)
    return bool(st.get("rend")) and bool(st.get("confirmed")), st
