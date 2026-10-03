"""
QGIS W27 暗色科技风（视觉重构）机械 —— "智慧城市数字孪生"视觉层级。

全部真实 UI（颜色对话框是**主窗口树里的 child 窗口 'Select Color'**，
W27 探路踩了三次才定位到；项目属性里的背景色对话框则是顶层 'Select Color'）：

  set_canvas_bg(hex)           Project → Properties… → Background color → hex
  set_layer_color(layer, hex)  单符号层 → 宽 Color 条 → Select Color → hex
  graduated_with_ramp(...)     Graduated + 色带菜单（按钮**右端** → All Color
                               Ramps → 名称）→ 字段/模式/Classify → 确定
  legend_item_font(hex)        图例 → 折叠主组 → Fonts and Text Formatting →
                               Item font（面板切到 Text Format）→ Color 行 → hex

关键教训：
  · QgsColorRampButton 的菜单在按钮**右边缘**（点中间开的是编辑器）；
  · 拾色器提交：编辑器内 HTML+Enter 不落盘（放弃自定义渐变，用内置 Inferno/Magma）；
  · 分类/分级控件的字段 combo 是可编辑的 → 写入前必须 Ctrl+A（否则 pop_maxpop_max…）。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
DBG = bool(__import__("os").environ.get("W27_DEBUG"))


def _shot(tag):
    if not DBG:
        return
    import pyautogui
    out = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode") / f"w27dbg_{tag}.png"
    try:
        pyautogui.screenshot().save(str(out))
    except Exception:
        pass
from client import call  # noqa: E402
from run_w18 import R, acquire  # noqa: E402
from scn_gui import btn, find_props, pick_combo  # noqa: E402
from scn_theme import _open_props, _set_spinner, combo_by_label  # noqa: E402
from run_w16 import find_layer_item  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402


def main_win():
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") == "QgisApp":
                return w
        except Exception:
            continue
    return None


def _parent_windows():
    out = []
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            cn = (w.element_info.class_name or "")
            if cn in ("QgisApp", "QgsLayoutDesignerDialog"):
                out.append(w)
        except Exception:
            continue
    return out


def find_select_color(timeout=8.0):
    """颜色对话框：先查顶层，再查主窗口/布局设计器树里的 child 'Select Color'。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                if (w.window_text() or "").strip() == "Select Color" and w.is_visible():
                    return w
            except Exception:
                continue
        for parent in _parent_windows():
            for c in parent.descendants():
                try:
                    if c.element_info.control_type == "Window" \
                            and (c.window_text() or "").strip() == "Select Color" \
                            and c.is_visible():
                        return c
                except Exception:
                    continue
        time.sleep(0.3)
    return None


def apply_color_dialog(hexv):
    """拾色器两种形态：
    1) 模态 'Select Color' 对话框（图层属性等，主窗口树里的 child）；
    2) 布局设计器右栏的**内联拾色器**（'Select ... Color' 替换面板，
       底部有 'HTML notation' 编辑框；写 hex + Enter 即生效，再点返回箭头退出）。
    """
    sc = find_select_color(timeout=4.0)
    if sc is not None:
        ed = None
        for x in sc.descendants(control_type="Edit"):
            v = (x.window_text() or "").strip()
            if v.startswith("#") and len(v) == 7:
                ed = x
        if ed is None:
            return False
        ed.click_input()
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        ed.set_edit_text(hexv)
        time.sleep(0.3)
        ok = next((b for b in sc.descendants(control_type="Button")
                   if (b.window_text() or "").strip() == "确定"), None)
        if ok is None:
            return False
        ok.click_input()
        time.sleep(0.8)
        return True

    from map_gui import designer as find_designer
    d = find_designer(timeout=2.0)
    if d is None:
        return False
    ed = None
    for c in d.descendants(control_type="Edit"):
        v = (c.window_text() or "").strip()
        if v.startswith("#") and len(v) == 7:
            ed = c
            break
    if ed is None:
        return False
    ed.click_input()
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "a")
    ed.set_edit_text(hexv)
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(0.6)
    _exit_inline_picker(d)
    return True


def _exit_inline_picker(d):
    for c in d.descendants(control_type="Button"):
        try:
            rr = R(c)
            if not (c.window_text() or "").strip() and 2160 < rr[0] < 2215 \
                    and 755 < rr[1] < 805 and rr[2] < 40:
                c.click_input()
                time.sleep(0.6)
                return True
        except Exception:
            continue
    return False


def _menu_nav(win, menu_name, item_text, part=False):
    from map_gui import _menu_item
    return _menu_item(win, menu_name, item_text, exact=not part)


def set_canvas_bg(hexv):
    """项目背景色（暗色底图的地基）。"""
    w = main_win()
    w.set_focus()
    time.sleep(0.3)
    if not _menu_nav(w, "Project", "Properties", part=True):
        return False
    dlg = None
    for _ in range(20):
        time.sleep(0.4)
        for c in w.descendants():
            try:
                if c.element_info.control_type == "Window" \
                        and "Project Properties" in (c.window_text() or ""):
                    dlg = c
                    break
            except Exception:
                continue
        if dlg:
            break
    if dlg is None:
        return False
    bgbtn = next((b for b in dlg.descendants(control_type="Button")
                  if (b.window_text() or "").strip() == "Background color"), None)
    if bgbtn is None:
        return False
    bgbtn.click_input()
    time.sleep(1.0)
    okc = apply_color_dialog(hexv)
    okp = btn(dlg, "确定")
    if okp is not None:
        okp.invoke()
        time.sleep(1.0)
    return okc and okp is not None


def _wide_color_button(dlg):
    cands = []
    for b in dlg.descendants(control_type="Button"):
        try:
            rr = R(b)
            if rr[2] > 200 and 150 < rr[1] < 380:
                cands.append((rr[1], b))
        except Exception:
            continue
    cands.sort(key=lambda x: x[0])
    return cands[0][1] if cands else None


def set_layer_color(layer, hexv):
    """单符号层：宽 Color 条 → Select Color → hex → 确定。"""
    try:
        dlg = _open_props(layer)
    except Exception:
        dlg = None
    if dlg is None:
        return False
    try:
        dlg.set_focus()
    except Exception:
        pass
    time.sleep(0.3)
    wide = _wide_color_button(dlg)
    if wide is None:
        return False
    wide.click_input()
    time.sleep(1.0)
    okc = apply_color_dialog(hexv)
    okp = btn(dlg, "确定")
    if okp is not None:
        okp.invoke()
        time.sleep(0.8)
    return okc and okp is not None


def _pick_ramp(ramp_btn, ramp_name):
    """色带按钮右端菜单：Recent 或 All Color Ramps 子菜单里选 name。"""
    r = R(ramp_btn)
    pyautogui.click(r[0] + r[2] - 6, r[1] + r[3] // 2)
    time.sleep(1.0)
    items = call({"op": "menu_items"}).get("items") or []
    names = [it["text"] for it in items if not it["sep"]]

    def nav_to(want, lst):
        for _ in range(len(lst) + 4):
            cur = call({"op": "menu_items"}).get("current") or ""
            if cur and want in cur:
                return True
            pyautogui.press("down")
            time.sleep(0.1)
        return False

    if any(n == ramp_name for n in names):
        if not nav_to(ramp_name, items):
            pyautogui.press("esc")
            return False
        pyautogui.press("enter")
        time.sleep(0.8)
        return True
    if not any("All Color Ramps" in (n or "") for n in names):
        pyautogui.press("esc")
        return False
    if not nav_to("All Color Ramps", items):
        pyautogui.press("esc")
        return False
    pyautogui.press("right")
    time.sleep(0.8)
    sub = call({"op": "menu_items"}).get("items") or []
    snames = [it["text"] for it in sub if not it["sep"]]
    if ramp_name not in snames:
        pyautogui.press("esc")
        return False
    if not nav_to(ramp_name, sub):
        pyautogui.press("esc")
        return False
    pyautogui.press("enter")
    time.sleep(0.8)
    return True


def graduated_with_ramp(layer, field, ramp, method="color", mode="quantile",
                        classes=None):
    """Graduated + 内置色带（如 Inferno/Magma）。"""
    st = {}
    t0 = time.time()
    dlg = _open_props(layer)
    if dlg is None:
        return False, {"err": "no properties dialog"}
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["rend"] = pick_combo(combos[0], "graduated")
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
    rb = next((b for b in dlg.descendants(control_type="Button")
               if (b.window_text() or "").strip() == "Color ramp"), None)
    st["ramp"] = _pick_ramp(rb, ramp) if rb is not None else False

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
    cb.invoke()
    time.sleep(1.3)
    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    ok = btn(dlg, "确定")
    if ok is not None:
        ok.invoke()
        time.sleep(1.0)
    st["total_s"] = round(time.time() - t0, 1)
    return bool(st.get("field")) and bool(st.get("ramp")), st


def _section_arrow(d, name):
    for c in d.descendants(control_type="Button"):
        try:
            if (c.window_text() or "").strip() == name \
                    and abs(R(c)[2] - 16) < 5:
                return c
        except Exception:
            continue
    return None


def legend_style(bg_hex, font_hex):
    """图例：深色面板 + 浅色字体（暗色科技风的"玻璃面板"）。

    分区头是 16x16 箭头 Button（文本=分区名）：
      先折叠 Main Properties / Legend Items → Background 挤进视野 →
      展开 → 勾选启用 → Color 色块 → hex；再展开 Fonts → Item font →
      Text Format 面板的 Color 色块（选该行最左的宽带按钮，别选 '…'）→ hex。
    """
    from map_gui import designer as find_designer
    st = {}
    d = find_designer()
    if d is None:
        return False, {"err": "no designer"}

    for name in ("Main Properties", "Legend Items"):
        a = _section_arrow(d, name)
        if a is not None:
            a.click_input()
            time.sleep(0.7)
    d = find_designer()
    st["bg_visible"] = False
    for _ in range(3):  # 防状态漂移：不在视野就再翻一次
        a = _section_arrow(d, "Background")
        if a is not None and 700 < R(a)[1] < 1400:
            st["bg_visible"] = True
            break
        for name in ("Main Properties", "Legend Items"):
            a2 = _section_arrow(d, name)
            if a2 is not None:
                a2.click_input()
                time.sleep(0.7)
        d = find_designer()
    if not st["bg_visible"]:
        return False, st
    a = _section_arrow(d, "Background")
    a.click_input()
    time.sleep(0.9)
    d = find_designer()
    _shot("bg_expanded")

    def _wide_button(dd):
        for c in dd.descendants(control_type="Button"):
            try:
                if (c.window_text() or "").strip() == "Background" \
                        and R(c)[2] > 200:
                    return c
            except Exception:
                continue
        return None

    wb = _wide_button(d)
    if wb is None:
        return False, st
    for c in d.descendants(control_type="CheckBox"):
        try:
            if (c.window_text() or "").strip() == "Background" \
                    and R(c)[2] > 300:
                try:
                    if not c.get_toggle_state():
                        c.toggle()
                        time.sleep(0.4)
                except Exception:
                    c.invoke()
                break
        except Exception:
            continue
    wb.click_input()
    time.sleep(1.5)
    _shot("bg_clicked")
    st["bg_color"] = apply_color_dialog(bg_hex)
    _shot("bg_after")

    d = find_designer()
    a = _section_arrow(d, "Fonts and Text Formatting")
    if a is not None:
        a.click_input()
        time.sleep(0.9)
    d = find_designer()
    ifont = None
    for c in d.descendants(control_type="Button"):
        try:
            if (c.window_text() or "").strip() == "Item font":
                ifont = c
        except Exception:
            continue
    if ifont is None:
        return False, st
    ifont.click_input()
    time.sleep(1.0)
    d = find_designer()
    lbl = None
    for t in d.descendants(control_type="Text"):
        try:
            if (t.window_text() or "").strip() == "Color" and R(t)[0] > 2100:
                lbl = t
                break
        except Exception:
            continue
    if lbl is None:
        return False, st
    lr = R(lbl)
    cands = []
    for b in d.descendants(control_type="Button"):
        try:
            rr = R(b)
            if abs(rr[1] - lr[1]) < 16 and rr[0] > lr[0] \
                    and (b.window_text() or "").strip() != "…":
                cands.append((rr[0], b))
        except Exception:
            continue
    if not cands:
        return False, st
    cands.sort(key=lambda x: x[0])
    cands[0][1].click_input()
    time.sleep(1.0)
    st["font_color"] = apply_color_dialog(font_hex)
    # 退出 Text Format（返回箭头：头部左侧的窄空 Button）——避免关闭设计器时 Qt 崩
    d = find_designer()
    for c in d.descendants(control_type="Button"):
        try:
            rr = R(c)
            if not (c.window_text() or "").strip() and 2160 < rr[0] < 2215 \
                    and 755 < rr[1] < 805 and rr[2] < 40:
                c.click_input()
                time.sleep(0.6)
                break
        except Exception:
            continue
    return bool(st.get("bg_color")) and bool(st.get("font_color")), st
