"""
QGIS W23 场景链构建块：Processing 算法对话框与层属性的通用 GUI 驱动。

W23 把 W18/W16 的单点链扩为 6 场景混合包，这里沉淀可复用机械：
  open_alg            工具箱搜索 → 双击（UIA 失败则像素）→ 等对话框
  pick_combo          combo 键盘 type-ahead 选值 + 读回验证（层/字段/渲染器通用）
  set_edit            数值参数（set_edit_text 失败则 ctrl+a 输入）+ 读回
  run_alg_close       Run invoke + 等 wait_fn 真值 + Close
  buffer_via_gui      Buffer：输入层 + Distance → 运行 → 'Buffered' 新层
  extract_via_gui     Extract by location：输入/参照层 + intersect → 'Extracted (location)'
  categorized_via_gui 层属性 → Categorized by <value> → Classify → 确定

只有环境重置与真值判定走桥；选择/运算/导出全部走真实 UI（UIA + 真鼠标键盘）。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w18 import acquire, find_child, R  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402

BUF_OUT = "Buffered"
LOC_OUT = "Extracted (location)"


def open_alg(search, timeout=40):
    """工具箱搜索 → 双击算法 → 返回对话框（run_w18 同源，标题即搜索词）。"""
    qg = acquire()
    tb = find_child(qg, "Processing Toolbox")
    if tb is None:
        qg.set_focus(); time.sleep(0.2)
        pyautogui.hotkey("ctrl", "alt", "t")
        time.sleep(3)
        qg = acquire(); tb = find_child(qg, "Processing Toolbox")
    ed = [x for x in tb.descendants(control_type="Edit") if R(x)[2] > 150][0]
    ed.set_edit_text(search)
    time.sleep(2.0)
    qg = acquire(); tb = find_child(qg, "Processing Toolbox")
    tgt = None
    for x in tb.descendants():          # 取最后一个匹配（树节点靠后）
        try:
            if (x.window_text() or "").strip() == search:
                tgt = x
        except Exception:
            pass
    if tgt is None:
        return None, "alg not found"
    try:
        tgt.double_click_input()
    except Exception:
        pass
    for _ in range(8):
        time.sleep(0.4)
        qg = acquire()
        dlg = find_child(qg, search) if qg else None
        if dlg:
            return dlg, "uia dblclick"
    try:
        r = tgt.rectangle()
        pyautogui.doubleClick(r.left + r.width() // 2, r.top + r.height() // 2)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.4)
        qg = acquire()
        dlg = find_child(qg, search) if qg else None
        if dlg:
            return dlg, "pixel dblclick"
    return None, "dialog not found"


def find_props(qg, layer=None):
    for w in qg.descendants():
        try:
            t = w.window_text() or ""
            if "Layer Properties" in t and w.is_visible() and w.is_enabled():
                if layer is None or layer in t:
                    return w
        except Exception:
            continue
    return None


def find_alg_win(qg, title_kw, need_combo=False, need_run=False):
    """同名 Window 可能有多个（参数页/结果页并存，W23 实测）→ 按内容挑。"""
    for x in qg.descendants():
        try:
            if x.element_info.control_type != "Window":
                continue
            if title_kw not in (x.window_text() or ""):
                continue
            if need_run:
                if btn(x, "Run") is None:
                    continue
            if need_combo and not x.descendants(control_type="ComboBox"):
                continue
            return x
        except Exception:
            continue
    return None


def btn(dlg, text):
    for b in dlg.descendants(control_type="Button"):
        try:
            if (b.window_text() or "").strip() == text:
                return b
        except Exception:
            pass
    return None


def pick_combo(combo, text, tries=3):
    """type-ahead 选值（Qt combo 打开后逐字跳转）→ Enter → 读回验证；
    失败兜底：全桌面搜 Qt 弹层 ListItem 点击。
    W23：Esc 只在真有 Qt 弹层时按 —— 弹层已关时 Esc 会关掉整个对话框（S1 实测）。"""
    low = text.lower()
    for _ in range(tries):
        try:
            combo.click_input()
            time.sleep(0.6)
            pyautogui.write(text, interval=0.06)
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(0.7)
        except Exception:
            pass
        try:
            cur = ((combo.window_text() or "") + (combo.selected_text() or "")).lower()
        except Exception:
            cur = ""
        if low in cur:
            return True
        # ListItem 兜底（限 Qt 弹层，防串台到别的应用）
        items = []
        for w in Desktop(backend="uia").windows(visible_only=True):
            cls = (w.element_info.class_name or "")
            if "Qt" not in cls:
                continue
            try:
                items += list(w.descendants(control_type="ListItem"))
            except Exception:
                pass
        hit = False
        for it in items:
            try:
                txt = (it.window_text() or "").strip().lower()
                if txt == low or txt.startswith(low):
                    it.click_input()
                    hit = True
                    time.sleep(0.5)
                    break
            except Exception:
                continue
        if items and not hit:               # 有 Qt 弹层但没点中 → 关弹层重试
            pyautogui.press("esc")
            time.sleep(0.3)
        try:
            cur = ((combo.window_text() or "") + (combo.selected_text() or "")).lower()
        except Exception:
            cur = ""
        if low in cur:
            return True
    return False


def set_edit(e, val, tries=3):
    for _ in range(tries):
        try:
            e.set_edit_text(val)
        except Exception:
            try:
                e.click_input()
                pyautogui.hotkey("ctrl", "a")
                pyautogui.write(str(val))
            except Exception:
                pass
        time.sleep(0.3)
        try:
            if str(val) in str(e.get_value() or ""):
                return True
        except Exception:
            return True
    return False


def uncheck_open_output(dlg):
    """W23 教训：该勾选（Open output file after running algorithm）控制**结果是否载入工程**，
    临时输出也靠它上图层 —— 必须保持默认勾选，取消勾选会导致算法跑完却无图层。"""
    for c in dlg.descendants(control_type="CheckBox"):
        try:
            if "Open output file" in (c.window_text() or ""):
                if not c.get_toggle_state():
                    try:
                        c.toggle()
                    except Exception:
                        c.invoke()
                    return True
                return True
        except Exception:
            continue
    return False


def run_alg_close(dlg, title_kw, wait_fn, attempts=3, wait_t=20.0):
    """Run invoke → 轮询 wait_fn（真值）→ 失败重试 → Close。
    W23：同名窗有参数页/结果页两个 → 只认带 Run 的；重试时结果页先 Change Parameters。"""
    st = {}
    t0 = time.time()
    ok = False
    for attempt in range(attempts):
        qg = acquire()
        d = find_alg_win(qg, title_kw, need_run=True)
        if d is None:                       # 结果页占着 → 回参数页
            d0 = find_alg_win(qg, title_kw)
            cp = btn(d0, "Change Parameters") if d0 is not None else None
            if cp is not None:
                try:
                    cp.invoke()
                except Exception:
                    cp.click_input()
                time.sleep(1.0)
                qg = acquire()
                d = find_alg_win(qg, title_kw, need_run=True)
        if d is None:
            st["err"] = "no Run window"
            return False, st
        run = btn(d, "Run")
        try:
            run.invoke()
        except Exception:
            try:
                d.set_focus(); time.sleep(0.3); run.click_input()
            except Exception:
                pass
        t1 = time.time()
        while time.time() - t1 < wait_t:
            time.sleep(0.4)
            if wait_fn():
                ok = True
                break
        if ok:
            break
    st["run_ms"] = (time.time() - t0) * 1000
    st["ok"] = ok
    # Close（底部行的 关闭，避免点到右上角 X 也能关但顺序更稳）
    try:
        qg = acquire()
        d = find_alg_win(qg, title_kw)
        if d is not None:
            closes = []
            for b in d.descendants(control_type="Button"):
                try:
                    if (b.window_text() or "").strip() in ("Close", "关闭"):
                        closes.append(b)
                except Exception:
                    pass
            if closes:
                closes = sorted(closes, key=lambda b: R(b)[1])  # 底部行优先
                try:
                    closes[-1].invoke()
                except Exception:
                    closes[-1].click_input()
    except Exception:
        pass
    time.sleep(0.6)
    return ok, st


def truth_layers():
    t = call({"op": "truth"})
    return (t.get("truth") or {}).get("layers", [])


def resolve_layer(prefix, timeout=25):
    """按前缀找实际层名 —— QGIS 3.44 临时输出带 UUID 后缀（Buffered_<uuid>）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for l in truth_layers():
            n = l["name"]
            if n == prefix or n.startswith(prefix + "_") \
                    or n.startswith(prefix + " (") or n.startswith(prefix + "2"):
                return n
        time.sleep(0.5)
    return None


def wait_layer(name, feat=None, timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for l in truth_layers():
            n = l["name"]
            hit = (n == name or n.startswith(name + "_")
                   or n.startswith(name + " ("))
            if hit and (feat is None or l["features"] == feat):
                return True
        time.sleep(0.5)
    return False


def buffer_via_gui(layer, dist=150000, expect_n=None):
    """Buffer：输入层 layer，Distance=dist（米）。返回 (ok, st)。"""
    st = {}
    t0 = time.time()
    dlg, how = open_alg("Buffer")
    st["open"] = how
    if dlg is None:
        return False, st
    time.sleep(0.6)
    qg = acquire(); dlg = find_alg_win(qg, "Buffer", need_combo=True)
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["pick"] = pick_combo(combos[0], layer)
    qg = acquire(); dlg = find_alg_win(qg, "Buffer", need_combo=True)
    edits = sorted([e for e in dlg.descendants(control_type="Edit") if R(e)[2] > 60],
                   key=lambda x: R(x)[1])
    st["dist"] = set_edit(edits[0], str(dist))
    qg = acquire(); dlg = find_alg_win(qg, "Buffer", need_combo=True)
    uncheck_open_output(dlg)
    qg = acquire(); dlg = find_alg_win(qg, "Buffer", need_combo=True)
    ok, rst = run_alg_close(dlg, "Buffer",
                            lambda: wait_layer(BUF_OUT, feat=expect_n))
    st.update(rst)
    st["total_s"] = round(time.time() - t0, 1)
    return ok and st.get("pick") and st.get("dist"), st


def extract_via_gui(inp, ref, pred="intersect"):
    """Extract by location：输入层 inp、参照层 ref、谓词 pred。返回 (ok, st)。"""
    st = {}
    t0 = time.time()
    dlg, how = open_alg("Extract by location")
    st["open"] = how
    if dlg is None:
        return False, st
    time.sleep(0.6)
    qg = acquire(); dlg = find_alg_win(qg, "Extract by location", need_combo=True)
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["pick_in"] = pick_combo(combos[0], inp)
    qg = acquire(); dlg = find_alg_win(qg, "Extract by location", need_combo=True)
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["pick_ref"] = pick_combo(combos[1], ref)
    # 谓词：确保 pred 勾选（默认 intersect 已勾；勾选其余会改变语义）
    qg = acquire(); dlg = find_alg_win(qg, "Extract by location", need_combo=True)
    try:
        for c in dlg.descendants(control_type="CheckBox"):
            txt = (c.window_text() or "").strip()
            if txt in ("intersect", "contain", "disjoint", "equal", "touch",
                       "overlap", "are within", "cross"):
                want = (txt == pred)
                if bool(c.get_toggle_state()) != want:
                    c.invoke()
                    time.sleep(0.2)
        st["pred"] = True
    except Exception as e:
        st["pred"] = str(e)
    qg = acquire(); dlg = find_alg_win(qg, "Extract by location", need_combo=True)
    uncheck_open_output(dlg)
    qg = acquire(); dlg = find_alg_win(qg, "Extract by location", need_combo=True)
    ok, rst = run_alg_close(dlg, "Extract by location",
                            lambda: wait_layer(LOC_OUT))
    st.update(rst)
    st["total_s"] = round(time.time() - t0, 1)
    return ok and bool(st.get("pick_in")) and bool(st.get("pick_ref")), st


def categorized_via_gui(layer, value):
    """层属性 → Symbology → Categorized by value → Classify → 确定。"""
    st = {}
    t0 = time.time()
    from run_w16 import find_layer_item
    qg = acquire()
    item = find_layer_item(qg, layer=layer)
    if item is None:
        return False, {"err": "no layer item"}
    r = item.rectangle()
    try:
        item.double_click_input()
    except Exception:
        pass
    dlg = None
    for _ in range(20):
        time.sleep(0.4)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg:
            break
    if dlg is None:                       # 像素双击兜底
        pyautogui.doubleClick(r.left + r.width() // 2, r.top + r.height() // 2)
        for _ in range(20):
            time.sleep(0.4)
            qg = acquire()
            dlg = find_props(qg, layer)
            if dlg:
                break
    if dlg is None:
        return False, {"err": "no properties dialog"}
    st["open_s"] = round(time.time() - t0, 1)

    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["rend"] = pick_combo(combos[0], "categ")
    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:                       # 中途被关（防崩）
        return False, st
    combos = sorted(dlg.descendants(control_type="ComboBox"), key=lambda x: R(x)[1])
    st["val"] = pick_combo(combos[1], value) if len(combos) > 1 else False
    qg = acquire(); dlg = find_props(qg, layer)
    if dlg is None:
        return False, st
    cb = btn(dlg, "Classify")
    if cb is None:
        # 关掉残留弹层重试一次
        pyautogui.press("esc")
        qg = acquire(); dlg = find_props(qg, layer)
        cb = btn(dlg, "Classify")
    if cb is None:
        return False, st
    cb.invoke()
    time.sleep(1.2)
    qg = acquire(); dlg = find_props(qg, layer)
    okb = btn(dlg, "确定")
    if okb is None:
        return False, st
    okb.invoke()
    time.sleep(1.0)
    st["total_s"] = round(time.time() - t0, 1)
    return bool(st.get("rend")) and bool(st.get("val")), st
