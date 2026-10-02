"""
QGIS W25 复杂示例：布局设计器（Print Layout）GUI 全链 —— "配一张主题地图"。

从分类设色到出图（PNG @300dpi）全程真实 UI（UIA + 真实鼠标/键盘/剪贴板）：

  new_layout          Project 菜单 → New Print Layout… → 布局设计器
  add_item            Add Item 菜单 → 项 → 页面单击 → New Item Properties（子窗口）
                      数值定位 X/Y/W/H → 确定
  set_label_text      标签文本（set_edit_text，实测支持中文）
  set_extent_canvas   Set to Map Canvas Extent
  export_image        Export as Image… → 原生保存对话框（剪贴板粘贴路径 → 保存）
                      → [覆盖确认 是] → Image Export Options → 保存 → PNG
  close_designer      关闭设计器（丢弃未保存变更）

W25 探路的四条硬经验（都进了机械）：
  1. 菜单加项（map/label/legend 同型）：单击页面 → "New Item Properties" 子窗口
     填 X/Y/W/H → 确定（数值定位，不靠拖拽像素）；
  2. 原生保存对话框是顶层 #32770；**逐字键入会丢反斜杠**（中文 IME 环境）→
     必须剪贴板粘贴路径；
  3. 输入焦点陷阱：其它应用会抢前台吃掉按键/点击 → 操作对话框前用
     win32 SetForegroundWindow 显式置前；
  4. 覆盖已存在文件时会弹"是/否"确认 → 需处理。
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402
import win32gui  # noqa: E402
import win32process  # noqa: E402
import win32clipboard as clip  # noqa: E402

pyautogui.FAILSAFE = False               # 自动化场景禁用角落保护（W25 实测踩到）

TITLE_TEXT = "中国公路等级专题图"


# --------------------------------------------------------------- primitives ---
def main_window():
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") == "QgisApp":
                return w
        except Exception:
            continue
    return None


def designer(timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                if (w.element_info.class_name or "") == "QgsLayoutDesignerDialog":
                    return w
            except Exception:
                continue
        time.sleep(0.5)
    return None


def _pid():
    w = main_window()
    if w is None:
        return None
    return win32process.GetWindowThreadProcessId(w.handle)[1]


def _tops(pid):
    out = []

    def cb(hwnd, _):
        try:
            _, p = win32process.GetWindowThreadProcessId(hwnd)
            if p == pid and win32gui.IsWindowVisible(hwnd):
                out.append((hwnd, win32gui.GetClassName(hwnd),
                            win32gui.GetWindowText(hwnd)))
        except Exception:
            pass
    win32gui.EnumWindows(cb, None)
    return out


def _fg(hwnd):
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    time.sleep(0.5)


def _clip_set(text):
    clip.OpenClipboard()
    clip.EmptyClipboard()
    clip.SetClipboardText(text)
    clip.CloseClipboard()


def _click(e):
    try:
        e.click_input()
    except Exception:
        r = e.rectangle()
        pyautogui.click((r.left + r.right) // 2, (r.top + r.bottom) // 2)


def _btn(win, text, startswith=False, min_top=None):
    for b in win.descendants(control_type="Button"):
        try:
            t = (b.window_text() or "").strip()
            if (t.startswith(text) if startswith else t == text):
                if min_top is not None and b.rectangle().top <= min_top:
                    continue
                return b
        except Exception:
            continue
    return None


def _menu_item(win, menu_name, item_text, exact=True):
    """打开菜单栏 menu_name → 读 menu_items → 反馈导航到 item_text → Enter。"""
    m = None
    for x in win.descendants(control_type="MenuItem"):
        try:
            if (x.window_text() or "").strip() == menu_name:
                m = x
                break
        except Exception:
            continue
    if m is None:
        return False
    _click(m)
    time.sleep(0.6)
    items = call({"op": "menu_items"}).get("items") or []
    tgt = next((it["text"] for it in items if not it["sep"]
                and ((it["text"] == item_text) if exact
                     else (item_text in (it["text"] or "")))), None)
    if tgt is None:
        pyautogui.press("esc")
        return False
    for _ in range(len(items) + 3):
        cur = call({"op": "menu_items"}).get("current") or ""
        if cur == tgt:
            break
        pyautogui.press("down")
        time.sleep(0.1)
    pyautogui.press("enter")
    return True


# ------------------------------------------------------------------ chains ---
def new_layout(timeout=20.0):
    """Project → New Print Layout…（默认名 Layout N）→ 设计器。返回 (designer, st)。"""
    st = {}
    t0 = time.time()
    w = main_window()
    if w is None:
        return None, {"err": "no main window"}
    w.set_focus()
    time.sleep(0.3)
    if not _menu_item(w, "Project", "New Print Layout", exact=False):
        return None, {"err": "no New Print Layout item"}
    st["menu_ms"] = round((time.time() - t0) * 1000, 1)
    d = designer(timeout=8)
    if d is None:                          # 名称对话框可能还在 → 默认名确认
        pyautogui.press("enter")
        time.sleep(1.0)
        d = designer(timeout=14)
    st["total_ms"] = round((time.time() - t0) * 1000, 1)
    return d, st


def setup_view(d):
    """最大化设计器 + Zoom Full（固定页面屏幕位置，供单击定位）。"""
    b = _btn(d, "最大化")
    if b is not None:
        _click(b)
        time.sleep(1.0)
    b = _btn(d, "Zoom Full")
    if b is not None:
        _click(b)
        time.sleep(0.8)


def _item_dialog(d, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for c in d.descendants():
            try:
                if c.element_info.control_type == "Window" \
                        and (c.window_text() or "").strip() == "New Item Properties":
                    return c
            except Exception:
                continue
        time.sleep(0.3)
    return None


def _fill_rect(dlg, x, y, w, h):
    texts, edits = [], []
    for c in dlg.descendants():
        try:
            ct = c.element_info.control_type
            nm = (c.window_text() or "").strip()
            if ct == "Text" and nm:
                texts.append((nm, c.rectangle()))
            elif ct == "Edit":
                edits.append((c.rectangle(), c))
        except Exception:
            continue
    for name, val in (("X", x), ("Y", y), ("Width", w), ("Height", h)):
        tr = next((r for t, r in texts if t == name), None)
        if tr is None:
            return False
        cand = [e for e in edits if abs(e[0].top - tr.top) < 14 and e[0].left > tr.left]
        cand.sort(key=lambda e: (abs(e[0].top - tr.top), e[0].left))
        if not cand:
            return False
        e = cand[0][1]
        e.click_input()
        time.sleep(0.15)
        e.set_edit_text(str(val))
        pyautogui.press("tab")
        time.sleep(0.2)
    return True


def add_item(d, name, click_xy, rect):
    """Add Item → name → 单击页面 → New Item Properties 数值定位 → 确定。"""
    d.set_focus()
    time.sleep(0.2)
    if not _menu_item(d, "Add Item", name):
        return False, {"err": f"no menu item {name}"}
    time.sleep(0.6)
    pyautogui.click(click_xy[0], click_xy[1])
    time.sleep(1.0)
    dlg = _item_dialog(d)
    if dlg is None:
        return False, {"err": "no New Item Properties dialog"}
    ok = _fill_rect(dlg, *rect)
    kb = _btn(dlg, "确定")
    if kb is None:
        return False, {"err": "no confirm button"}
    _click(kb)
    time.sleep(1.0)
    return ok, {}


def set_label_text(d, text, old="Lorem ipsum", timeout=8.0):
    """等 Item Properties 切到新 Label（Edit 显示默认文本）→ 替换文本。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        for c in d.descendants(control_type="Edit"):
            try:
                if (c.window_text() or "").strip() == old:
                    c.click_input()
                    time.sleep(0.2)
                    pyautogui.hotkey("ctrl", "a")
                    time.sleep(0.15)
                    c.set_edit_text(text)
                    time.sleep(0.4)
                    return True
            except Exception:
                continue
        time.sleep(0.4)
    return False


def set_extent_canvas(d):
    b = _btn(d, "Set to Map Canvas Extent")
    if b is None:
        return False
    _click(b)
    time.sleep(0.8)
    return True


def export_image(d, path, timeout=30.0):
    """Export as Image… → 原生保存（剪贴板路径）→ [覆盖确认] → 选项保存 → PNG。"""
    st = {}
    p = Path(path)
    if p.exists():
        p.unlink()
    b = _btn(d, "Export as Image", startswith=True)
    if b is None:
        return False, {"err": "no export button"}
    _click(b)
    pid = _pid()
    t0 = time.time()
    save = None
    while time.time() - t0 < 10:
        for h, c, t in _tops(pid):
            if c == "#32770" and t:
                save = h
                break
        if save:
            break
        time.sleep(0.4)
    if save is None:
        return False, {"err": "no save dialog"}
    _fg(save)
    fe = None
    for c in d.descendants(control_type="Edit"):
        try:
            r = c.rectangle()
            if 580 <= r.top <= 640 and r.left < 200 and r.right > 1000:
                fe = c
                break
        except Exception:
            continue
    if fe is None:
        return False, {"err": "no filename edit"}
    r0 = fe.rectangle()
    _clip_set(str(p))
    pyautogui.click((r0.left + r0.right) // 2, (r0.top + r0.bottom) // 2)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.6)
    sv = _btn(d, "保存(S)") or _btn(d, "保存", startswith=True, min_top=500)
    if sv is None:
        return False, {"err": "no save button"}
    _click(sv)
    time.sleep(1.8)

    t0 = time.time()
    while time.time() - t0 < 5:            # 覆盖确认（可选）
        yes = None
        for c in d.descendants(control_type="Button"):
            try:
                t = (c.window_text() or "").strip()
                if t in ("是(Y)", "是", "Yes") and c.rectangle().top > 300:
                    yes = c
                    break
            except Exception:
                continue
        if yes is not None:
            for h, c2, t in _tops(pid):
                if c2 == "#32770":
                    _fg(h)
            _click(yes)
            st["overwrite"] = True
            time.sleep(1.5)
            break
        time.sleep(0.3)

    opts = None
    t0 = time.time()
    while time.time() - t0 < 15:
        for h, c, t in _tops(pid):
            if t == "Image Export Options":
                opts = h
                break
        if opts:
            break
        time.sleep(0.4)
    if opts is None:
        return p.exists(), {"err": "no export options dialog"}
    _fg(opts)
    x0, y0, x1, y1 = win32gui.GetWindowRect(opts)
    sb = None
    for c in d.descendants(control_type="Button"):
        try:
            t = (c.window_text() or "").strip()
            r = c.rectangle()
            if t == "保存" and r.left >= x0 - 5 and r.top >= y0 - 5 \
                    and r.right <= x1 + 5 and r.bottom <= y1 + 5:
                sb = c
                break
        except Exception:
            continue
    if sb is None:
        return p.exists(), {"err": "no options save button"}
    _click(sb)
    time.sleep(3.0)
    st["export_ms"] = round((time.time() - t0) * 1000, 1)
    ok = p.exists() and p.stat().st_size > 0
    st["png_bytes"] = p.stat().st_size if p.exists() else 0
    return ok, st


def close_designer(d):
    """关闭设计器；未保存布局弹确认（保存/丢弃/取消）→ 丢弃。"""
    b = None
    for bb in d.descendants(control_type="Button"):
        try:
            if (bb.window_text() or "").strip() == "关闭" and bb.rectangle().top < 100:
                b = bb
                break
        except Exception:
            continue
    if b is None:
        return False
    _click(b)
    time.sleep(1.2)
    for c in d.descendants(control_type="Button"):
        try:
            t = (c.window_text() or "").strip()
            if t in ("丢弃", "放弃", "舍弃", "不保存", "Discard", "Don't Save",
                     "Close without Saving"):
                _click(c)
                time.sleep(0.8)
                return True
        except Exception:
            continue
    return True


def dismiss_prompts():
    """防御性清理残留提示（如 Save Project / 布局保存提示）→ 丢弃。"""
    pid = _pid()
    if pid is None:
        return False
    hit = False
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            _, p = win32process.GetWindowThreadProcessId(w.handle)
            if p != pid:
                continue
            for c in w.descendants(control_type="Button"):
                t = (c.window_text() or "").strip()
                if t in ("丢弃", "Don't Save"):
                    _fg(w.handle)
                    _click(c)
                    hit = True
                    time.sleep(0.8)
                    break
        except Exception:
            continue
    return hit
