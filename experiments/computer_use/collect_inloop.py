"""
QGIS W29：接环示范数据采集（确定性链当老师）。

把 W25/W26/W27 已验证的 UI 链（样式 → 制图 → 导出）改造成录制器：
每个真实 UI 交互（点击/键盘/菜单导航/拾色器/文件对话框）在执行前截全屏、
按链步骤语义生成英文模板指令，执行后记 bridge truth 摘要，
落盘为 DemoSample（data-model §1.2 契约，artifacts/computer_use_inloop/）。

动作空间与模型一致：click / type / scroll / move / hotkey / press
（scroll/move 本链不用；双击 = 两个 click 样本；可编辑 combo = click+ctrl+a+type+enter）。

用法: python experiments/computer_use/collect_inloop.py [--variants 20]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BR = ROOT / "experiments" / "qgis" / "qgis_bridge"
sys.path.insert(0, str(BR))
from client import call  # noqa: E402
from map_gui import (dismiss_prompts, _fg, _pid, _clip_set,  # noqa: E402
                     _tops, designer as find_designer)
from run_w16 import find_layer_item  # noqa: E402
from run_w18 import R, acquire  # noqa: E402
from scn_gui import btn, find_props  # noqa: E402
from scn_style import (_section_arrow, _wide_color_button,  # noqa: E402
                       find_select_color)
from scn_theme import combo_by_label, spinner_by_label  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402
import win32process  # noqa: E402

pyautogui.FAILSAFE = False

OUT = ROOT / "artifacts" / "computer_use_inloop"
SHOTS = OUT / "shots"
QGIS_BAT = r"D:\QGIS\bin\qgis-ltr.bat"
STARTUP = str(BR / "_startup.py")

BGS = ["#0A1220", "#0B1526", "#101D33", "#0E1A2E", "#122036", "#0D1B2A"]
PALETTES = [
    {"provinces": "#16283F", "roads": "#2C4A6E", "airports": "#4A6B8A",
     "ports": "#556B8C"},
    {"provinces": "#1A2F4A", "roads": "#2E4F78", "airports": "#3F6286",
     "ports": "#4A6E90"},
    {"provinces": "#14243A", "roads": "#28446A", "airports": "#42607E",
     "ports": "#4E6A88"},
    {"provinces": "#1E3049", "roads": "#32527A", "airports": "#4C6C8C",
     "ports": "#5A7290"},
    {"provinces": "#122033", "roads": "#24405F", "airports": "#3E5E7C",
     "ports": "#486482"},
]
RAMPS = ["Inferno", "Viridis", "Plasma", "Magma", "Cividis", "Blues"]
MODES = ["quantile", "equal", "jenks"]
CLASSES = [4, 5, 6, 7, 8]
TITLES = [
    "City Population Grades - Digital Twin Map",
    "Urban Population Ranking - Thematic Map",
    "City Scale Classification - Command View",
    "Population Grades - Dark Operations Map",
    "City Tier Map - Digital Twin Panel",
    "Urban Hierarchy - Night Operations View",
]
LEG_BGS = ["#0F1E33", "#12233B", "#0E1C30", "#14263E", "#101F36"]
LEG_FONTS = ["#DCE9F7", "#E3EEFB", "#D6E6F5", "#EAF2FC", "#DFEBFA"]

RECT_MAP = (0, 0, 297, 210)
RECT_TITLE = (10, 8, 160, 16)
RECT_LEGEND = (230, 26, 60, 92)
CLICK = {"Add Map": (800, 500), "Add Label": (700, 250),
         "Add Legend": (1300, 320)}


def ensure_qgis(timeout=50.0):
    try:
        call({"op": "ping"}, timeout=3)
        return False
    except Exception:
        pass
    env = dict(os.environ, PYQGIS_STARTUP=STARTUP)
    subprocess.Popen(["cmd", "/c", QGIS_BAT], env=env,
                     creationflags=0x00000008)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2.5)
        try:
            call({"op": "ping"}, timeout=3)
            time.sleep(2.0)
            call({"op": "win", "maximize": True})
            return True
        except Exception:
            continue
    return False


def main_qgis():
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") == "QgisApp":
                return w
        except Exception:
            continue
    return None


def make_params(seed, n):
    out = []
    for i in range(n):
        out.append({
            "bg": BGS[i % len(BGS)],
            "palette": PALETTES[(i // len(BGS)) % len(PALETTES)],
            "ramp": RAMPS[(i * 2 + 1) % len(RAMPS)],
            "mode": MODES[i % len(MODES)],
            "classes": CLASSES[i % len(CLASSES)],
            "title": TITLES[i % len(TITLES)],
            "legend_bg": LEG_BGS[i % len(LEG_BGS)],
            "legend_font": LEG_FONTS[(i * 3) % len(LEG_FONTS)],
        })
    return out


class Recorder:
    def __init__(self):
        self.chain = "v00"
        self.seq = 0
        self.buffer = []
        self.recording = True
        self.action_counts = Counter()
        self.phase_counts = Counter()

    def light_truth(self):
        try:
            t = (call({"op": "truth"}, timeout=5) or {}).get("truth") or {}
            layers = []
            for l in t.get("layers", []):
                r = l.get("renderer") or {}
                layers.append({"name": l.get("name"),
                               "renderer": r.get("type"),
                               "attr": r.get("attr"),
                               "ranges": r.get("ranges")})
            return {"bg": t.get("bg"), "scale": t.get("scale"),
                    "extent": t.get("extent"), "tool": t.get("tool"),
                    "layers": layers}
        except Exception:
            return None

    def snap(self, instruction, action, params, phase):
        if not self.recording:
            return None
        shot_name = f"{self.chain}_{self.seq:05d}.png"
        shot_path = SHOTS / shot_name
        try:
            pyautogui.screenshot().save(str(shot_path))
        except Exception:
            shot_path = None
        sample = {
            "id": f"qgis-{self.chain}-{self.seq:05d}",
            "screenshot": (str(shot_path.relative_to(ROOT)).replace("\\", "/")
                           if shot_path is not None else None),
            "instruction": instruction,
            "action": action,
            "params": params,
            "post_state": None,
            "meta": {"app": "qgis", "phase": phase,
                     "chain": f"collect_inloop/{self.chain}"},
        }
        self.seq += 1
        self.buffer.append(sample)
        self.action_counts[action] += 1
        self.phase_counts[phase] += 1
        return sample

    def click(self, x, y, instruction, phase):
        s = self.snap(instruction, "click",
                      {"x": int(x), "y": int(y)}, phase)
        pyautogui.click(int(x), int(y))
        time.sleep(0.35)
        if s is not None:
            s["post_state"] = self.light_truth()
        return s

    def click_el(self, el, instruction, phase):
        l, t, w, h = R(el)
        return self.click(l + w // 2, t + h // 2, instruction, phase)

    def hotkey(self, keys, instruction, phase):
        s = self.snap(instruction, "hotkey",
                      {"text": "+".join(keys)}, phase)
        pyautogui.hotkey(*keys)
        time.sleep(0.2)
        if s is not None:
            s["post_state"] = self.light_truth()
        return s

    def press(self, key, instruction, phase):
        s = self.snap(instruction, "press", {"text": key}, phase)
        pyautogui.press(key)
        time.sleep(0.15)
        if s is not None:
            s["post_state"] = self.light_truth()
        return s

    def type_text(self, text, instruction, phase, interval=0.03):
        s = self.snap(instruction, "type", {"text": str(text)}, phase)
        pyautogui.write(str(text), interval=interval)
        time.sleep(0.25)
        if s is not None:
            s["post_state"] = self.light_truth()
        return s

    def menu_item(self, win, menu_name, item_text, phase, exact=True):
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
        self.click_el(m, f"click the {menu_name} menu", phase)
        items = (call({"op": "menu_items"}, timeout=5) or {}).get("items") or []
        tgt = next((it["text"] for it in items if not it["sep"]
                    and ((it["text"] == item_text) if exact
                         else (item_text in (it["text"] or "")))), None)
        if tgt is None:
            self.press("esc", "press Escape to close the menu", phase)
            return False
        if not self.nav_menu(tgt, items, phase):
            self.press("esc", "press Escape to close the menu", phase)
            return False
        self.press("enter", f"press Enter to open the {tgt} menu item", phase)
        return True

    def nav_menu(self, want, items, phase):
        for _ in range(len(items) + 4):
            cur = (call({"op": "menu_items"}, timeout=5) or {}).get("current") or ""
            if cur and want in cur:
                return True
            self.press("down",
                       f"press Down to highlight the {want} menu item", phase)
            time.sleep(0.1)
        return False

    def open_props(self, layer, phase):
        qg = acquire()
        item = find_layer_item(qg, layer=layer)
        if item is None:
            return None
        l, t, w, h = R(item)
        cx, cy = l + w // 2, t + h // 2
        self.click(cx, cy,
                   f"click the {layer} layer in the layers panel", phase)
        time.sleep(0.1)
        self.click(cx, cy,
                   f"click the {layer} layer again to open its properties",
                   phase)
        for _ in range(20):
            time.sleep(0.4)
            qg = acquire()
            dlg = find_props(qg, layer)
            if dlg is not None:
                return dlg
        return None

    def pick_combo(self, combo, text, label, phase):
        low = text.lower()
        for _ in range(2):
            self.click_el(combo, f"click the {label} combo box", phase)
            time.sleep(0.4)
            self.hotkey(["ctrl", "a"],
                        f"select all text in the {label} combo box", phase)
            time.sleep(0.1)
            self.type_text(text,
                           f"type {text} into the {label} combo box",
                           phase, interval=0.05)
            time.sleep(0.2)
            self.press("enter",
                       f"press Enter to confirm the {label} combo box", phase)
            time.sleep(0.7)
            try:
                cur = ((combo.window_text() or "")
                       + (combo.selected_text() or "")).lower()
            except Exception:
                cur = ""
            if low in cur:
                return True
        return False

    def set_spinner(self, dlg, label, val, phase):
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
        l, t, w, h = R(target)
        self.click(l + w // 2, t + h // 2,
                   f"click the {label} spinner", phase)
        self.hotkey(["ctrl", "a"],
                    f"select all text in the {label} spinner", phase)
        self.type_text(str(val),
                       f"type {val} into the {label} spinner", phase)
        self.press("tab", "press Tab to leave the spinner", phase)
        return True

    def apply_color(self, hexv, phase):
        sc = find_select_color(timeout=2.5)
        if sc is not None:
            return self.color_modal(sc, hexv, phase)
        d = find_designer(timeout=1.5)
        if d is None:
            return False
        return self.color_inline(d, hexv, phase)

    def color_modal(self, sc, hexv, phase):
        ed = None
        for x in sc.descendants(control_type="Edit"):
            v = (x.window_text() or "").strip()
            if v.startswith("#") and len(v) == 7:
                ed = x
        if ed is None:
            return False
        l, t, w, h = R(ed)
        self.click(l + w // 2, t + h // 2,
                   "click the HTML notation field in the color dialog", phase)
        self.hotkey(["ctrl", "a"],
                    "select all text in the HTML notation field", phase)
        self.type_text(hexv,
                       f"type {hexv} into the HTML notation field", phase)
        ok = btn(sc, "确定")
        if ok is None:
            return False
        self.click_el(ok, "click the OK button in the color dialog", phase)
        time.sleep(0.6)
        return True

    def color_inline(self, d, hexv, phase):
        ed = None
        for c in d.descendants(control_type="Edit"):
            v = (c.window_text() or "").strip()
            if v.startswith("#") and len(v) == 7:
                ed = c
                break
        if ed is None:
            return False
        l, t, w, h = R(ed)
        self.click(l + w // 2, t + h // 2,
                   "click the HTML notation field in the color panel", phase)
        self.hotkey(["ctrl", "a"],
                    "select all text in the HTML notation field", phase)
        self.type_text(hexv,
                       f"type {hexv} into the HTML notation field", phase)
        self.press("enter", "press Enter to apply the color", phase)
        time.sleep(0.5)
        self.exit_inline(d, phase)
        return True

    def exit_inline(self, d, phase):
        for c in d.descendants(control_type="Button"):
            try:
                rr = R(c)
                if (not (c.window_text() or "").strip()
                        and 2160 < rr[0] < 2215 and 755 < rr[1] < 805
                        and rr[2] < 40):
                    self.click(rr[0] + rr[2] // 2, rr[1] + rr[3] // 2,
                               "click the back arrow to close the color panel",
                               phase)
                    return True
            except Exception:
                continue
        return False

    def set_canvas_bg(self, hexv, phase):
        w = main_qgis()
        if w is None:
            return False
        try:
            w.set_focus()
        except Exception:
            pass
        time.sleep(0.3)
        if not self.menu_item(w, "Project", "Properties", phase, exact=False):
            return False
        dlg = None
        for _ in range(20):
            time.sleep(0.4)
            for c in w.descendants():
                try:
                    if (c.element_info.control_type == "Window"
                            and "Project Properties" in (c.window_text() or "")):
                        dlg = c
                        break
                except Exception:
                    continue
            if dlg is not None:
                break
        if dlg is None:
            return False
        bgbtn = next((b for b in dlg.descendants(control_type="Button")
                      if (b.window_text() or "").strip() == "Background color"),
                     None)
        if bgbtn is None:
            return False
        self.click_el(bgbtn, "click the Background color button", phase)
        time.sleep(0.8)
        okc = self.apply_color(hexv, phase)
        okp = btn(dlg, "确定")
        if okp is not None:
            self.click_el(okp,
                          "click the OK button in the project properties dialog",
                          phase)
            time.sleep(0.8)
        return okc and okp is not None

    def set_layer_color(self, layer, hexv, phase):
        dlg = self.open_props(layer, phase)
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
        self.click_el(wide,
                      f"click the Color swatch of the {layer} layer", phase)
        time.sleep(0.8)
        okc = self.apply_color(hexv, phase)
        okp = btn(dlg, "确定")
        if okp is not None:
            self.click_el(okp,
                          f"click the OK button in the {layer} properties dialog",
                          phase)
            time.sleep(0.6)
        return okc and okp is not None

    def pick_ramp(self, ramp_btn, ramp_name, phase):
        r = R(ramp_btn)
        self.click(r[0] + r[2] - 6, r[1] + r[3] // 2,
                   "click the right edge of the Color ramp button to open the ramp menu",
                   phase)
        time.sleep(0.8)
        items = (call({"op": "menu_items"}, timeout=5) or {}).get("items") or []
        names = [it["text"] for it in items if not it["sep"]]
        if any(n == ramp_name for n in names):
            if self.nav_menu(ramp_name, items, phase):
                self.press("enter",
                           f"press Enter to select the {ramp_name} color ramp",
                           phase)
                time.sleep(0.6)
                return True
            self.press("esc", "press Escape to close the ramp menu", phase)
            return False
        if not any("All Color Ramps" in (n or "") for n in names):
            self.press("esc", "press Escape to close the ramp menu", phase)
            return False
        if not self.nav_menu("All Color Ramps", items, phase):
            self.press("esc", "press Escape to close the ramp menu", phase)
            return False
        self.press("right",
                   "press Right to open the All Color Ramps submenu", phase)
        time.sleep(0.7)
        sub = (call({"op": "menu_items"}, timeout=5) or {}).get("items") or []
        snames = [it["text"] for it in sub if not it["sep"]]
        if ramp_name not in snames:
            self.press("esc", "press Escape to close the ramp menu", phase)
            return False
        if not self.nav_menu(ramp_name, sub, phase):
            self.press("esc", "press Escape to close the ramp menu", phase)
            return False
        self.press("enter",
                   f"press Enter to select the {ramp_name} color ramp", phase)
        time.sleep(0.6)
        return True

    def graduated(self, layer, field, ramp, mode, classes, phase):
        dlg = self.open_props(layer, phase)
        if dlg is None:
            return False
        combos = sorted(dlg.descendants(control_type="ComboBox"),
                        key=lambda x: R(x)[1])
        if not self.pick_combo(combos[0], "graduated", "renderer", phase):
            return False
        time.sleep(0.8)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        if not self.pick_combo(combo_by_label(dlg, "Value"), field,
                               "value", phase):
            return False
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        rb = next((b for b in dlg.descendants(control_type="Button")
                   if (b.window_text() or "").strip() == "Color ramp"), None)
        if rb is None or not self.pick_ramp(rb, ramp, phase):
            return False
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        if not self.pick_combo(combo_by_label(dlg, "Method"), "color",
                               "method", phase):
            return False
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        if not self.pick_combo(combo_by_label(dlg, "Mode"), mode,
                               "mode", phase):
            return False
        if classes:
            qg = acquire()
            dlg = find_props(qg, layer)
            if dlg is not None:
                self.set_spinner(dlg, "Classes", classes, phase)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        cb = btn(dlg, "Classify")
        if cb is None:
            return False
        self.click_el(cb, "click the Classify button", phase)
        time.sleep(1.2)
        qg = acquire()
        dlg = find_props(qg, layer)
        if dlg is None:
            return False
        ok = btn(dlg, "确定")
        if ok is None:
            return False
        self.click_el(ok,
                      f"click the OK button in the {layer} properties dialog",
                      phase)
        time.sleep(0.8)
        return True

    def new_layout(self, phase):
        w = main_qgis()
        if w is None:
            return None
        try:
            w.set_focus()
        except Exception:
            pass
        time.sleep(0.3)
        if not self.menu_item(w, "Project", "New Print Layout", phase,
                              exact=False):
            return None
        d = find_designer(timeout=8)
        if d is None:
            self.press("enter",
                       "press Enter to confirm the default layout name", phase)
            time.sleep(1.0)
            d = find_designer(timeout=14)
        return d

    def setup_view(self, d, phase):
        b = None
        for x in d.descendants(control_type="Button"):
            try:
                if (x.window_text() or "").strip() == "最大化":
                    b = x
                    break
            except Exception:
                continue
        if b is not None:
            self.click_el(b,
                          "click the maximize button in the layout designer",
                          phase)
            time.sleep(0.8)
        b = None
        for x in d.descendants(control_type="Button"):
            try:
                if (x.window_text() or "").strip() == "Zoom Full":
                    b = x
                    break
            except Exception:
                continue
        if b is not None:
            self.click_el(b, "click the Zoom Full button in the layout designer",
                          phase)
            time.sleep(0.6)

    def fill_rect(self, dlg, x, y, w, h, phase):
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
            cand = [e for e in edits
                    if abs(e[0].top - tr.top) < 14 and e[0].left > tr.left]
            cand.sort(key=lambda e: (abs(e[0].top - tr.top), e[0].left))
            if not cand:
                return False
            e = cand[0][1]
            rr = e[0]
            self.click((rr.left + rr.right) // 2, (rr.top + rr.bottom) // 2,
                       f"click the {name} field", phase)
            self.hotkey(["ctrl", "a"],
                        f"select all text in the {name} field", phase)
            self.type_text(str(val), f"type {val} into the {name} field", phase)
            self.press("tab", "press Tab to move to the next field", phase)
            time.sleep(0.15)
        return True

    def add_item(self, d, name, click_xy, rect, phase):
        if not self.menu_item(d, "Add Item", name, phase):
            return False
        time.sleep(0.5)
        self.click(click_xy[0], click_xy[1],
                   f"click on the layout page to place the {name} item", phase)
        time.sleep(0.9)
        dlg = None
        t0 = time.time()
        while time.time() - t0 < 10:
            for c in d.descendants():
                try:
                    if (c.element_info.control_type == "Window"
                            and (c.window_text() or "").strip()
                            == "New Item Properties"):
                        dlg = c
                        break
                except Exception:
                    continue
            if dlg is not None:
                break
            time.sleep(0.3)
        if dlg is None:
            return False
        if not self.fill_rect(dlg, *rect, phase=phase):
            return False
        kb = btn(dlg, "确定")
        if kb is None:
            return False
        self.click_el(kb,
                      "click the OK button in the new item properties dialog",
                      phase)
        time.sleep(0.9)
        return True

    def activate_item_props(self, d, phase):
        for t in d.descendants(control_type="TabItem"):
            try:
                if (t.window_text() or "").strip() == "Item Properties":
                    self.click_el(t, "click the Item Properties tab", phase)
                    time.sleep(0.4)
                    return True
            except Exception:
                continue
        return False

    def set_title(self, d, text, phase):
        if not self.activate_item_props(d, phase):
            return False
        t0 = time.time()
        done = False
        while time.time() - t0 < 8 and not done:
            for c in d.descendants(control_type="Edit"):
                try:
                    if (c.window_text() or "").strip() == "Lorem ipsum":
                        l, t, w, h = R(c)
                        self.click(l + w // 2, t + h // 2,
                                   "click the label text field", phase)
                        self.hotkey(["ctrl", "a"],
                                    "select all text in the label text field",
                                    phase)
                        self.type_text(text,
                                       f"type {text} into the label text field",
                                       phase, interval=0.02)
                        done = True
                        break
                except Exception:
                    continue
            if not done:
                time.sleep(0.4)
        if not done:
            return False
        t0 = time.time()
        while time.time() - t0 < 4:
            for c in d.descendants(control_type="CheckBox"):
                try:
                    if (c.window_text() or "").strip() == "Render as HTML":
                        try:
                            if not c.get_toggle_state():
                                self.click_el(c,
                                              "click the Render as HTML checkbox",
                                              phase)
                                time.sleep(0.3)
                        except Exception:
                            pass
                        return True
                except Exception:
                    continue
            time.sleep(0.3)
        return True

    def set_extent_canvas(self, d, phase):
        b = None
        for x in d.descendants(control_type="Button"):
            try:
                if (x.window_text() or "").strip() == "Set to Map Canvas Extent":
                    b = x
                    break
            except Exception:
                continue
        if b is None:
            return False
        self.click_el(b, "click the Set to Map Canvas Extent button", phase)
        time.sleep(0.7)
        return True

    def legend_style(self, bg_hex, font_hex, phase):
        d = find_designer(timeout=10)
        if d is None:
            return False
        for name in ("Main Properties", "Legend Items"):
            a = _section_arrow(d, name)
            if a is not None:
                self.click_el(a,
                              f"click the {name} section arrow to collapse it",
                              phase)
                time.sleep(0.6)
        d = find_designer(timeout=5)
        bg_visible = False
        for _ in range(3):
            a = _section_arrow(d, "Background")
            if a is not None and 700 < R(a)[1] < 1400:
                bg_visible = True
                break
            for name in ("Main Properties", "Legend Items"):
                a2 = _section_arrow(d, name)
                if a2 is not None:
                    self.click_el(a2,
                                  f"click the {name} section arrow to expand it",
                                  phase)
                    time.sleep(0.6)
            d = find_designer(timeout=5)
        if not bg_visible:
            return False
        a = _section_arrow(d, "Background")
        if a is None:
            return False
        self.click_el(a,
                      "click the Background section arrow to expand it", phase)
        time.sleep(0.8)
        d = find_designer(timeout=5)
        wb = None
        for c in d.descendants(control_type="Button"):
            try:
                if ((c.window_text() or "").strip() == "Background"
                        and R(c)[2] > 200):
                    wb = c
                    break
            except Exception:
                continue
        if wb is None:
            return False
        for c in d.descendants(control_type="CheckBox"):
            try:
                if ((c.window_text() or "").strip() == "Background"
                        and R(c)[2] > 300):
                    try:
                        if not c.get_toggle_state():
                            self.click_el(c,
                                          "click the Background checkbox to enable it",
                                          phase)
                            time.sleep(0.3)
                    except Exception:
                        pass
                    break
            except Exception:
                continue
        self.click_el(wb, "click the Background color swatch", phase)
        time.sleep(1.2)
        bg_ok = self.apply_color(bg_hex, phase)
        d = find_designer(timeout=5)
        a = _section_arrow(d, "Fonts and Text Formatting")
        if a is not None:
            self.click_el(a,
                          "click the Fonts and Text Formatting section arrow to expand it",
                          phase)
            time.sleep(0.8)
        d = find_designer(timeout=5)
        ifont = None
        for c in d.descendants(control_type="Button"):
            try:
                if (c.window_text() or "").strip() == "Item font":
                    ifont = c
                    break
            except Exception:
                continue
        if ifont is None:
            return False
        self.click_el(ifont, "click the Item font button", phase)
        time.sleep(0.9)
        d = find_designer(timeout=5)
        lbl = None
        for t in d.descendants(control_type="Text"):
            try:
                if (t.window_text() or "").strip() == "Color" and R(t)[0] > 2100:
                    lbl = t
                    break
            except Exception:
                continue
        if lbl is None:
            return False
        lr = R(lbl)
        cands = []
        for b in d.descendants(control_type="Button"):
            try:
                rr = R(b)
                if (abs(rr[1] - lr[1]) < 16 and rr[0] > lr[0]
                        and (b.window_text() or "").strip() != "…"):
                    cands.append((rr[0], b))
            except Exception:
                continue
        if not cands:
            return False
        cands.sort(key=lambda x: x[0])
        self.click_el(cands[0][1],
                      "click the Color swatch in the text format panel", phase)
        time.sleep(0.9)
        font_ok = self.apply_color(font_hex, phase)
        d = find_designer(timeout=5)
        for c in d.descendants(control_type="Button"):
            try:
                rr = R(c)
                if (not (c.window_text() or "").strip()
                        and 2160 < rr[0] < 2215 and 755 < rr[1] < 805
                        and rr[2] < 40):
                    self.click(rr[0] + rr[2] // 2, rr[1] + rr[3] // 2,
                               "click the back arrow to leave the text format panel",
                               phase)
                    break
            except Exception:
                continue
        return bool(bg_ok) and bool(font_ok)

    def export_image(self, d, path, phase):
        p = Path(path)
        if p.exists():
            p.unlink()
        b = None
        for x in d.descendants(control_type="Button"):
            try:
                if (x.window_text() or "").strip().startswith("Export as Image"):
                    b = x
                    break
            except Exception:
                continue
        if b is None:
            return False
        self.click_el(b, "click the Export as Image button", phase)
        pid = _pid()
        save = None
        t0 = time.time()
        while time.time() - t0 < 10:
            for h, c, t in _tops(pid):
                if c == "#32770" and t:
                    save = h
                    break
            if save is not None:
                break
            time.sleep(0.4)
        if save is None:
            return False
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
            return False
        r0 = fe.rectangle()
        _clip_set(str(p))
        self.click((r0.left + r0.right) // 2, (r0.top + r0.bottom) // 2,
                   "click the file name field in the save dialog", phase)
        self.hotkey(["ctrl", "a"],
                    "select all text in the file name field", phase)
        self.hotkey(["ctrl", "v"],
                    "paste the export path into the file name field", phase)
        time.sleep(0.5)
        sv = None
        for x in d.descendants(control_type="Button"):
            try:
                t = (x.window_text() or "").strip()
                if t == "保存(S)" or (t.startswith("保存")
                                      and x.rectangle().top > 500):
                    sv = x
                    break
            except Exception:
                continue
        if sv is None:
            return False
        self.click_el(sv, "click the Save button", phase)
        time.sleep(1.6)
        t0 = time.time()
        while time.time() - t0 < 5:
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
                self.click_el(yes,
                              "click the Yes button to overwrite the existing file",
                              phase)
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
            if opts is not None:
                break
            time.sleep(0.4)
        if opts is None:
            return p.exists()
        _fg(opts)
        x0, y0, x1, y1 = None, None, None, None
        try:
            import win32gui
            x0, y0, x1, y1 = win32gui.GetWindowRect(opts)
        except Exception:
            pass
        sb = None
        for c in d.descendants(control_type="Button"):
            try:
                t = (c.window_text() or "").strip()
                r = c.rectangle()
                if (t == "保存" and x0 is not None and r.left >= x0 - 5
                        and r.top >= y0 - 5 and r.right <= x1 + 5
                        and r.bottom <= y1 + 5):
                    sb = c
                    break
            except Exception:
                continue
        if sb is None:
            return p.exists()
        self.click_el(sb, "click the Save button in the image export options dialog",
                      phase)
        time.sleep(3.0)
        return p.exists() and p.stat().st_size > 0

    def close_designer(self, d, phase):
        b = None
        for bb in d.descendants(control_type="Button"):
            try:
                if (bb.window_text() or "").strip() == "关闭" \
                        and bb.rectangle().top < 100:
                    b = bb
                    break
            except Exception:
                continue
        if b is None:
            return False
        self.click_el(b, "click the close button of the layout designer", phase)
        time.sleep(1.1)
        for c in d.descendants(control_type="Button"):
            try:
                t = (c.window_text() or "").strip()
                if t in ("丢弃", "放弃", "舍弃", "不保存", "Discard",
                         "Don't Save", "Close without Saving"):
                    self.click_el(c, "click the discard changes button", phase)
                    time.sleep(0.7)
                    return True
            except Exception:
                continue
        return True

    def reset(self):
        self.recording = False
        ensure_qgis()
        call({"op": "close_dialogs"})
        pyautogui.moveTo(1280, 700)
        pyautogui.press("esc")
        time.sleep(0.3)
        d = find_designer(timeout=1.0)
        if d is not None:
            try:
                self.close_designer(d, "layout")
            except Exception:
                pass
            time.sleep(2.0)
            ensure_qgis()
        dismiss_prompts()
        call({"op": "reset", "scene": "real"})
        time.sleep(0.6)
        self.recording = True

    def run_variant(self, i, params):
        chain = f"v{i:02d}"
        self.chain = chain
        self.seq = 0
        self.buffer = []
        self.reset()
        phase = "symbology"
        if not self.set_canvas_bg(params["bg"], phase):
            return False, "style_bg"
        for layer, hexv in params["palette"].items():
            if not self.set_layer_color(layer, hexv, phase):
                return False, f"style_{layer}"
        if not self.graduated("cities", "pop_max", params["ramp"],
                              params["mode"], params["classes"], phase):
            return False, "style_cities"
        phase = "layout"
        d = self.new_layout(phase)
        if d is None:
            return False, "layout"
        self.setup_view(d, phase)
        if not self.add_item(d, "Add Map", CLICK["Add Map"], RECT_MAP, phase):
            return False, "map"
        self.set_extent_canvas(d, phase)
        if not self.add_item(d, "Add Label", CLICK["Add Label"],
                             RECT_TITLE, phase):
            return False, "title"
        if not self.set_title(d, params["title"], phase):
            return False, "title"
        if not self.add_item(d, "Add Legend", CLICK["Add Legend"],
                             RECT_LEGEND, phase):
            return False, "legend"
        if not self.legend_style(params["legend_bg"], params["legend_font"],
                                 phase):
            return False, "legend"
        phase = "export"
        out_p = OUT / "exports" / f"{chain}.png"
        if not self.export_image(d, out_p, phase):
            return False, "export"
        if not self.close_designer(d, "layout"):
            return False, "close"
        return True, None

    def variant_truth(self, out_p, bg, classes):
        t = (call({"op": "truth"}) or {}).get("truth") or {}
        rend = next((l.get("renderer") for l in t.get("layers", [])
                     if l["name"] == "cities"), None)
        rend_ok = bool(rend) and rend.get("type") == "graduatedSymbol" \
            and rend.get("attr") == "pop_max" and rend.get("method") == "color" \
            and rend.get("ranges") == classes
        lays = (call({"op": "layouts"}) or {}).get("layouts") or []
        lay = lays[0] if lays else None
        items = (lay or {}).get("items") or []
        types = [i.get("type") for i in items]
        p = Path(out_p)
        png_ok = p.exists() and p.stat().st_size > 50000
        return {"renderer": rend, "render_ok": rend_ok, "bg": t.get("bg"),
                "bg_ok": t.get("bg") == bg.lower(), "n_layouts": len(lays),
                "items": types, "label": any(i.get("type") == "QgsLayoutItemLabel"
                                             for i in items),
                "legend": "QgsLayoutItemLegend" in types,
                "map": "QgsLayoutItemMap" in types, "png_ok": png_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=20)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    (OUT / "exports").mkdir(parents=True, exist_ok=True)
    ensure_qgis()
    try:
        call({"op": "win", "maximize": True})
    except Exception:
        pass
    rec = Recorder()
    params_list = make_params(args.seed, args.variants)
    manifest = {"screen": [pyautogui.size().width, pyautogui.size().height],
                "variants": []}
    files = []
    for i in range(args.variants):
        params = params_list[i]
        t0 = time.time()
        try:
            ok, fail = rec.run_variant(i, params)
        except Exception as e:
            ok, fail = False, f"exception: {e}"
        total = round(time.time() - t0, 1)
        path = OUT / f"demo_v{i:02d}.json"
        path.write_text(json.dumps(rec.buffer, ensure_ascii=False),
                        encoding="utf-8")
        files.append(path)
        truth = None
        if ok:
            try:
                truth = rec.variant_truth(OUT / "exports" / f"v{i:02d}.png",
                                          params["bg"], params["classes"])
                if not (truth.get("render_ok") and truth.get("bg_ok")
                        and truth.get("legend") and truth.get("map")
                        and truth.get("png_ok")):
                    ok = False
                    fail = "truth"
            except Exception as e:
                ok = False
                fail = f"truth: {e}"
        row = {"variant": i, "ok": bool(ok), "fail": fail,
               "samples": len(rec.buffer), "total_s": total,
               "truth": truth,
               "params": {"bg": params["bg"], "ramp": params["ramp"],
                          "mode": params["mode"], "classes": params["classes"],
                          "title": params["title"]}}
        manifest["variants"].append(row)
        mark = "PASS" if ok else f"FAIL({fail})"
        print(f"  v{i:02d}: {mark} {total}s samples={len(rec.buffer)}", flush=True)
        (OUT / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8")
    total_samples = sum(r["samples"] for r in manifest["variants"])
    splits = {"train": [], "val": []}
    for i, f in enumerate(files):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        splits["val" if i % args.val_every == args.val_every - 1 else "train"].append(rel)
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2),
                                     encoding="utf-8")
    manifest["total_samples"] = total_samples
    manifest["action_counts"] = dict(rec.action_counts)
    manifest["phase_counts"] = dict(rec.phase_counts)
    manifest["splits"] = {k: len(v) for k, v in splits.items()}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nW29 采集完成：{total_samples} 样本 "
          f"（train {len(splits['train'])} 链 / val {len(splits['val'])} 链）")
    print(f"动作分布: {dict(rec.action_counts)}")
    print(f"阶段分布: {dict(rec.phase_counts)}")
    print(f"已写出 {OUT / 'manifest.json'} / splits.json / demo_v*.json")


if __name__ == "__main__":
    main()
