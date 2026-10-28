"""
应用无关录制器（W28b / A' 路线）：双输入层 patch → 六要素示范样本。

捕获两层真实输入（软件无关）：
  pyautogui：click / doubleClick / dragTo / write / press / hotkey / scroll
  pywinauto ：BaseWrapper.click_input / double_click_input / type_keys、
              UIA 基类 invoke、EditWrapper.set_edit_text、ButtonWrapper.toggle

每个事件前后：全屏截图（与 v3 训练帧同规约）+ 观察（应用提供者注入：
QGIS 走 bridge truth/structure；Notepad 等无特权应用走 UIA 扫描）。
数据格式：docs/taskhub/c3ecabb0/data-model.md §1.2 DemoSample v2。

用法（collector 内）：
    rec = Recorder("notepad", OUT, obs_provider=notepad_observation)
    rec.install()
    rec.new_task("np_0001", task_name="write_note", chain="notepad_teacher")
    ... 教师链（真实 UI 操作自动落盘）...
    rec.end_task(ok=True, verify={"file_bytes": 123})
    rec.uninstall()
"""
import json
import time
from pathlib import Path

import pyautogui

pyautogui.FAILSAFE = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Recorder:
    def __init__(self, app, out_root, obs_provider=None, shot_scale=0.6,
                 settle=0.4, save_post=True):
        self.app = app
        self.out_root = Path(out_root) / app
        self.obs_provider = obs_provider
        self.shot_scale = float(shot_scale)
        self.settle = float(settle)
        self.save_post = bool(save_post)
        self.screen = [int(v) for v in pyautogui.size()]
        self.demos = []
        self.seq = 0
        self.task_id = None
        self.task_dir = None
        self.task_name = ""
        self.chain = ""
        self.phase = "idle"
        self.goal = ""
        self.perturbation = "nominal"
        self.failure = None
        self.instr_override = None
        self.paused = False
        self.installed = False
        self._in_event = False
        self._orig = {}

    # ---------------- lifecycle ----------------
    def install(self):
        if self.installed:
            return
        self._patch_pyautogui()
        self._patch_pywinauto()
        self.installed = True

    def uninstall(self):
        if not self.installed:
            return
        for obj, name, orig in self._orig.values():
            setattr(obj, name, orig)
        self._orig.clear()
        self.installed = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    # ---------------- context ----------------
    def set_phase(self, phase, goal=""):
        self.phase = phase
        self.goal = goal

    def set_perturbation(self, kind):
        self.perturbation = kind

    def tag_failure(self, kind, note="", instr=None):
        self.failure = {"kind": kind, "injected": True, "note": note}
        self.instr_override = instr

    def type_text(self, text, instruction=None):
        """文本输入：剪贴板粘贴执行（绕 IME/丢键），录制为 type 动作。"""
        import win32clipboard as _clip

        def _do(t):
            _clip.OpenClipboard()
            try:
                _clip.EmptyClipboard()
                _clip.SetClipboardText(t)
            finally:
                _clip.CloseClipboard()
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.45)

        instr = instruction or f"Type '{text}'"
        self._around(_do, (text,), {}, "type", {"text": str(text)},
                     instr, None)

    # ---------------- tasks ----------------
    def new_task(self, task_id, task_name="", chain=""):
        self.task_id = task_id
        self.task_name = task_name
        self.chain = chain
        self.task_dir = self.out_root / task_id
        (self.task_dir / "shots").mkdir(parents=True, exist_ok=True)
        self.demos = []
        self.seq = 0

    def end_task(self, ok, verify=None):
        if self.task_id is None:
            return
        manifest = {
            "task_id": self.task_id, "app": self.app,
            "task_name": self.task_name, "chain": self.chain,
            "perturbation": self.perturbation, "ok": bool(ok),
            "n_demos": len(self.demos), "screen": self.screen,
            "verify": verify or {},
        }
        (self.task_dir / "task.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self._flush_demos()
        idx = self.out_root / "index.json"
        tasks = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else []
        tasks.append(manifest)
        idx.write_text(json.dumps(tasks, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        self.task_id = None

    # ---------------- recording core ----------------
    def _shot(self, tag):
        from PIL import ImageGrab
        img = ImageGrab.grab()
        if self.shot_scale != 1.0:
            img = img.resize((int(self.screen[0] * self.shot_scale),
                              int(self.screen[1] * self.shot_scale)))
        p = self.task_dir / "shots" / f"{self.seq:04d}_{tag}.png"
        img.save(p)
        return str(p.resolve().relative_to(ROOT))

    def _obs(self):
        if not self.obs_provider:
            return {}
        try:
            return self.obs_provider()
        except Exception as e:  # 观察失败不阻断录制
            return {"error": f"{type(e).__name__}: {e}"}

    def _around(self, fn, args, kwargs, action, params, instruction,
                target_rect):
        """事件包装：前观察+截图 → 原语执行 → 稳定 → 后观察+截图 → 落盘。"""
        if self.paused or self.task_id is None or self._in_event:
            return fn(*args, **kwargs)
        self._in_event = True
        try:
            pre_obs = self._obs()
            pre_shot = self._shot("pre")
            err = None
            try:
                ret = fn(*args, **kwargs)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                ret = None
            time.sleep(self.settle)
            post_obs = self._obs()
            post_shot = self._shot("post") if self.save_post else None
            fail = self.failure
            self.failure = None
            self._append(action, params, instruction, target_rect,
                         pre_obs, pre_shot, post_obs, post_shot, fail, err)
            if err:
                raise Exception(err)
            return ret
        finally:
            self._in_event = False

    def _append(self, action, params, instruction, target_rect,
                pre_obs, pre_shot, post_obs, post_shot, failure, err):
        if self.instr_override:
            instruction = self.instr_override
            self.instr_override = None
        W, H = self.screen
        nc = None
        if "x" in params and "y" in params:
            nc = [round(params["x"] / W, 4), round(params["y"] / H, 4)]
        demo = {
            "id": f"{self.task_id}:{self.seq}",
            "screenshot": pre_shot,
            "instruction": instruction,
            "action": action,
            "params": params,
            "norm_coord": nc,
            "target_rect": target_rect,
            "screen": list(self.screen),
            "observation": pre_obs,
            "intent": {"task": self.task_name, "phase": self.phase,
                       "goal": self.goal},
            "result": {"screenshot": post_shot, "truth": post_obs,
                       "ok": err is None},
            "context": {"app": self.app, "scene": self.app,
                        "active_layer": None,
                        "window_rect": None,
                        "perturbation": self.perturbation,
                        "chain": self.chain},
            "failure": failure,
        }
        if err:
            demo["result"]["error"] = err
        self.demos.append(demo)
        self.seq += 1
        self._flush_demos()

    def _flush_demos(self):
        if self.task_dir is not None:
            (self.task_dir / "demo.json").write_text(
                json.dumps(self.demos, ensure_ascii=False, indent=2),
                encoding="utf-8")

    # ---------------- pyautogui layer ----------------
    def _patch_pyautogui(self):
        def wrap(mod, name, builder):
            orig = getattr(mod, name)
            self._orig[("pa", name)] = (mod, name, orig)

            def patched(*args, **kwargs):
                action, params, instruction = builder(*args, **kwargs)
                return self._around(orig, args, kwargs, action, params,
                                    instruction, None)
            patched.__name__ = name
            setattr(mod, name, patched)

        def b_click(x=None, y=None, *a, **k):
            x, y = self._xy(x, y)
            return ("click", {"x": int(x), "y": int(y)},
                    self._instr_phase("click", x, y))

        def b_dblclick(x=None, y=None, *a, **k):
            x, y = self._xy(x, y)
            return ("click", {"x": int(x), "y": int(y)},
                    self._instr_phase("double-click", x, y))

        def b_drag(x=None, y=None, *a, **k):
            x, y = self._xy(x, y)
            return ("move", {"x": int(x), "y": int(y)},
                    self._instr_phase("drag to", x, y))

        def b_write(text="", *a, **k):
            return ("type", {"text": str(text)},
                    f"Type '{text}'")

        def b_press(key="", *a, **k):
            return ("press", {"text": str(key)},
                    f"Press {key}")

        def b_hotkey(*keys, **k):
            combo = "+".join(str(k2) for k2 in keys)
            return ("hotkey", {"text": combo},
                    f"Press {combo}")

        def b_scroll(clicks=0, *a, **k):
            d = "up" if clicks > 0 else "down"
            return ("scroll", {"direction": d}, f"Scroll {d}")

        wrap(pyautogui, "click", b_click)
        wrap(pyautogui, "doubleClick", b_dblclick)
        wrap(pyautogui, "dragTo", b_drag)
        wrap(pyautogui, "write", b_write)
        wrap(pyautogui, "press", b_press)
        wrap(pyautogui, "hotkey", b_hotkey)
        wrap(pyautogui, "scroll", b_scroll)

    def _xy(self, x, y):
        if x is None or y is None:
            px, py = pyautogui.position()
            return (px if x is None else x, py if y is None else y)
        return (x, y)

    _PHASE_INSTR = {
        "write": "Click in the text editor",
        "save": "Click in the filename field",
        "layout_add_map": "Click on the page to place the map item",
        "layout_add_label": "Click on the page to place the label",
        "layout_add_legend": "Click on the page to place the legend",
        "canvas": "Click on the map canvas",
    }

    def _instr_phase(self, verb, x, y):
        tpl = self._PHASE_INSTR.get(self.phase)
        if tpl:
            return f"{verb} {tpl.split(' ', 1)[1]}" if verb == "click" \
                else f"{verb} {tpl}"
        return f"{verb} the target control at ({int(x)}, {int(y)})"

    # ---------------- pywinauto layer ----------------
    def _patch_pywinauto(self):
        from pywinauto.base_wrapper import BaseWrapper
        from pywinauto.controls.uia_controls import (
            ButtonWrapper, EditWrapper)

        uia_base = next(c for c in EditWrapper.__mro__
                        if "invoke" in c.__dict__)

        def wrap_cls(cls, name, builder):
            orig = getattr(cls, name)
            self._orig[(cls.__name__, name)] = (cls, name, orig)

            def patched(el, *args, **kwargs):
                action, params, instruction, rect = builder(el, *args,
                                                            **kwargs)
                return self._around(orig, (el,) + args, kwargs, action,
                                    params, instruction, rect)
            patched.__name__ = name
            setattr(cls, name, patched)

        def el_info(el):
            try:
                text = (el.window_text() or "").strip()
            except Exception:
                text = ""
            try:
                ct = el.element_info.control_type
            except Exception:
                ct = "Control"
            try:
                r = el.rectangle()
            except Exception:
                r = None
            return text, ct, r

        def rect_list(r):
            if r is None:
                return None
            return [int(r.left), int(r.top), int(r.right), int(r.bottom)]

        def center(r):
            if r is None:
                return (0, 0)
            return (int((r.left + r.right) // 2), int((r.top + r.bottom) // 2))

        def b_click_input(el, *a, **k):
            text, ct, r = el_info(el)
            x, y = center(r)
            instr = (f"Click the '{text}' {ct.lower()}" if text
                     else f"Click the {ct.lower()}")
            return ("click", {"x": x, "y": y}, instr, rect_list(r))

        def b_dblclick_input(el, *a, **k):
            text, ct, r = el_info(el)
            x, y = center(r)
            instr = (f"Double-click the '{text}' {ct.lower()}" if text
                     else f"Double-click the {ct.lower()}")
            return ("click", {"x": x, "y": y}, instr, rect_list(r))

        def b_type_keys(el, text="", *a, **k):
            t = str(text)
            if "{" in t:
                return ("hotkey", {"text": t}, f"Press {t}", None)
            return ("type", {"text": t}, f"Type '{t}'", None)

        def b_invoke(el, *a, **k):
            text, ct, r = el_info(el)
            x, y = center(r)
            instr = (f"Click the '{text}' {ct.lower()}" if text
                     else f"Click the {ct.lower()}")
            return ("click", {"x": x, "y": y}, instr, rect_list(r))

        def b_set_edit(el, text="", *a, **k):
            t = str(text)
            _, ct, r = el_info(el)
            instr = f"Type '{t}' into the {ct.lower()}"
            return ("type", {"text": t}, instr, rect_list(r))

        def b_toggle(el, *a, **k):
            text, ct, r = el_info(el)
            x, y = center(r)
            instr = (f"Toggle the '{text}' {ct.lower()}" if text
                     else f"Toggle the {ct.lower()}")
            return ("click", {"x": x, "y": y}, instr, rect_list(r))

        wrap_cls(BaseWrapper, "click_input", b_click_input)
        wrap_cls(BaseWrapper, "double_click_input", b_dblclick_input)
        wrap_cls(BaseWrapper, "type_keys", b_type_keys)
        wrap_cls(uia_base, "invoke", b_invoke)
        wrap_cls(EditWrapper, "set_edit_text", b_set_edit)
        wrap_cls(ButtonWrapper, "toggle", b_toggle)
