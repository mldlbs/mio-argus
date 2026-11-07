"""Bigemap 无特权执行基准 runner（T1 窗口层已接，T2–T5 待接）。

协议：docs/BIGEMAP_EXEC_BENCH.md
输出：JSONL（一行一步 + 一行一 trial），可直接喂给 exec_bench.py 汇总。

用法::

    python experiments/computer_use/bigemap_runner.py --probe          # 只读体检，不动窗口
    python experiments/computer_use/bigemap_runner.py --task T1 --trials 5
    python experiments/computer_use/bigemap_runner.py --task T1 --trials 5 --inject F2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pyautogui  # noqa: F401  （后续 T3/T4 用；同时启用 FAILSAFE 语义）
import win32api
import win32con
import win32gui
import win32process
from pywinauto import Desktop

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]

from exec_bench import validate  # noqa: E402

OUT_DIR = ROOT / "results" / "computer_use"
DL_ROOT = ROOT / "artifacts" / "bench_dl"

CFG_PATH = HERE / "bench_config.json"
CFG: dict = {}
try:
    CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
except Exception:
    CFG = {}

TITLE_KEY = CFG.get("app_title_key", "Bigemap GIS Office")
CANON = tuple(CFG.get("window_rect", (1344, 16, 2544, 916)))
SCREEN = tuple(CFG.get("screen", (2560, 1440)))
RECT_TOL = 2
DIALOG_TITLES = ("下载对话框", "导入边界", "选择文件", "定位", "提示", "警告", "错误")
MIN_MAIN_NODES = 200

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def _now() -> float:
    return time.perf_counter()


def _ms(t0: float, t1: float | None = None) -> float:
    return round(((t1 if t1 is not None else _now()) - t0) * 1000, 1)


def list_windows() -> list[dict]:
    out: list[dict] = []

    def cb(h, _):
        try:
            if not win32gui.IsWindowVisible(h):
                return
            L, T, R, B = win32gui.GetWindowRect(h)
            out.append({
                "hwnd": h,
                "cls": win32gui.GetClassName(h),
                "title": win32gui.GetWindowText(h),
                "rect": [L, T, R, B],
                "iconic": bool(win32gui.IsIconic(h)),
                "enabled": bool(win32gui.IsWindowEnabled(h)),
            })
        except Exception:
            pass

    win32gui.EnumWindows(cb, out)
    return out


def find_main() -> dict | None:
    cands = [w for w in list_windows()
             if w["cls"].startswith("Qt") and TITLE_KEY in w["title"]]
    return cands[0] if cands else None


def find_dialogs(main_hwnd: int | None = None) -> list[dict]:
    out = []
    for w in list_windows():
        if w["hwnd"] == main_hwnd:
            continue
        if not w["cls"].startswith("Qt"):
            continue
        if any(k in w["title"] for k in DIALOG_TITLES):
            out.append(w)
    return out


def window_state(h: int) -> dict:
    L, T, R, B = win32gui.GetWindowRect(h)
    fg = win32gui.GetForegroundWindow()
    return {
        "hwnd": h,
        "rect": [L, T, R, B],
        "minimized": bool(win32gui.IsIconic(h)),
        "visible": bool(win32gui.IsWindowVisible(h)),
        "enabled": bool(win32gui.IsWindowEnabled(h)),
        "foreground": fg == h,
        "foreground_hwnd": fg,
    }


def rect_ok(rect, canon=CANON, tol=RECT_TOL) -> bool:
    return all(abs(int(a) - int(b)) <= tol for a, b in zip(rect, canon))


def uia_nodes(h: int) -> int:
    return len(Desktop(backend="uia").window(handle=h).descendants())


def force_foreground(h: int, timeout: float = 3.0) -> bool:
    """SetForegroundWindow + AttachThreadInput 兜底；返回是否真的前台。"""
    if win32gui.GetForegroundWindow() == h:
        return True
    try:
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(h)
    except Exception:
        pass
    end = time.time() + timeout
    while time.time() < end:
        if win32gui.GetForegroundWindow() == h:
            return True
        fg = win32gui.GetForegroundWindow()
        if fg:
            fg_tid, _ = win32process.GetWindowThreadProcessId(fg)
            cur_tid = win32api.GetCurrentThreadId()
            attached = False
            try:
                win32api.AttachThreadInput(cur_tid, fg_tid, True)
                attached = True
                win32gui.BringWindowToTop(h)
                win32gui.SetForegroundWindow(h)
            except Exception:
                pass
            finally:
                if attached:
                    try:
                        win32api.AttachThreadInput(cur_tid, fg_tid, False)
                    except Exception:
                        pass
        if win32gui.GetForegroundWindow() != h:
            # 前台锁兜底：模拟一次 Alt 键，使本线程重新获得前台权限
            try:
                win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                time.sleep(0.02)
                win32gui.SetForegroundWindow(h)
            except Exception:
                pass
            finally:
                try:
                    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                except Exception:
                    pass
        time.sleep(0.15)
    return win32gui.GetForegroundWindow() == h


def restore_window(h: int, timeout: float = 4.0) -> dict:
    """最小化/摆放/前台，返回实际达成情况与耗时。"""
    t0 = _now()
    placement = win32gui.GetWindowPlacement(h)
    if win32gui.IsIconic(h) or (len(placement) > 1 and placement[1] == win32con.SW_SHOWMAXIMIZED):
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
    L, T, R, B = CANON
    win32gui.SetWindowPos(h, win32con.HWND_TOP, L, T, R - L, B - T,
                          win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE)
    end = time.time() + timeout
    while time.time() < end:
        if not win32gui.IsIconic(h) and rect_ok(win32gui.GetWindowRect(h)):
            break
        time.sleep(0.1)
    fg_ok = force_foreground(h)
    return {"ms": _ms(t0), "rect_ok": rect_ok(win32gui.GetWindowRect(h)),
            "foreground_ok": fg_ok}


def close_window(h: int, timeout: float = 1.0) -> bool:
    """UIA Close 优先，WM_CLOSE 兜底；条件轮询，不用固定 sleep。"""
    try:
        Desktop(backend="uia").window(handle=h).close()
    except Exception:
        try:
            win32gui.PostMessage(h, win32con.WM_CLOSE, 0, 0)
        except Exception:
            return False
    ok, _ = wait_for(lambda: not win32gui.IsWindow(h), timeout, 0.02)
    return ok


def inventory(d: Path) -> dict:
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): {"size": p.stat().st_size, "mtime": round(p.stat().st_mtime, 3)}
            for p in d.rglob("*") if p.is_file()}


def snapshot() -> dict:
    main = find_main()
    if main is None:
        raise SystemExit("找不到 Bigemap 主窗口（标题含 'Bigemap GIS Office'）")
    st = window_state(main["hwnd"])
    st["control"] = {"name": main["title"], "type": "Window",
                     "rect": st["rect"], "patterns": ["WindowPattern"]}
    return st


def reset(trial: str) -> dict:
    """R1–R6；返回 {ok, ms, pre_state, trial_dir, pre_files, closed}。"""
    t0 = _now()
    trial_dir = DL_ROOT / trial
    trial_dir.mkdir(parents=True, exist_ok=True)
    pre_files = inventory(trial_dir)

    main = find_main()
    if main is None:
        return {"ok": False, "ms": _ms(t0), "reason": "no_main_window"}

    closed = [d["title"] for d in find_dialogs(main["hwnd"]) if close_window(d["hwnd"])]
    left = find_dialogs(main["hwnd"])
    if left:
        return {"ok": False, "ms": _ms(t0), "reason": f"dialogs_left:{len(left)}"}

    r = restore_window(main["hwnd"])
    st = window_state(main["hwnd"])
    ok = r["rect_ok"] and st["visible"] and st["enabled"] and not st["minimized"]
    return {"ok": ok, "ms": _ms(t0), "reason": "" if ok else "window_not_ready",
            "trial_dir": str(trial_dir), "pre_files": pre_files,
            "closed": closed, "pre_state": st}


class Intruder:
    """F2 用：Win32 顶层窗口抢焦点，trial 结束即销毁（不依赖 Tk 事件循环）。"""

    def __init__(self, w: int = 520, h: int = 320):
        self.hwnd = win32gui.CreateWindowEx(
            0, "STATIC", "bench_intruder",
            win32con.WS_POPUP | win32con.WS_VISIBLE,
            40, 40, w, h, 0, 0, 0, None)
        win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 40, 40, w, h, 0)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        force_foreground(self.hwnd, timeout=1.0)
        time.sleep(0.2)

    def foreground(self) -> bool:
        return "bench_intruder" in win32gui.GetWindowText(win32gui.GetForegroundWindow())

    def close(self):
        try:
            win32gui.DestroyWindow(self.hwnd)
        except Exception:
            pass


def make_step(trial: str, arm: str, task: str, step: int, cls: str, pre: dict,
              post: dict, action: dict, expect: dict, observed: dict, verdict: str,
              fc: str, seg: dict, t0: float, layer: str = "L0",
              recovery: dict | None = None, silent: bool = False) -> dict:
    wall = _ms(t0)
    seg = dict(seg)
    seg.setdefault("verify", 0.0)
    used = sum(seg.values())
    if abs(used - wall) > 45:            # 把剩余时间记入 verify，保证闭合
        seg["verify"] = round(seg["verify"] + (wall - used), 1)
    return {
        "kind": "step", "trial": trial, "arm": arm, "task": task, "attempt": 1,
        "step": step, "layer": layer, "control_class": cls,
        "t0": round(t0, 3), "t1": round(t0 + wall / 1000, 3),
        "window": {k: pre.get(k) for k in ("hwnd", "rect", "minimized", "foreground")},
        "control": pre.get("control") or {"name": None, "type": "Window",
                                          "rect": pre.get("rect"), "patterns": []},
        "action": action, "expect": expect, "observed": observed,
        "verdict": verdict, "silent": silent, "failure_class": fc,
        "recovery": recovery, "segment_ms": {k: round(v, 1) for k, v in seg.items()},
        "wall_ms": wall,
    }


def _save(recs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bad = [(i, validate(r)) for i, r in enumerate(recs) if validate(r)]
    if bad:
        print(f"[warn] {len(bad)} 条记录不合法，示例 {bad[0]}", file=sys.stderr)
    print(f"[saved] {out_path}")


def wait_for(pred, timeout: float = 3.0, interval: float = 0.05) -> tuple[bool, float]:
    t0 = _now()
    while _ms(t0) < timeout * 1000:
        if pred():
            return True, _ms(t0)
        time.sleep(interval)
    return False, _ms(t0)


def locate_control(h: int, name: str, types=("Button", "SplitButton")):
    """UIA 定位一个具名控件：先 find-first（~0.25s），失败再全树枚举（~1s）。"""
    t0 = _now()
    win = Desktop(backend="uia").window(handle=h)
    ctrl = None
    for t in types:
        try:
            ctrl = win.child_window(title=name, control_type=t).wrapper_object()
            break
        except Exception:
            ctrl = None
    if ctrl is None:
        try:
            for c in win.descendants():
                try:
                    if (c.element_info.control_type in types
                            and (c.window_text() or "").strip() == name):
                        ctrl = c
                        break
                except Exception:
                    continue
        except Exception:
            pass
    ms = _ms(t0)
    if ctrl is None:
        return None, ms, None
    rc = ctrl.rectangle()
    return ctrl, ms, [rc.left, rc.top, rc.right, rc.bottom]


AOI_COLOR = (77, 144, 254)   # AOI 选中框：两像素实线，颜色稳定


def _detect_aoi_once(h: int, tol: int = 8) -> dict:
    import numpy as np
    L, T, R, B = win32gui.GetWindowRect(h)
    shot = pyautogui.screenshot(region=(L, T, R - L, B - T))
    arr = np.asarray(shot)[:, :, :3].astype(np.int16)
    m = ((abs(arr[:, :, 0] - AOI_COLOR[0]) <= tol)
         & (abs(arr[:, :, 1] - AOI_COLOR[1]) <= tol)
         & (abs(arr[:, :, 2] - AOI_COLOR[2]) <= 4))
    rows = m.sum(axis=1)
    cols = m.sum(axis=0)
    hh, ww = m.shape
    ry = [i for i, v in enumerate(rows) if v > 0.45 * ww]
    cx = [i for i, v in enumerate(cols) if v > 0.35 * hh]
    if len(ry) < 2 or len(cx) < 2:
        return {"found": False, "rows": len(ry), "cols": len(cx),
                "pixels": int(m.sum())}
    top, bot = min(ry), max(ry)
    left, right = min(cx), max(cx)
    if (right - left) < 200 or (bot - top) < 150:
        # 只有一条边（2px 线会被当成 top+bottom）→ 判为未找到，交给重试
        return {"found": False, "degenerate": True, "rows": len(ry),
                "cols": len(cx), "pixels": int(m.sum())}
    return {"found": True, "pixels": int(m.sum()),
            "rect": [L + left, T + top, L + right, T + bot],
            "center": [L + (left + right) // 2, T + (top + bot) // 2],
            "size": [right - left, bot - top]}


def detect_aoi(h: int, tol: int = 8, attempts: int = 3,
               interval: float = 0.12) -> dict:
    """AOI 检测带重试：关窗后地图重绘有 1–2 帧空窗，单次截图会假失败。"""
    t0 = _now()
    last: dict = {}
    for k in range(attempts):
        last = _detect_aoi_once(h, tol)
        last["attempt"] = k + 1
        if last.get("found"):
            return {**last, "ms": _ms(t0)}
        time.sleep(interval)
    return {**last, "ms": _ms(t0)}


def run_t2(arm: str, trials: int, out_path: Path, fault: str | None = None) -> dict:
    """T2：UIA 定位「导入边界」→ invoke 开窗 → 关窗。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T2-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T2",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL: {r.get('reason')}")
            continue

        main = find_main()
        h = main["hwnd"]
        pre = window_state(h)
        pre["control"] = {"name": "导入边界", "type": "Button", "rect": None,
                          "patterns": ["Invoke"]}
        occ = Occluder() if fault == "F8" else None
        if occ:
            pre["occluded"] = True

        # 1) L1 定位
        t0 = _now()
        ctrl, locate_ms, rect = locate_control(h, "导入边界")
        pre["control"]["rect"] = rect
        recs.append(make_step(
            trial, arm, "T2", 0, "standard", pre, None,
            {"kind": "locate", "source": "uia", "coords": None},
            {"pred": "control_named", "name": "导入边界", "fault": fault},
            {"found": ctrl is not None, "rect": rect, "enumerate_ms": locate_ms},
            "pass" if ctrl is not None else "fail",
            "none" if ctrl is not None else "wrong_control",
            {"reset": r["ms"], "perceive": locate_ms, "decide": 0.0, "act": 0.0,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if ctrl is None:
            if occ:
                occ.close()
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T2",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL locate")
            continue

        # 2) L2 invoke 开窗
        t0 = _now()
        invoke_ms = 0.0
        try:
            t_act = _now()
            ctrl.invoke()
            invoke_ms = _ms(t_act)
            appeared, appear_ms = wait_for(lambda: len(find_dialogs(h)) > 0, 3.0)
        except Exception:
            appeared, appear_ms = False, 0.0
        dlgs = find_dialogs(h)
        nodes = uia_nodes(dlgs[0]["hwnd"]) if dlgs else 0
        main_enabled = window_state(h)["enabled"]
        open_ok = appeared and nodes >= 10 and not main_enabled
        recs.append(make_step(
            trial, arm, "T2", 1, "standard", pre, None,
            {"kind": "invoke", "source": "uia", "coords": None},
            {"pred": "dialog_open", "title": "导入边界数据", "min_nodes": 10},
            {"appeared": appeared, "appear_ms": appear_ms, "nodes": nodes,
             "main_enabled": main_enabled},
            "pass" if open_ok else "fail", "none" if open_ok else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": invoke_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L2"))

        # 3) L2 关窗
        close_ms = gone_ms = 0.0
        gone = False
        if dlgs:
            t0 = _now()
            t_act = _now()
            close_window(dlgs[0]["hwnd"])
            close_ms = _ms(t_act)
            gone, gone_ms = wait_for(lambda: not find_dialogs(h), 2.0)
            recs.append(make_step(
                trial, arm, "T2", 2, "standard", pre, None,
                {"kind": "close", "source": "uia", "coords": None},
                {"pred": "dialog_closed", "pre": "open"},
                {"closed": gone, "left": len(find_dialogs(h)),
                 "main_enabled": window_state(h)["enabled"]},
                "pass" if gone else "fail", "none" if gone else "state_lost",
                {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": close_ms,
                 "verify": gone_ms, "recover": 0.0}, t0, layer="L2"))
        else:
            close_ms = 0.0

        success = open_ok and gone
        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T2",
                     "success": success, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        if occ:
            occ.close()
        runs += 1
        print(f"[{trial}] {'PASS' if success else 'FAIL'} locate={locate_ms:.0f}ms "
              f"invoke={invoke_ms:.0f}ms appear={appear_ms:.0f}ms nodes={nodes} "
              f"close={close_ms:.0f}ms wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def find_dataitem(h: int, name: str):
    t0 = _now()
    try:
        c = Desktop(backend="uia").window(handle=h).child_window(
            title=name, control_type="DataItem").wrapper_object()
        rc = c.rectangle()
        return c, _ms(t0), [rc.left, rc.top, rc.right, rc.bottom]
    except Exception:
        return None, _ms(t0), None


def find_level_row(h: int, level: int, max_scroll: int = 10):
    """在级别表里找目标行；不可见时滚动表格。返回 (row, ms, rect)。"""
    name = f"第{level}级"
    row, ms, rect = find_dataitem(h, name)
    if row:
        return row, ms, rect
    try:
        w = Desktop(backend="uia").window(handle=h)
        table = next((c for c in w.descendants()
                      if c.element_info.control_type == "Table"), None)
        if table is None:
            return None, ms, None
        rc = table.rectangle()
        for _ in range(max_scroll):
            pyautogui.moveTo((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
            pyautogui.scroll(-240)
            time.sleep(0.5)
            row, ms2, rect = find_dataitem(h, name)
            ms += ms2
            if row:
                return row, ms, rect
    except Exception:
        pass
    return None, ms, None


CHECKED_TH = 0.02   # 勾选框暗像素占比阈值（经验值：空 0.000 / 勾选 0.05）


def cell_checked_ratio(rect) -> float:
    """截图单元格左侧勾选框区域，返回暗像素占比。UIA 不暴露该勾选状态。"""
    import numpy as np
    l, t, r, b = rect
    shot = pyautogui.screenshot(region=(l, t, r - l, b - t))
    a = np.asarray(shot)[:, :, :3].astype(int)
    h, w = a.shape[:2]
    box = a[int(0.2 * h):int(0.85 * h), 2:min(int(0.25 * w), 22)]
    return round(float((box.sum(axis=2) < 300).mean()), 4)


def open_download(h: int):
    """检测 AOI → 双击中心 → 等下载对话框。返回 (aoi, ok, dbl_ms, appear_ms)。"""
    aoi = detect_aoi(h)
    if not aoi.get("found"):
        return aoi, False, 0.0, 0.0
    t0 = _now()
    pyautogui.doubleClick(*aoi["center"])
    dbl_ms = _ms(t0)
    ok, appear_ms = wait_for(lambda: len(find_dialogs(h)) > 0, 4.0)
    return aoi, ok, dbl_ms, appear_ms


def run_t4(arm: str, trials: int, out_path: Path) -> dict:
    """T4：画布定位 AOI（像素）→ 双击 → 下载对话框。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T4-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T4",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL: {r.get('reason')}")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)
        pre["control"] = {"name": "AOI", "type": "Canvas", "rect": None, "patterns": []}

        # 1) L1 视觉定位 AOI
        t0 = _now()
        aoi = detect_aoi(h)
        loc_ms = _ms(t0)
        recs.append(make_step(
            trial, arm, "T4", 0, "canvas", pre, None,
            {"kind": "detect", "source": "pixel", "coords": None},
            {"pred": "aoi_found", "color": list(AOI_COLOR)},
            aoi, "pass" if aoi.get("found") else "fail",
            "none" if aoi.get("found") else "wrong_control",
            {"reset": r["ms"], "perceive": loc_ms, "decide": 0.0, "act": 0.0,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if not aoi.get("found"):
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T4",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL aoi")
            continue

        # 2) L2 双击打开
        t0 = _now()
        _, opened, dbl_ms, appear_ms = open_download(h)
        dlgs = find_dialogs(h)
        nodes = uia_nodes(dlgs[0]["hwnd"]) if dlgs else 0
        ok_open = opened and nodes >= 80 and not window_state(h)["enabled"]
        pre_ctrl = dict(pre["control"]); pre_ctrl["rect"] = aoi["rect"]
        recs.append(make_step(
            trial, arm, "T4", 1, "canvas", {**pre, "control": pre_ctrl}, None,
            {"kind": "double_click", "source": "mouse", "coords": aoi["center"]},
            {"pred": "dialog_open", "title": "下载对话框", "min_nodes": 80},
            {"opened": opened, "appear_ms": appear_ms, "nodes": nodes,
             "main_enabled": window_state(h)["enabled"]},
            "pass" if ok_open else "fail", "none" if ok_open else "synthetic_rejected",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L2"))

        # 3) 清理
        t0 = _now()
        t_act = _now()
        closed = True
        for dd in find_dialogs(h):
            closed = close_window(dd["hwnd"]) and closed
        close_ms = _ms(t_act)
        recs.append(make_step(
            trial, arm, "T4", 2, "canvas", pre, None,
            {"kind": "close", "source": "uia", "coords": None},
            {"pred": "dialog_closed"},
            {"closed": closed, "left": len(find_dialogs(h))},
            "pass" if closed else "fail", "none" if closed else "state_lost",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": close_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L3"))

        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T4",
                     "success": bool(ok_open), "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        runs += 1
        print(f"[{trial}] {'PASS' if ok_open else 'FAIL'} aoi={aoi.get('size')} "
              f"loc={loc_ms:.0f}ms dbl={dbl_ms:.0f}ms nodes={nodes} wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def run_t3(arm: str, trials: int, out_path: Path) -> dict:
    """T3：下载对话框级别表，勾选「第16级」并用像素差分验证（UIA 无 pattern）。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T3-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL: {r.get('reason')}")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)

        # 0) 前置：打开下载对话框（L1，不计入 HR）
        t0 = _now()
        aoi, opened, dbl_ms, appear_ms = open_download(h)
        dlgs = find_dialogs(h)
        setup_ok = bool(opened and dlgs)
        recs.append(make_step(
            trial, arm, "T3", 0, "canvas", pre, None,
            {"kind": "double_click", "source": "mouse",
             "coords": aoi.get("center")},
            {"pred": "dialog_open", "detail": "setup"},
            {"opened": opened, "aoi": aoi, "nodes": uia_nodes(dlgs[0]["hwnd"]) if dlgs else 0},
            "pass" if setup_ok else "fail", "none" if setup_ok else "timeout",
            {"reset": r["ms"], "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L1"))
        if not setup_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL setup")
            continue

        dh = dlgs[0]["hwnd"]
        # 1) L2 定位行 + 勾选（先保证未勾选，再点）
        t0 = _now()
        row, find_ms, rect = find_dataitem(dh, "第16级")
        if row is None:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL row not found")
            continue

        t_pre = _now()
        pre_ratio = cell_checked_ratio(rect)
        pre_ms = _ms(t_pre)
        recover_ms = 0.0
        if pre_ratio > CHECKED_TH:            # 残留勾选：先清掉，保证不是空过
            t_rec = _now()
            pyautogui.click(rect[0] + 8, (rect[1] + rect[3]) // 2)
            wait_for(lambda: cell_checked_ratio(rect) <= CHECKED_TH, 1.5)
            pre_ratio = cell_checked_ratio(rect)
            recover_ms = _ms(t_rec)

        t_act = _now()
        pyautogui.click(rect[0] + 8, (rect[1] + rect[3]) // 2)
        click_ms = _ms(t_act)
        changed, wait_ms = wait_for(lambda: cell_checked_ratio(rect) > CHECKED_TH, 2.0)
        post_ratio = cell_checked_ratio(rect)
        ok = bool(changed and post_ratio > CHECKED_TH and pre_ratio <= CHECKED_TH)
        recs.append(make_step(
            trial, arm, "T3", 1, "table_row", pre, None,
            {"kind": "click", "source": "uia_rect", "coords": [rect[0] + 8, (rect[1] + rect[3]) // 2]},
            {"pred": "checkbox_checked", "pre_ratio": pre_ratio, "th": CHECKED_TH},
            {"pre_ratio": pre_ratio, "post_ratio": post_ratio, "rect": rect,
             "row": "第16级", "find_ms": find_ms},
            "pass" if ok else "fail", "none" if ok else "no_pattern",
            {"reset": 0.0, "perceive": round(find_ms + pre_ms, 1), "decide": 0.0,
             "act": click_ms, "verify": wait_ms, "recover": recover_ms}, t0, layer="L2"))

        t0 = _now()
        t_act = _now()
        closed = True
        for dd in find_dialogs(h):
            closed = close_window(dd["hwnd"]) and closed
        close_ms = _ms(t_act)
        recs.append(make_step(
            trial, arm, "T3", 2, "table_row", pre, None,
            {"kind": "close", "source": "uia", "coords": None},
            {"pred": "dialog_closed"},
            {"closed": closed, "left": len(find_dialogs(h))},
            "pass" if closed else "fail", "none" if closed else "state_lost",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": close_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L3"))

        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} find={find_ms:.0f}ms "
              f"pre={pre_ratio} post={post_ratio} click={click_ms:.0f}ms "
              f"wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


GDALINFO = Path(CFG.get("gdalinfo", r"D:\QGIS\bin\gdalinfo.exe"))
T5_ROOT = Path(CFG.get("download_root", r"F:\Program Files (x86)\BMDownload"))
AOI_BOUNDS = tuple(CFG.get("aoi_bounds", (116.2098, 39.9048, 116.2201, 39.9118)))
AOI_TOL_DEG = 2e-5                                    # ≈1 px @ 1.83 m/px


def set_clipboard(text: str, attempts: int = 6) -> bool:
    import win32clipboard
    for _ in range(attempts):
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception:
            time.sleep(0.2)
    return False


def paste_into(rect, value: str) -> bool:
    """点输入框 → Ctrl+A → 剪贴板粘贴（write() 会打乱内容，见 CANDIDATE_RECAP §8.7/B4）。"""
    pyautogui.click((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)
    time.sleep(0.06)
    pyautogui.hotkey("ctrl", "a")
    if not set_clipboard(value):
        return False
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.10)
    return True


def gdal_json(path: Path) -> dict | None:
    try:
        out = subprocess.run([str(GDALINFO), "-json", str(path)],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def raster_wgs84_bounds(path: Path, attempts: int = 3):
    j = None
    for k in range(attempts):
        j = gdal_json(path)
        if j:
            break
        time.sleep(0.4)
    if not j:
        return None
    ext = j.get("wgs84Extent")
    if ext:
        pts = ext.get("coordinates", [[]])[0]
    else:
        cc = j.get("cornerCoordinates") or {}
        pts = [cc.get("upperLeft"), cc.get("lowerRight")]
    pts = [p for p in pts if p]
    if not pts:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys), j


def wait_artifact(task_dir: Path, timeout: float = 180.0):
    t0 = _now()
    last: dict = {}
    stable = 0
    tifs: list[Path] = []
    while _ms(t0) < timeout * 1000:
        tifs = list(task_dir.glob("*.tif"))
        cur = {p.name: p.stat().st_size for p in tifs}
        if cur and cur == last and all(v > 10_000 for v in cur.values()):
            stable += 1
            if stable >= 2:
                return True, _ms(t0), tifs
        else:
            stable = 0
        last = cur
        time.sleep(0.1)
    return False, _ms(t0), tifs


def run_t5(arm: str, trials: int, out_path: Path, fault: str | None = None,
           level: int = 16, region=None) -> dict:
    """T5：AOI → 任务名 → 第16级 → 确定 → 产物 GDAL 校验。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T5-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        task_name = f"bench_{trial}"
        task_dir = T5_ROOT / task_name
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL: {r.get('reason')}")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)

        # 0) 打开下载对话框（setup：AOI 双击 或 区域面板图标）
        t0 = _now()
        if region:
            sel = region_select(h, region)
            exp_bounds = sel.get("bounds") or panel_bounds(h)
            icon = panel_icon_point(h, region[-1])
            if icon and 0 < icon[0] < 2500 and 0 < icon[1] < 1400:
                pyautogui.click(*icon)
            opened, appear_ms = wait_for(lambda: len(find_dialogs(h)) > 0, 5.0)
            dbl_ms = 0.0
            aoi = {"source": "region", "district": region[-1],
                   "bounds": exp_bounds, "center": icon}
        else:
            aoi, opened, dbl_ms, appear_ms = open_download(h)
            exp_bounds = AOI_BOUNDS
        dlgs = find_dialogs(h)
        setup_ok = bool(opened and dlgs)
        recs.append(make_step(
            trial, arm, "T5", 0, "canvas", pre, None,
            {"kind": "double_click", "source": "mouse", "coords": aoi.get("center")},
            {"pred": "dialog_open", "detail": "setup"},
            {"opened": opened, "aoi": aoi},
            "pass" if setup_ok else "fail", "none" if setup_ok else "timeout",
            {"reset": r["ms"], "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L1"))
        if not setup_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL setup")
            continue

        dh = dlgs[0]["hwnd"]
        # 应用会记住上次的 tab；T5 显式切回「影像」页
        try:
            w2 = Desktop(backend="uia").window(handle=dh)
            for c in w2.descendants():
                if c.element_info.control_type == "TabItem" and "影" in (c.window_text() or ""):
                    rc = c.rectangle()
                    pyautogui.click((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
                    time.sleep(0.6)
                    break
        except Exception:
            pass
        rec = None
        if fault in ("F1", "F4", "F6"):
            t_rec = _now()
            act = inject_fault(fault, dh, h)
            rec_ok = True
            if fault == "F1":
                rec_ok = restore_any(dh)
            elif fault == "F4":
                restore_window(h)
                _, ok2, _, _ = open_download(h)
                rec_ok = bool(ok2)
                time.sleep(0.35)
            cur = find_dialogs(h)
            if not cur:
                rec_ok = False
            else:
                dh = cur[0]["hwnd"]
            rec = {"injected": fault, "action": act, "ok": bool(rec_ok),
                   "ms": _ms(t_rec)}
            if not rec_ok:
                recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                             "success": False, "attempts": 1, "takeovers": 0,
                             "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
                print(f"[{trial}] FAIL recover")
                continue
        w = Desktop(backend="uia").window(handle=dh)
        edits = []
        for c in w.descendants():
            try:
                if c.element_info.control_type == "Edit":
                    edits.append((c, c.rectangle()))
            except Exception:
                pass
        edits.sort(key=lambda x: x[1].top)

        # 1) 任务名（唯一）
        t0 = _now()
        name_ok = False
        if edits:
            paste_into(edits[0][1], task_name)
            try:
                name_ok = (edits[0][0].get_value() or "").strip() == task_name
            except Exception:
                name_ok = False
        recs.append(make_step(
            trial, arm, "T5", 1, "standard", pre, None,
            {"kind": "paste", "source": "clipboard", "coords": None},
            {"pred": "text_set", "value": task_name},
            {"value": task_name, "ok": name_ok},
            "pass" if name_ok else "fail", "none" if name_ok else "state_lost",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": _ms(t0),
             "verify": 0.0, "recover": 0.0}, t0, layer="L2", recovery=rec))
        if not name_ok:
            for dd in find_dialogs(h):
                close_window(dd["hwnd"])
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL name")
            continue

        # 2) 勾选目标级别
        t0 = _now()
        row, find_ms, rect = find_level_row(dh, level)
        lvl_ok = False
        pre_ratio = post_ratio = -1.0
        if row is not None:
            pre_ratio = cell_checked_ratio(rect)
            if pre_ratio <= CHECKED_TH:
                pyautogui.click(rect[0] + 8, (rect[1] + rect[3]) // 2)
                wait_for(lambda: cell_checked_ratio(rect) > CHECKED_TH, 2.0)
            post_ratio = cell_checked_ratio(rect)
            lvl_ok = post_ratio > CHECKED_TH
        recs.append(make_step(
            trial, arm, "T5", 2, "table_row", pre, None,
            {"kind": "click", "source": "uia_rect",
             "coords": [rect[0] + 8, (rect[1] + rect[3]) // 2] if rect else None},
            {"pred": "checkbox_checked", "th": CHECKED_TH},
            {"pre_ratio": pre_ratio, "post_ratio": post_ratio, "rect": rect},
            "pass" if lvl_ok else "fail", "none" if lvl_ok else "no_pattern",
            {"reset": 0.0, "perceive": find_ms, "decide": 0.0,
             "act": max(0.0, _ms(t0) - find_ms), "verify": 0.0, "recover": 0.0},
            t0, layer="L2"))
        if not lvl_ok:
            for dd in find_dialogs(h):
                close_window(dd["hwnd"])
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL level")
            continue

        # 3) 确定
        t0 = _now()
        pre_open = bool(find_dialogs(h))
        clicked = False
        try:
            w.child_window(title="确定 Enter", control_type="Button").wrapper_object().invoke()
            clicked = True
        except Exception:
            clicked = False
        gone, gone_ms = wait_for(lambda: not find_dialogs(h), 8.0)
        ok_ok = bool(clicked and gone and pre_open)
        recs.append(make_step(
            trial, arm, "T5", 3, "standard", pre, None,
            {"kind": "invoke", "source": "uia", "coords": None},
            {"pred": "dialog_closed_after_ok", "pre_open": pre_open},
            {"clicked": clicked, "closed": gone, "pre_open": pre_open,
             "left": [d["title"] for d in find_dialogs(h)]},
            "pass" if ok_ok else ("vacuous" if not pre_open else "fail"),
            "none" if ok_ok else ("none" if not pre_open else "timeout"),
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": 0.0,
             "verify": gone_ms, "recover": 0.0}, t0, layer="L2"))

        # 4) 产物 + GDAL oracle
        t0 = _now()
        got, wait_ms, tifs = wait_artifact(task_dir, 10 if fault else 180)
        tif = tifs[0] if tifs else None
        art_ok = False
        detail = {}
        if tif is not None:
            sha = hashlib.sha256(tif.read_bytes()).hexdigest()
            rb = raster_wgs84_bounds(tif)
            if rb:
                west, south, east, north, j = rb
                dev = max(abs(west - exp_bounds[0]), abs(south - exp_bounds[1]),
                          abs(east - exp_bounds[2]), abs(north - exp_bounds[3]))
                art_ok = dev <= (1e-3 if region else AOI_TOL_DEG) and j.get("size", [0, 0])[0] > 0
                detail = {"bounds": [west, south, east, north], "expected": list(exp_bounds),
                          "dev_deg": dev,
                          "size": j.get("size"), "sha256": sha,
                          "bytes": tif.stat().st_size, "path": str(tif)}
            else:
                detail = {"path": str(tif), "sha256": sha,
                          "bytes": tif.stat().st_size, "gdal": "fail"}
        recs.append(make_step(
            trial, arm, "T5", 4, "standard", pre, None,
            {"kind": "wait_artifact", "source": "fs", "coords": None},
            {"pred": "raster_matches_aoi", "tol_deg": AOI_TOL_DEG},
            {"got": got, "files": [p.name for p in tifs], **detail},
            "pass" if art_ok else "fail", "none" if art_ok else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": 0.0,
             "verify": wait_ms, "recover": 0.0}, t0, layer="L4"))

        ok = bool(name_ok and lvl_ok and clicked and gone and art_ok)
        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial),
                     "artifact": detail or None})
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} name={name_ok} lvl={lvl_ok} "
              f"ok={clicked and gone} tif={tif.name if tif else None} "
              f"dev={detail.get('dev_deg')} wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


_OCR = None


def ocr_engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def shot_np(region):
    import numpy as np
    return np.asarray(pyautogui.screenshot(
        region=tuple(int(v) for v in region)))[:, :, :3].astype(int)


def ocr_lines(region):
    """截图 → OCR，返回 [{text, score, rect(屏幕坐标)}] 与耗时。"""
    t0 = _now()
    img = pyautogui.screenshot(region=tuple(int(v) for v in region))
    res, _ = ocr_engine()(img)
    ms = _ms(t0)
    out = []
    for box, text, score in (res or []):
        xs = [p[0] + region[0] for p in box]
        ys = [p[1] + region[1] for p in box]
        out.append({"text": text, "score": float(score),
                    "rect": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    return out, ms


def diff_fraction(a, b, th: int = 30) -> float:
    import numpy as np
    if a.shape != b.shape:
        return 1.0
    return float((np.abs(a - b).sum(axis=2) > th).mean())


OCC_RECT = (1600, 66, 520, 58)   # 覆盖「导入边界」按钮的置顶遮挡窗


class Occluder:
    """F8：不透明置顶窗，挡住像素通道；UIA 通道不受影响。"""

    def __init__(self, rect=OCC_RECT):
        import tkinter as tk
        L, T, W, H = rect
        self.root = tk.Tk()
        self.root.withdraw()
        self.top = tk.Toplevel(self.root)
        self.top.overrideredirect(True)
        self.top.geometry(f"{W}x{H}+{L}+{T}")
        self.top.configure(bg="black")
        self.top.attributes("-topmost", True)
        self.top.update()
        time.sleep(0.3)

    def close(self):
        try:
            self.top.destroy()
            self.root.destroy()
        except Exception:
            pass


def run_t2_a(trials: int, out_path: Path, fault: str | None = None) -> dict:
    """A 臂 T2：截图 + OCR 找「导入边界」→ 坐标点击 → 全窗像素差分验证。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T2-A-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T2",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL")
            continue
        main = find_main(); h = main["hwnd"]
        L, T, R, B = win32gui.GetWindowRect(h)
        pre = window_state(h)
        pre["control"] = {"name": None, "type": "Pixel", "rect": None, "patterns": []}
        occ = Occluder() if fault == "F8" else None
        if occ:
            pre["occluded"] = True
        before = shot_np((L, T, R - L, B - T))

        t0 = _now()
        lines, ocr_ms = ocr_lines((L, T, R - L, 150))
        hit = next((x for x in lines if "导入边界" in x["text"]), None)
        recs.append(make_step(
            trial, "A", "T2", 0, "standard", pre, None,
            {"kind": "ocr", "source": "pixel", "coords": None},
            {"pred": "text_found", "text": "导入边界", "fault": fault},
            {"hit": hit, "ocr_ms": ocr_ms, "lines": len(lines)},
            "pass" if hit else "fail", "none" if hit else "wrong_control",
            {"reset": r["ms"], "perceive": ocr_ms, "decide": 0.0, "act": 0.0,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if not hit:
            if occ:
                occ.close()
            recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T2",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL ocr")
            continue

        x0, y0, x1, y1 = hit["rect"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        t0 = _now()
        pyautogui.click(cx, cy)
        click_ms = _ms(t0)
        changed, wait_ms = wait_for(
            lambda: diff_fraction(before, shot_np((L, T, R - L, B - T))) > 0.05, 3.0)
        oracle = bool(find_dialogs(h))
        recs.append(make_step(
            trial, "A", "T2", 1, "standard", pre, None,
            {"kind": "click", "source": "pixel", "coords": [cx, cy]},
            {"pred": "screen_changed", "frac": 0.05},
            {"changed": changed, "oracle_dialog": oracle,
             "ocr_score": hit["score"]},
            "pass" if changed else "fail", "none" if changed else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": click_ms,
             "verify": wait_ms, "recover": 0.0}, t0, layer="L2"))
        for dd in find_dialogs(h):
            close_window(dd["hwnd"])
        if occ:
            occ.close()
        recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T2",
                     "success": oracle, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        runs += 1
        print(f"[{trial}] {'PASS' if oracle else 'FAIL'} ocr={ocr_ms:.0f}ms "
              f"click={click_ms:.0f}ms wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def run_t3_a(trials: int, out_path: Path) -> dict:
    """A 臂 T3：OCR 找「第16级」→ 点文字左侧勾选框 → 像素差分验证。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T3-A-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)

        t0 = _now()
        aoi, opened, dbl_ms, appear_ms = open_download(h)
        dlgs = find_dialogs(h)
        setup_ok = bool(opened and dlgs)
        recs.append(make_step(
            trial, "A", "T3", 0, "canvas", pre, None,
            {"kind": "double_click", "source": "mouse", "coords": aoi.get("center")},
            {"pred": "dialog_open", "detail": "setup"},
            {"opened": opened, "aoi": aoi},
            "pass" if setup_ok else "fail", "none" if setup_ok else "timeout",
            {"reset": r["ms"], "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L1"))
        if not setup_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL setup")
            continue

        dl = dlgs[0]["rect"]
        ocr_region = (dl[0], dl[1], dl[2] - dl[0], dl[3] - dl[1])
        t0 = _now()
        time.sleep(0.35)          # 等对话框首帧绘制完成，否则 OCR 读到空白
        lines, ocr_ms = ocr_lines(ocr_region)
        hit = next((x for x in lines if "第16" in x["text"]), None)
        if not hit:
            recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL ocr")
            continue

        x0, y0, x1, y1 = hit["rect"]
        band = (x0 - 20, y0 - 4, x1 + 4, y1 + 4)
        pre_ratio = cell_checked_ratio(band)
        recover_ms = 0.0
        if pre_ratio > CHECKED_TH:
            t_rec = _now()
            pyautogui.click(x0 - 12, (y0 + y1) // 2)
            wait_for(lambda: cell_checked_ratio(band) <= CHECKED_TH, 1.5)
            pre_ratio = cell_checked_ratio(band)
            recover_ms = _ms(t_rec)

        t_act = _now()
        pyautogui.click(x0 - 12, (y0 + y1) // 2)
        click_ms = _ms(t_act)
        changed, wait_ms = wait_for(lambda: cell_checked_ratio(band) > CHECKED_TH, 2.0)
        post_ratio = cell_checked_ratio(band)
        ok = bool(changed and post_ratio > CHECKED_TH and pre_ratio <= CHECKED_TH)
        recs.append(make_step(
            trial, "A", "T3", 1, "table_row", pre, None,
            {"kind": "click", "source": "pixel", "coords": [x0 - 12, (y0 + y1) // 2]},
            {"pred": "checkbox_checked", "pre_ratio": pre_ratio, "th": CHECKED_TH},
            {"pre_ratio": pre_ratio, "post_ratio": post_ratio, "band": band,
             "text": hit["text"], "score": hit["score"]},
            "pass" if ok else "fail", "none" if ok else "no_pattern",
            {"reset": 0.0, "perceive": ocr_ms, "decide": 0.0, "act": click_ms,
             "verify": wait_ms, "recover": recover_ms}, t0, layer="L2"))

        t0 = _now()
        t_act = _now()
        closed = True
        for dd in find_dialogs(h):
            closed = close_window(dd["hwnd"]) and closed
        close_ms = _ms(t_act)
        recs.append(make_step(
            trial, "A", "T3", 2, "table_row", pre, None,
            {"kind": "close", "source": "uia", "coords": None},
            {"pred": "dialog_closed"},
            {"closed": closed, "left": len(find_dialogs(h))},
            "pass" if closed else "fail", "none" if closed else "state_lost",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": close_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L3"))

        recs.append({"kind": "trial", "trial": trial, "arm": "A", "task": "T3",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} ocr={ocr_ms:.0f}ms "
              f"pre={pre_ratio} post={post_ratio} wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def run_t5_pixel(arm: str, trials: int, out_path: Path, fault: str | None = None) -> dict:
    """像素臂 T5（A=无恢复，C=重开+重 OCR）：OCR 定位任务名/级别/确定。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T5-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        task_name = f"bench_{trial}"
        task_dir = T5_ROOT / task_name
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)

        t0 = _now()
        aoi, opened, dbl_ms, appear_ms = open_download(h)
        dlgs = find_dialogs(h)
        setup_ok = bool(opened and dlgs)
        recs.append(make_step(
            trial, arm, "T5", 0, "canvas", pre, None,
            {"kind": "double_click", "source": "mouse", "coords": aoi.get("center")},
            {"pred": "dialog_open", "detail": "setup"},
            {"opened": opened, "aoi": aoi},
            "pass" if setup_ok else "fail", "none" if setup_ok else "timeout",
            {"reset": r["ms"], "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L1"))
        if not setup_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL setup")
            continue

        dl = dlgs[0]["rect"]
        ocr_region = (dl[0], dl[1], dl[2] - dl[0], dl[3] - dl[1])
        t0 = _now()
        time.sleep(0.35)
        lines, ocr_ms = ocr_lines(ocr_region)
        name_hit = next((x for x in lines if "任务名称" in x["text"]), None)
        row_hit = next((x for x in lines if "第16" in x["text"]), None)
        ok_hit = next((x for x in lines if x["text"].strip() == "确定"), None)
        found_ok = bool(name_hit and row_hit and ok_hit)
        recs.append(make_step(
            trial, arm, "T5", 1, "standard", pre, None,
            {"kind": "ocr", "source": "pixel", "coords": None},
            {"pred": "three_targets_found"},
            {"name": name_hit, "row": row_hit, "ok_btn": ok_hit, "ocr_ms": ocr_ms},
            "pass" if found_ok else "fail", "none" if found_ok else "wrong_control",
            {"reset": 0.0, "perceive": ocr_ms, "decide": 0.0, "act": 0.0,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if not found_ok:
            for dd in find_dialogs(h):
                close_window(dd["hwnd"])
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL ocr")
            continue

        # 故障注入（F1/F4/F6）+ 恢复
        rec = None
        if fault in ("F1", "F4", "F6"):
            t_rec = _now()
            act = inject_fault(fault, dlgs[0]["hwnd"], h)
            rec_ok = False
            if arm == "C":
                if fault == "F1":
                    restore_any(dlgs[0]["hwnd"])
                elif fault == "F4":
                    restore_window(h)
                    _, ok2, _, _ = open_download(h)
                    if ok2:
                        time.sleep(0.35)
                cur = find_dialogs(h)
                if cur:
                    dl = cur[0]["rect"]
                    lines2, _ = ocr_lines((dl[0], dl[1],
                                           dl[2] - dl[0], dl[3] - dl[1]))
                    name_hit = next((x for x in lines2 if "任务名称" in x["text"]), None)
                    row_hit = next((x for x in lines2 if "第16" in x["text"]), None)
                    ok_hit = next((x for x in lines2 if x["text"].strip() == "确定"), None)
                    rec_ok = bool(name_hit and row_hit and ok_hit)
            rec = {"injected": fault, "action": act, "ok": rec_ok, "ms": _ms(t_rec)}
            if arm == "C" and not rec_ok:
                for dd in find_dialogs(h):
                    close_window(dd["hwnd"])
                recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                             "success": False, "attempts": 1, "takeovers": 0,
                             "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
                print(f"[{trial}] FAIL recover")
                continue

        # 任务名：点「任务名称」标签右侧 → 粘贴
        t0 = _now()
        nx0, ny0, nx1, ny1 = name_hit["rect"]
        pyautogui.click(min(nx1 + 70, dl[2] - 30), (ny0 + ny1) // 2)
        time.sleep(0.15)
        pyautogui.hotkey("ctrl", "a")
        set_clipboard(task_name)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.25)
        paste_ms = _ms(t0)
        recs.append(make_step(
            trial, arm, "T5", 2, "standard", pre, None,
            {"kind": "click_paste", "source": "pixel",
             "coords": [min(nx1 + 70, dl[2] - 30), (ny0 + ny1) // 2]},
            {"pred": "task_name_set", "value": task_name},
            {"value": task_name},
            "pass", "none",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": paste_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L2", recovery=rec))

        # 勾选第16级
        t0 = _now()
        rx0, ry0, rx1, ry1 = row_hit["rect"]
        band = (rx0 - 20, ry0 - 4, rx1 + 4, ry1 + 4)
        pre_ratio = cell_checked_ratio(band)
        if pre_ratio <= CHECKED_TH:
            pyautogui.click(rx0 - 12, (ry0 + ry1) // 2)
            wait_for(lambda: cell_checked_ratio(band) > CHECKED_TH, 2.0)
        post_ratio = cell_checked_ratio(band)
        lvl_ok = post_ratio > CHECKED_TH
        recs.append(make_step(
            trial, arm, "T5", 3, "table_row", pre, None,
            {"kind": "click", "source": "pixel", "coords": [rx0 - 12, (ry0 + ry1) // 2]},
            {"pred": "checkbox_checked", "th": CHECKED_TH},
            {"pre_ratio": pre_ratio, "post_ratio": post_ratio, "band": band},
            "pass" if lvl_ok else "fail", "none" if lvl_ok else "no_pattern",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": _ms(t0),
             "verify": 0.0, "recover": 0.0}, t0, layer="L2"))
        if not lvl_ok:
            for dd in find_dialogs(h):
                close_window(dd["hwnd"])
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL level")
            continue

        # 确定
        t0 = _now()
        pre_open = bool(find_dialogs(h))
        ox0, oy0, ox1, oy1 = ok_hit["rect"]
        pyautogui.click((ox0 + ox1) // 2, (oy0 + oy1) // 2)
        click_ms = _ms(t0)
        gone, gone_ms = wait_for(lambda: not find_dialogs(h), 8.0)
        ok_ok = bool(gone and pre_open)
        recs.append(make_step(
            trial, arm, "T5", 4, "standard", pre, None,
            {"kind": "click", "source": "pixel", "coords": [(ox0 + ox1) // 2, (oy0 + oy1) // 2]},
            {"pred": "dialog_closed_after_ok", "pre_open": pre_open},
            {"closed": gone, "pre_open": pre_open,
             "left": [d["title"] for d in find_dialogs(h)]},
            "pass" if ok_ok else ("vacuous" if not pre_open else "fail"),
            "none" if ok_ok else ("none" if not pre_open else "timeout"),
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": click_ms,
             "verify": gone_ms, "recover": 0.0}, t0, layer="L2"))

        # 产物 + GDAL
        t0 = _now()
        got, wait_ms, tifs = wait_artifact(task_dir, 10 if fault else 180)
        tif = tifs[0] if tifs else None
        art_ok = False
        detail = {}
        if tif is not None:
            sha = hashlib.sha256(tif.read_bytes()).hexdigest()
            rb = raster_wgs84_bounds(tif)
            if rb:
                west, south, east, north, j = rb
                dev = max(abs(west - AOI_BOUNDS[0]), abs(south - AOI_BOUNDS[1]),
                          abs(east - AOI_BOUNDS[2]), abs(north - AOI_BOUNDS[3]))
                art_ok = dev <= AOI_TOL_DEG and j.get("size", [0, 0])[0] > 0
                detail = {"bounds": [west, south, east, north], "dev_deg": dev,
                          "size": j.get("size"), "sha256": sha,
                          "bytes": tif.stat().st_size, "path": str(tif)}
            else:
                detail = {"path": str(tif), "sha256": sha,
                          "bytes": tif.stat().st_size, "gdal": "fail"}
        recs.append(make_step(
            trial, arm, "T5", 5, "standard", pre, None,
            {"kind": "wait_artifact", "source": "fs", "coords": None},
            {"pred": "raster_matches_aoi", "tol_deg": AOI_TOL_DEG},
            {"got": got, "files": [p.name for p in tifs], **detail},
            "pass" if art_ok else "fail", "none" if art_ok else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": 0.0,
             "verify": wait_ms, "recover": 0.0}, t0, layer="L4"))

        ok = bool(lvl_ok and gone and art_ok)
        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T5",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial),
                     "artifact": detail or None})
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} ocr={ocr_ms:.0f}ms "
              f"tif={tif.name if tif else None} dev={detail.get('dev_deg')} "
              f"wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def restore_any(h: int) -> bool:
    if win32gui.IsIconic(h):
        win32gui.ShowWindow(h, win32con.SW_RESTORE)
        time.sleep(0.2)
    return force_foreground(h)


def inject_fault(fault: str, dh: int, h: int) -> str:
    """F1 最小化对话框 / F4 关对话框 / F6 移动对话框。"""
    if fault == "F1":
        win32gui.ShowWindow(dh, win32con.SW_MINIMIZE)
        time.sleep(0.45)
        return "minimize_dialog"
    if fault == "F4":
        for x in find_dialogs(h):
            close_window(x["hwnd"])
        time.sleep(0.35)
        return "close_dialog"
    if fault == "F6":
        L, T, R, B = win32gui.GetWindowRect(dh)
        win32gui.SetWindowPos(dh, win32con.HWND_TOP, L + 140, T + 90,
                              R - L, B - T,
                              win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE)
        time.sleep(0.3)
        return "move_dialog"
    return "none"


def oracle_row_checked(h: int) -> tuple[bool, list | None]:
    """评分用真值：UIA 找当前对话框里的第16级行，再量像素。不参与 A 臂控制。"""
    dlgs = find_dialogs(h)
    if not dlgs:
        return False, None
    row, _, rect = find_dataitem(dlgs[0]["hwnd"], "第16级")
    if rect is None:
        return False, None
    return cell_checked_ratio(rect) > CHECKED_TH, rect


def run_t3_fault(arm: str, fault: str, trials: int, out_path: Path) -> dict:
    """T3 故障注入：A 臂用 stale 坐标；B 臂恢复窗口/重开对话框/重取 rect。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T3{arm}-{fault}-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL")
            continue
        main = find_main(); h = main["hwnd"]
        pre = window_state(h)

        t0 = _now()
        aoi, opened, dbl_ms, appear_ms = open_download(h)
        dlgs = find_dialogs(h)
        setup_ok = bool(opened and dlgs)
        recs.append(make_step(
            trial, arm, "T3", 0, "canvas", pre, None,
            {"kind": "double_click", "source": "mouse", "coords": aoi.get("center")},
            {"pred": "dialog_open", "detail": "setup"},
            {"opened": opened, "aoi": aoi},
            "pass" if setup_ok else "fail", "none" if setup_ok else "timeout",
            {"reset": r["ms"], "perceive": 0.0, "decide": 0.0, "act": dbl_ms,
             "verify": appear_ms, "recover": 0.0}, t0, layer="L1"))
        if not setup_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL setup")
            continue

        dh = dlgs[0]["hwnd"]
        t0 = _now()
        if arm == "B":
            row, find_ms, rect = find_dataitem(dh, "第16级")
            band = rect
            coords = [rect[0] + 8, (rect[1] + rect[3]) // 2] if rect else None
            ocr_ms = 0.0
        else:
            time.sleep(0.35)
            dl = dlgs[0]["rect"]
            lines, ocr_ms = ocr_lines((dl[0], dl[1], dl[2] - dl[0], dl[3] - dl[1]))
            hit = next((x for x in lines if "第16" in x["text"]), None)
            find_ms = ocr_ms
            if hit:
                x0, y0, x1, y1 = hit["rect"]
                band = (x0 - 20, y0 - 4, x1 + 4, y1 + 4)
                coords = [x0 - 12, (y0 + y1) // 2]
            else:
                band = None
                coords = None
        if not coords:
            for dd in find_dialogs(h):
                close_window(dd["hwnd"])
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL locate")
            continue

        pre_ratio = cell_checked_ratio(band)
        if pre_ratio > CHECKED_TH:            # 清残留，保证不是空过
            pyautogui.click(*coords)
            wait_for(lambda: cell_checked_ratio(band) <= CHECKED_TH, 1.5)
            pre_ratio = cell_checked_ratio(band)

        # 注入 + （仅 B）恢复
        t_rec = _now()
        rec_action = "none"
        rec_ok = True
        rec_ms = 0.0
        if fault != "none":
            rec_action = inject_fault(fault, dh, h)
            if arm == "B":
                if fault == "F1":
                    restore_any(dh)
                if fault == "F4" or not find_dialogs(h):
                    restore_window(h)
                    _, ok2, _, _ = open_download(h)
                    rec_ok = bool(ok2)
                    rec_action += "+reopen" if ok2 else "+reopen_fail"
                    if ok2:
                        time.sleep(0.35)
                cur = find_dialogs(h)
                if cur:
                    _, _, rect = find_dataitem(cur[0]["hwnd"], "第16级")
                    if rect:
                        band = rect
                        coords = [rect[0] + 8, (rect[1] + rect[3]) // 2]
                        pre_ratio = cell_checked_ratio(band)
                    else:
                        rec_ok = False
                else:
                    rec_ok = False
            elif arm == "C":
                # 像素臂 + 恢复：恢复窗口/重开对话框后**重新截图 + 重新 OCR**
                if fault == "F1":
                    restore_any(dh)
                cur = find_dialogs(h)
                if fault == "F4" or not cur:
                    restore_window(h)
                    _, ok2, _, _ = open_download(h)
                    rec_ok = bool(ok2)
                    rec_action += "+reopen" if ok2 else "+reopen_fail"
                    cur = find_dialogs(h)
                if cur:
                    time.sleep(0.35)
                    dl2 = cur[0]["rect"]
                    lines2, _ = ocr_lines((dl2[0], dl2[1],
                                           dl2[2] - dl2[0], dl2[3] - dl2[1]))
                    hit2 = next((x for x in lines2 if "第16" in x["text"]), None)
                    if hit2:
                        x0, y0, x1, y1 = hit2["rect"]
                        band = (x0 - 20, y0 - 4, x1 + 4, y1 + 4)
                        coords = [x0 - 12, (y0 + y1) // 2]
                        pre_ratio = cell_checked_ratio(band)
                        rec_action += "+reocr"
                    else:
                        rec_ok = False
                else:
                    rec_ok = False
            rec_ms = _ms(t_rec)

        t_act = _now()
        pyautogui.click(*coords)
        click_ms = _ms(t_act)
        loop_ok, wait_ms = wait_for(lambda: cell_checked_ratio(band) > CHECKED_TH, 2.0)
        oracle_ok, oracle_rect = oracle_row_checked(h)
        silent = bool(loop_ok and not oracle_ok)
        ok = bool(oracle_ok and rec_ok)
        fc = "none" if ok else ("state_lost" if fault != "none" else "no_pattern")
        recs.append(make_step(
            trial, arm, "T3", 1, "table_row", pre, None,
            {"kind": "click", "source": "uia_rect" if arm == "B" else "pixel",
             "coords": coords},
            {"pred": "checkbox_checked", "fault": fault, "th": CHECKED_TH},
            {"pre_ratio": pre_ratio, "loop_ok": loop_ok, "oracle": oracle_ok,
             "oracle_rect": oracle_rect, "silent": silent, "cleared": rec_ok},
            "pass" if ok else "fail",
            fc,
            {"reset": 0.0, "perceive": find_ms, "decide": 0.0, "act": click_ms,
             "verify": wait_ms, "recover": rec_ms}, t0, layer="L2",
            recovery=({"injected": fault, "action": rec_action, "ok": ok,
                       "ms": rec_ms} if fault != "none" else None),
            silent=silent))

        t0 = _now()
        t_act = _now()
        closed = True
        for dd in find_dialogs(h):
            closed = close_window(dd["hwnd"]) and closed
        close_ms = _ms(t_act)
        recs.append(make_step(
            trial, arm, "T3", 2, "table_row", pre, None,
            {"kind": "close", "source": "uia", "coords": None},
            {"pred": "dialog_closed"},
            {"closed": closed, "left": len(find_dialogs(h))},
            "pass" if closed else "fail", "none" if closed else "state_lost",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": close_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L3"))

        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T3",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} loop={loop_ok} oracle={oracle_ok} "
              f"silent={silent} rec={rec_action if fault != 'none' else '-'} "
              f"wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


T6_DISTRICT = (116.1056, 39.8729, 116.2537, 39.9921)   # 石景山区边界
T6_TOL = 2e-3
OGR2OGR = Path(CFG.get("ogr2ogr", r"D:\QGIS\bin\ogr2ogr.exe"))


def region_select(h: int, path=("北京市", "北京城区", "石景山区")) -> dict:
    """选择行政区域：省(直辖市) → 市(城区) → 区县，按名称选择。"""
    t0 = _now()
    w = Desktop(backend="uia").window(handle=h)
    tab = next(c for c in w.descendants()
               if c.element_info.control_type == "TabItem"
               and "选择行政区域" in (c.window_text() or ""))
    rc = tab.rectangle()
    pyautogui.click((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
    time.sleep(0.5)
    combos = [c for c in w.descendants()
              if c.element_info.control_type == "ComboBox" and c.rectangle().top < 250]
    combos.sort(key=lambda c: (c.rectangle().top, c.rectangle().left))
    picked, err = [], None
    try:
        for idx, val in zip((2, 3, 4), path):
            ok = False
            for _ in range(2):
                combos[idx].select(val)
                time.sleep(0.6)
                cur = (combos[idx].selected_text() or "").strip()
                if cur == val:
                    ok = True
                    break
                time.sleep(0.4)
            if not ok:
                raise RuntimeError(f"select {val} -> {cur!r}")
            picked.append(val)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    time.sleep(0.4)
    return {"ms": _ms(t0), "picked": picked, "error": err,
            "bounds": panel_bounds(h), "district": path[-1]}


def panel_bounds(h: int):
    """解析区域面板上的经纬度范围（应用自报的 AOI 定义）。"""
    w = Desktop(backend="uia").window(handle=h)
    for c in w.descendants():
        try:
            if c.element_info.control_type != "Text":
                continue
            txt = (c.window_text() or "").strip()
            m = re.search(r"([\d.]+)[，,]\s*([\d.]+)\s*>>\s*([\d.]+)[，,]\s*([\d.]+)", txt)
            if m:
                x1, y1, x2, y2 = (float(x) for x in m.groups())
                # 面板顺序是 (西, 北) >> (东, 南)，统一成 (w, s, e, n)
                return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except Exception:
            continue
    return None


def panel_icon_point(h: int, district: str = "石景山区") -> list | None:
    """区域面板的下载图标：锚定面板标题文字，避免面板因布局上下移动。"""
    w = Desktop(backend="uia").window(handle=h)
    for c in w.descendants():
        try:
            if c.element_info.control_type == "Text" and \
               (c.window_text() or "").strip() == district:
                rc = c.rectangle()
                if rc.width() < 50 or rc.top < 50:
                    continue
                L, T, R, B = win32gui.GetWindowRect(h)
                return [R - 162, rc.top + 30]
        except Exception:
            continue
    return None


def run_t6(arm: str, trials: int, out_path: Path, fault: str | None = None,
           region=("北京市", "北京城区", "石景山区")) -> dict:
    """T6：行政区边界下载（省+区县，SHP/WGS84）→ OGR 校验 2 个面。"""
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T6-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        task_dir = T5_ROOT / trial
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T6",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL")
            continue
        main = find_main(); h = main["hwnd"]
        L, T, R, B = win32gui.GetWindowRect(h)
        pre = window_state(h)

        # 0) 选择区域
        t0 = _now()
        sel = region_select(h, region)
        sel_ok = bool(sel.get("bounds")) and not sel.get("error") and len(sel["picked"]) == 3
        recs.append(make_step(
            trial, arm, "T6", 0, "standard", pre, None,
            {"kind": "select_region", "source": "uia", "coords": None},
            {"pred": "district_selected", "value": region[-1]},
            sel, "pass" if sel_ok else "fail", "none" if sel_ok else "wrong_control",
            {"reset": r["ms"], "perceive": sel["ms"], "decide": 0.0, "act": 0.0,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if not sel_ok:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T6",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL region {region} {sel.get('error')}")
            continue

        # 1) 打开下载对话框（区域面板下载图标，锚定面板标题）
        t0 = _now()
        icon = panel_icon_point(h, region[-1])
        if icon:
            pyautogui.click(*icon)
        ok_open, open_ms = wait_for(lambda: len(find_dialogs(h)) > 0, 5.0)
        dlgs = find_dialogs(h)
        recs.append(make_step(
            trial, arm, "T6", 1, "standard", pre, None,
            {"kind": "click", "source": "uia_anchor", "coords": icon},
            {"pred": "dialog_open"}, {"opened": ok_open, "icon": icon},
            "pass" if ok_open else "fail", "none" if ok_open else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": open_ms,
             "verify": 0.0, "recover": 0.0}, t0, layer="L1"))
        if not ok_open:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T6",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
            print(f"[{trial}] FAIL open")
            continue

        dh = dlgs[0]["hwnd"]
        # 应用会记住上次的 tab；T5 显式切回「影像」页
        try:
            w2 = Desktop(backend="uia").window(handle=dh)
            for c in w2.descendants():
                if c.element_info.control_type == "TabItem" and "影" in (c.window_text() or ""):
                    rc = c.rectangle()
                    pyautogui.click((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
                    time.sleep(0.6)
                    break
        except Exception:
            pass
        rec = None
        if fault in ("F1", "F4", "F6"):
            t_rec = _now()
            act = inject_fault(fault, dh, h)
            okr = True
            if fault == "F1":
                okr = restore_any(dh)
            elif fault == "F4":
                time.sleep(0.3)
                icon = panel_icon_point(h)
                if icon:
                    pyautogui.click(*icon)
                ok2, _ = wait_for(lambda: len(find_dialogs(h)) > 0, 5.0)
                dlgs2 = find_dialogs(h)
                okr = bool(ok2 and dlgs2)
                if okr:
                    dh = dlgs2[0]["hwnd"]
            rec = {"injected": fault, "action": act, "ok": bool(okr),
                   "ms": _ms(t_rec)}
            if not okr:
                recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T6",
                             "success": False, "attempts": 1, "takeovers": 0,
                             "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial)})
                print(f"[{trial}] FAIL recover")
                continue
        w2 = Desktop(backend="uia").window(handle=dh)

        # 2) 切到行政边界页 + 配置 + 确定
        t0 = _now()
        cfg_ok = False
        detail = {}
        try:
            for c in w2.descendants():
                if c.element_info.control_type == "TabItem" and "行政边界" in (c.window_text() or ""):
                    rc = c.rectangle()
                    pyautogui.click((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
                    break
            time.sleep(0.6)
            fmt = [c for c in w2.descendants()
                   if c.element_info.control_type == "ComboBox"][0]
            fmt.select("Shape File (*.shp)")
            edits = sorted([c for c in w2.descendants()
                            if c.element_info.control_type == "Edit"],
                           key=lambda c: c.rectangle().top)
            paste_into(edits[0].rectangle(), trial)
            chk = next(c for c in w2.descendants()
                       if c.element_info.control_type == "CheckBox"
                       and (c.window_text() or "").strip() == "区县")
            if chk.get_toggle_state() == 0:
                rc = chk.rectangle()
                pyautogui.click((rc.left + rc.right) // 2, (rc.top + rc.bottom) // 2)
            time.sleep(0.2)
            w2.child_window(title="确定", control_type="Button").wrapper_object().click_input()
            cfg_ok = True
            detail = {"format": fmt.selected_text(), "name": trial,
                      "district_checked": chk.get_toggle_state()}
        except Exception as e:
            detail = {"error": f"{type(e).__name__}: {e}"}
        gone, gone_ms = wait_for(lambda: not find_dialogs(h), 10.0)
        recs.append(make_step(
            trial, arm, "T6", 2, "standard", pre, None,
            {"kind": "configure_confirm", "source": "uia", "coords": None},
            {"pred": "dialog_closed_after_ok"}, {**detail, "closed": gone},
            "pass" if (cfg_ok and gone) else "fail",
            "none" if (cfg_ok and gone) else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": 0.0,
             "verify": gone_ms, "recover": 0.0}, t0, layer="L2", recovery=rec))

        # 3) 产物 + OGR oracle
        t0 = _now()
        shp = None
        wait_ms = 0.0
        ok_file, wait_ms = wait_for(
            lambda: any(p.stat().st_size > 1000 for p in task_dir.glob("*.shp"))
            if task_dir.exists() else False, 90.0)
        shps = list(task_dir.glob("*.shp")) if task_dir.exists() else []
        art_ok = False
        art = {}
        if shps:
            shp = shps[0]
            gj = ROOT / "artifacts" / "bench_dl" / trial / "verify.geojson"
            gj.parent.mkdir(parents=True, exist_ok=True)
            o = None
            for _ in range(3):
                o = subprocess.run([str(OGR2OGR), "-f", "GeoJSON", str(gj), str(shp)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                if o.returncode == 0 and gj.exists():
                    break
                time.sleep(0.5)
            if o is not None and o.returncode == 0 and gj.exists():
                j = json.loads(gj.read_text(encoding="utf-8"))
                feats = j.get("features", [])
                bounds = []
                for f in feats:
                    xs, ys = [], []

                    def walk(c):
                        if not c:
                            return
                        if isinstance(c[0], (int, float)):
                            xs.append(c[0]); ys.append(c[1]); return
                        for p in c:
                            walk(p)
                    walk(f["geometry"]["coordinates"])
                    bounds.append((min(xs), min(ys), max(xs), max(ys)))
                hit = None
                exp = sel.get("bounds")
                if exp:
                    hit = next((b for b in bounds
                                if all(abs(b[k] - exp[k]) <= T6_TOL for k in range(4))), None)
                art_ok = len(feats) == 2 and hit is not None
                art = {"features": len(feats), "bounds": bounds, "expected": exp,
                       "district_match": hit is not None,
                       "shp_bytes": shp.stat().st_size,
                       "sha256": hashlib.sha256(shp.read_bytes()).hexdigest(),
                       "path": str(shp)}
            else:
                art = {"ogr2ogr": "fail", "stderr": o.stderr[:200]}
        recs.append(make_step(
            trial, arm, "T6", 3, "standard", pre, None,
            {"kind": "wait_artifact", "source": "fs", "coords": None},
            {"pred": "two_features_with_district", "tol": T6_TOL},
            {"got": ok_file, **art},
            "pass" if art_ok else "fail", "none" if art_ok else "timeout",
            {"reset": 0.0, "perceive": 0.0, "decide": 0.0, "act": 0.0,
             "verify": wait_ms, "recover": 0.0}, t0, layer="L4"))

        ok = bool(cfg_ok and gone and art_ok)
        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T6",
                     "success": ok, "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": _ms(t_trial),
                     "artifact": art or None})
        for dd in find_dialogs(h):
            close_window(dd["hwnd"])
        runs += 1
        print(f"[{trial}] {'PASS' if ok else 'FAIL'} sel={sel['ms']:.0f}ms "
              f"open={ok_open} feats={art.get('features')} dist={art.get('district_match')} "
              f"wall={_ms(t_trial)/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def run_t1(arm: str, trials: int, inject: str | None, out_path: Path) -> dict:
    recs: list[dict] = []
    runs = 0
    for i in range(trials):
        trial = f"T1-{arm}-{i:04d}-{uuid.uuid4().hex[:4]}"
        t_trial = _now()
        r = reset(trial)
        if not r["ok"]:
            recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T1",
                         "success": False, "attempts": 1, "takeovers": 0,
                         "reset_ok": False, "reset_ms": r["ms"], "wall_ms": r["ms"]})
            print(f"[{trial}] reset FAIL: {r.get('reason')}")
            continue

        main = find_main()
        h = main["hwnd"]
        pre = window_state(h)
        pre["control"] = {"name": main["title"], "type": "Window", "rect": pre["rect"],
                          "patterns": ["WindowPattern"]}

        # 任务本身：先最小化（T1 的定义），可选再叠加 F2 抢焦点
        t0 = _now()
        win32gui.ShowWindow(h, win32con.SW_MINIMIZE)
        time.sleep(0.4)
        minimized = window_state(h)
        intruder = None
        f2_ok = None
        if inject == "F2":
            intruder = Intruder()
            f2_ok = intruder.foreground()

        t_act = _now()
        rr = restore_window(h)
        act_ms = _ms(t_act)

        t_ver = _now()
        st = window_state(h)
        try:
            nodes = uia_nodes(h)
        except Exception:
            nodes = -1
        restored = (not st["minimized"]) and st["visible"] and rect_ok(st["rect"])
        fg_ok = st["foreground"]
        verify_ms = _ms(t_ver)
        if intruder is not None:
            intruder.close()

        seg = {"reset": r["ms"], "perceive": 0.0, "decide": 0.0,
               "act": act_ms, "verify": verify_ms, "recover": 0.0}
        expect = {"pred": "restored_and_foreground",
                  "pre": {"minimized": minimized["minimized"], "rect": minimized["rect"]}}
        observed = {"minimized": st["minimized"], "visible": st["visible"],
                    "rect": st["rect"], "rect_ok": rect_ok(st["rect"]),
                    "foreground_ok": fg_ok, "uia_nodes": nodes,
                    "f2_injected": f2_ok}
        verdict = "pass" if (restored and fg_ok and nodes >= MIN_MAIN_NODES) else "fail"
        fc = "none" if verdict == "pass" else (
            "focus_eaten" if not fg_ok else ("stale_rect" if not restored else "timeout"))
        recs.append(make_step(trial, arm, "T1", 0, "window", pre, st,
                              {"kind": "restore", "source": "win32", "coords": None},
                              expect, observed, verdict, fc, seg, t0,
                              recovery=({"injected": "F2", "action": "restore",
                                         "ok": bool(fg_ok and restored),
                                         "ms": act_ms} if inject == "F2" else None)))
        wall = _ms(t_trial)
        recs.append({"kind": "trial", "trial": trial, "arm": arm, "task": "T1",
                     "success": verdict == "pass", "attempts": 1, "takeovers": 0,
                     "reset_ok": True, "reset_ms": r["ms"], "wall_ms": wall,
                     "artifact": None})
        runs += 1
        print(f"[{trial}] {'PASS' if verdict == 'pass' else 'FAIL'} "
              f"restored={restored} fg={fg_ok} nodes={nodes} act={act_ms:.0f}ms wall={wall/1000:.1f}s")

    _save(recs, out_path)
    print(f"[done] trials={runs}/{trials}")
    return {"out": str(out_path), "trials": runs, "records": len(recs)}


def preflight() -> int:
    """开跑前体检：配置、依赖、屏幕、应用、AOI、GDAL/OGR、目录可写。"""
    rows: list[tuple[str, bool, str, bool]] = []

    def add(name, ok, detail="", fatal=True):
        rows.append((name, bool(ok), str(detail), fatal))

    add("config", CFG_PATH.exists(), CFG_PATH)
    for m in ("pywinauto", "pyautogui", "win32gui", "win32api", "numpy"):
        try:
            __import__(m)
            add(f"dep:{m}", True)
        except Exception as e:
            add(f"dep:{m}", False, type(e).__name__)
    try:
        __import__("rapidocr_onnxruntime")
        add("dep:rapidocr", True)
    except Exception:
        add("dep:rapidocr(optional)", False, "A/C 臂 OCR 不可用", fatal=False)

    w = win32api.GetSystemMetrics(0)
    h = win32api.GetSystemMetrics(1)
    add("screen", (w, h) == SCREEN, f"{w}x{h}（期望 {SCREEN[0]}x{SCREEN[1]}）")

    main = find_main()
    add("app", main is not None, TITLE_KEY)
    if main:
        add("window_rect", rect_ok(main["rect"]), main["rect"])
        aoi = detect_aoi(main["hwnd"])
        add("aoi", aoi.get("found"), f"size={aoi.get('size')}（T4/T5 需要）", fatal=False)

    for name, p in (("gdalinfo", GDALINFO), ("ogr2ogr", OGR2OGR)):
        ok = p.exists()
        ver = ""
        if ok:
            try:
                r = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=10)
                ok = r.returncode == 0
                out = (r.stdout or r.stderr or "").strip().splitlines()
                ver = out[0] if out else ""
            except Exception as e:
                ok, ver = False, type(e).__name__
        add(name, ok, f"{p} {ver}")

    for name, d in (("download_root", T5_ROOT), ("artifacts_root", DL_ROOT)):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".bench_preflight"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            add(name, True, d)
        except Exception as e:
            add(name, False, f"{d}: {e}")

    print(f"{'CHECK':26s} {'TAG':4s} DETAIL")
    for name, ok, detail, fatal in rows:
        print(f"{name:26s} {('PASS' if ok else ('FAIL' if fatal else 'WARN')):4s} {detail}")
    fatal_n = sum(1 for r in rows if not r[1] and r[3])
    print(f"\n{len(rows) - fatal_n}/{len(rows)} ok, fatal={fatal_n}")
    return 1 if fatal_n else 0


def env_report() -> dict:
    main = find_main()
    fg = win32gui.GetForegroundWindow()
    u = win32api.GetSystemMetrics
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "screen": [u(0), u(1)],
        "main": main,
        "dialogs": find_dialogs(main["hwnd"]) if main else [],
        "foreground": {"hwnd": fg, "title": win32gui.GetWindowText(fg)[:80]},
        "canonical": list(CANON),
        "rect_ok": rect_ok(main["rect"]) if main else False,
        "uia_nodes": uia_nodes(main["hwnd"]) if main else 0,
        "exe": main and win32gui.GetWindowText(main["hwnd"]),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Bigemap 无特权执行基准 runner（T1）")
    ap.add_argument("--probe", action="store_true", help="只读体检：窗口/UIA/对话框，不改状态")
    ap.add_argument("--preflight", action="store_true", help="开跑前体检：配置/依赖/GDAL/目录/应用/AOI")
    ap.add_argument("--task", default="T1", choices=["T1", "T2", "T3", "T4", "T5", "T6"])
    ap.add_argument("--arm", default="S", choices=["A", "B", "C", "S"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--inject", default=None, choices=[None, "F1", "F2", "F4", "F6", "F8"])
    ap.add_argument("--level", type=int, default=16, help="T5 下载级别（默认 16）")
    ap.add_argument("--source", default="aoi", choices=["aoi", "region"],
                    help="T5 入口：aoi=双击画布 AOI；region=区域面板下载图标")
    ap.add_argument("--region", default=CFG.get("t6_default_region", "北京市/北京城区/石景山区"),
                    help="省/市/区县，例如 河北省/石家庄市/长安区")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.preflight:
        raise SystemExit(preflight())

    if args.probe:
        rep = env_report()
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        if rep["main"] is None:
            raise SystemExit("[probe] Bigemap 未运行")
        if not rep["rect_ok"]:
            print(f"[probe] 窗口不在 canonical {CANON}，runner 会先摆放")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else (
        OUT_DIR / f"bigemap_bench_{args.arm}_{stamp}.jsonl")
    if args.task == "T6":
        region = tuple(x.strip() for x in re.split(r"[/>]", args.region) if x.strip())
        run_t6(args.arm, args.trials, out,
               args.inject if args.inject in ("F1", "F4", "F6") else None,
               region)
    elif args.task == "T3" and args.inject in ("F1", "F4", "F6"):
        run_t3_fault(args.arm, args.inject, args.trials, out)
    elif args.task == "T2" and args.arm == "A":
        run_t2_a(args.trials, out, args.inject if args.inject == "F8" else None)
    elif args.task == "T3" and args.arm == "A":
        run_t3_a(args.trials, out)
    elif args.task == "T5" and args.arm in ("A", "C"):
        run_t5_pixel(args.arm, args.trials, out,
                     args.inject if args.inject in ("F1", "F4", "F6") else None)
    elif args.task == "T5":
        region = None
        if args.source == "region":
            region = tuple(x.strip() for x in re.split(r"[/>]", args.region) if x.strip())
        run_t5(args.arm, args.trials, out,
               args.inject if args.inject in ("F1", "F4", "F6") else None,
               args.level, region)
    elif args.task == "T4":
        if args.inject:
            print("[warn] T4 暂不支持注入，已忽略 --inject", file=sys.stderr)
        run_t4(args.arm, args.trials, out)
    elif args.task == "T3":
        if args.inject:
            print("[warn] T3 暂不支持注入，已忽略 --inject", file=sys.stderr)
        run_t3(args.arm, args.trials, out)
    elif args.task == "T2":
        run_t2(args.arm, args.trials, out, args.inject if args.inject == "F8" else None)
    else:
        run_t1(args.arm, args.trials, args.inject, out)


if __name__ == "__main__":
    main()
