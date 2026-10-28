"""
Notepad 教师链（W28b E3 数据源）：写文本 → 文件菜单另存为 → UIA/磁盘验证。

A' 路线核心：Notepad 是**从未出现在 v3 训练分布**中的软件，
采集秒级/任务；其微调后对 QGIS 的零样本迁移是 A' 的决定性证据。

安全规约（2026-10-22 事故后）：
  - 绝不对恢复的旧标签页发 Ctrl+S（会静默覆盖用户文件）
  - 只用 文件 → 另存为（显式路径），并先 Ctrl+N 保证活动标签为空白
  - 关闭走 WM_CLOSE/Alt+F4，遇保存提示一律"不保存"

扰动（data-model §1.2 taxonomy）：
  nominal / drift / occlusion / focus_error

用法: python experiments/computer_use/collect_notepad.py --tasks 10
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from inloop_rec import Recorder  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402
import win32api  # noqa: E402
import win32clipboard as clip  # noqa: E402
import win32con  # noqa: E402
import win32gui  # noqa: E402
import win32process  # noqa: E402

pyautogui.FAILSAFE = False
WM_INPUTLANGCHANGEREQUEST = 0x0050

OUT = ROOT / "artifacts" / "computer_use_inloop"
NOTEPAD_DIR = OUT / "notepad"
FILES_DIR = NOTEPAD_DIR / "files"
SUMMARY = ROOT / "results" / "computer_use" / "collect_notepad_summary.json"

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "qgis is a free and open source geographic information system",
    "cross software transfer experiment sample note",
    "grounding verification and recovery loop",
    "a general desktop agent learns from teachers",
    "notepad teacher chain writes and saves this text",
    "model in the loop deterministic execution demo",
    "the fifth assessment report summary for policymakers",
]

VARIANTS = ["nominal", "drift", "drift", "occlusion", "focus_error"]
AFFORD_TYPES = {"Button", "Edit", "ComboBox", "MenuItem", "TabItem",
                "CheckBox", "Document"}

TARGET_HWND = None


# ------------------------------------------------------------ observation ---
def notepad_observation():
    """目标窗口 affordances + 真值；全屏窗口清单（含外来遮挡窗）。"""
    wins, afford = [], []
    edit_text = None
    canvas_rect = None
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            cls = w.element_info.class_name or ""
            if "tooltip" in cls.lower() or cls in ("Progman", "SysShadow"):
                continue
            r = w.rectangle()
            if int(r.right) - int(r.left) <= 0:
                continue
            wins.append({"class": cls,
                         "title": (w.window_text() or "")[:60],
                         "rect": [int(r.left), int(r.top),
                                  int(r.right), int(r.bottom)]})
            if TARGET_HWND is None or w.handle != TARGET_HWND:
                continue
            seen = set()
            best_area = 0
            for c in w.descendants():
                try:
                    ct = c.element_info.control_type
                    if ct not in AFFORD_TYPES:
                        continue
                    cr = c.rectangle()
                    aw = int(cr.right) - int(cr.left)
                    ah = int(cr.bottom) - int(cr.top)
                    if aw <= 0 or ah <= 0:
                        continue
                    raw = c.window_text() or ""
                    t = raw.strip()
                    if ct in ("Edit", "Document") and \
                            "文件名" not in t and "编码" not in t and \
                            "查找" not in t:
                        if aw * ah > best_area:
                            best_area = aw * ah
                            edit_text = raw[:400]
                            canvas_rect = [int(cr.left), int(cr.top),
                                            int(cr.right), int(cr.bottom)]
                    if not t or len(afford) >= 40:
                        continue
                    key = (t[:60], ct)
                    if key in seen:
                        continue
                    seen.add(key)
                    afford.append({
                        "text": t[:60], "cls": ct,
                        "rect": [int(cr.left), int(cr.top),
                                  int(cr.right), int(cr.bottom)]})
                except Exception:
                    continue
        except Exception:
            continue
    return {
        "truth": {"edit_text": edit_text},
        "affordances": afford[:40],
        "windows": wins,
        "canvas_rect": canvas_rect,
    }


# ---------------------------------------------------------------- helpers ---
def snapshot_notepad_handles():
    s = set()
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") == "Notepad":
                s.add(w.handle)
        except Exception:
            continue
    return s


def notepad_wins():
    return [w for w in Desktop(backend="uia").windows(visible_only=True)
            if (w.element_info.class_name or "") == "Notepad"]


def find_new_notepad(known, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for w in notepad_wins():
            try:
                if w.handle not in known:
                    return w
            except Exception:
                continue
        time.sleep(0.25)
    return None


def buttons(win):
    out = []
    for c in win.descendants():
        try:
            if c.element_info.control_type == "Button":
                t = (c.window_text() or "").strip()
                if t:
                    out.append((t, c))
        except Exception:
            continue
    return out


def force_close(win):
    try:
        win32gui.PostMessage(win.handle, win32con.WM_CLOSE, 0, 0)
    except Exception:
        return
    time.sleep(0.9)
    for t, c in buttons(win):
        if t in ("不保存", "Don't save", "Don't Save"):
            try:
                c.invoke()
            except Exception:
                pass
            return


def _clip_set(text):
    clip.OpenClipboard()
    clip.EmptyClipboard()
    clip.SetClipboardText(text)
    clip.CloseClipboard()


def set_geometry(hwnd, rng):
    x = rng.randint(80, 700)
    y = rng.randint(60, 320)
    w = rng.randint(760, 1150)
    h = rng.randint(500, 800)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, w, h, 0)
    time.sleep(0.4)
    return [x, y, x + w, y + h]


def launch_occluder(main_rect, rng):
    try:
        import tkinter as tk
    except Exception:
        return None
    mx0, my0, mx1, my1 = main_rect
    w = int((mx1 - mx0) * 0.42)
    h = int((my1 - my0) * 0.42)
    x = mx1 - w - 6
    y = my1 - h - 6
    root = tk.Tk()
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.attributes("-topmost", True)
    root.update()
    return root


def menu_item(win, text):
    for c in win.descendants():
        try:
            if c.element_info.control_type == "MenuItem" and \
                    (c.window_text() or "").strip().replace("...", "") == text:
                return c
        except Exception:
            continue
    return None


def find_save_dialog(win, timeout=12):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for c in win.descendants():
            try:
                if c.element_info.control_type == "Window" and \
                        ("另存为" in (c.window_text() or "")
                         or "Save As" in (c.window_text() or "")):
                    return c
            except Exception:
                continue
        time.sleep(0.4)
    return None


def dlg_file_edit(dlg):
    for c in dlg.descendants():
        try:
            if c.element_info.automation_id == "1001" and \
                    c.element_info.control_type == "Edit":
                return c
        except Exception:
            continue
    return None


def dlg_save_btn(dlg):
    for c in dlg.descendants():
        try:
            t = (c.window_text() or "").strip()
            if c.element_info.control_type == "Button" and \
                    (t.startswith("保存") or t.startswith("Save")):
                return c
        except Exception:
            continue
    return None


def read_edit_text(win):
    best, best_area = None, 0
    for c in win.descendants():
        try:
            if c.element_info.control_type not in ("Edit", "Document"):
                continue
            cr = c.rectangle()
            area = (int(cr.right) - int(cr.left)) * \
                   (int(cr.bottom) - int(cr.top))
            if area > best_area:
                best_area = area
                best = c.window_text() or ""
        except Exception:
            continue
    return best if best is not None else None


def wait_edit_text(win, expected, timeout=4.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        last = (read_edit_text(win) or "").replace("\r\n", "\n")
        if last.strip() == expected.strip():
            return last
        time.sleep(0.3)
    return last


def ensure_focus(win, tries=3):
    for _ in range(tries):
        try:
            if win32gui.GetForegroundWindow() == win.handle:
                return True
            win.set_focus()
            time.sleep(0.35)
        except Exception:
            time.sleep(0.2)
    try:
        return win32gui.GetForegroundWindow() == win.handle
    except Exception:
        return False


def cleanup_all_notepad():
    for w in notepad_wins():
        force_close(w)
    time.sleep(0.5)
    return len(notepad_wins())


# ------------------------------------------------------------ teacher chain ---
def run_task(rec, task_id, variant, rng):
    known = snapshot_notepad_handles()
    subprocess.Popen(["notepad.exe"])
    time.sleep(1.4)
    win = find_new_notepad(known)
    if win is None:
        return False, "no new notepad window"

    # 新建窗口（不录制）：避免 Ctrl+N 标签页与会话恢复干扰
    win.set_focus()
    time.sleep(0.4)
    fg_known = snapshot_notepad_handles()
    nw = None
    for _ in range(4):
        ensure_focus(win)
        fm = menu_item(win, "文件") or menu_item(win, "File")
        if fm is not None:
            try:
                fm.click_input()
            except Exception:
                pass
            time.sleep(0.7)
            nw = menu_item(win, "新建窗口") or menu_item(win, "New window")
            if nw is not None:
                break
        pyautogui.press("esc")
        time.sleep(0.4)
    if nw is None:
        force_close(win)
        return False, "no new-window item"
    nw.click_input()
    time.sleep(1.3)
    win2 = find_new_notepad(fg_known, timeout=8)
    if win2 is not None:
        win = win2
    else:
        force_close(win)
        return False, "no clean window"
    np_pid = win.element_info.process_id
    global TARGET_HWND
    TARGET_HWND = win.handle

    blank = False
    for _ in range(4):
        try:
            if (win.window_text() or "").strip().startswith(("无标题",
                                                             "Untitled")):
                blank = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not blank:
        force_close(win)
        return False, "active tab not blank"
    if (read_edit_text(win) or "").strip() != "":
        force_close(win)
        return False, "edit not empty"

    win.set_focus()
    time.sleep(0.3)
    hkl = win32api.LoadKeyboardLayout("00000409", win32con.KLF_ACTIVATE)
    win32api.PostMessage(win.handle, WM_INPUTLANGCHANGEREQUEST, 0, hkl)
    time.sleep(0.4)

    geo = set_geometry(win.handle, rng) if variant == "drift" else None
    occ = None
    if variant == "occlusion":
        obs = notepad_observation()
        main_rect = obs.get("canvas_rect")
        if main_rect:
            occ = launch_occluder(main_rect, rng)
            time.sleep(0.5)
            try:
                win32gui.SetForegroundWindow(win.handle)
            except Exception:
                pass
            time.sleep(0.3)

    out_p = FILES_DIR / f"{task_id}.txt"
    rec.new_task(task_id, task_name="write_note",
                 chain="notepad_teacher")
    rec.set_perturbation(variant)
    rec.set_phase("write", "Write a note in Notepad and save it to a file")

    try:
        obs = notepad_observation()
        edit = obs.get("canvas_rect")
        if edit is None:
            raise RuntimeError("no edit control")
        ex, ey = (edit[0] + edit[2]) // 2, (edit[1] + edit[3]) // 2
        jx = rng.randint(-100, 100) if variant == "drift" else 0
        jy = rng.randint(-50, 50) if variant == "drift" else 0
        ensure_focus(win)
        pyautogui.click(ex + jx, ey + jy)
        time.sleep(0.3)

        line1 = rng.choice(TEXTS)
        line2 = rng.choice(TEXTS)
        expected = f"{line1}\n{line2}"

        def _read(expect, timeout=3.0):
            t0 = time.time()
            last = ""
            while time.time() - t0 < timeout:
                try:
                    last = ((notepad_observation().get("truth") or {})
                            .get("edit_text") or "")
                    last = last.replace("\r\n", "\n").replace("\r", "\n")
                except Exception:
                    last = ""
                if last.strip() == expect.strip():
                    return last
                time.sleep(0.25)
            return last

        if not ensure_focus(win):
            raise RuntimeError("focus lost before typing")
        rec.type_text(line1)
        time.sleep(0.8)
        got1 = _read(line1)
        if got1.strip() == line1.strip():
            pyautogui.press("enter")
            time.sleep(0.4)
            rec.type_text(line2)
            time.sleep(0.8)
            typed = _read(expected)
        else:
            typed = got1
        if typed.strip() != expected.strip():
            # 教师端异常：暂停录制清空后重打（不把噪声当示范）
            rec.pause()
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("delete")
            time.sleep(0.4)
            rec.resume()
            rec.type_text(line1)
            time.sleep(0.8)
            got1 = _read(line1)
            if got1.strip() == line1.strip():
                pyautogui.press("enter")
                time.sleep(0.4)
                rec.type_text(line2)
                time.sleep(0.8)
                typed = _read(expected)
            else:
                typed = got1
            if typed.strip() != expected.strip():
                rec.tag_failure("input_mismatch",
                                note="typing mismatch twice")
                dbg = []
                try:
                    for c in win.descendants():
                        try:
                            if c.element_info.control_type in ("Edit",
                                                               "Document"):
                                rr = c.rectangle()
                                dbg.append((c.element_info.control_type,
                                            len(c.window_text() or ""),
                                            (int(rr.left), int(rr.top),
                                             int(rr.right), int(rr.bottom))))
                        except Exception:
                            pass
                    fg = win32gui.GetForegroundWindow()
                    try:
                        fg_cls = win32gui.GetClassName(fg)
                        fg_t = (win32gui.GetWindowText(fg) or "")[:40]
                    except Exception:
                        fg_cls, fg_t = "", ""
                    print("  dbg win:", win.handle,
                          repr((win.window_text() or "")[:40]),
                          "| fg:", fg, fg_cls, repr(fg_t),
                          "| docs:", dbg)
                except Exception as exc:
                    print("  dbg err:", exc)
                raise RuntimeError(f"typed mismatch: {typed[:40]!r}")

        if variant == "focus_error":
            pyautogui.hotkey("ctrl", "f")
            time.sleep(0.8)
            rec.tag_failure(
                "focus_error", note="stray Find bar",
                instr="Close the stray Find bar with Escape")
            pyautogui.press("esc")
            time.sleep(0.6)

        rec.set_phase("save", "Save the note as a text file")
        sa = None
        for _ in range(4):
            ensure_focus(win)
            fm = menu_item(win, "文件")
            if fm is None:
                fm = menu_item(win, "File")
            if fm is not None:
                try:
                    fm.click_input()
                except Exception:
                    pass
                time.sleep(0.7)
                sa = menu_item(win, "另存为")
                if sa is None:
                    sa = menu_item(win, "Save As")
                if sa is not None:
                    break
            pyautogui.press("esc")
            time.sleep(0.4)
        if sa is None:
            raise RuntimeError("no save-as item")
        sa.invoke()

        dlg = find_save_dialog(win)
        if dlg is None:
            raise RuntimeError("no save dialog")
        fe = dlg_file_edit(dlg)
        if fe is None:
            raise RuntimeError("no filename edit")
        fe.click_input()
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "a")
        _clip_set(str(out_p))
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.4)
        sb = dlg_save_btn(dlg)
        if sb is None:
            raise RuntimeError("no save button")
        sb.invoke()

        got = None
        t0 = time.time()
        while time.time() - t0 < 9:
            if out_p.exists():
                got = out_p.read_text(encoding="utf-8")
                if got:
                    break
            time.sleep(0.3)
        ok = (got or "").replace("\r\n", "\n").strip() == expected.strip()
        verify = {"file": str(out_p.relative_to(ROOT)),
                  "bytes": out_p.stat().st_size if out_p.exists() else 0,
                  "content_match": bool(ok)}

        pyautogui.hotkey("alt", "f4")
        time.sleep(0.8)
        rec.end_task(ok, verify)
        return ok, verify
    except Exception as e:
        try:
            rec.end_task(False, {"error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        return False, f"{type(e).__name__}: {e}"
    finally:
        if occ is not None:
            try:
                occ.destroy()
            except Exception:
                pass
        try:
            force_close(win)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    FILES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    left = cleanup_all_notepad()
    print(f"环境清理：残留记事本窗口 {left} 个")

    rec = Recorder("notepad", OUT, obs_provider=notepad_observation,
                   shot_scale=0.6, settle=0.35)
    rec.install()
    rows = []
    print(f"== notepad 教师链（{args.tasks} 任务，扰动循环）==")
    for i in range(args.tasks):
        task_id = f"np_{i + 1:04d}"
        variant = VARIANTS[i % len(VARIANTS)]
        t0 = time.time()
        ok, info = run_task(rec, task_id, variant, rng)
        row = {"task_id": task_id, "variant": variant, "ok": bool(ok),
               "info": str(info)[:200],
               "s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  {task_id} [{variant}] {'PASS' if ok else 'FAIL'}"
              f" {row['s']}s  {str(info)[:80]}")
    rec.uninstall()

    summary = {
        "tasks": len(rows), "pass": sum(1 for r in rows if r["ok"]),
        "variants": {v: sum(1 for r in rows if r["variant"] == v)
                     for v in VARIANTS},
        "n_demos": 0,
        "rows": rows,
    }
    try:
        idx = json.loads((NOTEPAD_DIR / "index.json")
                         .read_text(encoding="utf-8"))
        summary["n_demos"] = sum(m.get("n_demos", 0) for m in idx)
    except Exception:
        pass
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"notepad: {summary['pass']}/{summary['tasks']}"
          f"  demos={summary['n_demos']}")
    print(f"已写出 {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
