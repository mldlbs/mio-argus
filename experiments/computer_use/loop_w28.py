"""
W28 模型接环测距：v3 零样本在真桌面上预测动作（QGIS 干跑 + 安全闭环冒烟）。

把 MioArgus v3（ComputerUseModelV3，artifacts/checkpoints/computer_use_model_v3_real.pth）
接进"截图 → 模型 → 解析 → 执行"闭环，做两段测距：

  A. 干跑（不点鼠标）：真 QGIS 截屏 → 模型预测动作+坐标 → 判定预测点是否落在
     目标按钮的 UIA rect 内（目标取 UIA 可见的工具栏安全动作子集）。
  B. 安全闭环冒烟：仅当预测点落在 QGIS 主窗口内才执行点击（越界=不点，计 miss）；
     三个单步任务（Zoom Full / Zoom Last / Zoom to Native），
     真值一律由 qgis_bridge truth()（canvas extent / scale）判定，
     其中 Zoom to Native 的期望 scale 用桥的特权 act 校准一次。

坐标约定：模型输出 [0,1] 归一化（训练帧 1920x1080），按**当前屏幕尺寸**反投影。
用法: python experiments/computer_use/loop_w28.py [--trials 3] [--dry-only]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT / "mio"), str(ROOT),
                            str(ROOT / "experiments" / "qgis" / "qgis_bridge"))
                if p not in sys.path]

import numpy as np  # noqa: E402
import pyautogui  # noqa: E402
import torch  # noqa: E402
import win32gui  # noqa: E402
from pywinauto import Desktop  # noqa: E402

from mio_argus import (ACTION_MAP, ComputerUseModelV3,  # noqa: E402
                       get_image_processor, get_tokenizer)
from client import call  # noqa: E402

pyautogui.FAILSAFE = False

MODEL_PATH = ROOT / "artifacts" / "checkpoints" / "computer_use_model_v3_real.pth"
OUT = ROOT / "results" / "computer_use" / "w28_results.json"
MAX_LEN = 64
SCROLL_MAP = {0: "up", 1: "down", 2: "none"}

DRY_TARGETS = [
    ("click the Zoom Full button", "Zoom Full"),
    ("click the Zoom to Selection button", "Zoom to Selection"),
    ("click the Zoom Last button", "Zoom Last"),
    ("click the Zoom to Native Resolution button", "Zoom to Native Resolution (100%)"),
    ("click the Refresh button", "Refresh"),
    ("click the Undo button", "Undo"),
]
LIVE_TASKS = ["zoom_full", "zoom_last", "zoom_native"]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ QGIS/UIA ---
def qgis_window():
    """取在屏内的 QgisApp 窗口（可能同时存在 -32000 的最小化幽灵句柄）。"""
    best, best_area = None, -1
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") != "QgisApp":
                continue
            r = win32gui.GetWindowRect(w.handle)
            area = max(0, r[2] - r[0]) * max(0, r[3] - r[1])
            if r[0] > -1000 and area > best_area:   # 排除最小化（-32000）幽灵
                best, best_area = w, area
        except Exception:
            continue
    return best


def ensure_visible(qg):
    """Qt showMaximized 对最小化窗口静默无效（Win32 仍 -32000）→ 外部兜底。"""
    import win32con
    try:
        call({"op": "win", "maximize": True})
    except Exception:
        pass
    r = win32gui.GetWindowRect(qg.handle)
    if r[0] <= -1000:                            # 仍最小化 → 外部恢复
        win32gui.ShowWindow(qg.handle, win32con.SW_RESTORE)
        time.sleep(0.5)
        win32gui.ShowWindow(qg.handle, win32con.SW_MAXIMIZE)
        time.sleep(0.8)
    try:
        win32gui.SetForegroundWindow(qg.handle)
    except Exception:
        pass
    time.sleep(0.4)
    return win32gui.GetWindowRect(qg.handle)


def uia_buttons(qg):
    out = {}
    for b in qg.descendants(control_type="Button"):
        try:
            n = (b.window_text() or "").strip()
            r = b.rectangle()
        except Exception:
            continue
        if n and n not in out and r.width() >= 8 and r.height() >= 8:
            out[n] = (r.left, r.top, r.right, r.bottom)
    return out


def qgis_rect(qg):
    try:
        return win32gui.GetWindowRect(qg.handle)
    except Exception:
        return None


def focus_qgis(qg):
    try:
        win32gui.SetForegroundWindow(qg.handle)
    except Exception:
        pass
    time.sleep(0.4)


# --------------------------------------------------------------------- model ---
class Brain:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = get_tokenizer()
        self.imp = get_image_processor()
        self.model = ComputerUseModelV3(num_actions=6, freeze_vit=True,
                                        freeze_gpt2=True).to(self.device)
        state = torch.load(str(MODEL_PATH), map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()

    def predict(self, img, instruction):
        t0 = time.time()
        px = self.imp(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        enc = self.tok(instruction, max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        with torch.no_grad():
            o = self.model(px, enc["input_ids"].to(self.device),
                           enc["attention_mask"].to(self.device))
        aid = int(torch.argmax(o["action_logits"], dim=-1).item())
        coord = o["coord_pred"][0].cpu().numpy().astype(float).tolist()
        sid = int(torch.argmax(o["scroll_logits"], dim=-1).item())
        return {"action": ACTION_MAP[aid], "coord": coord,
                "scroll": SCROLL_MAP[sid], "ms": round((time.time() - t0) * 1000, 1)}


# ------------------------------------------------------------------ helpers ---
def dist_to_rect(x, y, r):
    dx = max(r[0] - x, 0.0, x - r[2])
    dy = max(r[1] - y, 0.0, y - r[3])
    return (dx * dx + dy * dy) ** 0.5


def ext_span(e):
    return max(e[2] - e[0], e[3] - e[1])


def ext_close(a, b, tol_frac=0.01):
    s = max(ext_span(b), 1e-9)
    return all(abs(a[i] - b[i]) <= tol_frac * s for i in range(4))


def sub_extent(e, frac):
    cx, cy = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
    hx, hy = (e[2] - e[0]) * frac / 2, (e[3] - e[1]) * frac / 2
    return [cx - hx, cy - hy, cx + hx, cy + hy]


def truth():
    return call({"op": "truth"})["truth"]


def layer_extent(tr, name):
    for l in tr.get("layers", []):
        if l["name"] == name:
            return l["extent"]
    return None


def do_step(task, instr, brain, screen, wrect, expect, trial):
    qg = qgis_window()
    if qg is None:
        return {"task": task, "trial": trial, "fail": "no_qgis_window"}
    wrect = ensure_visible(qg)
    img = pyautogui.screenshot()
    pred = brain.predict(img, instr)
    x = pred["coord"][0] * screen[0]
    y = pred["coord"][1] * screen[1]
    row = {"task": task, "trial": trial, "instruction": instr, "pred": pred,
           "px": [round(x, 1), round(y, 1)]}
    inside = (wrect[0] <= x <= wrect[2]) and (wrect[1] <= y <= wrect[3])
    row["in_window"] = inside
    if pred["action"] != "click":
        row.update({"clicked": False, "success": False, "fail": "action_wrong"})
    elif not inside:
        row.update({"clicked": False, "success": False, "fail": "out_of_window"})
    else:
        pyautogui.click(int(round(x)), int(round(y)))
        row["clicked"] = True
        time.sleep(1.0)
        tr = truth()
        ok = bool(expect(tr))
        row.update({"success": ok, "fail": None if ok else "truth_fail",
                    "scale": tr.get("scale"), "extent": tr.get("extent")})
    log(f"  [live] {task} t{trial}: {row.get('fail') or 'PASS'} "
        f"pred=({x:.0f},{y:.0f}) {pred['action']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--dry-only", action="store_true")
    args = ap.parse_args()

    call({"op": "reset", "scene": "real"})
    time.sleep(1.0)

    qg = qgis_window()
    if qg is None:
        log("[err] QGIS 窗口未找到")
        return
    wrect = ensure_visible(qg)
    screen = tuple(pyautogui.size())
    log(f"[env] screen={screen} qgis_rect={wrect}")

    brain = Brain()
    log(f"[model] v3 loaded on {brain.device}")

    rows_dry, rows_live = [], []

    # ---------- A. 干跑（不点鼠标） ----------
    log("== A. 干跑测距（UIA rect 为真值） ==")
    for trial in range(1, args.trials + 1):
        qg = qgis_window()
        wrect = ensure_visible(qg)
        buttons = uia_buttons(qg)
        img = pyautogui.screenshot()
        for instr, target in DRY_TARGETS:
            if target not in buttons:
                rows_dry.append({"trial": trial, "instruction": instr,
                                 "target": target, "hit": False,
                                 "fail": "target_not_in_uia"})
                continue
            pred = brain.predict(img, instr)
            x = pred["coord"][0] * screen[0]
            y = pred["coord"][1] * screen[1]
            r = buttons[target]
            hit = (r[0] <= x <= r[2]) and (r[1] <= y <= r[3])
            row = {"trial": trial, "instruction": instr, "target": target,
                   "rect": list(r), "pred": pred, "px": [round(x, 1), round(y, 1)],
                   "hit": bool(hit), "dist_px": round(dist_to_rect(x, y, r), 1),
                   "action_ok": pred["action"] == "click"}
            rows_dry.append(row)
            log(f"  [dry] t{trial} {target}: hit={hit} d={row['dist_px']}px "
                f"a={pred['action']}")
        time.sleep(0.2)

    # ---------- B. 安全闭环冒烟 ----------
    if not args.dry_only:
        log("== B. 安全闭环冒烟（越界不点；bridge truth 判定） ==")
        tr0 = truth()
        base = layer_extent(tr0, "provinces") or layer_extent(tr0, "roads")
        if base is None:
            log("[err] 找不到 provinces/roads 图层 extent")
            return
        # Zoom Full 的期望 extent 不能用单层 extent（canvas 全量=全图层并集，
        # W28 首测 oracle 因此误判）→ 特权 act 触发一次，读 extent 作期望。
        call({"op": "set_extent", "box": sub_extent(base, 0.4)})
        time.sleep(0.5)
        call({"op": "act", "kind": "action", "text": "Zoom Full"})
        time.sleep(1.2)
        full = truth().get("extent")
        if full is None:
            log("[err] Zoom Full 校准失败")
            return
        log(f"[calib] Zoom Full extent = {[round(v) for v in full]}")
        A = sub_extent(full, 0.4)
        B = sub_extent(full, 0.7)

        # Zoom to Native 的期望 scale：特权 act 校准一次
        call({"op": "set_extent", "box": B})
        time.sleep(0.5)
        call({"op": "act", "kind": "action", "text": "Zoom to Native Resolution (100%)"})
        time.sleep(1.2)
        S_native = truth().get("scale")
        log(f"[calib] Zoom to Native scale = {S_native}")

        # 0) Oracle 通路自检：用 UIA 真值坐标点一次 Zoom Full →
        #    验证"真点击 + bridge truth"机械本身可用（模型零样本从不输出 click，
        #    否则执行路径不会被走到）。
        qg = qgis_window()
        ensure_visible(qg)
        buttons = uia_buttons(qg)
        call({"op": "set_extent", "box": A})
        time.sleep(0.5)
        r = buttons.get("Zoom Full")
        oracle = {"task": "oracle_zoom_full", "trial": 0, "instruction": "click the Zoom Full button"}
        if r:
            ox, oy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
            pyautogui.click(ox, oy)
            time.sleep(1.0)
            ok = ext_close(truth()["extent"], full)
            oracle.update({"clicked": True, "px": [ox, oy], "success": bool(ok),
                           "fail": None if ok else "truth_fail"})
        else:
            oracle.update({"clicked": False, "success": False, "fail": "target_not_in_uia"})
        rows_live.append(oracle)
        log(f"  [oracle] zoom_full: {oracle.get('fail') or 'PASS'}")

        for trial in range(1, args.trials + 1):
            # 1) Zoom Full：缩到中心 40% → 模型点 → extent 应回到 full
            call({"op": "set_extent", "box": A})
            time.sleep(0.5)
            rows_live.append(do_step(
                "zoom_full", "click the Zoom Full button", brain, screen, wrect,
                lambda tr: ext_close(tr["extent"], full), trial))
            # 2) Zoom Last：A → B → 模型点 → extent 应回到 A
            call({"op": "set_extent", "box": A})
            time.sleep(0.4)
            call({"op": "set_extent", "box": B})
            time.sleep(0.4)
            rows_live.append(do_step(
                "zoom_last", "click the Zoom Last button", brain, screen, wrect,
                lambda tr: ext_close(tr["extent"], A), trial))
            # 3) Zoom to Native：先缩到 A → 模型点 → scale 应回到 S_native
            call({"op": "set_extent", "box": sub_extent(full, 0.5)})
            time.sleep(0.4)
            rows_live.append(do_step(
                "zoom_native", "click the Zoom to Native Resolution button",
                brain, screen, wrect,
                lambda tr: (S_native is not None
                            and abs(tr.get("scale", 0) - S_native) <= 0.01 * S_native),
                trial))

    # ---------- 汇总 ----------
    n_dry = len(rows_dry)
    dry_hits = sum(1 for r in rows_dry if r.get("hit"))
    dry_act = sum(1 for r in rows_dry if r.get("action_ok"))
    model_live = [r for r in rows_live if r.get("trial") != 0]
    n_live = len(model_live)
    live_ok = sum(1 for r in model_live if r.get("success"))
    fails = Counter(r.get("fail") for r in model_live if not r.get("success"))
    oracle_row = next((r for r in rows_live if r.get("trial") == 0), None)
    summary = {
        "dry_n": n_dry, "dry_action_acc": round(dry_act / n_dry, 3) if n_dry else None,
        "dry_hit_rate": round(dry_hits / n_dry, 3) if n_dry else None,
        "dry_dist_px_median": round(float(np.median(
            [r["dist_px"] for r in rows_dry if "dist_px" in r])), 1)
        if any("dist_px" in r for r in rows_dry) else None,
        "live_n": n_live,
        "live_success_rate": round(live_ok / n_live, 3) if n_live else None,
        "live_fails": dict(fails),
        "oracle_ok": bool(oracle_row and oracle_row.get("success")),
    }
    out = {
        "meta": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "ckpt_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else None,
            "device": str(brain.device), "screen": list(screen),
            "qgis_rect": list(wrect) if wrect else None,
            "trials": args.trials, "ts": datetime.now().isoformat(timespec="seconds"),
        },
        "dry": rows_dry, "live": rows_live, "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("== 汇总 ==")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"已写出 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
