"""
W31 + 复用：用 GUI-Actor-2B 当定位器，跑真任务闭环（验证器在环）

与 w31_loop.py 的唯一差别是定位来源：那边是自研 v3，这边是现成的 GUI-Actor-2B。
任务、校准、判定、重试协议完全相同，因此两次的成功率可直接对比。

必须在隔离 venv 里跑（transformers 4.51.3 + gui_actor 代码包）：
  set HF_ENDPOINT=https://hf-mirror.com
  set PYTHONPATH=<GUI-Actor-main>/src
  artifacts/audit_env/Scripts/python.exe experiments/computer_use/w31_loop_ga.py --trials 3 --attempts 5

输出：results/computer_use/w31_ga_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "experiments" / "qgis" / "qgis_bridge"),
                str(ROOT / "mio"), str(ROOT)]

import pyautogui  # noqa: E402

from client import call  # noqa: E402
from loop_w28 import (ensure_visible, ext_close, layer_extent,  # noqa: E402
                      qgis_window, sub_extent, truth, uia_buttons)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "w31_ga_results.json"
SYS_MSG = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
           "You need to perform a series of pyautogui actions to complete the task.")
TASKS = {
    "zoom_full": "click the Zoom Full button",
    "zoom_last": "click the Zoom Last button",
    "zoom_native": "click the Zoom to Native Resolution button",
}
UIA_NAME = {
    "zoom_full": "Zoom Full",
    "zoom_last": "Zoom Last",
    "zoom_native": "Zoom to Native Resolution (100%)",
}


def log(m):
    print(m, flush=True)


class GuiActor:
    def __init__(self, name="microsoft/GUI-Actor-2B-Qwen2-VL", attn="sdpa"):
        import torch
        from transformers import AutoProcessor
        from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(name)
        self.tok = self.proc.tokenizer
        self.model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map=self.device,
            attn_implementation=attn).eval()
        self.vram = int(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
        log(f"[model] GUI-Actor on {self.device} vram≈{self.vram}MB")

    def predict(self, img, instruction, topk=1):
        from gui_actor.inference import inference
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYS_MSG}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": instruction}]},
        ]
        t0 = time.time()
        pred = inference(conv, self.model, self.tok, self.proc,
                         use_placeholder=True, topk=topk)
        pts = pred.get("topk_points") or []
        return {"coord": [float(pts[0][0]), float(pts[0][1])] if pts else None,
                "action": "click", "ms": round((time.time() - t0) * 1000, 1)}


def calibrate():
    tr0 = truth()
    base = layer_extent(tr0, "provinces") or layer_extent(tr0, "roads")
    if base is None:
        raise RuntimeError("找不到 provinces/roads 图层 extent")
    call({"op": "set_extent", "box": sub_extent(base, 0.4)})
    time.sleep(0.5)
    call({"op": "act", "kind": "action", "text": "Zoom Full"})
    time.sleep(1.2)
    full = truth().get("extent")
    A, B = sub_extent(full, 0.4), sub_extent(full, 0.7)
    call({"op": "set_extent", "box": B})
    time.sleep(0.5)
    call({"op": "act", "kind": "action", "text": "Zoom to Native Resolution (100%)"})
    time.sleep(1.2)
    return {"full": full, "A": A, "B": B, "s_native": truth().get("scale")}


def prepare(task, cal):
    try:
        call({"op": "close_dialogs"})
    except Exception:
        pass
    if task == "zoom_full":
        call({"op": "set_extent", "box": cal["A"]})
    elif task == "zoom_last":
        call({"op": "set_extent", "box": cal["A"]})
        time.sleep(0.4)
        call({"op": "set_extent", "box": cal["B"]})
    else:
        call({"op": "set_extent", "box": cal["B"]})
    time.sleep(0.8)


def verify(task, cal, pre=None):
    tr = truth()
    if task == "zoom_full":
        e = tr.get("extent")
        return bool(e) and ext_close(e, cal["full"], 0.01), tr
    if task == "zoom_last":
        e = tr.get("extent")
        return bool(e) and ext_close(e, cal["A"], 0.01), tr
    s = tr.get("scale")
    ok = bool(s) and bool(cal["s_native"]) and abs(s - cal["s_native"]) / cal["s_native"] < 0.01
    if ok and pre is not None and pre.get("scale"):
        ok = abs(s - pre["scale"]) / pre["scale"] > 0.01
    return ok, tr


def run_task(task, instr, cal, brain, screen, attempts):
    prepare(task, cal)
    qg = qgis_window()
    wrect = ensure_visible(qg)
    pre = truth()
    rows = []
    for a in range(1, attempts + 1):
        img = pyautogui.screenshot()
        pred = brain.predict(img, instr)
        if pred["coord"] is None:
            rows.append({"attempt": a, "fail": "no_output", "success": False})
            continue
        x, y = pred["coord"][0] * screen[0], pred["coord"][1] * screen[1]
        inside = wrect[0] <= x <= wrect[2] and wrect[1] <= y <= wrect[3]
        row = {"attempt": a, "px": [round(x, 1), round(y, 1)], "in_window": inside,
               "ms": pred["ms"]}
        if not inside:
            row.update({"clicked": False, "success": False, "fail": "out_of_window"})
        else:
            pyautogui.click(int(round(x)), int(round(y)))
            time.sleep(1.0)
            ok, tr = verify(task, cal, pre)
            row.update({"clicked": True, "success": ok,
                        "fail": None if ok else "truth_fail", "scale": tr.get("scale")})
        rows.append(row)
        log(f"    a{a}: {row.get('fail') or 'PASS'} ({x:.0f},{y:.0f}) {pred['ms']:.0f}ms")
        if row["success"]:
            break
    return rows


def run_oracle(task, cal):
    prepare(task, cal)
    qg = qgis_window()
    ensure_visible(qg)
    b = uia_buttons(qg).get(UIA_NAME[task])
    if not b:
        return {"ok": False, "fail": "uia_missing"}
    pyautogui.click((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
    time.sleep(1.2)
    ok, _ = verify(task, cal)
    return {"ok": bool(ok), "fail": None if ok else "truth_fail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    call({"op": "win", "maximize": True})
    qg = qgis_window()
    if qg is None:
        import collect_inloop as CI
        log("[env] 启动 QGIS")
        CI.ensure_qgis()
        time.sleep(2)
        qg = qgis_window()
    if qg is None:
        from pywinauto import Desktop
        for w in Desktop(backend="uia").windows(visible_only=True):
            try:
                if (w.element_info.class_name or "") == "QgisApp":
                    qg = w
                    break
            except Exception:
                continue
    if qg is None:
        log("[err] 找不到 QGIS 窗口")
        return 1
    screen = tuple(pyautogui.size())
    ensure_visible(qg)

    cal = calibrate()
    log(f"[calib] native scale={cal['s_native']}")
    oracle = {t: run_oracle(t, cal) for t in TASKS}
    for t, r in oracle.items():
        log(f"[oracle] {t}: {'PASS' if r['ok'] else r['fail']}")

    brain = GuiActor()
    trials = []
    for task, instr in TASKS.items():
        for k in range(1, args.trials + 1):
            t0 = time.time()
            rows = run_task(task, instr, cal, brain, screen, args.attempts)
            ok = any(r["success"] for r in rows)
            trials.append({"task": task, "trial": k, "success": ok,
                           "success_at": next((r["attempt"] for r in rows if r["success"]), None),
                           "attempts": len(rows), "elapsed_s": round(time.time() - t0, 1),
                           "rows": rows})
            log(f"  [ga] {task} t{k}: {'PASS' if ok else 'FAIL'} {time.time() - t0:.1f}s")

    n = len(trials)
    summary = {"n": n, "success": sum(1 for r in trials if r["success"]),
               "rate": sum(1 for r in trials if r["success"]) / n,
               "success_at_1": sum(1 for r in trials if r["success_at"] == 1) / n,
               "mean_attempts": sum(r["attempts"] for r in trials) / n,
               "fails": {
                   "no_output": sum(1 for r in trials for x in r["rows"] if x.get("fail") == "no_output"),
                   "out_of_window": sum(1 for r in trials for x in r["rows"] if x.get("fail") == "out_of_window"),
                   "truth_fail": sum(1 for r in trials for x in r["rows"] if x.get("fail") == "truth_fail")}}
    log(f"[summary] {summary['success']}/{n} = {summary['rate']:.2f} fails={summary['fails']}")

    payload = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": args.trials,
                        "attempts": args.attempts, "locator": "GUI-Actor-2B",
                        "vram_mb": brain.vram, "screen": list(screen)},
               "calibration": {"s_native": cal["s_native"]},
               "oracle": oracle, "trials": trials, "summary": summary}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
