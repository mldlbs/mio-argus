"""
W31：真任务闭环（验证器在环，FR-6）

问的问题不是"模型点得准不准"，而是**验证器能不能把不准救回来**：
每次尝试都用 bridge 真值判定成败，失败就重截屏再试，最多 K 次。

三个任务（都能被 bridge 机械判定）：
  zoom_full   当前 extent → 全图 extent
  zoom_last   B → A（回到上一个视图）
  zoom_native scale → 标定的 native scale

两个臂：
  model  模型驱动（截图→动作→点击→真值判定→重试）
  oracle UIA 真值坐标点一次（脚本基线，验证机械本身可用）

指标：success@1..K、总成功率、正确任务/分钟、失败归因（action_wrong / out_of_window / truth_fail）。

用法:
    python experiments/computer_use/w31_loop.py --trials 3 --attempts 5
    python experiments/computer_use/w31_loop.py --dry   # 只校准 + oracle，不跑模型
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

from loop_w28 import (ensure_visible, ext_close, layer_extent,  # noqa: E402
                      qgis_window, sub_extent, truth, uia_buttons)
from client import call  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "w31_results.json"


def make_brain(ckpt):
    """W31 用微调后的 checkpoint（W30），不是 W28 的基线模型。"""
    import torch
    from mio_argus import (ACTION_MAP, ComputerUseModelV3, get_image_processor,
                           get_tokenizer)
    from loop_w28 import MAX_LEN, SCROLL_MAP

    class Brain:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = get_tokenizer()
            self.imp = get_image_processor()
            self.model = ComputerUseModelV3(num_actions=6, freeze_vit=True,
                                            freeze_gpt2=True).to(self.device)
            state = torch.load(str(ckpt), map_location=self.device)
            self.model.load_state_dict(state["model"] if "model" in state else state,
                                       strict=False)
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

    return Brain()


def log(m):
    print(m, flush=True)


def calibrate():
    """用特权 act 标定三个任务的期望终态（W28 的教训：不能拿单层 extent 当全图）。"""
    tr0 = truth()
    base = layer_extent(tr0, "provinces") or layer_extent(tr0, "roads")
    if base is None:
        raise RuntimeError("找不到 provinces/roads 图层 extent")
    call({"op": "set_extent", "box": sub_extent(base, 0.4)})
    time.sleep(0.5)
    call({"op": "act", "kind": "action", "text": "Zoom Full"})
    time.sleep(1.2)
    full = truth().get("extent")
    if full is None:
        raise RuntimeError("Zoom Full 校准失败")
    A = sub_extent(full, 0.4)
    B = sub_extent(full, 0.7)
    call({"op": "set_extent", "box": B})
    time.sleep(0.5)
    call({"op": "act", "kind": "action", "text": "Zoom to Native Resolution (100%)"})
    time.sleep(1.2)
    s_native = truth().get("scale")
    return {"full": full, "A": A, "B": B, "s_native": s_native}


def prepare(task, cal):
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
    # 反空过：准备状态若已满足目标，则要求状态相对准备时确实发生了变化
    if ok and pre is not None and pre.get("scale"):
        ok = abs(s - pre["scale"]) / pre["scale"] > 0.01
    return ok, tr


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


def run_model_task(task, instr, cal, brain, screen, attempt_cap):
    prepare(task, cal)
    qg = qgis_window()
    wrect = ensure_visible(qg)
    pre = truth()
    rows = []
    for attempt in range(1, attempt_cap + 1):
        img = pyautogui.screenshot()
        pred = brain.predict(img, instr)
        x, y = pred["coord"][0] * screen[0], pred["coord"][1] * screen[1]
        inside = wrect[0] <= x <= wrect[2] and wrect[1] <= y <= wrect[3]
        row = {"attempt": attempt, "pred": pred, "px": [round(x, 1), round(y, 1)],
               "in_window": inside}
        if pred["action"] != "click":
            row.update({"clicked": False, "success": False, "fail": "action_wrong"})
        elif not inside:
            row.update({"clicked": False, "success": False, "fail": "out_of_window"})
        else:
            pyautogui.click(int(round(x)), int(round(y)))
            time.sleep(1.0)
            ok, tr = verify(task, cal, pre)
            row.update({"clicked": True, "success": ok, "fail": None if ok else "truth_fail",
                        "scale": tr.get("scale")})
        rows.append(row)
        log(f"    a{attempt}: {row.get('fail') or 'PASS'} pred=({x:.0f},{y:.0f}) "
            f"[{pred['action']}]")
        if row["success"]:
            break
    return rows


def run_oracle(task, cal, screen):
    prepare(task, cal)
    qg = qgis_window()
    ensure_visible(qg)
    buttons = uia_buttons(qg)
    name = UIA_NAME[task]
    if name not in buttons:
        return {"ok": False, "fail": "uia_missing"}
    r = buttons[name]
    cx, cy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
    pyautogui.click(cx, cy)
    time.sleep(1.2)
    ok, tr = verify(task, cal)
    return {"ok": bool(ok), "px": [cx, cy], "fail": None if ok else "truth_fail"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--dry", action="store_true", help="只做校准 + oracle 基线")
    ap.add_argument("--ckpt", default=str(ROOT / "artifacts" / "checkpoints" / "w30_full.pth"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    call({"op": "win", "maximize": True})
    qg = qgis_window()
    if qg is None:
        import collect_inloop as CI
        log("[env] QGIS 未运行，正在启动（bridge 就绪约 20-40s）")
        CI.ensure_qgis()
        time.sleep(2)
        qg = qgis_window()
    if qg is None:
        # 最小化时窗口 rect 是 (-32000,...)，qgis_window() 会当幽灵句柄过滤掉；
        # 这里兜底取任意 QgisApp，再交给 ensure_visible 恢复。
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

    oracle = {}
    for task in TASKS:
        oracle[task] = run_oracle(task, cal, screen)
        log(f"[oracle] {task}: {'PASS' if oracle[task]['ok'] else oracle[task]['fail']}")

    payload = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": args.trials,
                        "attempts": args.attempts, "screen": list(screen)},
               "calibration": {"s_native": cal["s_native"], "full": cal["full"],
                               "A": cal["A"], "B": cal["B"]},
               "oracle": oracle, "trials": []}

    if not args.dry:
        brain = make_brain(args.ckpt)
        log(f"[model] v3 on {brain.device}")
        for task, instr in TASKS.items():
            for t in range(1, args.trials + 1):
                t0 = time.time()
                rows = run_model_task(task, instr, cal, brain, screen, args.attempts)
                hit = any(r["success"] for r in rows)
                payload["trials"].append({
                    "task": task, "trial": t, "success": hit,
                    "success_at": next((r["attempt"] for r in rows if r["success"]), None),
                    "attempts": len(rows), "elapsed_s": round(time.time() - t0, 1),
                    "rows": rows})
                log(f"  [model] {task} t{t}: {'PASS' if hit else 'FAIL'} "
                    f"attempts={len(rows)} {time.time() - t0:.1f}s")

    tr = payload["trials"]
    if tr:
        total = len(tr)
        payload["summary"] = {
            "n_tasks": total,
            "success": sum(1 for r in tr if r["success"]),
            "rate": sum(1 for r in tr if r["success"]) / total,
            "success_at_1": sum(1 for r in tr if r["success_at"] == 1) / total,
            "success_at_k": sum(1 for r in tr if r["success"]) / total,
            "mean_attempts": sum(r["attempts"] for r in tr) / total,
            "mean_elapsed_s": sum(r["elapsed_s"] for r in tr) / total,
            "fails": {
                "action_wrong": sum(1 for r in tr for x in r["rows"] if x.get("fail") == "action_wrong"),
                "out_of_window": sum(1 for r in tr for x in r["rows"] if x.get("fail") == "out_of_window"),
                "truth_fail": sum(1 for r in tr for x in r["rows"] if x.get("fail") == "truth_fail"),
            },
        }
        log(f"[summary] success {payload['summary']['success']}/{total} "
            f"= {payload['summary']['rate']:.2f}, "
            f"fails={payload['summary']['fails']}")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
