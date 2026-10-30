"""
W28b 迁移评测（离线）：同一套指标量 E1/E2/E3，结论写 results/computer_use/transfer.json

设计：
- 评测完全离线：只读 demo.json / frames.json 里已落盘的截图与标签，不需要软件在跑。
- 指标口径与 W28 干跑一致：动作准确率 / 坐标命中（落在 target_rect 内）/ 距离中位 / scroll 准确率；
  另加 majority-class 基线，避免"全押 press"这类退化预测把准确率抬高。
- 按标签（tag）合并写入 transfer.json，一次实验三臂各自追加，最后统一出结论。
- `--capture-qgis` 是一次性评测集准备（唯一需要 QGIS 在跑的路径）：按 W28 的 6 条工具栏指令
  抓一帧真界面 + UIA 实测控件 rect，落盘后所有评测都可离线复现。

用法:
    # 0) 一次性：抓 QGIS 评测集（需要 QGIS，用 qgis-ltr.bat + bridge startup）
    python experiments/computer_use/eval_transfer.py --capture-qgis

    # 1) 评测某一臂
    python experiments/computer_use/eval_transfer.py --ckpt artifacts/checkpoints/inloop_notepad.pth \
        --app qgis --tag E3 --note "notepad -> qgis zero-shot"

    # 2) 只看某个 split（val_tasks 从训练侧 meta 读，避免在训练见过的任务上报数）
    python experiments/computer_use/eval_transfer.py --ckpt ... --app notepad --tag E1 \
        --train-meta artifacts/checkpoints/inloop_notepad.json --split val
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from train_inloop import DATA_ROOT, evaluate, load_demos  # noqa: E402
from mio_argus import ACTION_MAP, ComputerUseModelV3, get_image_processor, get_tokenizer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "transfer.json"
EVAL_QGIS = DATA_ROOT / "eval_qgis"
QGIS_BAT = r"D:\QGIS\bin\qgis-ltr.bat"
BRIDGE_DIR = ROOT / "experiments" / "qgis" / "qgis_bridge"

# W28 干跑同题（指令 / UIA 按钮名候选 / 期望动作）
DRY_TARGETS = [
    ("click the Zoom Full button", ["Zoom Full"], "click"),
    ("click the Zoom to Selection button", ["Zoom to Selection"], "click"),
    ("click the Zoom Last button", ["Zoom Last"], "click"),
    ("click the Zoom to Native Resolution button", ["Zoom to Native Resolution (100%)", "Zoom to Native"], "click"),
    ("click the Refresh button", ["Refresh"], "click"),
    ("click the Undo button", ["Undo"], "click"),
]


# ------------------------------------------------------------------ capture ---
def ensure_qgis(timeout=90.0):
    sys.path.insert(0, str(BRIDGE_DIR))
    from client import call
    try:
        call({"op": "ping"}, timeout=3)
        return call, False
    except Exception:
        pass
    env = dict(os.environ, PYQGIS_STARTUP=str(BRIDGE_DIR / "_startup.py"))
    subprocess.Popen(["cmd", "/c", QGIS_BAT], env=env, creationflags=0x00000008)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2.5)
        try:
            call({"op": "ping"}, timeout=3)
            time.sleep(2.5)
            call({"op": "win", "maximize": True})
            return call, True
        except Exception:
            continue
    raise RuntimeError("QGIS bridge 未就绪（_startup.log 看桥加载）")


def find_qgis_window():
    from pywinauto import Desktop
    import win32gui
    best, best_area = None, -1
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if (w.element_info.class_name or "") != "QgisApp":
                continue
            r = win32gui.GetWindowRect(w.handle)
            area = max(0, r[2] - r[0]) * max(0, r[3] - r[1])
            if r[0] > -1000 and area > best_area:
                best, best_area = w, area
        except Exception:
            continue
    return best


def uia_buttons(qg):
    out = {}
    for b in qg.descendants(control_type="Button"):
        try:
            name = (b.window_text() or "").strip()
            r = b.rectangle()
        except Exception:
            continue
        if name and name not in out and r.width() >= 8 and r.height() >= 8:
            out[name] = [int(r.left), int(r.top), int(r.right), int(r.bottom)]
    return out


def capture_qgis():
    import pyautogui
    import win32gui

    call, launched = ensure_qgis()
    print(f"[capture] qgis launched={launched}")
    qg = None
    buttons = {}
    deadline = time.time() + 40
    while time.time() < deadline:
        qg = qg or find_qgis_window()
        if qg is None:
            time.sleep(2)
            continue
        try:
            call({"op": "win", "maximize": True})
        except Exception:
            pass
        try:
            win32gui.SetForegroundWindow(qg.handle)
        except Exception:
            pass
        time.sleep(1.0)
        buttons = uia_buttons(qg)
        if any(any(alias in buttons for alias in aliases) for _, aliases, _ in DRY_TARGETS):
            break
        time.sleep(2)

    if not buttons:
        raise RuntimeError("UIA 没读到任何按钮（QGIS 主窗口未就绪？）")

    rect = win32gui.GetWindowRect(qg.handle)
    shot = pyautogui.screenshot()
    EVAL_QGIS.mkdir(parents=True, exist_ok=True)
    frame = EVAL_QGIS / "frame_000.png"
    shot.save(frame)

    w, h = shot.size
    rows, missing = [], []
    for i, (instruction, aliases, action) in enumerate(DRY_TARGETS):
        name = next((a for a in aliases if a in buttons), None)
        if name is None:
            name = next((b for b in buttons if any(a.lower() in b.lower() for a in aliases)), None)
        if name is None:
            missing.append({"instruction": instruction, "aliases": aliases})
            continue
        rows.append({
            "id": f"qgis_eval:{i:02d}",
            "screenshot": str(frame.relative_to(ROOT)),
            "instruction": instruction,
            "action": action,
            "params": {},
            "target_rect": buttons[name],
            "screen": [w, h],
            "meta": {"app": "qgis", "phase": "toolbar_dry", "chain": "eval_transfer/capture",
                     "uia_name": name, "qgis_rect": list(rect)},
        })

    payload = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qgis_rect": list(rect),
        "screen": [w, h],
        "n_buttons": len(buttons),
        "rows": rows,
        "missing": missing,
        "buttons_sample": dict(list(sorted(buttons.items()))[:80]),
    }
    (EVAL_QGIS / "frames.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"[capture] rows={len(rows)} missing={len(missing)} -> {EVAL_QGIS / 'frames.json'}")
    for r in rows:
        print(f"  {r['instruction']}: uia={r['meta']['uia_name']} rect={r['target_rect']}")
    return 0 if rows else 1


# --------------------------------------------------------------------- eval ---
def load_eval_rows(app, data_root, train_meta=None, split="all", mix=0, use_frames=True):
    """app=qgis 时优先用 eval_qgis/frames.json（W28 同题），否则走 demo.json。

    mix>0 时再混入 N 条真实采集的 QGIS 样本（按指令去重），让动作准确率不再被
    "全是 click" 的单一标签抬成 100%——否则预测 click 是免费的。
    """
    rows, skipped = load_demos(data_root, [app])
    frames = data_root / "eval_qgis" / "frames.json"
    if app == "qgis" and frames.exists() and use_frames:
        payload = json.loads(frames.read_text(encoding="utf-8"))
        rows = [dict(r, _app="qgis", _task="qgis_eval", _screenshot=str(ROOT / r["screenshot"]))
                for r in payload["rows"]]
        skipped = []

    if mix > 0:
        pool, skipped_mix = load_demos(data_root, [app])
        seen = {r["instruction"] for r in rows}
        added = 0
        for row in pool:
            if added >= mix:
                break
            if row["instruction"] in seen:
                continue
            seen.add(row["instruction"])
            row = dict(row, _task=row["_task"] + "_mix")
            rows.append(row)
            added += 1
        skipped = skipped + skipped_mix

    if train_meta:
        meta = json.loads(Path(train_meta).read_text(encoding="utf-8"))
        val_tasks = set(meta.get("val_tasks") or [])
        if split == "val":
            rows = [r for r in rows if r["_task"] in val_tasks]
        elif split == "train":
            rows = [r for r in rows if r["_task"] not in val_tasks]
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-qgis", action="store_true")
    ap.add_argument("--ckpt", default=str(ROOT / "artifacts" / "checkpoints" / "inloop_notepad.pth"))
    ap.add_argument("--app", default="qgis")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--train-meta", default="")
    ap.add_argument("--split", default="all", choices=["all", "val", "train"])
    ap.add_argument("--mix-demos", type=int, default=0,
                    help="额外混入 N 条真实采集样本（按指令去重），避免 action_acc 被单一标签抬成 100%%")
    ap.add_argument("--ignore-eval-frames", action="store_true",
                    help="qgis 时不要用 eval_qgis/frames.json 覆盖，直接用 demo.json（W30 留出链评测）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shuffle-instructions", action="store_true",
                    help="对照：把指令在样本间轮换，预测不变即说明模型没在用指令定位")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="")
    ap.add_argument("--head", default="v3", choices=["v3", "grid", "spatial"])
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    if args.capture_qgis:
        return capture_qgis()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, skipped = load_eval_rows(args.app, Path(args.data_root), args.train_meta or None,
                                   args.split, args.mix_demos,
                                   use_frames=not args.ignore_eval_frames)
    print(f"[data] app={args.app} split={args.split} n={len(rows)} skipped={len(skipped)}")
    if not rows:
        print("[data] 空评测集，终止")
        return 1

    if args.shuffle_instructions:
        instr = [r["instruction"] for r in rows]
        instr = instr[1:] + instr[:1]
        for r, text in zip(rows, instr):
            r["instruction"] = text
        print("[data] 指令已轮换（对照臂）")

    if args.head == "grid":
        from grid_model import GridModel
        model = GridModel(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True,
                          grid=args.grid)
    elif args.head == "spatial":
        from grid_model import SpatialModel
        model = SpatialModel(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True)
    else:
        model = ComputerUseModelV3(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True)
    ckpt = Path(args.ckpt)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    print(f"[ckpt] {ckpt.name} missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"[ckpt] missing={missing[:5]} unexpected={unexpected[:5]}")

    metrics = evaluate(model, rows, device, get_tokenizer(), get_image_processor(),
                       img_size=args.img_size)
    metrics["app"] = args.app
    metrics["split"] = args.split
    metrics["n_rows"] = len(rows)
    metrics["ckpt"] = str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt)
    metrics["note"] = args.note
    metrics["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"[eval] tag={args.tag or 'unnamed'} n={metrics['n']} "
          f"action_acc={metrics['action_acc']:.3f} "
          f"majority={metrics['majority_baseline']} "
          f"hit={metrics['hit_rate']} coord_med_px={metrics['coord_med_px']}")

    out = Path(args.out)
    payload = {"meta": {}, "runs": {}}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
    payload.setdefault("meta", {})
    payload["meta"].update({"updated_at": metrics["ts"],
                            "qgis_eval": str((Path(args.data_root) / "eval_qgis" / "frames.json").relative_to(ROOT))})
    payload.setdefault("runs", {})[args.tag or "unnamed"] = metrics
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
