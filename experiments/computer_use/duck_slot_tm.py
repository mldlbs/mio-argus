"""DUCK 槽位四分类 · 方案 A（模板匹配）。

本体（按裁定）：
    duck     鸭子          —— 参与鸭子召回率
    padlock  挂锁          —— 不参与
    empty    空槽          —— 不参与
    other    王冠/绿点/分隔符等装饰 —— 确认不属于目标类别
    unknown  证据不足（与 other 不同：other 是"确认不是"，unknown 是"判断不了"）

复用 ui_icons 的截图与多尺度匹配思路；模板库按类别分目录存放：
    templates/duck/*.png  templates/padlock/*.png  templates/other/*.png

门禁口径（docs/DUCK_EXEC_PROTOCOL.md）：
    扰动稳定性 ≥95%、鸭子召回 ≥95%、挂锁/空槽/other 误识为鸭子 ≤1%，
    并且必须同时报告**有效分类覆盖率**（防止用 unknown 刷稳定性）。

用法：
    python experiments/computer_use/duck_slot_tm.py capture --frames 3 --out results/computer_use/slot_raw
    python experiments/computer_use/duck_slot_tm.py calibrate --raw ... --labels ... --templates ...
    python experiments/computer_use/duck_slot_tm.py bench --raw ... --templates ... --labels ...
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import duck_slots as ds  # noqa: E402
import ui_icons  # noqa: E402

OUT_DIR = ROOT / "results" / "computer_use"
CLASSES = ("duck", "padlock", "empty", "other")
SCALES = tuple(round(v, 2) for v in np.arange(0.80, 1.26, 0.10))
MIN_SCORE = 0.62          # 低于此分一律 unknown
MIN_MARGIN = 0.05         # 第一名与第二名差距过小 → unknown


# ---------------------------------------------------------------- 页面断言

def page_guard(hwnd: int) -> tuple[bool, str]:
    """采集前置：必须是鸭子页且没有弹窗，否则拒绝采集（防止采到弹窗内容）。"""
    texts = ui_icons.grab(hwnd)
    if texts is None:
        return False, "no-window"
    import bigemap_runner as bench
    import win32gui
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    lines, _ = bench.ocr_lines((l, t, r - l, b - t))
    txt = " ".join(x["text"] for x in lines if x["text"].strip())
    for bad in ("物品已绑定", "TOMATO FIESTA", "ABANDONED CART", "合并进行中",
                "鸭子繁殖完成", "MAINTENANCE", "每日奖励", "SWEET HALLOWEEN"):
        if bad in txt:
            return False, f"blocked:{bad}"
    if not any(k in txt for k in ("点击喂食", "已准备好", "繁殖进行中", "这是一只新鸭子")):
        return False, "not-duck-page"
    return True, "ok"


# ---------------------------------------------------------------- 模板匹配

def _prep(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.equalizeHist(g)          # 亮度归一化
    return cv2.GaussianBlur(g, (3, 3), 0)


class Templates:
    def __init__(self, root: Path):
        self.lib: dict[str, list[tuple[str, np.ndarray]]] = {}
        for cls in CLASSES:
            d = root / cls
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.png")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                self.lib.setdefault(cls, []).append((p.name, _prep(img)))
        self.sizes = sorted({t.shape[0] for v in self.lib.values() for _, t in v})

    def ready(self) -> bool:
        return any(len(v) >= 1 for k, v in self.lib.items() if k in ("duck", "padlock"))

    def match(self, crop: np.ndarray) -> dict:
        g = _prep(crop)
        best = {"cls": "unknown", "score": 0.0, "template": "", "reason": ""}
        per_class: dict[str, float] = {}
        for cls, items in self.lib.items():
            score, name = 0.0, ""
            for tname, tpl in items:
                for sc in SCALES:
                    tt = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                    if tt.shape[0] > g.shape[0] or tt.shape[1] > g.shape[1]:
                        continue
                    res = cv2.matchTemplate(g, tt, cv2.TM_CCOEFF_NORMED)
                    s = float(res.max())
                    if s > score:
                        score, name = s, tname
            per_class[cls] = round(score, 3)
            if score > best["score"]:
                best = {"cls": cls, "score": round(score, 3), "template": name, "reason": ""}

        ranked = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
        top_cls, top = ranked[0] if ranked else ("unknown", 0.0)
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        if top < MIN_SCORE:
            return {"cls": "unknown", "score": round(top, 3), "template": best["template"],
                    "reason": f"score_below_{MIN_SCORE}", "per_class": per_class}
        if top - second < MIN_MARGIN:
            return {"cls": "unknown", "score": round(top, 3), "template": best["template"],
                    "reason": "margin_too_small", "per_class": per_class}
        return {"cls": top_cls, "score": round(top, 3), "template": best["template"],
                "reason": "ok", "per_class": per_class}


# ---------------------------------------------------------------- 基准集

def capture(frames: int, out: Path) -> int:
    hwnd = ui_icons.find_window()
    if not hwnd:
        print("找不到窗口")
        return 2
    ok, why = page_guard(hwnd)
    if not ok:
        print(f"拒绝采集：{why}（必须先回到干净的鸭子页）")
        return 3
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("f*_s*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    man = {"ts": datetime.now().isoformat(timespec="seconds"), "frames": []}
    for k in range(frames):
        bgr = ui_icons.grab(hwnd)
        if bgr is None:
            continue
        slots = ds.segment_slots(bgr)
        rec = {"frame": k, "slots": []}
        for i, s in enumerate(slots):
            x, y, w, h = s["box"]
            name = f"f{k:03d}_s{i:02d}.png"
            cv2.imwrite(str(out / name), bgr[y:y + h, x:x + w])
            rec["slots"].append({"i": i, "file": name, "box": s["box"]})
        man["frames"].append(rec)
        print(f"frame {k}: {len(slots)} slots")
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    return 0


def _perturb(img: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "orig": img,
        "bright+20": np.clip(img.astype(np.int16) + 20, 0, 255).astype(np.uint8),
        "bright-20": np.clip(img.astype(np.int16) - 20, 0, 255).astype(np.uint8),
        "blur": cv2.GaussianBlur(img, (3, 3), 0),
        "noise": np.clip(img + np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8),
        "shift": np.roll(img, 2, axis=1),
        "combo": cv2.GaussianBlur(np.clip(img.astype(np.int16) + 12, 0, 255).astype(np.uint8), (3, 3), 0),
    }


def bench(raw: Path, templates_dir: Path, labels_path: Path) -> int:
    """同一原图的全部扰动属于同一组；报告稳定性/召回/误识/覆盖率。"""
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    man = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    lib = Templates(templates_dir)
    if not lib.ready():
        print("模板库为空：先准备 templates/{duck,padlock,other}/*.png")
        return 3

    rows = []
    for fr in man["frames"]:
        for s in fr["slots"]:
            key = f"{fr['frame']}.{s['i']}"
            if key not in labels:
                continue
            img = cv2.imread(str(raw / s["file"]))
            if img is None:
                continue
            truth = labels[key]
            for kind, pert in _perturb(img).items():
                pred = lib.match(pert)
                rows.append({"group": key, "kind": kind, "truth": truth,
                             "pred": pred["cls"], "score": pred["score"],
                             "reason": pred["reason"]})

    n = max(1, len(rows))
    # 稳定性：同一组内各扰动预测与 orig 一致的比例
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    stable = tot = 0
    for g in by_group.values():
        base = next((x["pred"] for x in g if x["kind"] == "orig"), None)
        for x in g:
            if x["kind"] == "orig":
                continue
            tot += 1
            stable += (x["pred"] == base)
    # 分类指标（只统计 orig）
    orig = [r for r in rows if r["kind"] == "orig"]
    confusion = {t: {p: 0 for p in (*CLASSES, "unknown")} for t in CLASSES}
    for r in orig:
        if r["truth"] in confusion:
            confusion[r["truth"]][r["pred"]] += 1
    duck_total = sum(confusion["duck"].values())
    rep = {
        "n_classes_rows": len(orig),
        "perturbations": tot,
        "stability": round(stable / max(1, tot), 4),
        "coverage": round(sum(1 for r in orig if r["pred"] != "unknown") / max(1, len(orig)), 4),
        "unknown_ratio": round(sum(1 for r in orig if r["pred"] == "unknown") / max(1, len(orig)), 4),
        "duck_recall": round(confusion["duck"]["duck"] / max(1, duck_total), 4),
        "padlock_as_duck": round(confusion["padlock"]["duck"] / max(1, sum(confusion["padlock"].values())), 4),
        "empty_as_duck": round(confusion["empty"]["duck"] / max(1, sum(confusion["empty"].values())), 4),
        "other_as_duck": round(confusion["other"]["duck"] / max(1, sum(confusion["other"].values())), 4),
        "confusion": confusion,
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    (OUT_DIR / "slot_tm_bench.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **rep},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def sequence(frames: int, interval: float, templates_dir: Path) -> int:
    """连续帧验证：只读，检查同一槽位跨帧分类是否稳定、槽位数是否抖动。"""
    import time
    lib = Templates(templates_dir)
    if not lib.ready():
        print("模板库为空")
        return 3
    hwnd = ui_icons.find_window()
    if not hwnd:
        print("找不到窗口")
        return 2

    per_index: dict[int, list[str]] = {}
    counts, scores = [], []
    for k in range(frames):
        bgr = ui_icons.grab(hwnd)
        if bgr is None:
            time.sleep(interval)
            continue
        slots = ds.segment_slots(bgr)
        counts.append(len(slots))
        for i, s in enumerate(slots):
            r = lib.match(s["crop"])
            per_index.setdefault(i, []).append(r["cls"])
            scores.append(r["score"])
        time.sleep(interval)

    cons, flips, n_idx = [], 0, 0
    for i, seq in per_index.items():
        if len(seq) < 2:
            continue
        n_idx += 1
        mode = max(set(seq), key=seq.count)
        cons.append(seq.count(mode) / len(seq))
        flips += sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    rep = {
        "frames": len(counts),
        "slots_per_frame": {"min": min(counts) if counts else 0,
                            "max": max(counts) if counts else 0,
                            "mean": round(sum(counts) / max(1, len(counts)), 2)},
        "indices_seen": n_idx,
        "cross_frame_consistency": round(sum(cons) / max(1, len(cons)), 4),
        "class_flips": flips,
        "score_min": round(min(scores), 3) if scores else 0,
        "score_median": round(float(np.median(scores)), 3) if scores else 0,
        "unknown_rate": round(
            sum(1 for seq in per_index.values() for c in seq if c == "unknown")
            / max(1, sum(len(s) for s in per_index.values())), 4),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    (OUT_DIR / "slot_tm_sequence.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **rep},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--frames", type=int, default=3)
    c.add_argument("--out", default=str(OUT_DIR / "slot_raw"))
    b = sub.add_parser("bench")
    b.add_argument("--raw", default=str(OUT_DIR / "slot_raw"))
    b.add_argument("--templates", default=str(HERE / "templates"))
    b.add_argument("--labels", default=str(OUT_DIR / "slot_raw" / "labels.json"))
    g = sub.add_parser("guard")
    s = sub.add_parser("sequence")
    s.add_argument("--frames", type=int, default=30)
    s.add_argument("--interval", type=float, default=0.2)
    s.add_argument("--templates", default=str(HERE / "templates"))
    args = ap.parse_args()
    if args.cmd == "sequence":
        return sequence(args.frames, args.interval, Path(args.templates))
    if args.cmd == "guard":
        hwnd = ui_icons.find_window()
        print(page_guard(hwnd) if hwnd else (False, "no-window"))
        return 0
    if args.cmd == "capture":
        return capture(args.frames, Path(args.out))
    if args.cmd == "bench":
        return bench(Path(args.raw), Path(args.templates), Path(args.labels))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
