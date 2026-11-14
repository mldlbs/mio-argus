"""DUCK 槽位三分类（P1-1）。

业务类别：duck / padlock / empty；证据不足一律 unknown，不强行归类。

特征（多证据，不依赖单一轮廓）：
  ink_ratio    墨水占比（相对背景）
  comps        连通域数量
  aspect       主体外接框宽高比
  fill         主体填充率
  top_ratio    上部宽度 / 下部宽度（挂锁有细环，比值显著小）
  sat          墨水像素平均饱和度（彩色鸭更高）
  edge         墨水区域边缘密度（内部纹理）

门禁口径（见 docs/DUCK_EXEC_PROTOCOL.md）：
  鸭子召回率 / 挂锁误识率 / 空槽误识率 / unknown 比例 分开报告。

用法：
  python experiments/computer_use/duck_slots.py debug            # 打印当前帧每格特征
  python experiments/computer_use/duck_slots.py capture --frames 5 --out results/computer_use/slot_testset
  python experiments/computer_use/duck_slots.py eval   --set results/computer_use/slot_testset
  python experiments/computer_use/duck_slots.py perturb --set results/computer_use/slot_testset
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

import ui_icons  # noqa: E402

OUT_DIR = ROOT / "results" / "computer_use"
ROW_BAND = (0.658, 0.732)

# 判据阈值（先用真实帧校准，再进入门禁）
TH = {
    "empty_ink": 0.05,        # ink_ratio 低于此值判 empty
    "padlock_top": 0.62,      # top_ratio 低于此值倾向 padlock
    "duck_aspect": (0.70, 1.60),
    "duck_fill": (0.28, 0.75),
    "min_conf": 0.60,         # 最高分类置信度低于此值 → unknown
    "margin": 0.12,           # 最高与次高之差低于此值 → unknown
}


def _band(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    h = bgr.shape[0]
    y0, y1 = int(ROW_BAND[0] * h), int(ROW_BAND[1] * h)
    return bgr[y0:y1], y0


def locate_row(bgr: np.ndarray) -> int:
    """在 0.58–0.78 高度范围内找图标最多的一行，返回该行中心 y（像素）。"""
    h = bgr.shape[0]
    best_runs, best_y = -1, int(0.691 * h)
    for y in range(int(0.58 * h), int(0.78 * h), 4):
        strip = bgr[y:y + 10]
        if strip.shape[0] < 6:
            continue
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        col = (gray > 55).mean(axis=0) > 0.30
        runs, prev = 0, False
        for v in col:
            if v and not prev:
                runs += 1
            prev = v
        if runs > best_runs:
            best_runs, best_y = runs, y + 5
    return best_y


def segment_slots(bgr: np.ndarray) -> list[dict]:
    """按列投影切出槽位框（含全部图标，不区分类型）。"""
    cy = locate_row(bgr)
    y0 = max(0, cy - 20)
    y1 = min(bgr.shape[0], cy + 20)
    band = bgr[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bright = (gray > 55).astype(np.uint8)
    col = bright.mean(axis=0) > 0.12
    runs, start = [], None
    for x, v in enumerate(col):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(col) - 1))

    slots = []
    W = bright.shape[1]
    for a, b in runs:
        w = b - a + 1
        # 尺寸过滤 + 贴边剔除（行两端是页面白边，不是槽位）
        if not (10 <= w <= 60):
            continue
        if a <= 3 or b >= W - 4:
            continue
        seg = bright[:, a:b + 1]
        rows = np.where(seg.mean(axis=1) > 0.2)[0]
        if len(rows) < 5:
            continue
        top, bot = int(rows.min()), int(rows.max())
        hgt = bot - top + 1
        if not (12 <= hgt <= 40):
            continue
        # 近白细条（渲染白边）直接丢弃
        if float(gray[top:bot + 1, a:b + 1].mean()) > 200:
            continue
        slots.append({"box": [int(a), int(top + y0), int(w), int(hgt)],
                      "crop": band[max(0, top - 2):bot + 3, max(0, a - 1):b + 2]})
    return slots


def slot_features(crop: np.ndarray) -> dict:
    if crop.size == 0:
        return {"ink_ratio": 0.0, "comps": 0, "aspect": 0.0, "fill": 0.0,
                "top_ratio": 0.0, "sat": 0.0, "edge": 0.0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 亮度归一化 + 轻度平滑：对整体明暗/噪声不敏感（扰动测试曾把稳定性打到 4.7%）
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    g = (g - lo) / max(1e-6, hi - lo)
    ink = (g > 0.35).astype(np.uint8)
    ink_ratio = float(ink.mean())

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    idx = [i for i in range(1, n) if stats[i][4] >= 12]
    if not idx:
        return {"ink_ratio": round(ink_ratio, 3), "comps": 0, "aspect": 0.0, "fill": 0.0,
                "top_ratio": 0.0, "sat": 0.0, "edge": 0.0, "solidity": 0.0,
                "lr_sym": 0.0}
    comps = [stats[i] for i in idx]

    big_idx = max(idx, key=lambda i: stats[i][4])
    big = stats[big_idx]
    w, hh, area = int(big[2]), int(big[3]), int(big[4])
    aspect = w / max(1, hh)
    fill = area / max(1, w * hh)

    # 实心度：圆点≈1.0，鸭子有喙/颈凹陷，明显小于 1
    mask_big = (labels == big_idx).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        solidity = float(cv2.contourArea(c) / ha) if ha > 0 else 0.0

    # 左右对称度：挂锁（细环+方体）高度对称；鸭子有喙，偏一侧
    bx, by, bw, bh = int(big[0]), int(big[1]), int(big[2]), int(big[3])
    body = (labels[by:by + bh, bx:bx + bw] == big_idx)
    if body.size:
        mir = body[:, ::-1]
        union = np.logical_or(body, mir).sum()
        lr_sym = float(np.logical_and(body, mir).sum() / union) if union else 0.0
    else:
        lr_sym = 0.0

    # 行方向的墨水宽度分布 → 上下宽度比
    row_w = ink.sum(axis=1).astype(float)
    nz = np.where(row_w > 0)[0]
    if len(nz) >= 4:
        top_part = row_w[nz.min(): nz.min() + max(1, len(nz) // 3)]
        bot_part = row_w[nz.min() + len(nz) // 3: nz.max() + 1]
        top_ratio = float(top_part.mean() / max(1e-6, bot_part.mean()))
    else:
        top_ratio = 0.0

    sat = float(hsv[:, :, 1][ink > 0].mean()) if ink.sum() else 0.0
    edge = float((cv2.Canny(gray, 40, 120) > 0).mean())

    return {"ink_ratio": round(ink_ratio, 3), "comps": len(comps),
            "aspect": round(aspect, 3), "fill": round(fill, 3),
            "top_ratio": round(top_ratio, 3), "sat": round(sat, 1),
            "edge": round(edge, 4), "solidity": round(solidity, 3),
            "lr_sym": round(lr_sym, 3)}


def _ramp(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def classify(f: dict) -> dict:
    """决策规则分类。

    实测（同一帧 18 格）显示唯一稳定的判别证据是 top_ratio（上部宽度/下部宽度）：
      鸭子 0.83–0.89 ｜ 挂锁 0.52 ｜ 中间区间不可靠
    因此用显式阈值 + 中间带拒绝，而不是加权求和（求和会让低饱和灰鸭被判成挂锁）。
    """
    ink = f["ink_ratio"]
    if ink < TH["empty_ink"]:
        return {"cls": "empty", "conf": round(1.0 - ink / TH["empty_ink"], 3),
                "evidence": {**f, "why": "ink_below_empty"}}

    top = f["top_ratio"]
    ev = {**f}

    if top >= 0.68:
        conf = 0.60 + 0.40 * _ramp(top, 0.68, 0.86)
        return {"cls": "duck", "conf": round(min(1.0, conf), 3),
                "evidence": {**ev, "rule": "top_ratio>=0.68"}}
    if top <= 0.58:
        conf = 0.60 + 0.40 * _ramp(0.58 - top, 0.0, 0.20)
        return {"cls": "padlock", "conf": round(min(1.0, conf), 3),
                "evidence": {**ev, "rule": "top_ratio<=0.58"}}
    return {"cls": "unknown", "conf": round(0.5 + 0.2 * (1 - abs(top - 0.63) / 0.05), 3),
            "evidence": {**ev, "rule": "top_ratio_ambiguous"}}



def analyze_frame(bgr: np.ndarray) -> list[dict]:
    out = []
    for i, s in enumerate(segment_slots(bgr)):
        c = classify(slot_features(s["crop"]))
        c.update(index=i, box=s["box"])
        out.append(c)
    return out


def capture(frames: int, out: Path) -> int:
    hwnd = ui_icons.find_window()
    if not hwnd:
        print("找不到窗口")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"ts": datetime.now().isoformat(timespec="seconds"), "frames": []}
    for k in range(frames):
        bgr = ui_icons.grab(hwnd)
        if bgr is None:
            continue
        slots = analyze_frame(bgr)
        rec = {"frame": k, "slots": []}
        for s in slots:
            i = s["index"]
            x, y, w, h = s["box"]
            name = f"f{k:03d}_s{i:02d}.png"
            cv2.imwrite(str(out / name), bgr[y:y + h, x:x + w])
            rec["slots"].append({"i": i, "file": name, "box": s["box"],
                                 "pred": s["cls"], "conf": s["conf"],
                                 "evidence": s["evidence"]})
        manifest["frames"].append(rec)
        print(f"frame {k}: {len(slots)} slots -> " +
              " ".join(f"{s['index']}:{s['cls']}" for s in slots))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    return 0


def _metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(pred, label)]"""
    classes = ["duck", "padlock", "empty", "unknown"]
    conf = {a: {b: 0 for b in classes} for a in classes}
    for p, l in pairs:
        conf[p][l] += 1
    out = {"n": len(pairs), "confusion": conf}
    for c in ["duck", "padlock", "empty"]:
        tp = conf[c][c]
        pred_pos = sum(conf[c])            # 预测为该类的总数
        real = sum(conf[r][c] for r in classes)  # 实际为该类的总数
        out[f"{c}_recall"] = round(tp / real, 4) if real else None
        out[f"{c}_precision"] = round(tp / pred_pos, 4) if pred_pos else None
        out[f"{c}_false_pos"] = pred_pos - tp
    out["unknown_ratio"] = round(conf["unknown"][l] if False else
                                 sum(conf["unknown"].values()) / max(1, len(pairs)), 4)
    out["accuracy"] = round(sum(conf[c][c] for c in classes) / max(1, len(pairs)), 4)
    return out


def eval_set(path: Path) -> int:
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    labels_path = path / "labels.json"
    if not labels_path.exists():
        print(f"缺少人工标签 {labels_path}；先生成 labels.json（frame.slot -> class）")
        return 2
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    pairs, flips = [], 0
    for fr in man["frames"]:
        for s in fr["slots"]:
            key = f"{fr['frame']}.{s['i']}"
            if key in labels:
                pairs.append((s["pred"], labels[key]))
    m = _metrics(pairs)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    report = {"ts": datetime.now().isoformat(timespec="seconds"),
              "set": str(path), "metrics": m}
    (OUT_DIR / "slot_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def perturb_set(path: Path) -> int:
    """扰动稳定性：亮度/模糊/噪声/平移，分类结果应保持不变。"""
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    base, cases = {}, []
    for fr in man["frames"]:
        for s in fr["slots"]:
            img = cv2.imread(str(path / s["file"]))
            if img is None:
                continue
            base[(fr["frame"], s["i"])] = s["pred"]
            for name, fn in (
                ("bright+20", lambda im: np.clip(im.astype(np.int16) + 20, 0, 255).astype(np.uint8)),
                ("bright-20", lambda im: np.clip(im.astype(np.int16) - 20, 0, 255).astype(np.uint8)),
                ("blur", lambda im: cv2.GaussianBlur(im, (3, 3), 0)),
                ("noise", lambda im: np.clip(im + np.random.normal(0, 6, im.shape), 0, 255).astype(np.uint8)),
                ("shift", lambda im: np.roll(im, 2, axis=1)),
            ):
                p = classify(slot_features(fn(img)))["cls"]
                cases.append({"kind": name, "same": p == base[(fr["frame"], s["i"])]})
    rate = sum(1 for c in cases if c["same"]) / max(1, len(cases))
    print(json.dumps({"perturbations": len(cases),
                      "stability": round(rate, 4)}, ensure_ascii=False, indent=2))
    return 0


def debug() -> int:
    hwnd = ui_icons.find_window()
    if not hwnd:
        print("找不到窗口")
        return 2
    bgr = ui_icons.grab(hwnd)
    if bgr is None:
        print("截不到窗口")
        return 2
    for s in analyze_frame(bgr):
        e = s["evidence"]
        print(f"#{s['index']:02d} {s['cls']:8s} conf={s['conf']:.2f} "
              f"ink={e['ink_ratio']:.3f} comps={e['comps']} aspect={e['aspect']:.2f} "
              f"fill={e['fill']:.2f} top={e['top_ratio']:.2f} sat={e['sat']:.0f} "
              f"sol={e['solidity']:.2f} sym={e['lr_sym']:.2f} scores={e.get('scores')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("debug")
    c = sub.add_parser("capture")
    c.add_argument("--frames", type=int, default=3)
    c.add_argument("--out", default=str(OUT_DIR / "slot_testset"))
    e = sub.add_parser("eval")
    e.add_argument("--set", default=str(OUT_DIR / "slot_testset"))
    p = sub.add_parser("perturb")
    p.add_argument("--set", default=str(OUT_DIR / "slot_testset"))
    args = ap.parse_args()
    if args.cmd == "debug":
        return debug()
    if args.cmd == "capture":
        return capture(args.frames, Path(args.out))
    if args.cmd == "eval":
        return eval_set(Path(args.set))
    if args.cmd == "perturb":
        return perturb_set(Path(args.set))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
