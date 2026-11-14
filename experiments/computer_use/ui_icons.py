"""图标 / 红点识别：多尺度模板匹配 + 红色角标检测 + 关联 + 调试图。

为什么要这一层：
  OCR 只认字，写死坐标只认位置。图标和「红点」既没有字、又会随布局/角标数字变化，
  所以必须单独有一层「外观 → 名字」的绑定。这里不训练模型：
    - 图标：灰度 NCC（TM_CCOEFF_NORMED，对亮度线性变化不敏感）+ 多尺度搜索
    - 红点：HSV 红色掩码 + 连通域 + 形状过滤 + 最近图标关联

模板目录：experiments/computer_use/duck_icons/<name>.png
  模板必须裁到字形本身，**不要带角标**（角标数字会变，带进去会掉分）。

用法（仓库根目录）：
    python experiments/computer_use/ui_icons.py --harvest     # 一次性：按 NAV 锚点裁 5 个导航图标
    python experiments/computer_use/ui_icons.py --debug       # 识别当前窗口 + 写标注图/JSON
    python experiments/computer_use/ui_icons.py --selftest    # 缩放/压暗 + 红点关联自测
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEMPLATE_DIR = HERE / "duck_icons"
OUT_DIR = ROOT / "results" / "computer_use"

# 导航图标的一次性引导锚点（归一化到窗口；只用于 --harvest，不参与运行时定位）
HARVEST_BOXES = {
    "market": (0.2605, 0.8570, 0.0480, 0.0270),
    "egg":    (0.3765, 0.8570, 0.0359, 0.0270),
    "duck":   (0.4781, 0.8570, 0.0518, 0.0315),
    "gods":   (0.6199, 0.8570, 0.0518, 0.0270),
    "task":   (0.7131, 0.8570, 0.0359, 0.0270),
}

RED_LO = ((0, 150, 110), (170, 150, 110))
RED_HI = ((10, 255, 255), (180, 255, 255))
BADGE_MIN_AREA = 45
BADGE_MAX_W = 36
BADGE_MAX_H = 36
BADGE_MIN_FILL = 0.50


@dataclass
class Detection:
    name: str
    score: float
    center: tuple[int, int]
    box: tuple[int, int, int, int]
    scale: float

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 窗口/图像

def find_window(substrs=("DUCK", "鸭")) -> int | None:
    import win32gui

    hits: list[int] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and any(s.upper() in title.upper() for s in substrs):
            hits.append(hwnd)

    win32gui.EnumWindows(cb, None)
    return hits[0] if hits else None


def grab(hwnd: int) -> np.ndarray | None:
    """截窗口，返回 BGR（窗口被最小化/不可见时返回 None）。"""
    import pyautogui
    import win32gui

    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return None
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    if r - l <= 100 or b - t <= 100:
        return None
    img = np.array(pyautogui.screenshot(region=(l, t, r - l, b - t)))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


# ------------------------------------------------------------------ 图标匹配

class IconMatcher:
    def __init__(self, template_dir: Path = TEMPLATE_DIR, threshold: float = 0.78):
        self.template_dir = Path(template_dir)
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = {}
        self.scales = tuple(round(v, 2) for v in np.arange(0.70, 1.46, 0.05))
        self.load()

    def load(self) -> "IconMatcher":
        self.templates.clear()
        if not self.template_dir.exists():
            return self
        for p in sorted(self.template_dir.glob("*.png")):
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is not None and min(g.shape) >= 6:
                self.templates[p.stem] = g
        return self

    def names(self) -> list[str]:
        return sorted(self.templates)

    def find(self, gray: np.ndarray, names=None, region=None, threshold=None,
             max_per_name: int = 1) -> list[Detection]:
        """在 gray 的 region=(x,y,w,h) 里找模板；返回按分数降序的命中。"""
        if gray is None or not self.templates:
            return []
        min_score = self.threshold if threshold is None else threshold
        h, w = gray.shape[:2]
        x0, y0, rw, rh = region if region else (0, 0, w, h)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        rw, rh = min(w - x0, int(rw)), min(h - y0, int(rh))
        if rw < 8 or rh < 8:
            return []
        roi = gray[y0:y0 + rh, x0:x0 + rw]
        wanted = set(names) if names else set(self.templates)
        out: list[Detection] = []

        # 平坦/低对比区域（纯色底、背景水印）的 NCC 会退化或假高分，先用局部标准差剔掉
        f = roi.astype(np.float64)
        s1 = cv2.integral(f)
        s2 = cv2.integral(f * f)

        for name in sorted(wanted & set(self.templates)):
            tpl = self.templates[name]
            best = (-1.0, (0, 0), tpl.shape, 1.0)
            for s in self.scales:
                t = cv2.resize(tpl, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
                if t.shape[0] > roi.shape[0] or t.shape[1] > roi.shape[1]:
                    continue
                res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
                th, tw = t.shape[:2]
                n = float(tw * th)
                sums1 = (s1[th:, tw:] - s1[:-th, tw:] - s1[th:, :-tw] + s1[:-th, :-tw])
                sums2 = (s2[th:, tw:] - s2[:-th, tw:] - s2[th:, :-tw] + s2[:-th, :-tw])
                var = np.maximum(sums2 / n - (sums1 / n) ** 2, 0.0)
                std = np.sqrt(var)
                res[std < max(6.0, 0.45 * float(t.std()))] = -1.0
                _, mx, _, ml = cv2.minMaxLoc(res)
                if mx > best[0]:
                    best = (mx, ml, t.shape, s)
            if best[0] < min_score:
                continue

            score, loc, shape, s = best
            if max_per_name <= 1:
                cx = x0 + loc[0] + shape[1] // 2
                cy = y0 + loc[1] + shape[0] // 2
                out.append(Detection(name, round(float(score), 3), (cx, cy),
                                     (x0 + loc[0], y0 + loc[1], shape[1], shape[0]), s))
                continue

            # 同一模板多个实例：在最佳尺度上取所有峰，再做 NMS
            t = cv2.resize(tpl, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= min_score)
            peaks = sorted(((float(res[y, x]), int(x), int(y)) for y, x in zip(ys, xs)),
                           reverse=True)
            kept: list[tuple[int, int, int]] = []
            for sc, px, py in peaks:
                if len(kept) >= max_per_name:
                    break
                if all(abs(px - kx) > shape[1] * 0.6 or abs(py - ky) > shape[0] * 0.6
                       for kx, ky in [(k[1], k[2]) for k in kept]):
                    kept.append((px, py, sc))
            for px, py, sc in kept:
                cx = x0 + px + shape[1] // 2
                cy = y0 + py + shape[0] // 2
                out.append(Detection(name, round(float(sc), 3), (cx, cy),
                                     (x0 + px, y0 + py, shape[1], shape[0]), s))

        out.sort(key=lambda d: d.score, reverse=True)
        return out


# ------------------------------------------------------------------ 红点

def find_badges(bgr: np.ndarray) -> list[dict]:
    """找红色角标：HSV 红掩码 → 连通域 → 形状过滤。返回 box/center/area/fill。"""
    if bgr is None:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(RED_LO[0]), np.array(RED_HI[0]))
    m2 = cv2.inRange(hsv, np.array(RED_LO[1]), np.array(RED_HI[1]))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < BADGE_MIN_AREA or w > BADGE_MAX_W or h > BADGE_MAX_H:
            continue
        fill = area / float(max(1, w * h))
        if fill < BADGE_MIN_FILL:
            continue
        if not (0.5 <= w / float(max(1, h)) <= 2.2):
            continue
        out.append({"box": [x, y, w, h], "center": [round(cents[i][0], 1), round(cents[i][1], 1)],
                    "area": area, "fill": round(fill, 2)})
    out.sort(key=lambda b: b["area"], reverse=True)
    return out


def associate(badges: list[dict], dets: list[Detection], max_dist: float = 48.0) -> list[dict]:
    """把红点挂到最近的图标上；超出 max_dist 的不挂。原地补 near/dist 并返回 badges。"""
    for b in badges:
        bx, by = b["center"]
        best = None
        for d in dets:
            cx, cy = d.center
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            # 角标只会在图标的右上：排除明显在左/下的
            if bx < cx - 6 or by > cy + 10:
                continue
            if best is None or dist < best[1]:
                best = (d.name, dist)
        b["near"] = best[0] if best and best[1] <= max_dist else None
        b["dist"] = round(best[1], 1) if best else None
    return badges


# ------------------------------------------------------------------ 标注

def annotate(bgr: np.ndarray, dets: list[Detection], badges: list[dict], out_path: Path) -> Path:
    img = bgr.copy()
    for d in dets:
        x, y, w, h = d.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 220, 0), 1)
        cv2.putText(img, f"{d.name} {d.score:.2f}", (x, max(10, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
    for b in badges:
        if not b.get("near"):
            continue          # 未关联的红色块（装饰、图标本身）不画，避免调试图变噪声
        x, y, w, h = b["box"]
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 2)
        cv2.putText(img, f"dot->{b['near']}", (max(0, x - 24), min(img.shape[0] - 2, y + h + 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path


# ------------------------------------------------------------------ 引导：裁模板

def harvest(bgr: np.ndarray, template_dir: Path = TEMPLATE_DIR) -> list[dict]:
    if bgr is None:
        raise RuntimeError("图像不可用")
    gray = to_gray(bgr)
    h, w = gray.shape
    template_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, (cx, cy, bw, bh) in HARVEST_BOXES.items():
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        crop = gray[y1:y2, x1:x2]
        if min(crop.shape) < 6:
            continue
        p = template_dir / f"{name}.png"
        cv2.imwrite(str(p), crop)
        saved.append({"name": name, "path": str(p.relative_to(ROOT)), "box": [x1, y1, x2 - x1, y2 - y1]})
    return saved


# ------------------------------------------------------------------ CLI

def _window_or_die(hwnd_arg: int | None) -> int:
    hwnd = hwnd_arg or find_window()
    if not hwnd:
        raise SystemExit("找不到游戏窗口（标题含 DUCK/鸭）")
    return hwnd


def cmd_debug(bgr: np.ndarray, out_dir: Path = OUT_DIR) -> dict:
    matcher = IconMatcher()
    if bgr is None:
        raise SystemExit("图像不可用")
    gray = to_gray(bgr)
    h, w = gray.shape
    t0 = time.perf_counter()
    dets = matcher.find(gray, region=(0, int(0.78 * h), w, int(0.16 * h)))
    ms = (time.perf_counter() - t0) * 1000
    badges = associate(find_badges(bgr), dets,
                       max_dist=max(32.0, 0.10 * bgr.shape[1]))
    png = annotate(bgr, dets, badges, out_dir / "duck_icons_debug.png")
    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window": [int(v) for v in (w, h)],
        "templates": matcher.names(),
        "match_ms": round(ms, 1),
        "icons": [d.as_dict() for d in dets],
        "badges": badges,
        "debug_png": str(png),
    }
    (out_dir / "duck_icons_debug.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def cmd_selftest(bgr: np.ndarray, out_dir: Path = OUT_DIR) -> dict:
    """缩放 ±20% + 压暗：图标仍应命中同一位置；红点应挂到 egg/task。"""
    matcher = IconMatcher()
    if bgr is None:
        raise SystemExit("图像不可用")
    gray = to_gray(bgr)
    h, w = gray.shape
    region = (0, int(0.78 * h), w, int(0.16 * h))

    base = matcher.find(gray, region=region)
    base_by = {d.name: d.center for d in base}
    cases = []
    for s in (1.0, 0.8, 1.2):
        img = cv2.resize(gray, None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        bh, bw = img.shape
        dets = matcher.find(img, region=(0, int(0.78 * bh), bw, int(0.16 * bh)))
        got = {d.name: d.center for d in dets}
        rows = []
        for name, (cx, cy) in base_by.items():
            exp = (cx * s, cy * s)
            c = got.get(name)
            ok = bool(c) and abs(c[0] - exp[0]) <= 8 and abs(c[1] - exp[1]) <= 8
            rows.append({"name": name, "found": bool(c),
                         "err_px": None if not c else round(((c[0] - exp[0]) ** 2 + (c[1] - exp[1]) ** 2) ** 0.5, 1),
                         "score": next((d.score for d in dets if d.name == name), None),
                         "ok": ok})
        cases.append({"scale": s, "icons": rows,
                      "all_ok": all(r["ok"] for r in rows) and len(rows) == len(base_by)})

    dim = cv2.convertScaleAbs(gray, alpha=0.6, beta=-20)
    dets_dim = matcher.find(dim, region=region)
    dim_by = {d.name: d.score for d in dets_dim}

    badges = associate(find_badges(bgr), base)
    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window": [int(w), int(h)],
        "templates": matcher.names(),
        "base": [d.as_dict() for d in base],
        "scaled": cases,
        "dimmed_scores": dim_by,
        "badges": badges,
        "pass": bool(cases[0]["all_ok"] and cases[1]["all_ok"] and cases[2]["all_ok"]
                     and base and all(v >= 0.8 for v in dim_by.values())
                     and any(b.get("near") == "egg" for b in badges)
                     and any(b.get("near") == "task" for b in badges)),
    }
    (out_dir / "duck_icons_selftest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hwnd", type=int, default=0)
    ap.add_argument("--image", default="", help="离线截图；给定时不抓窗口")
    ap.add_argument("--harvest", action="store_true", help="按引导锚点裁导航模板（一次性）")
    ap.add_argument("--debug", action="store_true", help="识别当前窗口并写标注图")
    ap.add_argument("--selftest", action="store_true", help="缩放/压暗 + 红点关联自测")
    args = ap.parse_args()

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"读不到图像：{args.image}")
    else:
        hwnd = _window_or_die(args.hwnd or None)
        bgr = grab(hwnd)
        if bgr is None:
            raise SystemExit("窗口不可用（最小化或被隐藏）")

    if args.harvest:
        saved = harvest(bgr)
        print(json.dumps({"saved": saved}, ensure_ascii=False, indent=2))
        return 0
    if args.selftest:
        print(json.dumps(cmd_selftest(bgr), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(cmd_debug(bgr), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
