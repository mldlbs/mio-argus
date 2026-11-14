"""自动特征提取：只从 episode 原始观测算，不含任何"这条为什么失败"的知识。

特征分五族（供搜索组合）：
  D_*  区域像素差分（3×3 网格 + 全图）
  E_*  区域亮度能量（动作前/后）
  T_*  文本变化（长度、bigram Jaccard、行数）
  G_*  动作几何（落点坐标、与最近图标的距离、图标数）
  A_*  动作元信息（类型、目标文本长度、识别通道）
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

GRID = [(i / 3, (i + 1) / 3) for i in range(3)]     # 3×3


def _load(base: Path, rel: str | None):
    if not rel:
        return None
    p = (base / rel) if not Path(rel).is_absolute() else Path(rel)
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def _crop(img, y0, y1, x0, x1):
    if img is None:
        return None
    H, W = img.shape[:2]
    c = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    return c if c.size else None


def _bigrams(s: str) -> set[str]:
    s = "".join(s.split())
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def _jaccard(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def icon_centers(img) -> list[float]:
    """槽位带图标中心（相对 x），用于几何特征。"""
    if img is None:
        return []
    H, W = img.shape[:2]
    sub = _crop(img, 0.686, 0.708, 0, 1)
    if sub is None:
        return []
    v = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
    thr = max(45, int(np.median(v)) + 12)
    col = (v > thr).mean(axis=0) > 0.22
    out, i, n = [], 0, len(col)
    while i < n:
        if col[i]:
            j = i
            while j + 1 < n and col[j + 1]:
                j += 1
            if j - i + 1 >= 6:
                out.append(round(((i + j) / 2) / W, 3))
            i = j + 1
        else:
            i += 1
    return out


def extract(ep: dict, base: Path) -> dict:
    """返回 {feature_name: float}。全部可计算、无人类语义标签。"""
    f: dict[str, float] = {}
    a = _load(base, ep["before"].get("frame"))
    b = _load(base, ep["after"].get("frame"))

    # ---- D_*：区域差分
    diffs = []
    if a is not None and b is not None:
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
        whole = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        f["D_whole"] = round(whole, 3)
        for ry, (y0, y1) in enumerate(GRID):
            for cx, (x0, x1) in enumerate(GRID):
                ca, cb = _crop(a, y0, y1, x0, x1), _crop(b, y0, y1, x0, x1)
                if ca is None or cb is None:
                    continue
                d = float(np.abs(ca.astype(np.int16) - cb.astype(np.int16)).mean())
                f[f"D_r{ry}c{cx}"] = round(d, 3)
                diffs.append(d)
        f["D_mean"] = round(float(np.mean(diffs)), 3) if diffs else 0.0
        f["D_max"] = round(float(np.max(diffs)), 3) if diffs else 0.0
    else:
        f["D_whole"] = f["D_mean"] = f["D_max"] = 0.0

    # ---- E_*：能量
    for tag, img in (("b", a), ("a", b)):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
        f[f"E_{tag}_std"] = round(float(g.std()), 2) if g is not None else 0.0
        mid = _crop(g, 0.28, 0.66, 0.22, 0.78) if g is not None else None
        f[f"E_{tag}_mid"] = round(float(mid.std()), 2) if mid is not None else 0.0

    # ---- T_*：文本
    tb = (ep["before"].get("text") or "").strip()
    ta = (ep["after"].get("text") or "").strip()
    f["T_len_b"] = len(tb)
    f["T_len_a"] = len(ta)
    f["T_len_delta"] = len(ta) - len(tb)
    f["T_len_ratio"] = round(len(ta) / max(1, len(tb)), 3)
    f["T_jaccard"] = round(_jaccard(tb, ta), 3)
    f["T_changed"] = 0.0 if tb == ta else 1.0
    f["T_lines_b"] = len(tb.split())
    f["T_lines_a"] = len(ta.split())
    f["T_lines_delta"] = f["T_lines_a"] - f["T_lines_b"]

    # ---- G_*：几何
    act = ep.get("action") or {}
    x = act.get("x")
    centers = icon_centers(a)
    f["G_n_icons"] = float(len(centers))
    if x is not None and centers:
        f["G_icon_dist"] = round(min(abs(c - float(x)) for c in centers), 4)
        f["G_icon_dist_n"] = round(f["G_icon_dist"] / 0.0558, 2)
    else:
        f["G_icon_dist"] = -1.0
        f["G_icon_dist_n"] = -1.0
    f["G_x"] = float(x) if x is not None else -1.0
    f["G_y"] = float(act.get("y")) if act.get("y") is not None else -1.0

    # ---- A_*：动作元信息
    f["A_is_click"] = 1.0 if act.get("type") == "click" else 0.0
    f["A_is_recognize"] = 1.0 if act.get("type") == "recognize" else 0.0
    f["A_target_len"] = float(len((act.get("text_target") or "").strip()))
    f["A_fast"] = 1.0 if (ep["after"].get("ocr_channel") == "fast") else 0.0
    return f


def extract_all(episodes: list[dict], base: Path) -> dict[str, dict]:
    return {ep["id"]: extract(ep, base) for ep in episodes}


if __name__ == "__main__":
    OUT = Path("results") / "computer_use" / "exp_loop"
    eps = [json.loads(l) for l in (OUT / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    feats = extract_all(eps, OUT)
    (OUT / "features.json").write_text(json.dumps(feats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"episodes={len(feats)} features={len(next(iter(feats.values())))}")
