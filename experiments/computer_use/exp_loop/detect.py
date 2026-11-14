"""归因层：只吃 episode 原始观测，吐「失败假设 + 证据数值」。

原则：
  - 每个 detector 是纯函数，输入 episode（可加基目录读帧），输出 (是否成立, 证据)。
  - detector 不含任何"这是点中缝隙"之类的结论，只报可测量的现象。
  - 阈值全部可调；具体取值由验证器在数据上选，不写死在人类判断里。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# 槽位带常量（与 duck_guard 保持一致；这里不 import，避免拉起重依赖）
ROW_BAND = (0.686, 0.708)
ROW_CY = 0.6975


def _load(base: Path, rel: str | None):
    if not rel:
        return None
    p = (base / rel) if not Path(rel).is_absolute() else Path(rel)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def _frame_diff(a, b, region=None) -> float | None:
    """两帧的平均绝对差（0-255）。region=(y0,y1,x0,x1) 相对比例。"""
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    if region is not None:
        H, W = a.shape[:2]
        y0, y1, x0, x1 = region
        a = a[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        b = b[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        if a.size == 0:
            return None
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def _pixel_energy(img, region=None) -> float | None:
    """画面能量的代理量：亮度标准差（判断"不是纯黑/空白"）。"""
    if img is None:
        return None
    if region is not None:
        H, W = img.shape[:2]
        y0, y1, x0, x1 = region
        img = img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        if img.size == 0:
            return None
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())


def icon_centers(img, band=ROW_BAND, min_width: int = 6) -> list[float]:
    """槽位带图标中心（相对 x）。用于判断"落点是否在图标上"。"""
    if img is None:
        return []
    H, W = img.shape[:2]
    y0, y1 = int(band[0] * H), int(band[1] * H)
    sub = img[y0:y1, :]
    if sub.size == 0:
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
            if j - i + 1 >= min_width:
                out.append(round(((i + j) / 2) / W, 3))
            i = j + 1
        else:
            i += 1
    return out


# ---------------------------------------------------------------- detectors
# 每个 detector: fn(ep, base, **params) -> (fired, evidence)

def d_no_visual_change(ep: dict, base: Path, thr: float = 0.8,
                       region=(0.28, 0.62, 0.25, 0.75)) -> tuple[bool, dict]:
    """动作后画面几乎不变（默认只看卡片区域，避免倒计时等噪声）。"""
    a = _load(base, ep["before"].get("frame"))
    b = _load(base, ep["after"].get("frame"))
    d = _frame_diff(a, b, region)
    if d is None:
        return False, {"frame_diff": None}
    return d < thr, {"frame_diff": round(d, 3), "threshold": thr}


def d_text_unchanged(ep: dict, base: Path, **_) -> tuple[bool, dict]:
    """动作前后页面文本完全相同（而期望是变化）。"""
    exp = (ep.get("expected") or {}).get("kind")
    if exp not in ("selection_change", "text_change", "state_change"):
        return False, {"expected": exp}
    before = (ep["before"].get("text") or "").strip()
    after = (ep["after"].get("text") or "").strip()
    return before == after and bool(before), {"len": len(before)}


def d_empty_recognition(ep: dict, base: Path,
                        min_energy: float = 12.0) -> tuple[bool, dict]:
    """识别通道返回空串，但画面明显有内容（能量高于阈值）。"""
    if (ep.get("action") or {}).get("type") != "recognize":
        return False, {"action": (ep.get("action") or {}).get("type")}
    txt = (ep["after"].get("text") or "").strip()
    ch = ep["after"].get("ocr_channel")
    img = _load(base, ep["after"].get("frame"))
    e = _pixel_energy(img)
    if ch != "fast" or txt != "":
        return False, {"channel": ch, "text_len": len(txt), "energy": e}
    return (e is not None and e >= min_energy), {"channel": ch, "energy": e,
                                                 "min_energy": min_energy}


def d_target_is_sentence(ep: dict, base: Path, max_chars: int = 10,
                         **_) -> tuple[bool, dict]:
    """被点目标的文本很长（像句子/说明），不像按钮标签。"""
    t = ((ep.get("action") or {}).get("text_target") or "").strip()
    if not t:
        return False, {"text_len": 0}
    return len(t) > max_chars, {"text_len": len(t), "max_chars": max_chars,
                                "sample": t[:24]}


def d_click_off_icon(ep: dict, base: Path, tol: float = 0.012,
                     **_) -> tuple[bool, dict]:
    """点击落点不在任何图标中心（相对 x 距离都超过 tol）。"""
    act = ep.get("action") or {}
    if act.get("type") != "click" or act.get("x") is None:
        return False, {"action": act.get("type")}
    img = _load(base, ep["before"].get("frame"))
    centers = icon_centers(img)
    if not centers:
        return False, {"icons": 0}
    nearest = min(abs(c - float(act["x"])) for c in centers)
    return nearest > tol, {"nearest": round(nearest, 4), "tol": tol,
                           "icons": len(centers)}


DETECTORS = {
    "no_visual_change": d_no_visual_change,
    "text_unchanged": d_text_unchanged,
    "empty_recognition": d_empty_recognition,
    "target_is_sentence": d_target_is_sentence,
    "click_off_icon": d_click_off_icon,
}


def attribute(episodes: list[dict], base: Path, params: dict | None = None) -> list[dict]:
    """对每个 episode 跑全部 detector，输出假设列表（不含人类结论）。"""
    params = params or {}
    out = []
    for ep in episodes:
        fired = []
        for name, fn in DETECTORS.items():
            try:
                ok, ev = fn(ep, base, **(params.get(name) or {}))
            except Exception as exc:      # 单个 detector 出错不影响其它
                ok, ev = False, {"error": str(exc)[:120]}
            if ok:
                fired.append({"detector": name, "evidence": ev})
        out.append({"episode": ep["id"], "hypotheses": fired})
    return out
