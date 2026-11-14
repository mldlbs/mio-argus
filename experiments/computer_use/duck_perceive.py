"""DUCK 只读感知门禁（P0）。

连续 N 帧，只做 L0 操作：截图 / OCR / 连通域统计。**不点击、不聚焦、不拖拽。**

输出指标（对应 docs/DUCK_EXEC_PROTOCOL.md 第 5 节）：
  page_accuracy      人工标签 vs 识别结果（需 --label-page）
  duck_recall        彩色缩略图检出数 / 人工标注数（需 --label-ducks）
  position_jitter    同一图标跨帧中心点标准差
  state_change       计时器 / 队列读数变化次数
  rejection_rate     尺寸不合格被丢弃的连通域占比

用法：
    python experiments/computer_use/duck_perceive.py --runs 10 --interval 1.0
    python experiments/computer_use/duck_perceive.py --runs 100 --label-page duck --label-ducks 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import win32gui  # noqa: E402

import bigemap_runner as bench  # noqa: E402
import ui_icons  # noqa: E402

OUT_DIR = ROOT / "results" / "computer_use"

# ---------------------------------------------------------------- 页面规则
# 每组是「任选其一」；一个页面匹配 = 所有组都命中
PAGE_RULES: list[tuple[str, list[list[str]]]] = [
    ("maintenance", [["MAINTENANCE"]]),
    ("modal", [["ABANDONED CART", "TOMATO FIESTA", "KICKED OUT",
                "鸭子繁殖完成", "合并进行中", "获得鸭子的几率提高",
                "质押这只鸭子", "绑定的物品"]]),
    ("collection", [["SWEET HALLOWEEN"], ["已收集"]]),
    ("daily_reward", [["每日奖励"], ["领取"]]),
    ("ducker_pass", [["王子鸭主", "国王鸭主"]]),
    ("event_halloween", [["HALLOWEEN"], ["CANDY PATH", "GUARANTEED PRIZE", "SKIP ANIMATIONS"]]),
    ("merge_duck", [["合并鸭子"]]),
    ("market", [["DUCKS", "我的市场"]]),
    ("task", [["Challenge Week", "广告", "Pack"]]),
    ("egg", [["鸭群奖励", "下一颗蛋", "下一颖蛋", "自动合并", "合井等级", "合并等级"]]),
    ("duck", [["点击喂食", "已准备好", "繁殖进行中", "这是一只新鸭子", "请勿打扰",
               "鸭子正在出售", "已在市场队列中", "给它安排一个槽位"],
              ["市场", "鸭子", "任务", "GOOS"]]),
]

ROW_BAND = (0.672, 0.716)      # 槽位行（相对窗口高度）
TIMER_BAND = (0.72, 0.76)      # 「重置倒计时」
QUEUE_BAND = (0.755, 0.80)     # 「下一颗蛋及后续 N 颗」


def classify_page(text: str) -> tuple[str, float, list[str]]:
    best, best_score, evidence = "unknown", 0.0, []
    for page, groups in PAGE_RULES:
        hit = 0
        ev = []
        for group in groups:
            for pat in group:
                if pat in text:
                    hit += 1
                    ev.append(pat)
                    break
        score = hit / len(groups)
        if score > best_score:
            best, best_score, evidence = page, score, ev
    return (best, round(best_score, 3), evidence) if best_score == 1.0 else ("unknown", round(best_score, 3), evidence)


def detect_row_icons(bgr: np.ndarray) -> tuple[list[dict], dict]:
    """槽位行两类目标分开统计。

    - slots: 亮度投影得到的图标槽位（鸭子 + 挂锁，忽略颜色）
    - colored: 高饱和色块（红点 / 彩色鸭）

    早期版本只数高饱和块，结果数到的是红点而不是鸭子 —— 这个错误由门禁第一轮抓出来。
    """
    h = bgr.shape[0]
    y0, y1 = int(ROW_BAND[0] * h), int(ROW_BAND[1] * h)
    band = bgr[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bright = (gray > 55).astype(np.uint8)

    # 列投影 → 连续图标段
    col = bright.mean(axis=0)
    on = col > 0.12
    runs, start = [], None
    for x, v in enumerate(on):
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(on) - 1))

    slots = []
    for a, b in runs:
        w = b - a + 1
        if not (12 <= w <= 70):
            continue
        seg = bright[:, a:b + 1]
        rows = np.where(seg.mean(axis=1) > 0.2)[0]
        if len(rows) < 6:
            continue
        cy = float(rows.mean()) + y0
        slots.append({"box": [int(a), int(rows.min() + y0), int(w), int(rows.max() - rows.min() + 1)],
                      "center": [round((a + b) / 2, 1), round(cy, 1)]})

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    colored_mask = ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 90)).astype(np.uint8)
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(colored_mask, 8)
    colored = 0
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area >= 60 and 6 <= w <= 40 and 6 <= hh <= 40:
            colored += 1

    return slots, {"slots": len(slots), "colored": colored}



def ocr_bands(hwnd: int) -> dict:
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    w, h = r - l, b - t
    full, _ = bench.ocr_lines((l, t, w, h))
    text = " ".join(x["text"] for x in full if x["text"].strip())

    def band(frac):
        y0 = t + int(frac[0] * h)
        y1 = t + int(frac[1] * h)
        lines, _ = bench.ocr_lines((l, y0, w, max(10, y1 - y0)))
        return " ".join(x["text"] for x in lines if x["text"].strip())

    return {"text": text, "timer": band(TIMER_BAND), "queue": band(QUEUE_BAND)}


def jitter(frames: list[list[dict]]) -> float:
    """相邻帧按最近邻匹配图标中心，返回 dx/dy 的标准差（像素）。"""
    deltas = []
    for a, b in zip(frames, frames[1:]):
        for pa in a:
            best = None
            for pb in b:
                d = abs(pa["center"][0] - pb["center"][0]) + abs(pa["center"][1] - pb["center"][1])
                if best is None or d < best[0]:
                    best = (d, pb)
            if best and best[0] <= 14:
                deltas.append((best[1]["center"][0] - pa["center"][0],
                               best[1]["center"][1] - pa["center"][1]))
    if len(deltas) < 2:
        return 0.0
    arr = np.array(deltas, dtype=float)
    return round(float(np.sqrt(arr[:, 0].var() + arr[:, 1].var())), 2)


def cmd_pagecheck(rounds: int, per_page: int) -> int:
    """页面识别准确率：以“我导航到了哪一页”为真值，逐帧比对分类结果。"""
    import time
    from duck_guard import Game, find_game_window
    hwnd = find_game_window()
    if not hwnd:
        print("找不到窗口")
        return 2
    g = Game(hwnd)
    g.pin()
    NAV = {"duck": (0.478, 0.879), "egg": (0.388, 0.877),
           "market": (0.261, 0.885), "task": (0.727, 0.879)}
    per: dict[str, list[bool]] = {}
    for _ in range(rounds):
        for name, (rx, ry) in NAV.items():
            g.click(rx, ry, wait=1.2)
            for _ in range(per_page):
                page, conf, _ = classify_page(g.text())
                per.setdefault(name, []).append(page == name)
                time.sleep(0.25)
    rep = {k: round(sum(v) / max(1, len(v)), 4) for k, v in per.items()}
    overall = round(sum(sum(v) for v in per.values()) / max(1, sum(len(v) for v in per.values())), 4)
    out = {"per_page": rep, "overall_accuracy": overall,
           "samples": sum(len(v) for v in per.values())}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "page_accuracy.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_statecheck(samples: int, gap: float) -> int:
    """状态变化识别：倒计时递减是否被正确读出（语义单调性，不依赖标签）。"""
    import time
    from duck_guard import Game, find_game_window
    hwnd = find_game_window()
    if not hwnd:
        print("找不到窗口")
        return 2
    g = Game(hwnd)
    g.pin()
    g.nav("duck")
    seq = []
    stamps = []
    for _ in range(samples):
        t = g.text()
        m = re.search(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*[:：]\s*(\d{2})", t)
        if m:
            h, mi, s = (int(x) for x in m.groups())
            seq.append(h * 3600 + mi * 60 + s)
            stamps.append(time.time())
        time.sleep(gap)
    checks = []
    for i in range(1, len(seq)):
        elapsed = stamps[i] - stamps[i - 1]
        drop = seq[i - 1] - seq[i]
        checks.append(abs(drop - elapsed) <= 4 and drop > 0)
    out = {"samples": len(seq), "readings": seq[:6],
           "deltas": [round(stamps[i] - stamps[i - 1], 1) for i in range(1, len(seq))],
           "monotonic_ok": sum(checks),
           "change_accuracy": round(sum(checks) / max(1, len(checks)), 4)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "state_change.json").write_text(
        json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--label-page", default="", help="人工标注的页面名，用于算 page_accuracy")
    ap.add_argument("--label-ducks", type=int, default=-1, help="人工标注的可见鸭子数")
    ap.add_argument("--out", default="")
    sub = ap.add_subparsers(dest="cmd")
    pc = sub.add_parser("pagecheck")
    pc.add_argument("--rounds", type=int, default=2)
    pc.add_argument("--per-page", type=int, default=5)
    sc = sub.add_parser("statecheck")
    sc.add_argument("--samples", type=int, default=6)
    sc.add_argument("--gap", type=float, default=8.0)
    args = ap.parse_args()
    if args.cmd == "pagecheck":
        return cmd_pagecheck(args.rounds, args.per_page)
    if args.cmd == "statecheck":
        return cmd_statecheck(args.samples, args.gap)

    hwnd = ui_icons.find_window()
    if not hwnd:
        print("找不到小程序窗口")
        return 2

    frames, page_seq, icons_seq = [], [], []
    comp: dict = {}
    timer_seq, queue_seq = [], []
    t0 = time.time()

    for i in range(args.runs):
        bgr = ui_icons.grab(hwnd)
        if bgr is None:
            time.sleep(args.interval)
            continue
        ocr = ocr_bands(hwnd)
        page, conf, ev = classify_page(ocr["text"])
        icons, c = detect_row_icons(bgr)
        comp = {k: comp.get(k, 0) + c[k] for k in c}

        page_seq.append({"page": page, "confidence": conf, "evidence": ev})
        icons_seq.append(icons)
        timer_seq.append(ocr["timer"])
        queue_seq.append(ocr["queue"])
        frames.append({
            "i": i, "page": page, "conf": conf,
            "slots": len(icons),
            "timer": ocr["timer"][:24], "queue": ocr["queue"][:24],
        })
        print(f"[{i+1}/{args.runs}] page={page}({conf}) slots={len(icons)} "
              f"timer={ocr['timer'][:18]!r} queue={ocr['queue'][:18]!r}")
        time.sleep(args.interval)

    n = max(1, len(page_seq))
    hist: dict[str, int] = {}
    for p in page_seq:
        hist[p["page"]] = hist.get(p["page"], 0) + 1

    counts = [len(x) for x in icons_seq]
    timer_changes = sum(1 for a, b in zip(timer_seq, timer_seq[1:]) if a != b)
    queue_changes = sum(1 for a, b in zip(queue_seq, queue_seq[1:]) if a != b)

    report = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "window": hwnd,
        "runs_requested": args.runs,
        "frames": len(page_seq),
        "duration_s": round(time.time() - t0, 1),
        "page": {
            "hist": hist,
            "unknown_rate": round(hist.get("unknown", 0) / n, 4),
            "mean_confidence": round(float(np.mean([p["confidence"] for p in page_seq])), 3) if page_seq else 0,
        },
        "row_icons": {
            "mean": round(float(np.mean(counts)), 2) if counts else 0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "jitter_px": jitter(icons_seq),
        },
        "components": {**comp,
                       "per_frame_slots": round(sum(counts) / max(1, len(counts)), 2),
                       "per_frame_colored": round(comp.get("colored", 0) / max(1, len(counts)), 2)},
        "state": {"timer_changes": timer_changes, "queue_changes": queue_changes},
        "labels": {},
        "frames_detail": frames[-8:],
    }

    if args.label_page:
        ok = sum(1 for p in page_seq if p["page"] == args.label_page)
        report["labels"]["page_accuracy"] = round(ok / n, 4)
        report["labels"]["page_label"] = args.label_page
    if args.label_ducks >= 0:
        report["labels"]["duck_recall"] = round(
            float(np.mean(counts)) / max(1, args.label_ducks), 4) if counts else 0
        report["labels"]["duck_label"] = args.label_ducks

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"duck_perception_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
