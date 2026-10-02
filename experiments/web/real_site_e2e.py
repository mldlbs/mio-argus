"""
real_site_e2e：真站像素端到端对照（struct vs pixel）

REAL_SITE §5 高优（原 REPORT §12 "WinSim → 真实站点" 的真站通道版）：
同一 6 站 × 4 任务、同一关键词选择策略（keyword_pick），只换感知通道：

  - struct 臂：v3 快照（DOM/无障碍名）→ keyword_pick → prepare/hit/locator 点击；
  - pixel 臂：截图 → RapidOCR 文本框 ∪ cv2 几何框 → 同一 keyword_pick
    （tag 由几何猜：宽矮 → input）→ 页坐标 mouse.click。选前不碰 DOM
    （代码顺序保证：截图/OCR/选中完成后才做快照，且快照只用于评分）。

公平性纪律（REPORT §12，P4 教训）：
  - 两臂共用同一 oracle 目标（DOM oracle_pick），oracle 只评分、绝不进像素选择；
  - 选择策略、关键词、评分标准两臂一致；像素臂不给任务特判；
  - engage 语义对称：点击前"最上层元素属于 oracle"（struct=hit 测试，pixel=同判据）。

指标（site × task；oracle≥0 为有效试次 n_valid）：
  found     struct: 感知里可见 oracle 目标（=oracle_pick≥0，对齐 §3.7 口径）
            pixel:  检测框覆盖 oracle 矩形（在 n_valid 上计）
  pick      关键词选中
  correct   选中者 == oracle（struct: rid 相同；pixel: 框与 oracle 矩形相交）
  hit / click_ok（struct 专属机制阶段）
  engage    端到端成功 = correct ∧ 点击前最上层属于 oracle
  url_changed 点击效果（次级）

用法:
    python experiments/web/real_site_e2e.py --quick
    python experiments/web/real_site_e2e.py
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright, Error as PWError
from rapidocr_onnxruntime import RapidOCR

from real_site_agent import SITES, TASKS, keyword_pick, oracle_pick
from real_site_reach import (snapshot_v3, flat_table, prepare_target,
                             hit_test, attempt_click)

VIEW = dict(width=1000, height=700)
OCR = RapidOCR()

# 点击点处最上层元素是否属于 oracle（顶层 frame，评分用）
ENGAGE_JS = r"""
([x, y, rid]) => {
  const o = document.querySelector('[data-reach-id="' + rid + '"]');
  if (!o) return {ok: false, why: 'no-oracle'};
  const or = o.getBoundingClientRect();
  const el = document.elementFromPoint(x, y);
  if (!el) return {ok: false, why: 'none'};
  const r = el.getBoundingClientRect();
  const inOrc = (v, lo, hi) => v >= lo - 2 && v <= hi + 2;
  const leafInside = inOrc(r.x, or.x, or.x + or.width) &&
                     inOrc(r.y, or.y, or.y + or.height) &&
                     inOrc(r.x + r.width, or.x, or.x + or.width) &&
                     inOrc(r.y + r.height, or.y, or.y + or.height);
  const leaf = (el.closest('[data-reach-id]') || {}).getAttribute
               ? (el.closest('[data-reach-id]').getAttribute('data-reach-id'))
               : '';
  return {ok: leaf === rid || leafInside, leaf: leaf || el.tagName.toLowerCase()};
}
"""


def detect_pixel(png):
    """像素感知：OCR 文本框（命名候选）∪ cv2 几何框（仅覆盖度/评分）。"""
    t0 = time.perf_counter()
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    cands, all_boxes, n_cv = [], [], 0
    t1 = time.perf_counter()
    ocr_res, _ = OCR(arr)
    ocr_ms = (time.perf_counter() - t1) * 1000
    for it in (ocr_res or []):
        pts, text, score = it
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        text = (text or "").strip()
        if w < 4 or h < 4 or not text or score < 0.5:
            continue
        tag = "input" if (h <= 40 and w >= 60) else (
            "button" if h <= 40 else "text")
        cands.append(dict(name=text[:60], tag=tag, role="", is_logo=False,
                          href="", box=[x, y, w, h], src="ocr"))
        all_boxes.append([x, y, w, h])
    t2 = time.perf_counter()
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 60, 160),
                       np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 60 <= w * h <= 24000 and 0.1 <= w / max(h, 1) <= 12 \
                and w <= 950 and h <= 250:
            all_boxes.append([x, y, w, h])
            n_cv += 1
    cv_ms = (time.perf_counter() - t2) * 1000
    return cands, all_boxes, dict(total_ms=(time.perf_counter() - t0) * 1000,
                                  ocr_ms=ocr_ms, cv_ms=cv_ms,
                                  n_ocr=len(cands), n_cv=n_cv)


def rect_overlap(a, b):
    """中心互含或 IoU≥0.25（a/b 为 [x,y,w,h]）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return False
    if (bx <= ax + aw / 2 <= bx + bw and by <= ay + ah / 2 <= by + bh) or \
       (ax <= bx + bw / 2 <= ax + aw and ay <= by + bh / 2 <= ay + ah):
        return True
    return inter / min(aw * ah, bw * bh) >= 0.25


def run_struct(b, site, url, task):
    """结构通道端到端：v3 → oracle(评分) → keyword_pick → hit → click。"""
    row = dict(arm="struct", site=site, task=task, load_ok=False)
    pg = b.new_page(viewport=VIEW)
    try:
        pg.goto(url, wait_until="load", timeout=30000)
        pg.wait_for_timeout(1200)
        row["load_ok"] = True
        v3, _, snap_ms = snapshot_v3(pg)
        tab = flat_table(v3)
        oi = oracle_pick(tab, task)
        ki = keyword_pick(tab, task)
        row.update(snap_ms=snap_ms, found=oi >= 0, valid=oi >= 0,
                   pick=ki >= 0, n_v3=len(v3))
        if oi >= 0:
            orc = v3[oi]
            row["oracle_rid"] = orc["rid"]
            row["correct"] = ki >= 0 and v3[ki]["rid"] == orc["rid"]
        else:
            row["correct"] = False
        if ki >= 0:
            e = prepare_target(v3[ki], pg.frames)
            ht = hit_test(e, pg.frames)
            before = pg.url
            ck = attempt_click(e, pg.frames)
            row.update(hit=ht["hit"], occluder=ht["occluder"],
                       attempted=True, click_ok=ck["click_ok"],
                       click_err=ck["click_err"], url_changed=pg.url != before)
            row["engage"] = bool(row.get("correct") and ht["hit"]
                                 and ck["click_ok"])
            row["engage_any"] = bool(ht["hit"] and ck["click_ok"])
        else:
            row.update(hit=False, attempted=False, click_ok=False,
                       engage=False, engage_any=False, url_changed=False)
        if row["valid"]:
            row["reason"] = ("ok" if row["engage"]
                             else ("no_pick" if not row["pick"]
                                   else ("wrong_target"
                                         if not row.get("correct")
                                         else ("occluded"
                                               if not row.get("hit")
                                               else "click_timeout"))))
    except Exception as e2:
        row["error"] = f"{type(e2).__name__}: {str(e2)[:120]}"
    pg.close()
    return row


def run_pixel(b, site, url, task):
    """像素通道端到端：截图 → OCR/cv2 → keyword_pick（不碰 DOM）→
    鼠标点击；点击后才做快照取 oracle（仅评分）。"""
    row = dict(arm="pixel", site=site, task=task, load_ok=False)
    pg = b.new_page(viewport=VIEW)
    try:
        pg.goto(url, wait_until="load", timeout=30000)
        pg.wait_for_timeout(1200)
        row["load_ok"] = True
        png = pg.screenshot()
        cands, boxes, det = detect_pixel(png)
        row.update(det, n_cand=len(cands))
        ki = keyword_pick(cands, task)
        row["pick"] = ki >= 0
        clicked_box = None
        if ki >= 0:
            clicked_box = cands[ki]["box"]
            row["picked_name"] = cands[ki]["name"][:40]
        # —— 评分基准：此后才碰 DOM（oracle 只评分）——
        v3, _, snap_ms = snapshot_v3(pg)
        tab = flat_table(v3)
        oi = oracle_pick(tab, task)
        row.update(snap_ms=snap_ms, found=oi >= 0, valid=oi >= 0)
        if oi >= 0:
            orc = v3[oi]
            orc_rect = [orc["px"], orc["py"], orc["w"], orc["h"]]
            row["oracle_rid"] = orc["rid"]
            row["found"] = any(rect_overlap(bx, orc_rect) for bx in boxes)
            row["correct"] = bool(clicked_box
                                  and rect_overlap(clicked_box, orc_rect))
            if clicked_box:
                cx = clicked_box[0] + clicked_box[2] / 2
                cy = clicked_box[1] + clicked_box[3] / 2
                # 点击前最上层判定（对称于 struct 的 hit 测试）
                if orc["frame_idx"] == 0:
                    top = pg.evaluate(ENGAGE_JS, [cx, cy, orc["rid"]])
                else:
                    top = dict(ok=rect_overlap([cx - 1, cy - 1, 2, 2],
                                               orc_rect), leaf="iframe")
                row["top_ok"] = bool(top.get("ok"))
                row["top_leaf"] = top.get("leaf", "")
                before = pg.url
                try:
                    pg.mouse.click(cx, cy)
                    row["click_ok"] = True
                    row["click_err"] = ""
                except PWError as e3:
                    row["click_ok"] = False
                    row["click_err"] = str(e3).split("\n")[0][:80]
                pg.wait_for_timeout(400)
                row["url_changed"] = pg.url != before
            else:
                row.update(top_ok=False, click_ok=False, url_changed=False)
            row["engage"] = bool(row["correct"] and row.get("top_ok")
                                 and row.get("click_ok"))
            row["engage_any"] = bool(row.get("top_ok")
                                     and row.get("click_ok"))
        else:
            row.update(found=False, correct=False, engage=False,
                       engage_any=False, click_ok=False, url_changed=False)
        if row["valid"]:
            row["reason"] = ("ok" if row["engage"]
                             else ("no_pick" if not row["pick"]
                                   else ("wrong_target"
                                         if not row.get("correct")
                                         else ("not_visible"
                                               if not row.get("found")
                                               else "occluded"))))
    except Exception as e2:
        row["error"] = f"{type(e2).__name__}: {str(e2)[:120]}"
    pg.close()
    return row


def summarize(rows):
    def rate(xs):
        return sum(xs) / max(1, len(xs))
    out = {}
    print(f"\n{'site':<10}{'task':<7} | struct f/p/c/h/k/e | "
          f"pixel f/p/c/k/e | reasons")
    print("-" * 96)
    by = {}
    for r in rows:
        by.setdefault((r["site"], r["task"]), {})[r["arm"]] = r
    for (site, task), ab in by.items():
        s, p = ab.get("struct", {}), ab.get("pixel", {})
        def g(d, k):
            v = d.get(k)
            return "-" if v is None else ("1" if v else "0")
        print(f"{site:<10}{task:<7} | "
              f"{g(s,'found')}{g(s,'pick')}{g(s,'correct')}"
              f"{g(s,'hit')}{g(s,'click_ok')}{g(s,'engage')}      | "
              f"{g(p,'found')}{g(p,'pick')}{g(p,'correct')}"
              f"{g(p,'engage')}   | "
              f"s:{s.get('reason','?')} p:{p.get('reason','?')}")
    for arm in ("struct", "pixel"):
        rs = [r for r in rows if r["arm"] == arm and r["load_ok"]]
        valid = [r for r in rs if r.get("valid")]
        d = dict(
            n=len(rs), n_valid=len(valid),
            found_all=rate([r.get("found", False) for r in rs]),
            pick=rate([r.get("pick", False) for r in valid]),
            correct=rate([r.get("correct", False) for r in valid]),
            engage=rate([r.get("engage", False) for r in valid]),
            engage_any=rate([r.get("engage_any", False) for r in valid]),
            click_ok=rate([r.get("click_ok", False)
                           for r in valid if r.get("pick")]),
            url_changed=rate([r.get("url_changed", False)
                              for r in valid if r.get("pick")]),
        )
        if arm == "struct":
            d["found_valid"] = rate([r.get("found", False) for r in valid])
            d["hit"] = rate([r.get("hit", False)
                             for r in valid if r.get("pick")])
        else:
            d["found_valid"] = rate([r.get("found", False) for r in valid])
            d["ocr_ms"] = float(np.median([r["ocr_ms"] for r in valid])) \
                if valid else None
        reasons = {}
        for r in valid:
            reasons[r.get("reason", "?")] = reasons.get(
                r.get("reason", "?"), 0) + 1
        d["reasons"] = reasons
        out[arm] = d
        print(f"\n[{arm}] n={d['n']} valid={d['n_valid']}  "
              f"found {d['found_all']:.3f} → pick {d['pick']:.3f} → "
              f"correct {d['correct']:.3f} → engage {d['engage']:.3f} "
              f"(engage_any {d['engage_any']:.3f})  reasons={reasons}")
    if out.get("struct") and out.get("pixel"):
        print(f"\n端到端差（struct − pixel，valid 上 engage）: "
              f"{out['struct']['engage'] - out['pixel']['engage']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sites = dict(SITES)
    if args.quick:
        sites = dict(list(sites.items())[:3])

    rows = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for site, url in sites.items():
            for task in TASKS:
                rows.append(run_struct(b, site, url, task))
                rows.append(run_pixel(b, site, url, task))
                s, p = rows[-2], rows[-1]
                print(f"[{site}/{task}] struct:"
                      f"{'ok' if s.get('load_ok') else 'FAIL'}"
                      f" e={int(bool(s.get('engage')))} | pixel:"
                      f"{'ok' if p.get('load_ok') else 'FAIL'}"
                      f" e={int(bool(p.get('engage')))} "
                      f"ocr={p.get('ocr_ms', 0):.0f}ms")
        b.close()

    summary = summarize(rows)
    suffix = "_quick" if args.quick else ""
    out = Path(f"results/web/real_site_e2e_results{suffix}.json")
    out.write_text(json.dumps(dict(rows=rows, summary=summary),
                              indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
