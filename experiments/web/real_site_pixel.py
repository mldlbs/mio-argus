"""
real_site_pixel：真站上的**像素端到端**对照（REAL_SITE §5 高优）

结构侧基线（§3.7）：pick 19/32=0.594 → hit|pick 0.895 → click|attempt 0.842；快照 11.9 ms。
像素臂的公平性设计（避免 P4 式信息不对称）：
  · 感知只有 `page.screenshot()`（1000×700 视口）——不看 DOM/可访问性树；
  · 两条选择通道，都用**手工先验**（与 struct 侧的关键词先验对称）：
      - `ocr-kw`：rapidocr 整屏读文 → 给候选配名 → **复用同一个关键词规则**（主臂）；
      - `geom`  ：经典 cv2 候选框（边缘+连通域）→ 逐任务几何先验（副臂）；
  · 动作是坐标真点击；评估用 DOM oracle（只判分、不参与任务）。
  · **指标按视口内 oracle 条件化**：像素看不清视口外（这是通道的硬约束，单列报告）。

用法:
    python experiments/web/real_site_pixel.py --quick
    python experiments/web/real_site_pixel.py
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

from real_site_agent import SITES, TASKS, oracle_pick, keyword_pick
from real_site_reach import SNAPSHOT_JS_V3, HIT_JS, snapshot_v3, flat_table

W, H = 1000, 700
MAX_BOXES = 160
_OCR = None


# ---------------- 经典 cv2 候选框 ----------------
def detect_boxes(png_bytes):
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    boxes = []
    # (a) 边缘轮廓
    edges = cv2.dilate(cv2.Canny(gray, 60, 160), np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 18 <= w and 12 <= h and 220 <= w * h <= 0.85 * W * H and w / h <= 25 and h / w <= 12:
            boxes.append((x, y, w, h))
    # (b) 非白连通域（实心块 / 细文本 / 大轮廓环）
    mask = (gray < 246).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 16 or h < 10 or area < 180 or w * h > 0.85 * W * H:
            continue
        if w / h > 30 or h / w > 15:
            continue
        fill = area / (w * h)
        if fill >= 0.25 or h <= 34 or (w * h >= 3000 and w >= 80 and h >= 24):
            boxes.append((x, y, w, h))
    # 去重（IoU>0.85 合并）按面积降序截断
    boxes.sort(key=lambda b: -b[2] * b[3])
    kept = []
    for b in boxes:
        dup = False
        for k in kept:
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            if ix * iy > 0.85 * min(b[2] * b[3], k[2] * k[3]):
                dup = True
                break
        if not dup:
            kept.append(b)
        if len(kept) >= MAX_BOXES:
            break
    return kept, (time.perf_counter() - t0) * 1000


def geom_pick(boxes, task):
    if not boxes:
        return -1
    cxs = [b[0] + b[2] / 2 for b in boxes]
    if task == "search":
        cand = [i for i, b in enumerate(boxes)
                if b[1] < 0.22 * H and b[2] >= 0.15 * W and b[2] / b[3] >= 4]
        return max(cand, key=lambda i: (boxes[i][2], -boxes[i][1])) if cand else -1
    if task == "login":
        cand = [i for i, b in enumerate(boxes)
                if b[1] < 0.14 * H and cxs[i] > 0.65 * W and b[2] <= 0.24 * W]
        return min(cand, key=lambda i: (boxes[i][0], boxes[i][1])) if cand else -1
    if task == "home":
        cand = [i for i, b in enumerate(boxes)
                if b[1] < 0.14 * H and 0.10 * W < cxs[i] < 0.55 * W
                and b[2] <= 0.24 * W and b[3] <= 0.07 * H]
        return min(cand, key=lambda i: (boxes[i][1], boxes[i][0])) if cand else -1
    cand = [i for i, b in enumerate(boxes)
            if b[1] < 0.14 * H and cxs[i] < 0.30 * W and 0.4 <= b[2] / b[3] <= 3.0]
    return min(cand, key=lambda i: (boxes[i][1], boxes[i][0])) if cand else -1


# ---------------- OCR 臂 ----------------
def ocr_texts(png_bytes):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    t0 = time.perf_counter()
    res = _OCR(img)
    ms = (time.perf_counter() - t0) * 1000
    out = []
    if res and res[0]:
        for box, txt, score in res[0]:
            pts = np.array(box)
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            out.append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, text=str(txt)))
    return out, ms


def ocr_pick(texts, task):
    """关键词打分（与 struct 侧 keyword_pick 同规则）；search 给"宽而扁"的文本框加分。"""
    kw = TASKS[task]["kw"]
    best, bs = -1, 0.0
    for i, t in enumerate(texts):
        s = sum(1.0 for k in kw if k in t["text"].lower())
        if task == "search" and t["w"] >= 120 and t["h"] <= 60:
            s += 0.5
        if s > bs:
            best, bs = i, s
    return best if bs > 0 else -1


def overlap(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def center_in(box, outer):
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    return (outer["x"] <= cx <= outer["x"] + outer["w"] and
            outer["y"] <= cy <= outer["y"] + outer["h"])


def as_box(t):
    return dict(x=t[0], y=t[1], w=t[2], h=t[3])


def eval_site(b, name, url):
    """每个 (site, task) 独立装载：截图/检测/OCR/快照/命中/点击都在同一页面状态下（与 struct 侧协议一致）。"""
    rec = dict(site=name, url=url, load_ok=False, tasks=[])
    for task in TASKS:
        pg = b.new_page(viewport={"width": W, "height": H})
        try:
            pg.goto(url, wait_until="load", timeout=30000)
            pg.wait_for_timeout(1500)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:80]}"
            pg.close()
            return rec
        rec["load_ok"] = True
        shot = pg.screenshot()
        boxes, det_ms = detect_boxes(shot)
        texts, ocr_ms = ocr_texts(shot)
        # OCR 期间页面可能漂移（像素臂的"过期坐标"问题）→ 再抓一帧量漂移
        shot2 = pg.screenshot()
        a = cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_GRAYSCALE)
        b2 = cv2.imdecode(np.frombuffer(shot2, np.uint8), cv2.IMREAD_GRAYSCALE)
        diff = float(np.abs(a.astype(np.int16) - b2.astype(np.int16)).mean()) if a.shape == b2.shape else 255.0
        v3, _, snap_ms = snapshot_v3(pg)
        tab3 = flat_table(v3)
        rec.update(n_boxes=len(boxes), det_ms=det_ms, n_ocr=len(texts), ocr_ms=ocr_ms,
                   n_struct=len(v3), snap_ms=snap_ms, drift=diff)

        oi = oracle_pick(tab3, task)
        row = dict(task=task, oracle=oi >= 0, _drift=bool(diff > 1.5))
        if oi < 0:
            rec["tasks"].append(row)
            pg.close()
            continue
        e = v3[oi]
        oracle = dict(x=e["px"], y=e["py"], w=e["w"], h=e["h"])
        row["oracle_in_view"] = (0 <= oracle["x"] and 0 <= oracle["y"]
                                 and oracle["x"] + oracle["w"] <= W
                                 and oracle["y"] + oracle["h"] <= H)
        row["oracle"] = dict(x=round(oracle["x"]), y=round(oracle["y"]),
                             w=round(oracle["w"]), h=round(oracle["h"]))
        row["det"] = any(overlap(as_box(b), oracle) >= 0.4 or center_in(as_box(b), oracle)
                         for b in boxes)
        row["ocr_named"] = any(center_in(t, oracle) for t in texts)
        rid = e["rid"]
        picked = {}
        # 先测两臂的 pick / 命中（**不点击**，避免先点的一臂污染后一臂）
        for arm in ("geom", "ocr"):
            if arm == "geom":
                pi = geom_pick(boxes, task)
                pbox = as_box(boxes[pi]) if pi >= 0 else None
            else:
                pi = ocr_pick(texts, task)
                pbox = dict(texts[pi]) if pi >= 0 else None
            row[f"{arm}_pick"] = pi >= 0
            if pbox is None:
                continue
            picked[arm] = pbox
            row[f"{arm}_is_oracle"] = bool(overlap(pbox, oracle) >= 0.4
                                           or center_in(pbox, oracle)
                                           or center_in(oracle, pbox))
            cx, cy = int(pbox["x"] + pbox["w"] / 2), int(pbox["y"] + pbox["h"] / 2)
            ht = None
            for _ in range(2):
                try:
                    ht = pg.evaluate(HIT_JS, [cx, cy, rid])
                    break
                except Exception:
                    time.sleep(0.4)
            if ht is None:
                ht = dict(hit=False, occluder="eval-error")
            row[f"{arm}_hit"] = bool(ht.get("hit"))
            row[f"{arm}_occluder"] = ht.get("occluder", "")
        # 只点主臂（ocr）；若 OCR 没选到则不点
        pbox = picked.get("ocr")
        if pbox is not None:
            cx, cy = int(pbox["x"] + pbox["w"] / 2), int(pbox["y"] + pbox["h"] / 2)
            before = pg.url
            try:
                pg.mouse.click(cx, cy)
                pg.wait_for_timeout(600)
                row["ocr_clicked"] = True
                row["ocr_url_changed"] = pg.url != before
            except Exception:
                row["ocr_clicked"] = False
                row["ocr_url_changed"] = False
        rec["tasks"].append(row)
        pg.close()
    return rec


def summarize(records):
    loaded = [r for r in records if r["load_ok"]]
    allt = [t for r in loaded for t in r["tasks"] if t.get("oracle")]
    inv = [t for t in allt if t.get("oracle_in_view")]
    s = dict(sites=len(loaded), tasks_oracle=len(allt), in_view=len(inv),
             offscreen=len(allt) - len(inv))
    s["det_in_view"] = sum(1 for t in inv if t.get("det"))
    s["ocr_named_in_view"] = sum(1 for t in inv if t.get("ocr_named"))
    for arm in ("geom", "ocr"):
        pick = [t for t in inv if t.get(f"{arm}_pick")]
        clean = [t for t in pick if not t.get("_drift")]
        hits = sum(1 for t in pick if t.get(f"{arm}_hit"))
        hits_clean = sum(1 for t in clean if t.get(f"{arm}_hit"))
        url = sum(1 for t in pick if t.get(f"{arm}_url_changed"))
        s[arm] = dict(pick=len(pick), pick_rate=len(pick) / max(1, len(inv)),
                      pick_is_oracle=sum(1 for t in pick if t.get(f"{arm}_is_oracle")),
                      hit=hits, hit_rate=hits / max(1, len(pick)),
                      hit_clean=hits_clean, hit_clean_rate=hits_clean / max(1, len(clean)),
                      n_clean=len(clean),
                      url_changed=url, url_rate=url / max(1, len(pick)))
    drifts = [r.get("drift", 0.0) for r in loaded if r.get("drift") is not None]
    s["drift_median"] = sorted(drifts)[len(drifts) // 2] if drifts else None
    s["drift_tasks"] = sum(1 for t in allt if t.get("_drift"))
    for k in ("det_ms", "ocr_ms", "snap_ms", "n_boxes", "n_ocr"):
        xs = sorted(r.get(k) or 0 for r in loaded)
        s[f"{k}_median"] = xs[len(xs) // 2] if xs else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sites = dict(SITES)
    if args.quick:
        sites = dict(list(sites.items())[:3])

    records = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, url in sites.items():
            r = eval_site(b, name, url)
            records.append(r)
            print(f"[{name}] load={'ok' if r['load_ok'] else 'FAIL'} "
                  f"boxes={r.get('n_boxes')} det={r.get('det_ms', 0):.0f}ms "
                  f"n_ocr={r.get('n_ocr')} ocr={r.get('ocr_ms', 0):.0f}ms "
                  f"struct={r.get('n_struct')} snap={r.get('snap_ms', 0):.0f}ms", flush=True)
        b.close()

    s = summarize(records)
    print(f"\n汇总（{s['sites']} 站，oracle {s['tasks_oracle']}；视口内 {s['in_view']}，"
          f"视口外 {s['offscreen']}）")
    print(f"  像素检测覆盖（视口内）：{s['det_in_view']}/{s['in_view']}")
    print(f"  OCR 读出目标文本（视口内）：{s['ocr_named_in_view']}/{s['in_view']}")
    for arm in ("geom", "ocr"):
        a = s[arm]
        print(f"  [{arm:4}] pick {a['pick']}/{s['in_view']} = {a['pick_rate']:.3f}"
              f"（是 oracle {a['pick_is_oracle']}）"
              f" → hit {a['hit']}/{a['pick']} = {a['hit_rate']:.3f}"
              f"（无漂移子集 {a['hit_clean']}/{a['n_clean']} = {a['hit_clean_rate']:.3f}）"
              f" → url 变化 {a['url_changed']}/{a['pick']} = {a['url_rate']:.3f}")
    print(f"  漂移：中位 {s['drift_median']:.2f}（灰度均差），漂移任务 {s['drift_tasks']}/{s['tasks_oracle']}")
    print(f"  成本：检测 {s['det_ms_median']:.0f} ms / OCR {s['ocr_ms_median']:.0f} ms "
          f"/ 结构快照 {s['snap_ms_median']:.0f} ms（框 {s['n_boxes_median']} / 文本 "
          f"{s['n_ocr_median']} 条）")
    Path("results/web").mkdir(parents=True, exist_ok=True)
    Path("results/web/real_site_pixel_results.json").write_text(
        json.dumps(dict(records=records, summary=s), indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 results/web/real_site_pixel_results.json")


if __name__ == "__main__":
    main()
