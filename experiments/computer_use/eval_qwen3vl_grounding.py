"""qwen3-vl:8b 本地 grounding 评测（216 条冻结清单，零成本，只用时间）。

判据与 `audit_eval.py` 一致：预测点落在 `rect_img` 内即命中。
输出逐条预测，支持断点续跑（已完成的 id 跳过）。

用法：
    python experiments/computer_use/eval_qwen3vl_grounding.py --limit 5     # 冒烟
    python experiments/computer_use/eval_qwen3vl_grounding.py               # 全量
"""
import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts/computer_use_inloop/audit_manifest.json"
URL = "http://127.0.0.1:11434/api/generate"
NUM = re.compile(r"-?\d+(?:\.\d+)?")

PROMPT = (
    "You are looking at a screenshot of a desktop application.\n"
    "Instruction: {ins}\n"
    "Return the CENTER of the UI element to act on, as pixel coordinates in THIS image "
    "({w}x{h}, origin = top-left).\n"
    "Answer with exactly two numbers separated by a comma: x, y"
)


def ask(img_bytes, ins, w, h, model, ctx, predict, timeout):
    body = {
        "model": model,
        "prompt": PROMPT.format(ins=ins, w=w, h=h),
        "images": [base64.b64encode(img_bytes).decode()],
        "stream": False,
        "think": False,
        "options": {"num_predict": predict, "temperature": 0, "num_ctx": ctx},
    }
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError:
        body.pop("think", None)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    return d, time.time() - t0


def parse_xy(text, w, h):
    nums = [float(v) for v in NUM.findall(text or "")]
    if len(nums) < 2:
        return None
    x, y = nums[0], nums[1]
    if not (0 <= x <= w and 0 <= y <= h):
        # 偶尔给出归一化值（0..1）
        if 0 <= x <= 1 and 0 <= y <= 1:
            x, y = x * w, y * h
        else:
            x, y = min(max(x, 0), w), min(max(y, 0), h)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--predict", type=int, default=96)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="results/computer_use/qwen3vl_grounding.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"][:args.limit] if args.limit else man["rows"]
    out_path = ROOT / args.out

    done = {}
    if args.resume and out_path.exists():
        for r in json.loads(out_path.read_text(encoding="utf-8"))["rows"]:
            done[r["id"]] = r
        print(f"[resume] 已有 {len(done)} 条")

    results = []
    t_start = time.time()
    for i, row in enumerate(rows):
        if row["id"] in done:
            results.append(done[row["id"]])
            continue
        img = Path(row["image"])
        w, h = row["image_size"]
        d, wall = ask(img.read_bytes(), row["instruction"], w, h,
                      args.model, args.ctx, args.predict, args.timeout)
        xy = parse_xy(d.get("response"), w, h)
        x1, y1, x2, y2 = row["rect_img"]
        ok = bool(xy and x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2)
        rec = {"id": row["id"], "instruction": row["instruction"],
               "xy": [round(xy[0], 1), round(xy[1], 1)] if xy else None,
               "rect_img": [round(v, 1) for v in row["rect_img"]],
               "hit": ok, "wall_s": round(wall, 1),
               "img_tok": d.get("prompt_eval_count"), "out_tok": d.get("eval_count"),
               "raw": (d.get("response") or "").strip()[:120]}
        results.append(rec)
        if (i + 1) % 10 == 0 or args.limit:
            hits = sum(r["hit"] for r in results)
            print(f"  [{i+1}/{len(rows)}] hit={hits}/{len(results)} "
                  f"({hits/max(len(results),1):.3f}) last={rec['wall_s']}s", flush=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"model": args.model, "n": len(results), "rows": results},
                ensure_ascii=False, indent=1), encoding="utf-8")

    hits = sum(r["hit"] for r in results)
    walls = [r["wall_s"] for r in results if r.get("wall_s")]
    summary = {
        "model": args.model, "n": len(results), "hits": hits,
        "acc": hits / max(len(results), 1),
        "median_wall_s": float(np.median(walls)) if walls else None,
        "total_min": round((time.time() - t_start) / 60, 1),
        "unparsed": sum(1 for r in results if r["xy"] is None),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": results},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[out] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
