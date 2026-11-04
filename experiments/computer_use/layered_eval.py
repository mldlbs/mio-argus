"""
阶段2：分层视觉识别消融（不训练，直接用现成 2B grounder）

变体（同一批 216 条冻结样本、同一命中判据）：
  A1 整屏输入          直接在全图上预测
  A3 粗定位+裁剪       用 A1 的点为中心裁 S×S，再预测，映射回原图
  A4 两级放大          A3 之后再以 A3 的点为中心裁更小区域预测
  A5 重叠切片          3×3 重叠切片各自预测，映射回原图后做一致性聚类

坐标一律归一化 [0,1]（相对输入图），由本脚本统一映射与判命中，provider 内不做换算。
必须在隔离 venv 里跑（transformers 4.51.3 + gui_actor）：
  set PYTHONPATH=<GUI-Actor-main>/src
  artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A3
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from gui_actor.inference import inference

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "computer_use_inloop"
MANIFEST = ART / "audit_manifest.json"
PRED = ART / "audit_predictions"

SYS_MSG = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
           "You need to perform a series of pyautogui actions to complete the task.")


class Grounder:
    def __init__(self, name="microsoft/GUI-Actor-2B-Qwen2-VL", attn="sdpa"):
        self.proc = AutoProcessor.from_pretrained(name)
        self.tok = self.proc.tokenizer
        self.model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=attn).eval()
        self.vram = int(torch.cuda.max_memory_allocated() / 1e6)

    def predict(self, img, instruction, topk=1):
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYS_MSG}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": instruction}]},
        ]
        t0 = time.time()
        out = inference(conv, self.model, self.tok, self.proc,
                        use_placeholder=True, topk=topk)
        pts = out.get("topk_points") or []
        ms = (time.time() - t0) * 1000
        return [[float(x), float(y)] for x, y in pts], ms


def crop(img, cx, cy, size):
    W, H = img.size
    half = size // 2
    x0 = max(0, min(W - 1, int(cx) - half))
    y0 = max(0, min(H - 1, int(cy) - half))
    x1 = min(W, x0 + size)
    y1 = min(H, y0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    return img.crop((x0, y0, x1, y1)), (x0, y0, x1 - x0, y1 - y0)


def to_full(pt, box, img_size):
    x0, y0, w, h = box
    return [(x0 + pt[0] * w) / img_size[0], (y0 + pt[1] * h) / img_size[1]]


def cluster_pick(pts, tol_frac=0.02):
    """重叠切片的一致性聚类：选邻居最多的那个点。"""
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    best, best_n = pts[0], -1
    for p in pts:
        n = sum(1 for q in pts if math.dist(p, q) < tol_frac)
        if n > best_n:
            best, best_n = p, n
    return best


def run(g, rows, variant, crop1=512, crop2=256, grid=3, overlap=0.5, upscale=0):
    out = []
    for i, r in enumerate(rows):
        img = Image.open(ROOT / r["image"]).convert("RGB")
        W, H = img.size
        t0 = time.time()
        pts, lat = g.predict(img, r["instruction"])
        p1 = pts[0] if pts else None

        def zoom(c):
            return c.resize((upscale, upscale), Image.LANCZOS) if upscale else c

        if variant == "A1" or p1 is None:
            final = p1
        elif variant == "A3":
            cx, cy = p1[0] * W, p1[1] * H
            c, box = crop(img, cx, cy, crop1)
            q, _ = g.predict(zoom(c), r["instruction"])
            final = to_full(q[0], box, (W, H)) if q else p1
        elif variant == "A4":
            cx, cy = p1[0] * W, p1[1] * H
            c1, b1 = crop(img, cx, cy, crop1)
            q1, _ = g.predict(zoom(c1), r["instruction"])
            if not q1:
                final = p1
            else:
                p3 = to_full(q1[0], b1, (W, H))
                cx2, cy2 = p3[0] * W, p3[1] * H
                c2, b2 = crop(img, cx2, cy2, crop2)
                q2, _ = g.predict(zoom(c2), r["instruction"])
                final = to_full(q2[0], b2, (W, H)) if q2 else p3
        elif variant == "A5":
            step = 1.0 / (grid - overlap * (grid - 1)) if grid > 1 else 1.0
            tw, th = W * step, H * step
            raw = []
            for gy in range(grid):
                for gx in range(grid):
                    x0 = min(W - tw, gx * tw * (1 - overlap))
                    y0 = min(H - th, gy * th * (1 - overlap))
                    box = (int(x0), int(y0), int(tw), int(th))
                    c = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
                    c = zoom(c)
                    q, _ = g.predict(c, r["instruction"])
                    if q:
                        raw.append(to_full(q[0], box, (W, H)))
            final = cluster_pick(raw) or p1
        else:
            final = p1

        L, T, R, B = r["rect_img"]
        hit = bool(final) and (L <= final[0] * W <= R and T <= final[1] * H <= B)
        dist = None
        if final:
            cx, cy = (L + R) / 2, (T + B) / 2
            dist = math.dist((final[0] * W, final[1] * H), (cx, cy))
        out.append({"id": r["id"], "variant": variant, "hit": hit,
                    "px": [round(final[0] * W, 1), round(final[1] * H, 1)] if final else None,
                    "dist": round(dist, 1) if dist is not None else None,
                    "latency_ms": round((time.time() - t0) * 1000, 1)})
        if (i + 1) % 20 == 0:
            print(f"  {variant} {i + 1}/{len(rows)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A1", choices=["A1", "A3", "A4", "A5"])
    ap.add_argument("--crop1", type=int, default=512)
    ap.add_argument("--crop2", type=int, default=256)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--upscale", type=int, default=0, help="裁剪后放大到该边长（真正的 zoom）")
    args = ap.parse_args()

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"][:args.limit] if args.limit else man["rows"]
    print(f"[data] n={len(rows)} variant={args.variant}")

    g = Grounder()
    print(f"[model] loaded vram≈{g.vram}MB")
    res = run(g, rows, args.variant, args.crop1, args.crop2, args.grid,
              upscale=args.upscale)
    n = len(res)
    hit = sum(1 for r in res if r["hit"])
    print(f"[{args.variant}] top1={hit}/{n}={hit / n:.3f} "
          f"med_dist={sorted(r['dist'] for r in res if r['dist'] is not None)[n // 2]:.0f}px")

    PRED.mkdir(parents=True, exist_ok=True)
    p = PRED / f"layered_{args.variant}.json"
    p.write_text(json.dumps({"provider": f"gui_actor_{args.variant}",
                             "meta": {"variant": args.variant, "crop1": args.crop1,
                                      "crop2": args.crop2, "grid": args.grid,
                                      "n": n, "topk": 1},
                             "rows": res}, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
