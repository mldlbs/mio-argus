"""
W30-B：免训练两阶段（zoom）评测

背景：单阶段所有 head 都停在 5.6% 的平凡基线。文献给的判据是"目标要占 ~3 个视觉 token"
（AdaGround 2026），我们 224 输入下 32px 按钮只占 0.17 token。这里做最小验证：

  1) 粗定位：现有模型在全屏上给一个点（中位误差 84px）
  2) 以该点为中心裁 S×S，放大到 crop_img
  3) 二次定位（同一模型，不训练），把点映射回原图
  4) 量命中率

同时跑 oracle 裁剪（以真值点为中心）作为**二次定位能力上界**：
  - oracle 高、预测裁剪低 → 瓶颈在粗定位
  - 两者都低 → 二次定位本身不行（放大后仍是 OOD）

用法:
    python experiments/computer_use/zoom_eval.py --ckpt artifacts/checkpoints/w30_grid.pth \
        --head grid --crop 400 --crop-img 768
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from train_inloop import DATA_ROOT, DemoDataset, load_demos  # noqa: E402
from mio_argus import ACTION_MAP, ComputerUseModelV3, get_image_processor, get_tokenizer  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "zoom_eval.json"


def build_model(head, grid, ckpt, device):
    if head == "grid":
        from grid_model import GridModel
        m = GridModel(num_actions=len(ACTION_MAP), grid=grid)
    elif head == "spatial":
        from grid_model import SpatialModel
        m = SpatialModel(num_actions=len(ACTION_MAP))
    else:
        m = ComputerUseModelV3(num_actions=len(ACTION_MAP))
    state = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(state, strict=False)
    return m.to(device).eval()


@torch.no_grad()
def predict_box(model, img, instr, tok, ip, img_size, device):
    """返回归一坐标 (x,y)。img 是 PIL。"""
    im = img.convert("RGB").resize((img_size, img_size))
    px = ip(images=im, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(device)
    enc = tok(instr, return_tensors="pt", truncation=True, max_length=64)
    out = model(px, enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return out["coord_pred"][0].tolist()


def crop_center(img, cx, cy, size):
    w, h = img.size
    half = size // 2
    x0 = max(0, min(w - 1, int(cx) - half))
    y0 = max(0, min(h - 1, int(cy) - half))
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    return img.crop((x0, y0, x1, y1)), (x0, y0, x1 - x0, y1 - y0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "artifacts" / "checkpoints" / "w30_grid.pth"))
    ap.add_argument("--head", default="grid", choices=["v3", "grid", "spatial"])
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop", type=int, default=400, help="裁剪边长（原图像素）")
    ap.add_argument("--crop-img", type=int, default=768, help="裁剪后送入模型的分辨率")
    ap.add_argument("--splits", default=str(DATA_ROOT / "qgis_splits.json"))
    ap.add_argument("--train-meta", default=str(ROOT / "artifacts" / "checkpoints" / "w30_grid.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, _ = load_demos(DATA_ROOT, ["qgis"])
    sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    val = [r for r in rows if r["_task"] in set(sp["val"])]
    val = [r for r in val if r["action"] in ("click", "move") and r.get("target_rect")]
    if args.limit:
        val = val[:args.limit]
    print(f"[data] val click+rect n={len(val)}")

    model = build_model(args.head, args.grid, args.ckpt, device)
    tok, ip = get_tokenizer(), get_image_processor()

    # 1) 粗定位（全屏）
    coarse = []
    for i, r in enumerate(val):
        img = Image.open(r["_screenshot"])
        nx, ny = predict_box(model, img, r["instruction"], tok, ip, args.img_size, device)
        coarse.append((nx * img.size[0], ny * img.size[1]))
        if (i + 1) % 100 == 0:
            print(f"  coarse {i + 1}/{len(val)}", flush=True)

    def hit_of(px, py, rect):
        return int(rect[0] <= px <= rect[2] and rect[1] <= py <= rect[3])

    def rect_in_image_space(r, img):
        """rect 是全屏坐标，截图可能被缩放存盘（shot_scale），统一到截图空间再比。"""
        screen = r.get("screen") or [img.size[0], img.size[1]]
        s = img.size[0] / screen[0]
        R = r["target_rect"]
        return [R[0] * s, R[1] * s, R[2] * s, R[3] * s]

    def run(mode):
        hits = 0
        errs = []
        for i, r in enumerate(val):
            img = Image.open(r["_screenshot"])
            rect = rect_in_image_space(r, img)
            W, H = img.size
            if mode == "oracle":
                cx = (rect[0] + rect[2]) / 2
                cy = (rect[1] + rect[3]) / 2
            else:
                cx, cy = coarse[i]
            crop, (x0, y0, cw, ch) = crop_center(img, cx, cy, args.crop)
            nx, ny = predict_box(model, crop, r["instruction"], tok, ip, args.crop_img, device)
            px, py = x0 + nx * cw, y0 + ny * ch
            hits += hit_of(px, py, rect)
            errs.append(((px - (rect[0] + rect[2]) / 2) ** 2 + (py - (rect[1] + rect[3]) / 2) ** 2) ** 0.5)
            if (i + 1) % 100 == 0:
                print(f"  {mode} {i + 1}/{len(val)}", flush=True)
        errs.sort()
        return {"mode": mode, "n": len(val), "hit": hits, "hit_rate": hits / len(val),
                "dist_center_med": errs[len(errs) // 2]}

    # 2) 粗定位基线（不裁剪）；coarse 已是截图坐标，rect 要换成截图空间
    def base_stats():
        hits, errs = 0, []
        for i, r in enumerate(val):
            img = Image.open(r["_screenshot"])
            rect = rect_in_image_space(r, img)
            px, py = coarse[i]
            hits += hit_of(px, py, rect)
            cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
            errs.append(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
        errs.sort()
        return hits, errs[len(errs) // 2]

    base_hits, base_med = base_stats()
    base = {"mode": "coarse_only", "n": len(val), "hit": base_hits,
            "hit_rate": base_hits / len(val), "dist_center_med": base_med}
    print(f"[coarse] hit {base_hits}/{len(val)} = {base['hit_rate']:.3f} "
          f"med {base['dist_center_med']:.0f}px", flush=True)

    zoom = run("zoom")
    oracle = run("oracle")
    print(f"[zoom]   hit {zoom['hit']}/{zoom['n']} = {zoom['hit_rate']:.3f} "
          f"med {zoom['dist_center_med']:.0f}px", flush=True)
    print(f"[oracle] hit {oracle['hit']}/{oracle['n']} = {oracle['hit_rate']:.3f} "
          f"med {oracle['dist_center_med']:.0f}px", flush=True)

    payload = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ckpt": args.ckpt,
                        "head": args.head, "img_size": args.img_size,
                        "crop": args.crop, "crop_img": args.crop_img},
               "coarse_only": base, "zoom": zoom, "oracle_crop": oracle}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
