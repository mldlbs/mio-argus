"""A0-G6（误差版）：在臂自己的验证集上，看图像到底帮不帮忙。

与 `ablate_modalities.py`（输出敏感度）互补：这里直接测**误差**。
对同一批 val 行，分别用 正确图 / 灰图 / 换图 / 换指令 评测，
报 action_acc、coord MAE、命中率，以及 ΔMAE（图变差 → MAE 变差才说明图像有用）。

用法：
    python experiments/computer_use/eval_arm_g6.py --ckpt artifacts/checkpoints/inloop_f1_all.pth \
        --apps qgis,notepad --val-ratio 0.25 --seed 42

输出：results/computer_use/arm_g6_<ckpt>.json
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mio"))
sys.path.insert(0, str(ROOT / "experiments" / "computer_use"))

from train_inloop import (  # noqa: E402
    ACTION_MAP, DATA_ROOT, DemoDataset, load_demos, split_by_task,
)
from mio_argus import ComputerUseModelV3, get_image_processor, get_tokenizer  # noqa: E402

GRAY = Path(tempfile.gettempdir()) / "mio_gray_224.png"
Image.new("RGB", (224, 224), (128, 128, 128)).save(GRAY)


class VariantDataset(DemoDataset):
    """在不改 DemoDataset 的前提下，替换图像或指令。"""

    def __init__(self, rows, tokenizer, image_processor, mode, max_len=64, img_size=None):
        swapped = [dict(r) for r in rows]
        if mode == "gray":
            for r in swapped:
                r["_screenshot"] = str(GRAY)
        elif mode == "swap_img":
            perm = np.random.default_rng(0).permutation(len(rows))
            for i, r in enumerate(swapped):
                r["_screenshot"] = rows[perm[i]]["_screenshot"]
        elif mode == "swap_text":
            perm = np.random.default_rng(1).permutation(len(rows))
            for i, r in enumerate(swapped):
                r["instruction"] = rows[perm[i]]["instruction"]
        elif mode != "orig":
            raise ValueError(mode)
        super().__init__(swapped, tokenizer, image_processor, max_len=max_len, img_size=img_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--apps", default="qgis,notepad")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--head", default="v3", choices=["v3", "grid", "spatial"])
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    rows, _ = load_demos(Path(args.data_root), apps)
    if args.splits:
        sp = json.loads(Path(args.splits).read_text(encoding="utf-8"))
        tr, va = set(sp.get("train") or []), set(sp.get("val") or [])
        val_rows = [r for r in rows if r["_task"] in va]
    else:
        _, val_rows, _ = split_by_task(rows, args.val_ratio, args.seed)
    if not val_rows:
        print("[error] val 为空")
        return 1
    print(f"[data] val rows={len(val_rows)} tasks={len({r['_task'] for r in val_rows})} "
          f"coord={sum(1 for r in val_rows if r.get('target_rect'))}")

    if args.head == "grid":
        from grid_model import GridModel
        model = GridModel(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True,
                          grid=args.grid)
    elif args.head == "spatial":
        from grid_model import SpatialModel
        model = SpatialModel(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True)
    else:
        model = ComputerUseModelV3(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.to(device).eval()

    tok, proc = get_tokenizer(), get_image_processor()
    res = {}
    for mode in ("orig", "gray", "swap_img", "swap_text"):
        ds = VariantDataset(val_rows, tok, proc, mode, img_size=args.img_size)
        m = evaluate_with_loader(model, ds, ds.rows, device, args.batch_size)
        res[mode] = m
        print(f"[{mode:<9}] n={m['n']} acc={m.get('action_acc'):.3f} "
              f"coord_mae={m.get('coord_mae_norm')} hit={m.get('hit_rate')} "
              f"(hit_n={m.get('hit_n')})", flush=True)

    def delta(a, b):
        if a.get("coord_mae_norm") is None or b.get("coord_mae_norm") is None:
            return None
        return round(b["coord_mae_norm"] - a["coord_mae_norm"], 4)

    # 输出敏感度（用 samples 里的像素坐标）
    def sens(key):
        a, b = res["orig"]["samples"], res[key]["samples"]
        n = min(len(a), len(b))
        d = [abs(a[i]["xy_pred_px"][0] - b[i]["xy_pred_px"][0]) +
             abs(a[i]["xy_pred_px"][1] - b[i]["xy_pred_px"][1]) for i in range(n)]
        return float(np.mean(d)) if d else None

    img_effect = max(v for v in (sens("gray"), sens("swap_img")) if v is not None)
    txt_effect = sens("swap_text")
    ratio = img_effect / max(txt_effect or 1e-9, 1e-9)

    # G6b：配对 bootstrap —— 换灰图后逐行误差真的变差吗？
    def paired_ci(key, n_boot=2000, seed=0):
        a = {r["id"]: r["err"] for r in res["orig"]["coord_rows"] if r["id"] is not None}
        b = {r["id"]: r["err"] for r in res[key]["coord_rows"] if r["id"] is not None}
        ids = [i for i in a if i in b]
        if len(ids) < 10:
            return {"n": len(ids), "delta": None, "ci": None}
        d = np.array([b[i] - a[i] for i in ids])          # >0 表示图被破坏后误差变大
        rng = np.random.default_rng(seed)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        return {"n": len(ids), "delta": float(d.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

    g6b = {"gray": paired_ci("gray"), "swap_img": paired_ci("swap_img")}
    image_helpful = bool(g6b["gray"]["ci"] and g6b["gray"]["ci"][0] > 0)

    result = {
        "ckpt": args.ckpt, "apps": apps, "img_size": args.img_size,
        "val_rows": len(val_rows),
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk not in ("samples", "coord_rows")}
                    for k, v in res.items()},
        "delta_mae": {"gray": delta(res["orig"], res["gray"]),
                      "swap_img": delta(res["orig"], res["swap_img"]),
                      "swap_text": delta(res["orig"], res["swap_text"])},
        "pixel_sensitivity": {"gray": sens("gray"), "swap_img": sens("swap_img"),
                              "swap_text": txt_effect},
        "image_over_text": ratio,
        "g6_sensitivity": "IMAGE_USED" if ratio >= 0.10 else "IMAGE_IGNORED",
        "g6b_image_helpful": image_helpful,
        "g6b_paired": g6b,
    }
    print(json.dumps({k: result[k] for k in ("delta_mae", "image_over_text",
                                             "g6_sensitivity", "g6b_image_helpful",
                                             "g6b_paired")}, ensure_ascii=False, indent=2))

    out = Path(args.out) if args.out else \
        Path("results/computer_use") / f"arm_g6_{Path(args.ckpt).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


def evaluate_with_loader(model, ds, rows, device, batch_size):
    """用给定 dataset 评测（复制 evaluate 的统计口径，仅替换数据源）。

    额外返回逐行坐标误差 coord_rows（含 id），供配对 bootstrap 使用。
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    total = correct = coord_n = 0
    coord_err, hit, hit_n = [], 0, 0
    samples, coord_rows = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["image"].to(device), batch["input_ids"].to(device),
                            batch["attention_mask"].to(device))
            pred = outputs["action_logits"].argmax(1).cpu()
            labels = batch["action"]
            total += labels.numel()
            correct += int((pred == labels).sum())
            coord_pred = outputs["coord_pred"].cpu().tolist()
            batch_rows = rows[offset:offset + labels.numel()]
            for row, cp in zip(batch_rows, coord_pred):
                screen = row.get("screen") or [2560, 1440]
                if len(samples) < 200:
                    samples.append({"id": row.get("id"),
                                    "xy_pred_px": [round(cp[0] * screen[0], 1),
                                                   round(cp[1] * screen[1], 1)]})
            mask = batch["coord_mask"].bool()
            if mask.any():
                cpred = outputs["coord_pred"].cpu()[mask]
                ctrue = batch["coords"][mask]
                errs = (cpred - ctrue).norm(dim=1).tolist()
                coord_err.extend(errs)
                coord_n += int(mask.sum())
                keep = mask.nonzero(as_tuple=False).flatten().tolist()
                for i, e in zip(keep, errs):
                    coord_rows.append({"id": batch_rows[i].get("id"), "err": float(e)})
                for i, p in zip(keep, cpred.tolist()):
                    rect = batch_rows[i].get("target_rect")
                    if not rect:
                        continue
                    screen = batch_rows[i].get("screen") or [2560, 1440]
                    x, y = p[0] * screen[0], p[1] * screen[1]
                    hit_n += 1
                    hit += int(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])
            offset += labels.numel()

    return {"n": total, "action_acc": correct / max(total, 1),
            "coord_n": coord_n,
            "coord_mae_norm": float(np.mean(coord_err)) if coord_err else None,
            "coord_med_norm": float(np.median(coord_err)) if coord_err else None,
            "hit_n": hit_n, "hit_rate": (hit / hit_n) if hit_n else None,
            "samples": samples, "coord_rows": coord_rows}


if __name__ == "__main__":
    sys.exit(main())
