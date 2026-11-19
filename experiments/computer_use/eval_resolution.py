"""220M 模型在不同观察分辨率下的 click 准确率（真实 GUI / 合成 UI）。

用途：A2 阶段 B 的第 1–2 项。只做**单步 click 定位**，不做闭环、不训练。

真实 GUI：`artifacts/computer_use_inloop/audit_manifest.json`（216 条冻结样本），
          判据与 `audit_eval.py` 相同：预测点落在 `rect_img` 内即命中。
合成 UI ：`p0_env.make_sample` 生成 200 条单步任务，预测点落在目标 slot 内即命中。

用法：
    python experiments/computer_use/eval_resolution.py --env real --res 224,1024
    python experiments/computer_use/eval_resolution.py --env synthetic --res 224,1024
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mio"))
sys.path.insert(0, str(ROOT / "experiments" / "paper"))

from mio_argus import (ACTION_MAP, ComputerUseModelV3, get_image_processor,  # noqa: E402
                       get_tokenizer)

MODEL_PATH = ROOT / "artifacts/checkpoints/computer_use_model_v3_real.pth"
MANIFEST = ROOT / "artifacts/computer_use_inloop/audit_manifest.json"


def load_model(device):
    m = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state.get("model", state))
    m.eval()
    return m


def prep(img: Image.Image, res: int, proc):
    img = img.convert("RGB").resize((res, res))
    return proc(images=img, return_tensors="pt", do_resize=False)["pixel_values"]


def predict(model, tok, img_tensor, instruction, device):
    enc = tok(instruction, return_tensors="pt", truncation=True, max_length=64,
              padding="max_length")
    with torch.no_grad():
        out = model(img_tensor.to(device), enc["input_ids"].to(device),
                    enc["attention_mask"].to(device))
    coord = out["coord_pred"].squeeze(0).cpu().numpy()
    action = int(torch.argmax(out["action_logits"], -1).item())
    return coord, action


def run_real(model, tok, proc, res, device, limit=0):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"][:limit] if limit else man["rows"]
    model.interpolate_pos = res != 224
    hits, actions, lat = 0, {}, []
    detail = []
    for i, r in enumerate(rows):
        img = Image.open(r["image"]).convert("RGB")
        W, H = img.size
        t0 = time.time()
        coord, act = predict(model, tok, prep(img, res, proc), r["instruction"], device)
        lat.append((time.time() - t0) * 1000)
        px, py = coord[0] * W, coord[1] * H
        x1, y1, x2, y2 = r["rect_img"]
        ok = x1 <= px <= x2 and y1 <= py <= y2
        hits += int(ok)
        actions[ACTION_MAP[act]] = actions.get(ACTION_MAP[act], 0) + 1
        detail.append({"id": r["id"], "hit": bool(ok), "px": round(float(px), 1),
                       "py": round(float(py), 1), "action": ACTION_MAP[act]})
        if (i + 1) % 50 == 0:
            print(f"  [{res}] {i+1}/{len(rows)} acc={hits/(i+1):.3f}", flush=True)
    lat.sort()
    return {"env": "real", "res": res, "n": len(rows), "hits": hits,
            "acc": hits / len(rows), "latency_p50_ms": round(lat[len(lat) // 2], 1),
            "actions": actions, "rows": detail}


def run_synthetic(model, tok, proc, res, device, n=200, seed=999):
    from p0_env import EnvCfg, make_sample
    cfg = EnvCfg(grid=6, img_size=160, label_vocab=36, fill=0.40, jitter=4,
                 color_by_label=True)
    model.interpolate_pos = res != 224
    hits = 0
    detail = []
    for i in range(n):
        s = make_sample(np.random.default_rng(seed * 100000 + i), cfg)
        arr = (s.image.transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        coord, act = predict(model, tok, prep(img, res, proc), s.instruction, device)
        px, py = coord[0] * 160, coord[1] * 160
        bx1, by1, bx2, by2 = s.target_bbox
        ok = bx1 <= px <= bx2 and by1 <= py <= by2
        hits += int(ok)
        detail.append({"i": i, "hit": bool(ok)})
    return {"env": "synthetic", "res": res, "n": n, "hits": hits, "acc": hits / n,
            "rows": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=("real", "synthetic"), required=True)
    ap.add_argument("--res", default="224,1024")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_list = [int(v) for v in args.res.split(",")]
    model, tok, proc = load_model(device), get_tokenizer(), get_image_processor()
    print(f"[setup] device={device} ckpt={MODEL_PATH.name} res={res_list}", flush=True)

    out = []
    for res in res_list:
        t0 = time.time()
        r = (run_real(model, tok, proc, res, device, args.limit) if args.env == "real"
             else run_synthetic(model, tok, proc, res, device, args.n))
        r["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0) if torch.cuda.is_available() else None
        r["seconds"] = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats()
        out.append(r)
        print(f"[result] {args.env} @{res}: {r['hits']}/{r['n']} = {r['acc']:.4f}  "
              f"{r['seconds']}s  peakVRAM={r['peak_vram_mb']}MB", flush=True)

    path = Path(args.out or f"results/computer_use/resolution_{args.env}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {path}")


if __name__ == "__main__":
    main()
