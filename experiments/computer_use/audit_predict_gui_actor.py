"""
审计 provider：GUI-Actor-2B（在隔离 venv 里跑，transformers 4.51.3）

必须用隔离环境：GUI-Actor 的建模代码依赖 transformers 4.x 的内部结构，
主环境是 5.x。本脚本只负责产出与 audit_eval 相同的 predictions JSON，
评分仍由 audit_eval.py --score 统一完成（同一批样本、同一套判据）。

用法（在 artifacts/audit_env 里）：
  set HF_ENDPOINT=https://hf-mirror.com
  set PYTHONPATH=<GUI-Actor-main>/src
  python experiments/computer_use/audit_predict_gui_actor.py --limit 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "mio"), str(ROOT)]

ROOT_ART = ROOT / "artifacts" / "computer_use_inloop"
MANIFEST = ROOT_ART / "audit_manifest.json"
PRED_DIR = ROOT_ART / "audit_predictions"

SYS_MSG = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
           "You need to perform a series of pyautogui actions to complete the task.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/GUI-Actor-2B-Qwen2-VL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--out", default=str(PRED_DIR / "gui_actor_2b.json"))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
    from gui_actor.inference import inference

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"][:args.limit] if args.limit else man["rows"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation=args.attn).eval()
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"[gui_actor] loaded {args.model} attn={args.attn} vram≈{vram:.0f}MB", flush=True)

    out = []
    for i, r in enumerate(rows):
        img = Image.open(ROOT / r["image"]).convert("RGB")
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYS_MSG}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": r["instruction"]}]},
        ]
        t0 = time.time()
        try:
            pred = inference(conv, model, tok, proc, use_placeholder=True, topk=args.topk)
            pts = [[float(x), float(y)] for x, y in pred.get("topk_points", [])]
            raw = "topk_points"
            ok = bool(pts)
        except Exception as e:
            pts, raw, ok = [], f"err: {e}", False
        lat = (time.time() - t0) * 1000
        out.append({"id": r["id"], "points": pts, "latency_ms": round(lat, 1),
                    "vram_mb": int(vram), "raw": raw, "parse_ok": ok})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)} pts={pts[:1]}", flush=True)

    payload = {"provider": "gui_actor_2b", "meta": {"model": args.model, "n": len(out),
                                                    "topk": args.topk, "attn": args.attn},
               "rows": out}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[gui_actor] rows={len(out)} parsed={sum(1 for x in out if x['parse_ok'])} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
