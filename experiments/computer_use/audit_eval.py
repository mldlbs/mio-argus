"""
W31 复用审计：冻结样本 + 统一评价（对事不对模型）

设计约束（按审计规格）：
  - 样本冻结：只用 val 两条链（w26_v2 / w27）中带 target_rect 的 click 样本，共 216 条；
    清单落盘后不再变动。
  - 坐标映射唯一：**所有 provider 输出归一化 [0,1] 坐标（相对截图）**，
    由本脚本统一映射到截图像素、统一判命中，provider 内不做任何坐标换算。
  - 判据唯一：点落在 target_rect（换算到截图空间）内即命中；top-k = 前 k 个点任一命中。
  - 失败分类：no_output（无输出/解析失败）、invalid（NaN 或越界）、wrong（有效但未命中）。

三个入口：
  python audit_eval.py --build-manifest          # 生成 audit_manifest.json
  python audit_eval.py --predict structure       # 结构基线（UIA/bridge affordance）→ predictions
  python audit_eval.py --score                   # 汇总所有 predictions/*.json → 表 + json

provider 预测文件格式（predictions/<name>.json）：
  {"provider": "uground", "meta": {...},
   "rows": [{"id": "...", "points": [[x,y], ...], "latency_ms": 123, "vram_mb": 4200,
             "raw": "...", "parse_ok": true}, ...]}
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "mio"), str(ROOT)]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ART = ROOT / "artifacts" / "computer_use_inloop"
RES = ROOT / "results" / "computer_use"
MANIFEST = ART / "audit_manifest.json"
PRED_DIR = ART / "audit_predictions"

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


# ---------------------------------------------------------------- manifest ---
def build_manifest():
    from train_inloop import DATA_ROOT, load_demos
    sp = json.loads((ART / "qgis_splits.json").read_text(encoding="utf-8"))
    rows, _ = load_demos(DATA_ROOT, ["qgis"])
    val_tasks = set(sp["val"])
    out = []
    for r in rows:
        if r["_task"] not in val_tasks:
            continue
        if r["action"] not in ("click", "move") or not r.get("target_rect"):
            continue
        from PIL import Image
        img = Image.open(r["_screenshot"])
        screen = r.get("screen") or [img.size[0], img.size[1]]
        s = img.size[0] / screen[0]
        R = r["target_rect"]
        rect_img = [R[0] * s, R[1] * s, R[2] * s, R[3] * s]
        min_side_full = min(R[2] - R[0], R[3] - R[1])
        out.append({
            "id": r["id"],
            "task": r["_task"],
            "image": r["_screenshot"],
            "image_size": list(img.size),
            "screen": screen,
            "instruction": r["instruction"],
            "rect_full": R,
            "rect_img": rect_img,
            "target_min_side_full_px": round(min_side_full, 1),
            "affordances": (r.get("observation") or {}).get("affordances") or [],
        })
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(out),
               "source": "val chains w26_v2 + w27, click/move with target_rect",
               "rows": out}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = Counter(size_bucket(x["target_min_side_full_px"]) for x in out)
    print(f"[manifest] n={len(out)} -> {MANIFEST}")
    print(f"[manifest] 目标尺寸分桶 {dict(sizes)}")
    return 0


def size_bucket(px):
    if px < 50:
        return "small(<50px)"
    if px < 100:
        return "medium(50-100px)"
    return "large(>=100px)"


# ------------------------------------------------------------- structure ---
def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def lexical_score(instr, label):
    a, b = Counter(tokens(instr)), Counter(tokens(label))
    if not a or not b:
        return 0.0
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def predict_structure(topk=5):
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows, skipped = [], 0
    for r in man["rows"]:
        cands = [a for a in (r.get("affordances") or []) if a.get("text") and a.get("rect")]
        if not cands:
            rows.append({"id": r["id"], "points": [], "parse_ok": False, "raw": "no_affordances"})
            skipped += 1
            continue
        scored = sorted(cands, key=lambda a: -lexical_score(r["instruction"], a["text"]))[:topk]
        pts = []
        for a in scored:
            l, t, w, h = a["rect"]
            cx, cy = l + w / 2, t + h / 2
            pts.append([cx / r["screen"][0], cy / r["screen"][1]])
        rows.append({"id": r["id"], "points": pts, "latency_ms": 0.0, "vram_mb": 0,
                     "raw": [a["text"] for a in scored], "parse_ok": True})
    out = {"provider": "structure_uia", "meta": {"n": len(rows), "topk": topk},
           "rows": rows}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    p = PRED_DIR / "structure_uia.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[structure] rows={len(rows)} no_affordances={skipped} -> {p}")
    return 0


# ----------------------------------------------------------------- uground ---
UGROUND_PROMPT = """Your task is to help the user identify the precise coordinates (x, y) of a specific area on the screen.

- Your response should aim to point to the center or a representative point within the described area/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on the likely content.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {desc}

Answer:"""


def _parse_xy(text):
    """UGround 输出 (x, y)，坐标域 [0,1000)。返回归一化 [0,1] 或 None。"""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return [x / 1000.0, y / 1000.0]


def predict_uground(model_path, limit=0, dtype="bfloat16"):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["rows"][:limit] if limit else man["rows"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)

    proc = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device).eval()
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"[uground] loaded {model_path} on {device}, vram≈{vram:.0f}MB")

    out = []
    for i, r in enumerate(rows):
        img = Image.open(ROOT / r["image"] if not Path(r["image"]).is_absolute() else r["image"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": UGROUND_PROMPT.format(desc=r["instruction"])}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        lat = (time.time() - t0) * 1000
        ans = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        pt = _parse_xy(ans)
        out.append({"id": r["id"], "points": [pt] if pt else [], "latency_ms": round(lat, 1),
                    "vram_mb": int(vram), "raw": ans.strip()[:120], "parse_ok": pt is not None})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} last={ans.strip()[:40]!r}", flush=True)
    payload = {"provider": "uground_v1_2b",
               "meta": {"model": model_path, "n": len(out), "topk": 1,
                        "dtype": dtype, "device": device},
               "rows": out}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    p = PRED_DIR / "uground_v1_2b.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for x in out if x["parse_ok"])
    print(f"[uground] rows={len(out)} parsed={ok} -> {p}")
    return 0


# ------------------------------------------------------------------ score ---
def score_provider(pred, man_rows):
    by_id = {r["id"]: r for r in man_rows}
    stats = {"n": 0, "top1": 0, "topk": 0, "topk_k": pred.get("meta", {}).get("topk", 5),
             "no_output": 0, "invalid": 0, "wrong": 0, "latency": [], "vram_mb": 0,
             "by_size": {}}
    for row in pred["rows"]:
        m = by_id.get(row["id"])
        if m is None:
            continue
        stats["n"] += 1
        pts = row.get("points") or []
        stats["latency"].append(row.get("latency_ms") or 0.0)
        stats["vram_mb"] = max(stats["vram_mb"], row.get("vram_mb") or 0)
        valid = []
        for p in pts:
            try:
                x, y = float(p[0]), float(p[1])
            except Exception:
                continue
            if math.isnan(x) or math.isnan(y) or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            valid.append((x, y))
        if not pts:
            stats["no_output"] += 1
        elif not valid:
            stats["invalid"] += 1
        W, H = m["image_size"]
        rect = m["rect_img"]

        def hit(p):
            return rect[0] <= p[0] * W <= rect[2] and rect[1] <= p[1] * H <= rect[3]

        h1 = bool(valid) and hit(valid[0])
        hk = any(hit(p) for p in valid)
        stats["top1"] += int(h1)
        stats["topk"] += int(hk)
        if not h1:
            stats["wrong"] += 1
        b = size_bucket(m["target_min_side_full_px"])
        d = stats["by_size"].setdefault(b, {"n": 0, "top1": 0})
        d["n"] += 1
        d["top1"] += int(h1)
    n = max(stats["n"], 1)
    lat = sorted(stats["latency"])
    return {
        "provider": pred.get("provider"),
        "n": stats["n"],
        "top1": stats["top1"] / n,
        "topk": stats["topk"] / n,
        "topk_k": stats["topk_k"],
        "no_output": stats["no_output"],
        "invalid": stats["invalid"],
        "wrong": stats["wrong"],
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "latency_p90_ms": lat[int(len(lat) * 0.9)] if lat else None,
        "vram_mb": stats["vram_mb"],
        "by_size": {k: {"n": v["n"], "top1": v["top1"] / max(v["n"], 1)}
                    for k, v in sorted(stats["by_size"].items())},
    }


def structure_availability(man_rows):
    """结构通道天花板：目标中心是否落在某个 UIA 控件框内（信息在不在，而非匹配得多好）。"""
    n = inside = 0
    for r in man_rows:
        a = r.get("affordances") or []
        if not a:
            continue
        n += 1
        L, T, R, B = r["rect_full"]
        cx, cy = (L + R) / 2, (T + B) / 2
        if any(x["rect"][0] <= cx <= x["rect"][0] + x["rect"][2]
               and x["rect"][1] <= cy <= x["rect"][1] + x["rect"][3] for x in a):
            inside += 1
    return {"n": n, "inside": inside, "rate": inside / max(n, 1)}


def do_score():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = sorted(PRED_DIR.glob("*.json"))
    if not files:
        print("[score] 没有预测文件")
        return 1
    results = []
    for f in files:
        pred = json.loads(f.read_text(encoding="utf-8"))
        s = score_provider(pred, man["rows"])
        s["file"] = f.name
        results.append(s)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "manifest_n": man["n"],
               "structure_availability": structure_availability(man["rows"]),
               "results": results}
    p = RES / "audit_results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'provider':22s} {'n':>4s} {'top1':>7s} {'top5':>7s} {'p50ms':>8s} {'VRAM MB':>8s} "
          f"{'no_out':>6s} {'invalid':>7s} {'wrong':>6s}")
    for r in results:
        print(f"{str(r['provider']):22s} {r['n']:4d} {r['top1']*100:6.1f}% {r['topk']*100:6.1f}% "
              f"{(r['latency_p50_ms'] or 0):8.0f} {r['vram_mb']:8d} "
              f"{r['no_output']:6d} {r['invalid']:7d} {r['wrong']:6d}")
        for b, v in r["by_size"].items():
            print(f"    {b:20s} n={v['n']:3d} top1={v['top1']*100:5.1f}%")
    print(f"\n[out] {p}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--predict", default="", choices=["", "structure", "uground", "gui_actor"])
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model", default="osunlp/UGround-V1-2B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    if args.build_manifest:
        return build_manifest()
    if args.predict == "structure":
        return predict_structure(args.topk)
    if args.predict == "uground":
        return predict_uground(args.model, args.limit, args.dtype)
    if args.score:
        return do_score()
    print("需要 --build-manifest / --predict / --score 之一")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
