"""
方向 B 基准：候选召回（B1）+ 候选排序（B2）

数据：同一批 216 条冻结样本（audit_manifest.json），候选来自当帧记录的 affordance 列表。
已知缺陷：录制时 struct[:60] 截断 + 含容器控件，本脚本先按"叶子化"过滤，并把缺陷量化。

B1 召回：目标中心是否落在某个候选矩形内（label 是否存在）
B2 排序：对召回成功的样本，比较
    lexical  词面 IDF 加权（零模型基线）
    qwen     Qwen2.5-1.5B-Instruct 在 top-10 短名单里选一个
  指标：Top-1 / Top-3 / MRR / 候选数 / 缺失率

用法:
    python experiments/computer_use/candidate_bench.py --stage b1
    python experiments/computer_use/candidate_bench.py --stage b2 --ranker lexical
    python experiments/computer_use/candidate_bench.py --stage b2 --ranker qwen
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
ART = ROOT / "artifacts" / "computer_use_inloop"
MANIFEST = ART / "audit_manifest.json"
OUT = ROOT / "results" / "computer_use" / "candidate_bench.json"

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
STOP = {"click", "double", "the", "on", "in", "of", "to", "a", "an", "button",
        "menuitem", "menu", "edit", "combobox", "tab", "item", "treeitem", "checkbox"}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def toks(text):
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP]


def load():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return man


def candidates(row, max_area_frac=0.25):
    """叶子化：去掉覆盖超过 25% 屏幕的容器，按矩形去重。"""
    W, H = row["screen"]
    seen, out = set(), []
    for a in row.get("affordances") or []:
        l, t, w, h = a["rect"]
        if w <= 0 or h <= 0:
            continue
        if (w * h) / (W * H) > max_area_frac:
            continue
        key = (l, t, w, h)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def label_index(row, cands):
    L, T, R, B = row["rect_full"]
    cx, cy = (L + R) / 2, (T + B) / 2
    for i, a in enumerate(cands):
        l, t, w, h = a["rect"]
        if l <= cx <= l + w and t <= cy <= t + h:
            return i
    return None


def stage_b1():
    man = load()
    n = 0
    have = 0
    counts = []
    rows = []
    for r in man["rows"]:
        cands = candidates(r)
        n += 1
        counts.append(len(cands))
        idx = label_index(r, cands)
        if idx is not None:
            have += 1
        rows.append({"id": r["id"], "n_cand": len(cands), "label_idx": idx})
    counts.sort()
    payload = {
        "stage": "b1",
        "n": n, "recall": have / max(n, 1), "hit": have,
        "cand_median": counts[len(counts) // 2] if counts else 0,
        "cand_p90": counts[int(len(counts) * 0.9)] if counts else 0,
        "note": "候选来自录制时的 affordance（struct[:60] 截断 + 含容器），叶子化后过滤面积>25% 屏幕者",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[B1] n={n} recall={have}/{n}={have / n:.3f} "
          f"cand_median={payload['cand_median']} p90={payload['cand_p90']}")
    print(f"[out] {OUT}")
    return 0


def idf_map(man):
    df = Counter()
    for r in man["rows"]:
        for a in candidates(r):
            for t in set(toks(a.get("text", ""))):
                df[t] += 1
    total = max(len(man["rows"]), 1)
    return {t: math.log(1 + total / (1 + c)) for t, c in df.items()}


def lexical_scores(instr, cands, idf):
    it = set(toks(instr))
    out = []
    for a in cands:
        ct = set(toks(a.get("text", "")))
        out.append(sum(idf.get(t, 0.0) for t in (it & ct)))
    return out


def rank_metrics(scores, label, k=3):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    rank = order.index(label) + 1 if label in order else None
    return {
        "top1": rank == 1,
        "top3": rank is not None and rank <= k,
        "rr": 1.0 / rank if rank else 0.0,
        "rank": rank,
    }


def stage_b2(ranker, limit=0):
    man = load()
    idf = idf_map(man)
    rows = [r for r in man["rows"] if label_index(r, candidates(r)) is not None]
    if limit:
        rows = rows[:limit]
    print(f"[B2] 可排序样本 n={len(rows)} ranker={ranker}")

    model = tok = None
    if ranker == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen2.5-1.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto").to("cuda").eval()

    res = []
    for i, r in enumerate(rows):
        cands = candidates(r)
        idx = label_index(r, cands)
        if ranker == "lexical":
            scores = lexical_scores(r["instruction"], cands, idf)
        else:
            # 先取词面 top-10 短名单，再让模型选
            lex = lexical_scores(r["instruction"], cands, idf)
            short = sorted(range(len(cands)), key=lambda j: -lex[j])[:10]
            lines = "\n".join(f"{k + 1}. {cands[j].get('text', '')}" for k, j in enumerate(short))
            msgs = [{"role": "system", "content": "Pick the control that fulfils the instruction. Reply with only its number."},
                    {"role": "user", "content": f"Instruction: {r['instruction']}\n\nControls:\n{lines}\n\nNumber:"}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            import torch
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=6, do_sample=False)
            raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            nums = re.findall(r"\d+", raw)
            pick = int(nums[0]) - 1 if nums and 1 <= int(nums[0]) <= len(short) else None
            scores = [0.0] * len(cands)
            if pick is not None:
                scores[short[pick]] = 1.0
        m = rank_metrics(scores, idx)
        m["id"] = r["id"]
        m["n_cand"] = len(cands)
        res.append(m)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(res)}", flush=True)

    n = len(res)
    payload = {
        "stage": "b2", "ranker": ranker, "n": n,
        "top1": sum(1 for x in res if x["top1"]) / max(n, 1),
        "top3": sum(1 for x in res if x["top3"]) / max(n, 1),
        "mrr": sum(x["rr"] for x in res) / max(n, 1),
        "cand_median": sorted(x["n_cand"] for x in res)[n // 2] if n else 0,
        "rows": res,
    }
    p = ROOT / "results" / "computer_use" / f"candidate_b2_{ranker}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[B2:{ranker}] top1={payload['top1']:.3f} top3={payload['top3']:.3f} "
          f"mrr={payload['mrr']:.3f} cand_median={payload['cand_median']}")
    print(f"[out] {p}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="b1", choices=["b1", "b2"])
    ap.add_argument("--ranker", default="lexical", choices=["lexical", "qwen"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "b1":
        return stage_b1()
    return stage_b2(args.ranker, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
