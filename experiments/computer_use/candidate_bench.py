"""
方向 B 基准：候选召回（B1）+ 候选排序（B2）

支持两套候选口径，用于同帧对照：
  --cands affordances          新口径：actions + widgets，无 60 截断，按 (text,rect) 去重
  --cands affordances_legacy   旧口径：前 60 条 text 控件（复现 34.7% 的问题）

B1：目标中心是否落在某个候选矩形内（召回）；并按控件类别分桶统计。
B2：在召回成功的样本上比较排序器
    lexical  词面 IDF（零模型基线）
    qwen     Qwen2.5-1.5B-Instruct 在词面 top-10 短名单里选
  指标：Top-1 / Top-3 / MRR / 候选数

用法:
    python experiments/computer_use/candidate_bench.py --stage b1 --manifest artifacts/.../cand_manifest.json
    python experiments/computer_use/candidate_bench.py --stage b2 --ranker lexical --cands affordances
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "computer_use_inloop"
DEFAULT_MANIFEST = ART / "audit_manifest.json"

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
STOP = {"click", "double", "the", "on", "in", "of", "to", "a", "an", "button",
        "menuitem", "menu", "edit", "combobox", "tab", "item", "treeitem", "checkbox"}

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def toks(text):
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP]


def cand_key(row, name):
    return row.get(name) or []


CONTAINER_CLS = {"QgisApp", "QMainWindow", "QDialog", "QgsDockWidget", "QWidget"}


def is_usable(a, screen):
    """容器不算候选：全窗口矩形会让召回虚高（89.8% 那次就是它）。"""
    r = a.get("rect") or [0, 0, 0, 0]
    W, H = (screen or [2560, 1440])[:2]
    if r[2] <= 0 or r[3] <= 0:
        return False
    if (r[2] * r[3]) / float(W * H) > 0.25:
        return False
    return (a.get("cls") or "") not in CONTAINER_CLS


def dedupe(cands, screen=None):
    seen, out = set(), []
    for a in cands:
        r = a.get("rect")
        if not r or r[2] <= 0 or r[3] <= 0:
            continue
        if screen is not None and not is_usable(a, screen):
            continue
        k = ((a.get("text") or "").strip(), tuple(r))
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def cls_bucket(a):
    cls = (a.get("cls") or "") + " " + (a.get("path") or "")
    low = cls.lower()
    if "action" in (a.get("kind") or ""):
        return "action"
    if "toolbutton" in low or "toolbar" in low:
        return "toolbar"
    if "menu" in low:
        return "menu"
    if "lineedit" in low or "spinbox" in low or "combobox" in low:
        return "input"
    if "tree" in low or "itemview" in low:
        return "tree"
    return "other"


def label_index(row, cands):
    L, T, R, B = row["rect_full"]
    cx, cy = (L + R) / 2, (T + B) / 2
    for i, a in enumerate(cands):
        l, t, w, h = a["rect"]
        if l <= cx <= l + w and t <= cy <= t + h:
            return i
    return None


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_b1(man, cands_name):
    n = have = 0
    counts, bucket_n, bucket_hit = [], Counter(), Counter()
    for r in man["rows"]:
        cands = dedupe(cand_key(r, cands_name), r.get("screen"))
        n += 1
        counts.append(len(cands))
        idx = label_index(r, cands)
        if idx is None:
            continue
        have += 1
        b = cls_bucket(cands[idx])
        bucket_n[b] += 1
        bucket_hit[b] += 1
    # 分母：每类目标的总数（用 label 是否存在来近似，未命中时按 rect 归属判定不了类别）
    counts.sort()
    out = {"stage": "b1", "cands": cands_name, "n": n, "hit": have,
           "recall": have / max(n, 1),
           "cand_median": counts[len(counts) // 2] if counts else 0,
           "cand_p90": counts[int(len(counts) * 0.9)] if counts else 0,
           "hit_by_bucket": dict(bucket_n)}
    p = ROOT / "results" / "computer_use" / f"cand_b1_{cands_name}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[B1:{cands_name}] n={n} recall={have}/{n}={have / n:.3f} "
          f"cand_median={out['cand_median']} p90={out['cand_p90']}")
    print(f"[B1:{cands_name}] 命中按类别 {dict(bucket_n)}")
    print(f"[out] {p}")
    return 0


def idf_map(man, cands_name):
    df = Counter()
    for r in man["rows"]:
        for a in dedupe(cand_key(r, cands_name)):
            for t in set(toks(a.get("text", ""))):
                df[t] += 1
    total = max(len(man["rows"]), 1)
    return {t: math.log(1 + total / (1 + c)) for t, c in df.items()}


def lexical_scores(instr, cands, idf):
    it = set(toks(instr))
    return [sum(idf.get(t, 0.0) for t in (it & set(toks(a.get("text", ""))))) for a in cands]


def rank_metrics(scores, label, k=3):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    rank = order.index(label) + 1 if label in order else None
    return {"top1": rank == 1, "top3": rank is not None and rank <= k,
            "rr": 1.0 / rank if rank else 0.0, "rank": rank}


def stage_b2(man, cands_name, ranker, limit=0):
    rows = []
    for r in man["rows"]:
        c = dedupe(cand_key(r, cands_name), r.get("screen"))
        if c:
            rows.append((r, c))
    if limit:
        rows = rows[:limit]
    recallable = [(r, c) for r, c in rows if label_index(r, c) is not None]
    print(f"[B2:{cands_name}] 样本 {len(rows)}，可召回 {len(recallable)}，ranker={ranker}")

    idf = idf_map(man, cands_name)
    model = tok = None
    if ranker == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen2.5-1.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto").to("cuda").eval()

    res = []
    for i, (r, cands) in enumerate(recallable):
        idx = label_index(r, cands)
        if ranker == "lexical":
            scores = lexical_scores(r["instruction"], cands, idf)
        else:
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
        m.update({"id": r["id"], "n_cand": len(cands), "bucket": cls_bucket(cands[idx])})
        res.append(m)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(res)}", flush=True)

    n = len(res)
    out = {"stage": "b2", "ranker": ranker, "cands": cands_name, "n": n,
           "top1": sum(1 for x in res if x["top1"]) / max(n, 1),
           "top3": sum(1 for x in res if x["top3"]) / max(n, 1),
           "mrr": sum(x["rr"] for x in res) / max(n, 1),
           "rows": res}
    p = ROOT / "results" / "computer_use" / f"cand_b2_{cands_name}_{ranker}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[B2:{cands_name}:{ranker}] top1={out['top1']:.3f} top3={out['top3']:.3f} "
          f"mrr={out['mrr']:.3f}")
    print(f"[out] {p}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="b1", choices=["b1", "b2"])
    ap.add_argument("--cands", default="affordances",
                    choices=["affordances", "affordances_legacy"])
    ap.add_argument("--ranker", default="lexical", choices=["lexical", "qwen"])
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    man = load_manifest(args.manifest)
    if args.stage == "b1":
        return stage_b1(man, args.cands)
    return stage_b2(man, args.cands, args.ranker, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
