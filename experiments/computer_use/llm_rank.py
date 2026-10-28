"""
语义上限探针：小指令模型（Qwen2.5-1.5B-Instruct）零样本从候选控件里选一个

为什么做：`affordance_retriever.py` 的训练头学不动（19 条查询），无法判断是
"数据不够"还是"这类语义对齐做不到"。这个探针用**不训练**的方式测上限：
把候选控件名列表直接给模型，让它按意图选一个。

两套指令：
  - open    原始指令（字面含控件名）——应接近满分，作为探针自检
  - noleak  人手写的不含控件名的改写（affordance_pick.NO_LEAK）——真实 grounding

指标：top1（6 条），并打印每条的选择，便于人读归因。

用法:
    python experiments/computer_use/llm_rank.py
    python experiments/computer_use/llm_rank.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from affordance_pick import NO_LEAK  # noqa: E402
from train_inloop import DATA_ROOT  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "llm_rank.json"

SYS = ("You are given a screenshot description task: pick the ONE control that fulfils the "
       "instruction. Reply with only its number, nothing else.")


def load_cases():
    payload = json.loads((DATA_ROOT / "eval_qgis" / "frames.json").read_text(encoding="utf-8"))
    names = list(payload["buttons_sample"].keys())
    cases = []
    for r in payload["rows"]:
        target = r["meta"]["uia_name"]
        cases.append({"open": r["instruction"],
                      "noleak": NO_LEAK.get(r["instruction"], r["instruction"]),
                      "target": target, "names": names})
    return cases


def parse_choice(text, k):
    nums = re.findall(r"\d+", text)
    for n in nums:
        i = int(n) - 1
        if 0 <= i < k:
            return i
    low = text.lower()
    return None, low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device).eval()
    print(f"[model] {args.model} on {device}")

    cases = load_cases()
    report = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": args.model,
                       "n": len(cases), "chance": 1.0 / len(cases[0]["names"])}, "rows": []}
    for mode in ("open", "noleak"):
        hit = 0
        for c in cases:
            names = c["names"]
            listing = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(names))
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": f"Instruction: {c[mode]}\n\nControls:\n{listing}\n\nNumber:"}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     temperature=None, top_p=None, top_k=None)
            raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_choice(raw, len(names))
            idx = parsed[0] if isinstance(parsed, tuple) else parsed
            pick = names[idx] if idx is not None else None
            ok = pick == c["target"]
            hit += int(ok)
            report["rows"].append({"mode": mode, "instruction": c[mode], "target": c["target"],
                                   "raw": raw.strip()[:40], "pick": pick, "ok": ok})
            print(f"  [{mode}] {'OK ' if ok else 'MISS'} {c[mode]!r} -> {pick}")
        report[f"{mode}_top1"] = hit / len(cases)
        print(f"[{mode}] top1={hit}/{len(cases)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
