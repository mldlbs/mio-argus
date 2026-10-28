"""
指令改写层（离线）：给每条示范指令生成 2 条**不含控件名**的改写

用途：W28b 的结论是"开卷指令下的检索分数是假的"（§8），且训练查询太少（19 条）。
一条示范若能展开成 K 个检索对（原句 + 改写），训练查询就能 ×3，而不用重采 UI。

做法：本地 Qwen2.5-1.5B-Instruct（离线，无 API）逐条改写，再做强过滤：
  - 拒绝含原句引号内控件名（大小写不敏感）的改写
  - 拒绝含候选控件名（来自当帧 affordances / QGIS 按钮）的改写
  - 拒绝过短 / 过长 / 与原句重复的
过滤后不足 2 条则用规则模板兜底并标记 `fallback=true`，不伪装成模型输出。

产出：artifacts/computer_use_inloop/instruction_variants.json
  {"<原句>": [{"text": "...", "source": "llm"|"rule", "leak": false}, ...]}

用法:
    python experiments/computer_use/gen_paraphrase.py
    python experiments/computer_use/gen_paraphrase.py --no-llm   # 只用规则模板
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

from train_inloop import DATA_ROOT, load_demos  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = DATA_ROOT / "instruction_variants.json"
QUOTED_RE = re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,40})['\"\u2018\u2019\u201c\u201d]")

# 规则兜底：把"点名字"改写成"点角色"。不追求语言质量，只保证不含控件名。
RULES = [
    (r"^click in the (.+)$", "place the cursor in the {0}"),
    (r"^click the (.+) menu$", "open the menu in the menu bar"),
    (r"^click the '(.+)' menuitem$", "choose that item from the open menu"),
    (r"^click the '(.+)' edit$", "focus the text field"),
    (r"^click the '(.+)' button$", "confirm with the dialog button"),
    (r"^click the (.+)$", "click the matching control"),
    (r"^Press (.+)$", "use the keyboard shortcut for that action"),
    (r"^Type (.+)$", "enter the required text"),
]


def widget_names(instruction, cands):
    names = [m.strip() for m in QUOTED_RE.findall(instruction) if m.strip()]
    names += [c for c in cands if len(c) >= 2]
    return [n for n in names if n]


def leaks(text, names):
    low = text.lower()
    return any(n.lower() in low for n in names if len(n) >= 2)


def rule_variants(instruction):
    out = []
    for pat, tmpl in RULES:
        m = re.match(pat, instruction, re.IGNORECASE)
        if m:
            out.append(tmpl.format(*m.groups()))
            break
    if not out:
        out.append("perform the same action using the visible controls")
    out.append("do it with the control that matches the goal")
    return out


class Qwen:
    def __init__(self, device, name="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map=None).to(device).eval()
        self.device = device

    def ask(self, instruction, n=2):
        sys_p = ("You rewrite UI instructions. Return exactly {n} rewrites, one per line, "
                 "no numbering, no quotes. Each rewrite must express the SAME goal but must "
                 "NOT contain the name of any button, menu, field, file or dialog.").format(n=n)
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": f"Instruction: {instruction}"}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with __import__("torch").no_grad():
            out = self.model.generate(**enc, max_new_tokens=96, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        new = out[0][enc["input_ids"].shape[1]:]
        raw = self.tok.decode(new, skip_special_tokens=True)
        lines = [re.sub(r"^\s*[\d\-\.\*\)]+\s*", "", ln).strip().strip('"').strip("'")
                 for ln in raw.splitlines()]
        return [ln for ln in lines if len(ln) > 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows, _ = load_demos(DATA_ROOT, ["notepad"])
    by_instr = {}
    for r in rows:
        cands = [b["text"] for b in (r.get("observation") or {}).get("affordances") or []]
        by_instr.setdefault(r["instruction"], set()).update(widget_names(r["instruction"], cands))
    frames = DATA_ROOT / "eval_qgis" / "frames.json"
    if frames.exists():
        payload = json.loads(frames.read_text(encoding="utf-8"))
        names = list(payload["buttons_sample"].keys())
        for r in payload["rows"]:
            by_instr.setdefault(r["instruction"], set()).update(widget_names(r["instruction"], names))
    print(f"[data] unique instructions={len(by_instr)}")

    qwen = None
    if not args.no_llm:
        import torch
        qwen = Qwen(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    result = {}
    for i, (instr, names) in enumerate(sorted(by_instr.items()), 1):
        picked, fallback = [], False
        if qwen is not None:
            for cand in qwen.ask(instr, args.n):
                if (not leaks(cand, names) and cand.lower() not in {x["text"].lower() for x in picked}
                        and cand.lower() != instr.lower() and len(cand) <= 120):
                    picked.append({"text": cand, "source": "llm", "leak": False})
        if len(picked) < args.n:
            fallback = True
            for cand in rule_variants(instr):
                if len(picked) >= args.n:
                    break
                if not leaks(cand, names) and cand.lower() not in {x["text"].lower() for x in picked}:
                    picked.append({"text": cand, "source": "rule", "leak": False})
        result[instr] = picked
        srcs = ",".join(p["source"] for p in picked)
        print(f"[{i}/{len(by_instr)}] {instr!r} -> {[p['text'] for p in picked]} ({srcs}"
              f"{', fallback' if fallback else ''})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    n_llm = sum(1 for v in result.values() for p in v if p["source"] == "llm")
    n_rule = sum(1 for v in result.values() for p in v if p["source"] == "rule")
    print(f"[out] {out}  llm={n_llm} rule={n_rule} instructions={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
