"""
W28b 表征改造第二步：指令 × 候选控件的语义检索头（affordance retriever）

背景：`affordance_pick.py` 证明换掉坐标回归后同题 0/6 → 6/6，但那是词面匹配开卷；
把指令里的控件名去掉就掉到 3/6。本脚本测"能不能用语义编码器补上这 3 分"。

三个臂：
  R1 lexical   词面精确余弦（affordance_pick 的 P1，参照）
  R2 frozen    T5-base 编码器均值池化余弦（**不训练**，语义先验）
  R3 learned   在 R2 上接两个投影头，用 listwise cross-entropy 训练（19 条 Notepad 训练查询）
               评测时投影头只在 Notepad 上训过，QGIS 是无泄漏零样本

评测：
  - Notepad 留出 8 条（与 affordance_pick 同切法）
  - QGIS 6 条：开卷指令 + 无泄漏指令，各报一次

用法:
    python experiments/computer_use/affordance_retriever.py
    python experiments/computer_use/affordance_retriever.py --encoder gpt2 --steps 300
"""
import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from affordance_pick import NO_LEAK, bow_cos, notepad_candidates, qgis_candidates  # noqa: E402
from train_inloop import DATA_ROOT, load_demos  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "affordance_retriever.json"
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


# ------------------------------------------------------------------- encoders ---
class TextEncoder:
    """冻结编码器：T5 encoder 或 GPT-2，均值池化 + L2 归一。"""

    def __init__(self, name, device):
        from transformers import AutoTokenizer
        if name == "gpt2":
            from transformers import GPT2Model as Enc
            self.model = Enc.from_pretrained("gpt2")
            self.tok = AutoTokenizer.from_pretrained("gpt2")
            self.tok.pad_token = self.tok.eos_token
        else:
            from transformers import T5EncoderModel as Enc
            self.model = Enc.from_pretrained("t5-base")
            self.tok = AutoTokenizer.from_pretrained("t5-base")
        self.model.to(device).eval()
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=32,
                           return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            vec = F.normalize(vec, dim=-1)
            for t, v in zip(chunk, vec):
                self.cache[t] = v.detach()
        return torch.stack([self.cache[t] for t in texts])


# ------------------------------------------------------------------- training ---
class Proj(nn.Module):
    def __init__(self, dim_in=768, dim_out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_out), nn.GELU(), nn.Linear(dim_out, dim_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def train_retriever(enc, rows, steps, lr, temp, seed=0):
    torch.manual_seed(seed)
    device = enc.device
    pq, pc = Proj().to(device), Proj().to(device)
    params = list(pq.parameters()) + list(pc.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    data = []
    for r in rows:
        q = enc.embed([r["instruction"]])[0]
        c = enc.embed(r["cands"])
        data.append((q, c, r["label"]))

    for step in range(1, steps + 1):
        total = 0.0
        for q, c, label in data:
            scores = (pq(q) @ pc(c).T) / temp
            loss = F.cross_entropy(scores.unsqueeze(0),
                                   torch.tensor([label], device=scores.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if step % 50 == 0 or step == steps:
            train_acc = evaluate_learned(pq, pc, enc, rows, temp)["top1"]
            print(f"  step {step}: loss={total / len(data):.4f} train_top1={train_acc:.3f}")
    return pq, pc


def evaluate_learned(pq, pc, enc, rows, temp):
    pq.eval(), pc.eval()
    res = []
    with torch.no_grad():
        for r in rows:
            q = enc.embed([r["instruction"]])[0]
            c = enc.embed(r["cands"])
            scores = (pq(q) @ pc(c).T).cpu().numpy()
            order = np.argsort(-scores)
            rank_of = int(np.where(order == r["label"])[0][0]) + 1
            res.append({"top1": rank_of == 1, "top3": rank_of <= 3, "rr": 1.0 / rank_of,
                        "k": len(r["cands"]), "rank": rank_of,
                        "pick": r["cands"][order[0]], "target": r["cands"][r["label"]]})
    pq.train(), pc.train()
    return summarize("learned", res)


def eval_scorer(fn, rows, name):
    res = []
    for r in rows:
        scores = np.array([fn(r["instruction"], c) for c in r["cands"]], dtype=np.float64)
        order = np.argsort(-scores)
        rank_of = int(np.where(order == r["label"])[0][0]) + 1
        res.append({"top1": rank_of == 1, "top3": rank_of <= 3, "rr": 1.0 / rank_of,
                    "k": len(r["cands"]), "rank": rank_of,
                    "pick": r["cands"][order[0]], "target": r["cands"][r["label"]]})
    return summarize(name, res), res


def summarize(name, res):
    if not res:
        return {"arm": name, "n": 0}
    return {"arm": name, "n": len(res),
            "top1": float(np.mean([r["top1"] for r in res])),
            "top3": float(np.mean([r["top3"] for r in res])),
            "mrr": float(np.mean([r["rr"] for r in res])),
            "chance": float(np.mean([1.0 / r["k"] for r in res]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="t5", choices=["t5", "gpt2"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_rows, _ = load_demos(DATA_ROOT, ["notepad"])
    train_rows, val_rows = notepad_candidates(all_rows)
    qgis = qgis_candidates()
    qgis_nl = [dict(r, instruction=NO_LEAK.get(r["instruction"], r["instruction"])) for r in qgis]
    print(f"[data] notepad train={len(train_rows)} val={len(val_rows)}; qgis={len(qgis)}")

    enc = TextEncoder(args.encoder, device)
    print(f"[enc] {args.encoder} (frozen), dim=768")

    report = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "encoder": args.encoder,
                       "steps": args.steps, "lr": args.lr, "temp": args.temp,
                       "n_train": len(train_rows), "n_val": len(val_rows),
                       "n_qgis": len(qgis)}, "arms": {}, "detail": {}}

    def add(split, s, res=None):
        report["arms"][f"{s['arm']}@{split}"] = s
        if res is not None:
            report["detail"][f"{s['arm']}@{split}"] = res

    # R1 lexical
    for split, rows in (("notepad_val", val_rows), ("qgis_open", qgis), ("qgis_noleak", qgis_nl)):
        s, res = eval_scorer(bow_cos, rows, "R1_lexical")
        add(split, s, res)
        print(f"[{split}] R1_lexical: top1={s['top1']:.3f} top3={s['top3']:.3f} (n={s['n']})")

    # R2 frozen semantic
    def frozen(instr, cand):
        a, b = enc.embed([instr, cand])
        return float(a @ b)

    for split, rows in (("notepad_val", val_rows), ("qgis_open", qgis), ("qgis_noleak", qgis_nl)):
        s, res = eval_scorer(frozen, rows, "R2_frozen")
        add(split, s, res)
        print(f"[{split}] R2_frozen: top1={s['top1']:.3f} top3={s['top3']:.3f} (n={s['n']})")

    # R3 learned projections（只在 notepad train 上训）
    print(f"[train] R3_learned on {len(train_rows)} queries, {args.steps} steps")
    pq, pc = train_retriever(enc, train_rows, args.steps, args.lr, args.temp)
    for split, rows in (("notepad_train", train_rows), ("notepad_val", val_rows),
                        ("qgis_open", qgis), ("qgis_noleak", qgis_nl)):
        s = evaluate_learned(pq, pc, enc, rows, args.temp)
        s["arm"] = "R3_learned"
        add(split, s)
        print(f"[{split}] R3_learned: top1={s['top1']:.3f} top3={s['top3']:.3f} "
              f"mrr={s['mrr']:.3f} chance={s['chance']:.3f} (n={s['n']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
