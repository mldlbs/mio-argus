"""
W28b 表征改造复测：把"回归全屏坐标"换成"在候选控件里选一个"（affordance 排序）

背景：W28B_RESULTS §7 的结论是坐标目标没打通——模型对 OOD 帧退化成输出屏幕中心。
但采集数据里本来就有 UIA 控件清单（text + rect），所以定位不必是像素回归：
给定一帧 + 指令 + K 个候选控件，正确答案是**其中一个候选**，这是检索/排序问题。

三个臂（同一批候选，只差打分方式）：
  P0 coord      用 V3 模型预测的点击点，落在哪个候选框里就选哪个（需要 --ckpt；无则跳过）
  P1 bow        纯词面重叠（小写 token 余弦），不用任何模型
  P2 gpt2       GPT-2 均值池化句向量余弦（冻结，不训练）

评测集：
  - QGIS 零样本：eval_qgis/frames.json 的 55 个按钮，6 条工具栏指令（chance≈1.8%）
  - Notepad 留出：带 affordance 的 click 样本（按任务切分，与 train_inloop 同切法）

指标：top1 / top3 / MRR，并给 chance 参照。

用法:
    python experiments/computer_use/affordance_pick.py
    python experiments/computer_use/affordance_pick.py --ckpt artifacts/checkpoints/inloop_notepad.pth
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [p for p in (str(HERE), str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from train_inloop import DATA_ROOT, load_demos, split_by_task  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "results" / "computer_use" / "affordance_pick.json"
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def bow_vec(text):
    v = np.zeros(4096, dtype=np.float32)
    for tok in tokens(text):
        v[hash(tok) % 4096] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def cos(a, b):
    return float(np.dot(a, b))


# ------------------------------------------------------------------ datasets ---
def notepad_candidates(rows, val_ratio=0.25, seed=42):
    """带 affordance 的 click 样本；标签 = 包含点击点的候选框。"""
    usable = []
    for r in rows:
        aff = (r.get("observation") or {}).get("affordances") or []
        if r["action"] != "click" or not aff:
            continue
        nc = r.get("norm_coord")
        if not nc:
            continue
        screen = r.get("screen") or [2560, 1440]
        px, py = nc[0] * screen[0], nc[1] * screen[1]
        idx = next((i for i, b in enumerate(aff)
                    if b["rect"][0] <= px <= b["rect"][2] and b["rect"][1] <= py <= b["rect"][3]), None)
        if idx is None:
            continue
        usable.append({"row": r, "cands": [b["text"] for b in aff], "label": idx,
                       "task": r["_task"], "_task": r["_task"], "instruction": r["instruction"]})
    if not usable:
        return [], []
    train_rows, val_rows, _ = split_by_task(usable, val_ratio, seed)
    return train_rows, val_rows


def qgis_candidates():
    """QGIS 零样本：55 个 UIA 按钮，标签 = 该指令对应的按钮名。"""
    payload = json.loads((DATA_ROOT / "eval_qgis" / "frames.json").read_text(encoding="utf-8"))
    names = list(payload["buttons_sample"].keys())
    rows = []
    for r in payload["rows"]:
        target = r["meta"]["uia_name"]
        if target not in names:
            continue
        rows.append({"instruction": r["instruction"], "cands": names,
                     "label": names.index(target), "task": "qgis_eval"})
    return rows


# -------------------------------------------------------------------- scorers ---
class Gpt2Scorer:
    def __init__(self, device):
        from transformers import GPT2Model, GPT2Tokenizer
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.model = GPT2Model.from_pretrained("gpt2").to(device).eval()
        self.device = device

    def embed(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=32, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((out * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()


def rank(score_fn, row):
    scores = np.array([score_fn(row["instruction"], c) for c in row["cands"]], dtype=np.float32)
    order = np.argsort(-scores)
    rank_of_label = int(np.where(order == row["label"])[0][0]) + 1
    return {
        "top1": rank_of_label == 1,
        "top3": rank_of_label <= 3,
        "rr": 1.0 / rank_of_label,
        "rank": rank_of_label,
        "k": len(row["cands"]),
        "pick": row["cands"][order[0]],
        "target": row["cands"][row["label"]],
    }


def summarize(name, results):
    if not results:
        return {"arm": name, "n": 0}
    return {
        "arm": name,
        "n": len(results),
        "top1": float(np.mean([r["top1"] for r in results])),
        "top3": float(np.mean([r["top3"] for r in results])),
        "mrr": float(np.mean([r["rr"] for r in results])),
        "chance": float(np.mean([1.0 / r["k"] for r in results])),
    }


# ------------------------------------------------------------------- coord arm ---
def coord_picks(ckpt, rows, frames_payload):
    """用 V3 预测点 -> 落在哪个候选框就选哪个；QGIS 帧共用同一张图。"""
    from mio_argus import ACTION_MAP, ComputerUseModelV3, get_image_processor, get_tokenizer
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComputerUseModelV3(num_actions=len(ACTION_MAP))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.to(device).eval()
    tok, ip = get_tokenizer(), get_image_processor()

    out = []
    for r in rows:
        img = Image.open(r["_screenshot"]).convert("RGB").resize((224, 224))
        px = ip(images=img, return_tensors="pt")["pixel_values"].to(device)
        enc = tok(r["instruction"], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            o = model(px, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        cx, cy = o["coord_pred"][0].tolist()
        screen = r.get("screen") or [2560, 1440]
        x, y = cx * screen[0], cy * screen[1]
        rects = r["_rects"]
        pick = next((i for i, b in enumerate(rects)
                     if b[0] <= x <= b[2] and b[1] <= y <= b[3]), None)
        out.append({"instruction": r["instruction"], "cands": r["cands"],
                    "label": r["_label"], "pick_idx": pick,
                    "xy": [round(x, 1), round(y, 1)],
                    "task": r["_task"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "artifacts" / "checkpoints" / "inloop_notepad.pth"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_rows, _ = load_demos(DATA_ROOT, ["notepad"])
    train_rows, val_rows = notepad_candidates(all_rows)
    qgis_rows = qgis_candidates()
    print(f"[data] notepad usable={len(train_rows) + len(val_rows)} (train={len(train_rows)} val={len(val_rows)})")
    print(f"[data] qgis zero-shot n={len(qgis_rows)} K={len(qgis_rows[0]['cands']) if qgis_rows else 0}")

    gpt2 = Gpt2Scorer(device)
    cache = {}

    def gpt2_cos(instr, cand):
        if instr not in cache:
            cache[instr] = gpt2.embed([instr])[0]
        key = ("c", cand)
        if key not in cache:
            cache[key] = gpt2.embed([cand])[0]
        return cos(cache[instr], cache[key])

    arms = {
        "P1_bow": lambda i, c: cos(bow_vec(i), bow_vec(c)),
        "P2_gpt2": gpt2_cos,
    }

    report = {"meta": {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "notepad_train": len(train_rows), "notepad_val": len(val_rows),
                       "qgis_n": len(qgis_rows)}, "arms": {}}

    for split_name, rows in (("notepad_val", val_rows), ("qgis_zero_shot", qgis_rows)):
        for arm, fn in arms.items():
            res = [rank(fn, r) for r in rows]
            s = summarize(arm, res)
            report["arms"][f"{arm}@{split_name}"] = s
            print(f"[{split_name}] {arm}: top1={s['top1']:.3f} top3={s['top3']:.3f} "
                  f"mrr={s['mrr']:.3f} chance={s['chance']:.3f} (n={s['n']})")

    # 坐标臂（V3 预测点 -> 候选包含）
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        payload = json.loads((DATA_ROOT / "eval_qgis" / "frames.json").read_text(encoding="utf-8"))
        frame = str(ROOT / payload["rows"][0]["screenshot"])
        rects = payload["buttons_sample"]
        qrows = []
        for r in payload["rows"]:
            names = list(rects.keys())
            qrows.append({"instruction": r["instruction"], "cands": names,
                          "_screenshot": frame, "_rects": list(rects.values()),
                          "_label": names.index(r["meta"]["uia_name"]), "_task": "qgis_eval",
                          "screen": payload["screen"]})
        picks = coord_picks(ckpt, qrows, payload)
        res = [{"top1": p["pick_idx"] == p["label"], "top3": False, "rr": 0.0, "k": len(p["cands"]),
                "pick": p["cands"][p["pick_idx"]] if p["pick_idx"] is not None else None,
                "target": p["cands"][p["label"]]} for p in picks]
        s = summarize("P0_coord", res)
        s["note"] = "预测点不落在任何候选框时记 top1=False（W28b 的坐标塌缩）"
        report["arms"]["P0_coord@qgis_zero_shot"] = s
        report["P0_detail"] = picks
        print(f"[qgis_zero_shot] P0_coord: top1={s['top1']:.3f} (n={s['n']}) "
              f"picks={[p['cands'][p['pick_idx']] if p['pick_idx'] is not None else None for p in picks]}")

    report["meta"]["candidates"] = {
        "notepad": "observation.affordances（UIA 控件 text+rect）",
        "qgis": "eval_qgis/frames.json buttons_sample（55 个 UIA 按钮）",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
