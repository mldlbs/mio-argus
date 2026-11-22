"""模态有效性消融（A0-G6 门禁）：模型到底看不看图像？

对每条样本做 4 次前向：
  0. 原样（图 + 指令）
  1. 灰图（图像无信息，指令不变）
  2. 换图（错误图像，指令不变）
  3. 换指令（图像不变，错误指令）
报 |Δcoord|₁ 与相对特征变化；判定"图像是否被使用"。

用法：
    python experiments/computer_use/ablate_modalities.py --ckpt artifacts/checkpoints/xxx.pth
输出：
    results/computer_use/modality_ablation.json

判定（写进 A0-G6）：
    image_effect / text_effect < 0.10  → 图像通路失效（该臂不得进入比较实验）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mio"))
from mio_argus import ComputerUseModelV3, get_image_processor, get_tokenizer  # noqa: E402

DATA = ROOT / "artifacts/computer_use_data_real_balanced/balanced_training_data.json"


def load(ckpt, device):
    m = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state.get("model", state))
    m.eval()
    buf = {}
    m.out_proj.register_forward_hook(lambda mod, i, o: buf.__setitem__("f", i[0].detach()))
    return m, buf


def run(ckpt, n, res, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, buf = load(ckpt, device)
    tok, proc = get_tokenizer(), get_image_processor()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    rows = [r for r in data if r["action"] == "click"][:n]

    def img_of(r):
        p = Path(r["screenshot"])
        p = p if p.is_absolute() else ROOT / p
        return Image.open(p).convert("RGB").resize((res, res))

    gray = Image.new("RGB", (res, res), (128, 128, 128))

    def fwd(img, instr):
        px = proc(images=img, return_tensors="pt", do_resize=False)["pixel_values"]
        enc = tok(instr, return_tensors="pt", truncation=True, max_length=64,
                  padding="max_length")
        with torch.no_grad():
            o = model(px.to(device), enc["input_ids"].to(device),
                      enc["attention_mask"].to(device))
        return (o["coord_pred"][0].float().cpu().numpy(),
                int(o["action_logits"][0].argmax()),
                buf["f"].squeeze(0).float().cpu().numpy())

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    perm_t = np.random.default_rng(1).permutation(len(rows))
    acc, base, d = {}, [], {"gray": [], "swap_img": [], "swap_text": []}
    dfeat = {k: [] for k in d}
    accs = {k: 0 for k in ("orig", "gray", "swap_img", "swap_text")}

    for i, r in enumerate(rows):
        c0, a0, f0 = fwd(img_of(r), r["instruction"])
        c1, a1, f1 = fwd(gray, r["instruction"])
        c2, a2, f2 = fwd(img_of(rows[perm[i]]), r["instruction"])
        c3, a3, f3 = fwd(img_of(r), rows[perm_t[i]]["instruction"])
        base.append(np.abs(f0).mean())
        for k, (c, f) in zip(("gray", "swap_img", "swap_text"), ((c1, f1), (c2, f2), (c3, f3))):
            d[k].append(float(np.abs(c - c0).sum()))
            dfeat[k].append(float(np.abs(f - f0).mean()))
        accs["orig"] += int(a0 == 0)
        accs["gray"] += int(a1 == 0)
        accs["swap_img"] += int(a2 == 0)
        accs["swap_text"] += int(a3 == 0)

    b = float(np.mean(base))
    image_effect = max(float(np.mean(d["gray"])), float(np.mean(d["swap_img"])))
    text_effect = float(np.mean(d["swap_text"]))
    ratio = image_effect / max(text_effect, 1e-9)
    verdict = "IMAGE_USED" if ratio >= 0.10 else "IMAGE_IGNORED"

    result = {
        "ckpt": str(ckpt), "res": res, "n": len(rows),
        "delta_coord_l1": {k: float(np.mean(v)) for k, v in d.items()},
        "delta_feat_rel": {k: float(np.mean(v)) / b for k, v in dfeat.items()},
        "action_click_acc": {k: f"{v}/{len(rows)}" for k, v in accs.items()},
        "image_effect": image_effect, "text_effect": text_effect,
        "image_over_text": ratio, "verdict": verdict,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out_path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=84)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="results/computer_use/modality_ablation.json")
    a = ap.parse_args()
    run(a.ckpt, a.n, a.res, a.out)
