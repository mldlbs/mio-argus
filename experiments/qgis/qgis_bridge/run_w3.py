"""
QGIS W3：学习型结构策略 vs 关键词（动作空间上的"指令→动作"）

数据：QGIS 自己的动作空间（find_actions("") 拿到全部动作名）。
样本：每个动作 → 指令就是它的名字；候选 = 全部动作；目标 = 该动作。
划分：**按动作划分 held-out**（测试集动作名训练时完全没见过）。
对照：关键词基线（与 real_site_policy 同款）。

另测：W1/W2 的**润色指令**（"zoom in the map"）能否零样本对上 "Zoom In"。

用法:
    python experiments/qgis/qgis_bridge/run_w3.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

B = 1024


def bag(s):
    s = s.lower().strip()
    v = np.zeros(B, np.float32)
    if not s:
        return v
    if len(s) < 3:
        v[zlib.crc32(s.encode()) % B] += 1
    else:
        for i in range(len(s) - 2):
            v[zlib.crc32(s[i:i + 3].encode()) % B] += 1
    m = v.sum()
    return v / m if m > 0 else v


PARAPHRASES = {
    "activate the zoom in tool": "Zoom In",
    "activate the pan tool": "Pan Map",
    "zoom to the full extent of the cities layer": "Zoom Full",
    "open the python console": "Python Console",
    "select all features of the cities layer": "Select All Features",
    "clear the current selection": "Deselect Features from All Layers",
}


def keyword_pick(texts, instr):
    words = [w for w in instr.lower().replace(",", " ").split() if len(w) > 2]
    best, bs = None, 0.0
    for t in texts:
        s = sum(1.0 for w in words if w in t.lower())
        if s > bs:
            best, bs = t, s
    return best


def main():
    r = call({"op": "find_actions", "substr": ""})
    acts = [a["text"] for a in r["actions"] if a["text"].strip()]
    acts = sorted(set(acts))
    print(f"动作空间大小 = {len(acts)}")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    names = acts
    T = np.stack([bag(n) for n in names])
    X = torch.tensor(T)
    N = len(names)

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    cut = int(N * 0.8)
    tr, te = perm[:cut], perm[cut:]
    # 每个样本：指令 = 该动作名，目标 = 它的下标
    Xtr, Ytr = X[tr], torch.tensor(tr)

    class Net(nn.Module):
        def __init__(s, d=96):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(B, d), nn.ReLU(), nn.Linear(d, d))

        def forward(s, q):
            return s.enc(q) @ s.enc(X).T

    torch.manual_seed(0)
    m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    bs = 32
    for ep in range(60):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            idx = p[i:i + bs]
            loss = F.cross_entropy(m(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        acc_tr = float((m(Xtr).argmax(-1) == Ytr).float().mean())
        Xte = X[te]
        pred_te = m(Xte).argmax(-1)
        acc_te = float((pred_te == torch.tensor(te)).float().mean())

    kw_te = sum(1 for i in te if keyword_pick(names, names[i]) == names[i]) / len(te)

    print(f"\n{'':<22}{'learned':>10}{'keyword':>10}")
    print("-" * 44)
    print(f"{'指令→动作 (训练动作)':<22}{acc_tr:>10.3f}{'—':>10}")
    print(f"{'指令→动作 (未见动作)':<22}{acc_te:>10.3f}{kw_te:>10.3f}")
    print(f"  （chance = {1/N:.4f}）")

    print(f"\n{'润色指令（零样本）':<24}{'learned':>10}{'keyword':>10}")
    print("-" * 48)
    hit_l = hit_k = 0
    for instr, want in PARAPHRASES.items():
        if want not in names:
            print(f"  ({want} 不在动作空间里，跳过)")
            continue
        with torch.no_grad():
            p = names[int(m(torch.tensor(bag(instr))[None])[0].argmax())]
        k = keyword_pick(names, instr)
        hit_l += int(p == want)
        hit_k += int(k == want)
        print(f"  {instr[:34]:<36} {p[:16]:<16} {k[:16]:<16} 目标={want[:18]}")
    print(f"  命中：learned {hit_l}/{len(PARAPHRASES)}  keyword {hit_k}/{len(PARAPHRASES)}")

    (HERE / "w3_results.json").write_text(json.dumps({
        "n_actions": N,
        "instr_to_action_train": acc_tr,
        "instr_to_action_unseen": acc_te,
        "keyword_unseen": kw_te,
        "paraphrase_learned": hit_l, "paraphrase_keyword": hit_k,
        "paraphrase_n": len(PARAPHRASES),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w3_results.json")


if __name__ == "__main__":
    main()
