"""
QGIS W7：路由器式 hybrid（结构可用→struct；不可用→pixel）

W6 的 hybrid 是"朴素相加"，被弱像素分支拖累（0.727 < 0.854）。本轮改成**路由**：
  - 候选**有名字**且指令与某个名字有词重叠 → 走 struct 分支
  - 否则（名字被抹掉 = 模拟"没有可命名控件的区域"，如地图画布）→ 走 pixel 分支

在同一个 1200 样本集上量：
  normal（有名字）/ masked（名字抹掉）/ mixed（各半）

用法:
    python experiments/qgis/qgis_bridge/run_w7.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_w6 as W6  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def mask_names(s):
    """造"无结构区"：把候选名字全抹掉（模拟画布/不可命名控件）。"""
    t = dict(s)
    t["names"] = [""] * W6.MAXC
    return t


def has_name_match(s):
    """路由规则：指令关键词是否出现在某个候选名里。"""
    words = [w for w in s["instr"].lower().replace(",", " ").split() if len(w) > 2]
    for n in s["names"]:
        nl = (n or "").lower()
        if any(w in nl for w in words):
            return True
    return False


def train(model, samples, device, epochs=30):
    T = tuple(t.to(device) for t in W6.featurize(samples))
    m = model.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        p = torch.randperm(len(samples), device=device)
        for i in range(0, len(samples), 32):
            idx = p[i:i + 32]
            loss = F.cross_entropy(m(*[t[idx] for t in T[:5]]), T[5][idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def preds(m, samples, device):
    T = tuple(t.to(device) for t in W6.featurize(samples))
    m.eval()
    return m(*T[:5]).argmax(-1).cpu().numpy(), T[5].cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = list(np.load(W6.DATA, allow_pickle=True)["rows"])
    names_all = sorted({s["instr"] for s in rows})
    seen_names = set(names_all[:int(len(names_all) * 0.75)])
    seen_states = set(range(len(W6.STATES) // 2))
    tr = [s for s in rows if s["instr"] in seen_names and s["state"] in seen_states]
    te = [s for s in rows if s["instr"] in seen_names and s["state"] not in seen_states]
    print(f"train={len(tr)}  held-out 状态={len(te)}  （W6 同款划分）")

    ms = train(W6.Struct(), tr, device)
    mp = train(W6.Pixel(), tr, device)

    te_masked = [mask_names(s) for s in te]
    mixed = te[::2] + te_masked[::2]
    rng = np.random.default_rng(0)
    rng.shuffle(mixed)

    def evaluate(samples):
        ps, ys = preds(ms, samples, device)
        pp, _ = preds(mp, samples, device)
        # 路由：逐样本决定用哪支
        route_struct = np.array([has_name_match(s) for s in samples])
        pr = np.where(route_struct, ps, pp)
        ys_np = np.array([s["target"] for s in samples])
        return (float((ps == ys_np).mean()), float((pp == ys_np).mean()),
                float((pr == ys_np).mean()), float(route_struct.mean()))

    print(f"\n{'集合':<26}{'struct':>9}{'pixel':>9}{'router':>9}{'走struct比例':>13}")
    print("-" * 68)
    out = {}
    for name, S in (("normal（有名字，held-out 状态）", te),
                    ("masked（名字抹掉）", te_masked),
                    ("mixed（各半）", mixed)):
        a, b, c, r = evaluate(S)
        out[name] = dict(struct=a, pixel=b, router=c, struct_share=r)
        print(f"{name:<26}{a:>9.3f}{b:>9.3f}{c:>9.3f}{r:>13.2f}")
    print("-" * 68)
    print("注：masked 集合上 struct 分支必然失效（名字为空），router 应≈pixel；"
          "normal 上 router 应≈struct")

    print(f"\n对照 W6 的朴素相加 hybrid：normal 0.972 / held-out 状态 0.727")
    print(f"本轮 router：normal {out['normal（有名字，held-out 状态）']['router']:.3f} / "
          f"masked {out['masked（名字抹掉）']['router']:.3f}")

    (HERE / "w7_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w7_results.json")


if __name__ == "__main__":
    main()
