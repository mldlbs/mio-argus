"""E1-L · M1 预扫描 harness（新文件；不改 p3_env / p3_verifier / h4_density）。

目的：在**同一模型、同一任务流**上测**真实多步执行成功率**，
      检验候选 (干扰密度 d, 模型宽度 w) 能否让 H=1/2/4/8 全部落在 [0.2, 0.9]。

执行语义（本文件的核心契约）
---------------------------
1. **真多步**：一个 episode = 一条长度 H 的目标序列；第 t+1 步的输入是第 t 步执行后的屏幕，
   不是另一个独立单步样本。
2. **误差累积**：错误点击在被点格子上留红框、进度不前进；该状态**进入下一步输入**，
   直到 UI 重随机（ρ）才被清掉。没有"每步重置到正确状态"。
3. **H 参数**：任务脚本一次生成到 H_MAX=8，H 只截断执行长度 —— 同一 (seed, ρ) 下
   H=4 是 H=8 的**前缀**（同一任务、同一模型）。

状态机口径（两种掩码不同作用域，不得互相推导）
------------------------------------------------
| 量 | 含义 | UI 重随机（ρ）时 |
|---|---|---|
| `progress` | **累计**已完成的目标数（0..H），任务进度 | **保留**（任务目标是 H 个连续目标，不随布局改变） |
| `done[slot]` | **当前布局**中"这个格子已被点过" | 清零（布局换了，旧格子不再指同一元素） |
| `wrong[slot]` | **当前布局**中"这个格子被误点过" | 清零（同上） |

不变量：`progress == 前 t 步中正确的步数`；`success ⇔ progress == H`。
这与 `p3_env`/`h3b` 的语义一致（漂移只重置标记，不重置进度）。
4. **统计**：逐 (候选, seed, H) 报 成功数/总数/成功率/Wilson 95% CI，并报逐 step 正确率。

用法：
    python -m pytest experiments/paper/test_e1l_m1_prescan.py -q     # 逻辑测试，不训练
    python experiments/paper/e1l_m1_prescan.py --stage logic --tasks 20
    python experiments/paper/e1l_m1_prescan.py --stage full          # 12 次训练（需批准）
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from p0_env import LABELS, PALETTE, instruction_onehot
from p3_env import (BAR_BG, BAR_FG, BAR_Y0, BAR_Y1, BG, DONE, WRONG,
                    _font, _slot_box, fresh_labels)

IMG_SIZE = 160
GRID = 6
S = GRID * GRID
H_MAX = 8
HS = [1, 2, 4, 8]
RHO = 0.3
N_TASKS = 500

CANDS = [("d1w8", 1.0, 8), ("d1w12", 1.0, 12),
         ("d1w16", 1.0, 16), ("d1w24", 1.0, 24)]


# ------------------------------------------------------------------ 渲染

def make_bg(level, rng):
    if level == "flat":
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG)
    if level == "noise":
        return Image.fromarray(
            rng.integers(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), "RGB")
    raise ValueError(f"未知背景: {level}")


def gen_decoys(cfg, target_color, d, rng, top):
    """与 H4 同一条规则（2×2 子格、同调色板、排除目标色），格框换成带进度条的版本。

    d 为每格干扰数；d<1 时只在 round(36*d) 个格子里各放 1 个。
    """
    out = []
    k_full = int(math.floor(d))
    frac = d - k_full
    cells = set()
    if frac > 0:
        n_extra = int(round(cfg.num_slots * frac))
        cells = {int(v) for v in rng.choice(cfg.num_slots, size=n_extra, replace=False)}
    tgt = tuple(int(v) for v in target_color)
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = _slot_box(slot, cfg, top)
        w, h = (bx2 - bx1) // 2, (by2 - by1) // 2
        k = min(4, k_full + (1 if slot in cells else 0))
        if k <= 0:
            continue
        for j in rng.choice(4, size=k, replace=False):
            sx = bx1 + (int(j) % 2) * w
            sy = by1 + (int(j) // 2) * h
            ew = max(5, int(w * 0.70) - int(rng.integers(0, 3)))
            eh = max(5, int(h * 0.70) - int(rng.integers(0, 3)))
            ex1 = sx + int(rng.integers(0, max(1, w - ew)))
            ey1 = sy + int(rng.integers(0, max(1, h - eh)))
            color = None
            for _ in range(8):
                c = tuple(int(v) for v in PALETTE[int(rng.integers(0, len(PALETTE)))])
                if c != tgt:
                    color = c
                    break
            if color is not None:
                out.append(((ex1, ey1, min(ex1 + ew, IMG_SIZE - 1),
                             min(ey1 + eh, IMG_SIZE - 1)), color))
    return out


def render_state(cfg, labels, progress, H, wrong, done, target_label,
                 d=0.0, level="flat", decoy_seed=0, geom_seed=0, bar_total=None):
    """背景 → 网格 → 干扰项 → 元素 → 进度条。

    契约 1：`level="flat"`、`d=0`、`bar_total=H` 时与 `p3_env.render_ui(...)` **逐像素相同**
    （见 test_flat_zero_decoys_matches_p3_render_ui）。
    契约 2：**默认 `bar_total=H_MAX`**，即进度条按"总共 8 步"着色，与评估用的 H 无关 ——
    否则同一 progress 在 H=4/8 下会画出不同长度的条，观测分布随 H 变化，
    前缀一致性（test_prefix_invariance_across_H）就不成立。
    几何只由 `geom_seed` 决定，背景/干扰项用 `decoy_seed`，互不干扰。
    """
    rng = np.random.default_rng(geom_seed)
    top = BAR_Y1 + 4
    img = make_bg(level, np.random.default_rng(decoy_seed)).copy()
    dr = ImageDraw.Draw(img)

    for slot in range(cfg.num_slots):
        x1, y1, x2, y2 = _slot_box(slot, cfg, top)
        dr.rectangle([x1, y1, x2, y2], outline=(214, 218, 224))

    if d > 0:
        tgt_color = PALETTE[LABELS.index(target_label) % len(PALETTE)]
        for (x1, y1, x2, y2), color in gen_decoys(
                cfg, tgt_color, d, np.random.default_rng(decoy_seed + 1), top):
            dr.rectangle([x1, y1, x2, y2], fill=color, outline=(25, 25, 25))

    els = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = _slot_box(slot, cfg, top)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, IMG_SIZE - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, IMG_SIZE - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, IMG_SIZE - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, IMG_SIZE - 1))

        label = labels[slot]
        color = PALETTE[LABELS.index(label) % len(PALETTE)]
        if done[slot]:
            color = DONE
        outline = WRONG if wrong[slot] else (25, 25, 25)
        width = 3 if wrong[slot] else 1
        dr.rectangle([ex1, ey1, ex2, ey2], fill=color, outline=outline, width=width)
        if (ex2 - ex1) >= 30 and (ey2 - ey1) >= 14:
            f = _font(10)
            bb = dr.textbbox((0, 0), label, font=f)
            dr.text((ex1 + max(0, ((ex2 - ex1) - (bb[2] - bb[0])) // 2), ey1 + 2),
                    label, fill=(255, 255, 255), font=f)
        els.append((slot, (ex1, ey1, ex2, ey2), label,
                    tuple(int(v) for v in color)))

    x0, x1 = 3, IMG_SIZE - 4
    dr.rectangle([x0, BAR_Y0, x1, BAR_Y1], fill=BAR_BG, outline=(200, 204, 210))
    total = bar_total if bar_total is not None else H_MAX
    frac = progress / max(1, total)
    if frac > 0:
        xf = int(x0 + (x1 - x0) * frac)
        dr.rectangle([x0, BAR_Y0 + 1, max(x0 + 1, xf), BAR_Y1 - 1], fill=BAR_FG)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), els


# ------------------------------------------------------------- 任务脚本

@dataclass
class Step:
    labels: list
    target_slot: int
    drift: bool


def build_scripts(cfg, seed, rho, n_tasks, h_max=H_MAX):
    """一次生成到 h_max；H 只做截断，保证 H 之间的前缀一致（与 ρ 绑定）。"""
    scripts = []
    for i in range(n_tasks):
        rng = np.random.default_rng(seed * 1_000_003 + i + int(round(rho * 1000)) * 7919)
        labels = fresh_labels(cfg, rng)
        seq = []
        for t in range(h_max):
            drift = (t > 0 and rng.random() < rho)
            if drift:
                labels = fresh_labels(cfg, rng)
            seq.append(Step(list(labels), int(rng.integers(0, cfg.num_slots)), drift))
        scripts.append(seq)
    return scripts


# ---------------------------------------------------------------- rollout

def rollout(policy, cfg, script, H, d, seed, task_idx, rho, level="flat",
            keep_frames=False, keep_states=False):
    """在**同一条任务脚本**上前缀截断执行 H 步；返回轨迹与状态。

    `progress` 是累计量（不随 ρ 重置）；`wrong`/`done` 是当前布局的标记（随 ρ 清空）。
    `keep_states=True` 时返回每一步**渲染前**的 (progress, wrong, done)，供审计与测试。
    """
    wrong = [False] * cfg.num_slots
    done = [False] * cfg.num_slots
    progress = 0
    frames, actions, step_ok, states = [], [], [], []

    for t in range(H):
        st = script[t]
        if st.drift:                      # UI 重随机：旧标记随新布局一起消失（环境语义）
            wrong = [False] * cfg.num_slots
            done = [False] * cfg.num_slots

        geom = task_idx * 100 + t          # 与 H 无关 → 前缀一致
        decoy_seed = (seed * 1_000_003 + task_idx * 97 + t * 13
                      + int(round(d * 10)) * 17 + int(round(rho * 100)) * 29)
        if keep_states:
            states.append((progress, list(wrong), list(done)))
        img, els = render_state(cfg, st.labels, progress, H, wrong, done,
                                st.labels[st.target_slot], d=d, level=level,
                                decoy_seed=decoy_seed, geom_seed=geom)
        if keep_frames:
            frames.append(img)

        act = int(policy.act(img, st.labels[st.target_slot],
                             {"correct_slot": st.target_slot, "step": t}))
        actions.append(act)

        ok = (act == st.target_slot)
        step_ok.append(ok)
        if ok:
            done[act] = True
            progress += 1
        else:
            wrong[act] = True               # 标记**留在屏幕上**，进入下一步输入

    return {"success": progress == H, "progress": progress,
            "frames": frames, "actions": actions, "step_ok": step_ok,
            "states": states}


# ------------------------------------------------------------------ 政策

class OraclePolicy:
    """测试替身：按预设的"第几步故意错"出招。真实策略不得读 ctx。"""

    def __init__(self, wrong_steps=()):
        self.wrong_steps = set(wrong_steps)
        self.calls = 0

    def act(self, img, target, ctx):
        self.calls += 1
        correct = ctx["correct_slot"]
        if ctx["step"] in self.wrong_steps:
            return (correct + 1) % 36
        return correct


class NNPolicy:
    """真实策略：忽略 ctx，只用 (图像, 指令)。"""

    def __init__(self, model, device):
        import p3_verifier as P3
        self.model, self.device, self._P3 = model, device, P3

    def act(self, img, target, ctx=None):
        p = self._P3.pi_probs(self.model, img, target, self.device)
        return int(np.argmax(p))


# ------------------------------------------------------------------ 统计

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - half), min(1.0, c + half))


def evaluate(policy, cfg, scripts, H, d, seed, rho, level="flat"):
    succ, per_step = 0, []
    for ti, sc in enumerate(scripts):
        r = rollout(policy, cfg, sc, H, d, seed, ti, rho, level)
        succ += int(r["success"])
        per_step.append(r["step_ok"])
    n = len(scripts)
    lo, hi = wilson_ci(succ, n)
    acc_by_step = [float(np.mean([ps[t] for ps in per_step])) for t in range(H)]
    return {"H": H, "n": n, "success": succ, "rate": succ / n,
            "ci": [lo, hi], "acc_by_step": acc_by_step}


# ------------------------------------------------------------------- main

def parse_cands(spec):
    """候选解析：空/"all" = 全部；未知名字直接报错。

    默认必须是"全部"：曾经因为 CLI 默认只跑 `d1w16`，执行矩阵与批准的矩阵不一致，
    而脚本仍然照常输出结果（裁定 23 的同一类问题）。"""
    names = {c[0] for c in CANDS}
    if spec.strip().lower() in ("", "all"):
        return list(CANDS)
    want = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [w for w in want if w not in names]
    if bad:
        raise SystemExit(f"未知候选 {bad}；可用：{sorted(names)}")
    return [c for c in CANDS if c[0] in want]


def expand_matrix(cands, seeds, rhos, hs=HS):
    """执行矩阵（纯函数，可测试）：候选 × seed × ρ × H。

    **训练只按 (候选, seed)** —— ρ 是评估参数，不重训。
    返回的每一项都带 rho 与 tag（`name@r0.3`），保证两套曲线不会混。
    """
    return [{"cand": n, "d": d, "w": w, "seed": sd, "rho": r, "H": H,
             "tag": f"{n}@r{r}"}
            for (n, d, w) in cands for sd in seeds for r in rhos for H in hs]


def train_units(matrix):
    """需要训练的 (候选, seed) 组合（12 个 = 4 × 3，与 ρ 无关）。"""
    return sorted({(m["cand"], m["seed"]) for m in matrix})


def check_completeness(rows, matrix):
    """核对写出的行是否覆盖整个矩阵；返回缺失的 (tag, seed, H) 列表。"""
    have = {(r["cond"], r["seed"], r["x"]) for r in rows}
    want = {(m["tag"], m["seed"], m["H"]) for m in matrix}
    return sorted(want - have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("logic", "full"), default="logic")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cands", default="all")
    ap.add_argument("--rhos", default="0.3,0.0",
                    help="逗号分隔；0.3 为主条件，0.0 为对照（都会评估）")
    ap.add_argument("--train-mode", choices=("zero", "mixed"), default="zero")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="results/paper/e1l_m1_prescan.jsonl")
    ap.add_argument("--gate-out", default="results/paper/e1l_m1_prescan_gate.jsonl")
    args = ap.parse_args()

    cfg = EnvCfg6()
    seeds = [int(s) for s in args.seeds.split(",")]
    rhos = [float(v) for v in args.rhos.split(",")]
    cands = parse_cands(args.cands)
    matrix = expand_matrix(cands, seeds, rhos)
    units = train_units(matrix)
    print(f"[matrix] 候选 {[c[0] for c in cands]}；训练 {len(units)} 次（候选×seed），"
          f"评估 {len(matrix)} 格（×{len(rhos)} 个 ρ）；ρ = {rhos}", flush=True)

    if args.stage == "logic":
        for m in matrix:
            scripts = build_scripts(cfg, m["seed"], m["rho"], args.tasks)
            r = evaluate(OraclePolicy(), cfg, scripts, m["H"], m["d"], m["seed"], m["rho"])
            print(f"[logic] {m['tag']} s{m['seed']} H{m['H']}: rate={r['rate']:.3f} "
                  f"CI=[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        return

    import torch
    import h4_density as H4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for f in (args.out, args.gate_out):
        if Path(f).exists():
            Path(f).unlink()

    models = {}
    for name, s in units:
        d = next(m["d"] for m in matrix if m["cand"] == name)
        w = next(m["w"] for m in matrix if m["cand"] == name)
        tr = build_train_data(cfg, 3000, 100, d, s, mode=args.train_mode)
        models[(name, s)] = NNPolicy(H4.train(w, tr, args.epochs, device, seed=s), device)
        print(f"[train] {name} seed={s} 完成（累计 {len(models)}/{len(units)}）", flush=True)

    for m in matrix:
        policy = models[(m["cand"], m["seed"])]
        scripts = build_scripts(cfg, m["seed"], m["rho"], N_TASKS)
        r = evaluate(policy, cfg, scripts, m["H"], m["d"], m["seed"], m["rho"])
        row = {"cand": m["cand"], "tag": m["tag"], "d": m["d"], "w": m["w"],
               "seed": m["seed"], "H": m["H"], "rho": m["rho"], "n": r["n"],
               "success": r["success"], "rate": round(r["rate"], 4),
               "ci_lo": round(r["ci"][0], 4), "ci_hi": round(r["ci"][1], 4),
               "acc_by_step": [round(v, 4) for v in r["acc_by_step"]]}
        with Path(args.out).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with Path(args.gate_out).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cond": m["tag"], "x": m["H"], "seed": m["seed"],
                                 "value": row["rate"]}, ensure_ascii=False) + "\n")
        print(f"[eval] {m['tag']} s{m['seed']} H{m['H']}: {row['rate']:.3f} "
              f"CI=[{row['ci_lo']}, {row['ci_hi']}]", flush=True)

    rows = [json.loads(x) for x in Path(args.gate_out).read_text(
        encoding="utf-8").splitlines() if x.strip()]
    missing = check_completeness(rows, matrix)
    print(f"\n[check] gate 行数 {len(rows)} / 期望 {len(matrix)}；"
          f"缺失 {len(missing)}", flush=True)
    if missing:
        print("[check] 缺失示例:", missing[:5], flush=True)
        raise SystemExit(1)
    print("[check] 矩阵完整：4 候选 × 3 seed × 2 ρ × 4 H 全部产生", flush=True)


def build_train_data(cfg, n, split_seed, d, seed_for_decoys, mode="zero"):
    """单步训练数据；干扰项密度与评估一致。

    mode="zero"  ：progress 恒为 0（本轮 M1 用的原协议，默认，行为不变）
    mode="mixed" ：progress 覆盖评估中可达的中间状态（M1b 诊断臂 B）
                   H_eval ~ U{1,2,4,8}；t ~ U[0, H_eval-1]；progress ~ U[0, t]
                   （不取 progress=H，因为那一步在评估中不会作为"待决策"状态出现）
    """
    scripts = build_scripts(cfg, split_seed, 0.0, n)
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    for i in range(n):
        st = scripts[i][0]
        if mode == "mixed":
            mrng = np.random.default_rng(split_seed * 1_000_003 + 90_001 + i)
            h_eval = int(mrng.choice([1, 2, 4, 8]))
            t = int(mrng.integers(0, h_eval))
            prog = int(mrng.integers(0, t + 1))
        elif mode == "zero":
            prog = 0
        else:
            raise ValueError(f"未知 mode: {mode}")
        img, _ = render_state(cfg, st.labels, prog, 1, [False] * cfg.num_slots,
                              [False] * cfg.num_slots, st.labels[st.target_slot],
                              d=d, level="flat",
                              decoy_seed=seed_for_decoys * 1_000_003 + i,
                              geom_seed=i)
        imgs[i] = (img * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {st.labels[st.target_slot]}", len(LABELS))
        slots[i] = st.target_slot
    return {"images": imgs, "inst": inst, "slots": slots}


def EnvCfg6():
    from p0_env import EnvCfg
    return EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=S,
                  fill=0.40, jitter=4, color_by_label=True)


if __name__ == "__main__":
    main()
