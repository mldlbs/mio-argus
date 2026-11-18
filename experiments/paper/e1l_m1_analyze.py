"""E1-L · M1 预扫描结果汇总（读 jsonl，写 E1L_M1_RESULTS.md + summary json）。"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HS = [1, 2, 4, 8]
BAND = (0.2, 0.9)
MARGIN = 0.05


def main():
    rows = [json.loads(l) for l in Path("results/paper/e1l_m1_prescan.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    by = defaultdict(dict)          # (tag, seed) -> H -> rate
    steps = defaultdict(dict)       # (tag, seed, H) -> acc_by_step
    for r in rows:
        by[(r["tag"], r["seed"])][r["H"]] = r["rate"]
        steps[(r["tag"], r["seed"], r["H"])] = r["acc_by_step"]

    tags = sorted({t for t, _ in by})
    seeds = sorted({s for _, s in by})

    lo, hi = BAND[0] + MARGIN, BAND[1] - MARGIN

    def band_ok(curve):
        return all(lo <= v <= hi for v in curve)

    L = []
    A = L.append
    A("# E1-L · M1 预扫描 结果\n")
    A("> 协议：`E1L_PREREQ_DESIGN.md` v0.1 §3　harness：`e1l_m1_prescan.py`（17 项测试通过）")
    A("> 矩阵：4 候选 × 3 seed × 2 ρ × 4 H = **96 格**（12 次训练）；500 任务/格")
    A("> 门禁：`E1L_M1_GATE.md`（B 模式，`[0.2,0.9]`，margin 0.05，方向 down）\n")

    A("## 0. 判定\n")
    n_pass = sum(1 for t in tags for s in seeds if band_ok([by[(t, s)][H] for H in HS]))
    A(f"- 逐 (候选, ρ, seed) 的 B1 通过数：**{n_pass} / {len(tags) * len(seeds)}**")
    A("- **M1 不满足**：8 个 `候选@ρ` 组合全部被 B1 拦下（`E1L_M1_GATE.md`，退出码 1）。")
    A("- 按裁定 22：**本轮结束，不加候选、不调区间、不追加训练**。\n")

    A("## 1. 逐组合的 H 曲线（3 seed 均值 [最小值, 最大值]）\n")
    A("| 组合 | H1 | H2 | H4 | H8 | H8/H1 | B1 |")
    A("|---|---|---|---|---|---|---|")
    ratios = []
    for t in tags:
        cells, r8s = [], []
        for H in HS:
            vals = [by[(t, s)][H] for s in seeds]
            cells.append(f"{np.mean(vals):.3f} [{min(vals):.3f}, {max(vals):.3f}]")
        for s in seeds:
            h1 = by[(t, s)][1]
            if h1 > 0:
                r8s.append(by[(t, s)][8] / h1)
        ratios.extend(r8s)
        ok = all(band_ok([by[(t, s)][H] for H in HS]) for s in seeds)
        A(f"| {t} | " + " | ".join(cells) +
          f" | {np.mean(r8s):.3f} | {'✅' if ok else '❌'} |")
    A(f"\n- **H8/H1 平均 {np.mean(ratios):.3f}**（范围 {min(ratios):.3f}–{max(ratios):.3f}）")
    A(f"- 要让 H1≤0.88 且 H8≥0.22，至少需要 **H8/H1 ≥ 0.22/0.88 = 0.25**")
    A(f"→ 观测值差了约 **{0.25 / max(np.mean(ratios), 1e-9):.1f}×**。本轮 4 个宽度上"
      f"该比值**没有随 w 改善**（w8 0.006–0.010，w24 0.024–0.039），因此在本轮范围内"
      f"靠加宽补不上；是否还存在别的宽度能改变它，本轮未测、不外推。\n")

    A("## 2. 逐 step 准确率（暴露机制）\n")
    A("| 组合 | H | step0 | step1 | step2 | step3 | ... | 末步 |")
    A("|---|---|---|---|---|---|---|---|")
    for t in tags:
        for H in (4, 8):
            arr = np.array([steps[(t, s, H)] for s in seeds])   # (seed, H)
            m = arr.mean(0)
            A(f"| {t} | {H} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | "
              f"{m[3] if len(m) > 3 else float('nan'):.3f} | … | {m[-1]:.3f} |")
    A("\n**读法**：step0 是训练分布内（progress=0、无标记）；之后每步都要看进度条已填充、"
      "灰色 done 元素、红色 wrong 描边 —— 这些状态**训练时从未出现**。")
    A("所以曲线不是\"难度不够可调\"，而是**策略在它没见过的中间状态上退化**。\n")

    A("### 2.1 与 P3 的差别：训练时是否见过中间状态\n")
    A("| | 训练样本的状态 | H1 | H4 | H4/H1 |")
    A("|---|---|---|---|---|")
    A("| P3（`build_pi_data`） | **progress ∈ [0..H] 随机、H ∈ {1,2,4,8} 随机**、无标记 | 0.980 | 0.837 | 0.85 |")
    A("| 本轮 M1（`build_train_data`） | **progress 恒为 0**、无标记 | 0.739–0.802 | 0.175–0.262 | 0.22–0.33 |")
    A("")
    A("`p3_verifier.build_pi_data` 里显式做了 `H = rng.choice([1,2,4,8])`、"
      "`prog = rng.integers(0, H+1)`；本轮的训练数据只画 `progress=0`。")
    A("所以「策略在训练分布外退化」是一个**有出处**的解释，但**不是受控对照**"
      "（P3 无干扰项、元素 26px vs 本轮 24px、宽度也不同）。")
    A("")

    A("## 3. p^H 假设的实测否证（这正是裁定 23 提醒过的）\n")
    A("| 组合 | p（H1 实测） | p^8 预测 | H8 实测 | 偏差 |")
    A("|---|---|---|---|---|")
    for t in tags:
        p = np.mean([by[(t, s)][1] for s in seeds])
        pred = p ** 8
        act = np.mean([by[(t, s)][8] for s in seeds])
        A(f"| {t} | {p:.3f} | {pred:.3f} | {act:.3f} | {act / pred if pred else 0:.2f}× |")
    A("\n→ 除 `d1w8`（`p^8` 本身接近 0，比值失去意义）外，实测 H8 只有 `p^8` 的 "
      "**0.06×–0.22×**，即独立步假设**高估了 5–16 倍**。它在本环境不成立，"
      "M1 的设计必须按实测曲线做，不能按 `p^H` 反推难度。\n")

    A("## 4. ρ 对照（B-drift 量化）\n")
    A("| 候选 | H | ρ=0.3 | ρ=0.0 | 差 |")
    A("|---|---|---|---|---|")
    drift = []
    for c in sorted({t.split("@")[0] for t in tags}):
        for H in HS:
            a = np.mean([by[(f"{c}@r0.3", s)][H] for s in seeds])
            b = np.mean([by[(f"{c}@r0.0", s)][H] for s in seeds])
            drift.append(abs(a - b))
            A(f"| {c} | {H} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    A(f"\n- ρ 的最大影响 **{max(drift):.3f}**，平均 {np.mean(drift):.3f} → "
      "漂移不是曲线崩塌的原因；`B-drift` 的代价比预期小。\n")

    A("## 5. 结论边界\n")
    A("- 只回答\"用 (干扰密度, 宽度) 能否得到 H=1..8 的可测窗口\" → **不能**；")
    A("- 不涉及 E1-L 本身（crossing 仍未检验）；不推断 10⁹ 规模；")
    A("- 单一环境（6×6、160px、颜色映射标签、无文字），不外推到真实 GUI；")
    A("- 本轮的 96 格全部来自 12 个模型，seed 间波动可达 0.3（d1w8 s1 vs s2），"
      "小宽度的训练稳定性本身是噪声源；")
    A("- 本轮 H1（0.74–0.80 @ w16/w24）低于 H4 同配置单步（0.877），"
      "原因是进度条把格框从 26px 压到 24px（元素更小）。这是 harness 差异，"
      "**不能**把两边的绝对值直接相减；")
    A("- ρ=0.3 在长 H 上略优于 ρ=0.0，说明漂移清掉红色错误标记对策略**有利**；"
      "因此本环境里的误差累积成本主要来自**残留标记**，不是漂移本身。\n")

    A("## 6. 下一步（需另行批准，本轮不执行）\n")
    A("M1 的失败**不是**\"难度轴没调对\"，而是策略在中间状态上退化。证据链：")
    A("1. step0 准确率 0.53–0.80，末步降到 0.35–0.66（§2）；")
    A("2. `H8/H1` 只有 0.02（需要 0.25），且不随宽度改善（§1）；")
    A("3. ρ=0.3 在长 H 上**略优于** ρ=0.0（H4 +0.044、H8 +0.013）：漂移会清掉红色"
      "错误标记，说明**累积的 wrong 标记确实在伤害策略**；")
    A("4. P3 的训练协议见过 progress>0 的状态（§2.1），本轮没见过。")
    A("")
    A("**最小检验（未执行）**：1 候选（d1w16）× 1 seed，把训练数据改成"
      "`progress ∈ [0..H]`、`H ∈ {1,2,4,8}` 随机（与 P3 一致），看逐 step 准确率是否变平。")
    A("若变平 → 训练协议是主因，M1 需要按新协议重新预注册；若不变 → 需要另找瓶颈"
      "（如 decoys 在大宽度下的可分辨性）。")
    A("")
    A("**不做**：不放宽 `[0.2,0.9]`、不加候选、不在本轮续跑、不把这次结果当 E1-L 的证据。")

    Path("results/paper/E1L_M1_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    summary = {
        "matrix_cells": len(rows),
        "tags": tags,
        "band_pass": n_pass,
        "band_total": len(tags) * len(seeds),
        "h8_over_h1_mean": float(np.mean(ratios)),
        "required_h8_over_h1": 0.25,
        "max_drift_effect": float(max(drift)),
        "curves": {t: {str(s): {str(H): by[(t, s)][H] for H in HS} for s in seeds}
                   for t in tags},
    }
    Path("results/paper/e1l_m1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L[:26]))
    print(f"\n写出 E1L_M1_RESULTS.md / e1l_m1_summary.json（{len(rows)} 行）")


if __name__ == "__main__":
    main()
