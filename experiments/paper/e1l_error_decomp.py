"""E1-L 误差分解（只用已有日志，不训练）。

回答裁定 26 留下的问题：多步瓶颈是"状态分布外推"、"单步能力不足"，还是"随深度额外累积"？

可计算的量（现有 jsonl 足够）：
  m_t = 第 t 步边际准确率；P = ∏_{t<H} m_t（若各步独立，任务成功率应 ≈ P）
  S = 实测任务成功率
  I = S / P  ← 独立步比值：≈1 与"各步边际准确率之积"相容；<1 实际低于该基准
  注意：I 不是错误相关性的估计量（状态转移、条件准确率变化、任务难度异质都会造成偏离）

不可计算的量（现有日志不足，记为证据缺口，**不追加实验**）：
  决策类型分解（哪类标签/槽位易混）、t×state 联合分布、逐任务错误相关性。
"""
import json
import math
from pathlib import Path

HS = [1, 2, 4, 8]


def marginals(acc_by_t, H):
    return [c["ok"] / c["n"] if c["n"] else None for c in acc_by_t[:H]]


def independence_index(acc_by_t, succ, n, H):
    """I = S / ∏ m_t；任一步无数据返回 None。"""
    ms = marginals(acc_by_t, H)
    if len(ms) < H or any(m is None or m == 0 for m in ms):
        return None, None, None
    prod = 1.0
    for m in ms:
        prod *= m
    s = succ / n if n else None
    return (s, prod, (s / prod if prod else None))


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - half), min(1.0, c + half))


def main():
    rows = [json.loads(l) for l in Path("results/paper/e1l_m1b_diag.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {(r["arm"], r["H"]): r for r in rows}

    L = []
    A = L.append
    A("# E1-L 误差分解（只用已有日志）\n")
    A("> 数据：`e1l_m1b_diag.jsonl`（A/B 两臂）与 `e1l_m1_prescan.jsonl`（M1 的 96 格）")
    A("> **没有新增训练**。本文只把已有的逐 step 计数与成功率做一致性检验。\n")

    A("## 0. 结论\n")
    a = by[("A", 8)]
    b = by[("B", 8)]
    sa, pa, ia = independence_index(a["acc_by_t"], a["succ"], a["n"], 8)
    sb, pb, ib = independence_index(b["acc_by_t"], b["succ"], b["n"], 8)
    A(f"- **B 臂**：实测 H8 = {sb:.3f}，独立步预测 ∏m_t = {pb:.3f}，独立步比值 **{ib:.2f}** "
      f"→ 与独立步基准相容，**瓶颈是单步准确率本身（0.78→0.73）**。")
    A("  （≠\"错误彼此独立\"；只能说偏差很小、没有可测的额外多步损失）")
    A(f"- **A 臂**：实测 H8 = {sa:.3f}，∏m_t = {pa:.3f}，指数 **{ia:.2f}** "
      f"→ 除准确率低，还存在**额外多步损失**（与错误聚簇相容，但不等于证明了聚簇）。")
    A("- 因此\"状态分布外推\"与\"单步能力不足\"**不是二选一**：")
    A("  - 状态覆盖（M1b 的干预）主要改善**逐 step 水平**（D1/D3 已支持）；")
    A("  - 但即便把逐 step 拉平，`∏m_t` 仍只有 ~0.09 → **绝对能力不足是主因**；")
    A("  - A 臂 I=0.72 说明存在**额外的多步损失**；其来源（聚簇 / 状态转移 / 难度异质）"
      "在本轮日志下**无法分离**。\n")

    A("### 0.1 解释边界（必须与数字一起读）\n")
    A("`I = P(H8 成功) / ∏ m_t` 的含义只有三种：")
    A("- `I ≈ 1`：与\"各步边际准确率之积\"**相容**；")
    A("- `I < 1`：实际成功率**低于**独立步基准；")
    A("- `I > 1`：实际成功率**高于**独立步基准。")
    A("")
    A("**`I` 不是错误相关性的估计量。** 状态转移、条件准确率随 t 变化、任务难度异质，"
      "都会造成同样的偏离。因此：")
    A("- A 的 0.72 → 存在**额外的多步损失**（与错误聚簇**相容**，不构成证明）；")
    A("- B 的 0.97 → 与独立步基准相容，**不能**说\"错误之间独立\"；")
    A("- M1 的 14/24 → 只能描述**指数分布**，不能认定这 14 组都存在错误聚簇。")
    A("")

    A("## 1. 逐臂分解（H=8）\n")
    A("| 臂 | 实测 H8 | 95% CI | ∏m_t（独立步） | 独立步比值 | 解释 |")
    A("|---|---|---|---|---|---|")
    for arm, (s, p, idx) in (("A", (sa, pa, ia)), ("B", (sb, pb, ib))):
        r = by[(arm, 8)]
        lo, hi = wilson(r["succ"], r["n"])
        note = "低于独立步基准" if idx < 0.9 else "与基准相容"
        A(f"| {arm} | {s:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.3f} | {idx:.2f} | {note} |")
    A("\n> ⚠️ `∏m_t` 是**独立步假设下的示意值**，不是 H8 的预测或上界；"
      "它的作用只是把\"单步水平\"与\"额外相关性\"分开看。\n")

    A("## 2. 全部 H 的分解\n")
    A("| 臂 | H | 实测 | ∏m_t | 指数 |")
    A("|---|---|---|---|---|")
    for arm in ("A", "B"):
        for H in HS:
            r = by[(arm, H)]
            s, p, idx = independence_index(r["acc_by_t"], r["succ"], r["n"], H)
            A(f"| {arm} | {H} | {s:.3f} | {p:.3f} | {idx:.2f} |")
    A("")

    A("## 3. 状态条件准确率（H=1/4/8 池化，来自 M1b）\n")
    A("| 状态 | A | B | Δ |")
    A("|---|---|---|---|")
    pools = [1, 4, 8]
    for mk in ("none", "done_only", "wrong"):
        ka = sum(by[("A", h)]["by_marks"].get(mk, {"ok": 0, "n": 0})["ok"] for h in pools)
        na = sum(by[("A", h)]["by_marks"].get(mk, {"ok": 0, "n": 0})["n"] for h in pools)
        kb = sum(by[("B", h)]["by_marks"].get(mk, {"ok": 0, "n": 0})["ok"] for h in pools)
        nb = sum(by[("B", h)]["by_marks"].get(mk, {"ok": 0, "n": 0})["n"] for h in pools)
        ra = "—" if na == 0 else f"{ka / na:.3f}"
        rb = "—" if nb == 0 else f"{kb / nb:.3f}"
        d = "—" if na == 0 or nb == 0 else f"{kb / nb - ka / na:+.3f}"
        A(f"| {mk} | {ra} | {rb} | {d} |")
    A("")

    A("## 4. M1 的 96 格是否同一形态\n")
    m1 = [json.loads(l) for l in Path("results/paper/e1l_m1_prescan.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
    idx8 = []
    for r in m1:
        if r["H"] != 8:
            continue
        cells = [{"n": r["n"], "ok": round(v * r["n"])} for v in r["acc_by_step"]]
        s, p, idx = independence_index(cells, r["success"], r["n"], 8)
        if idx is not None:
            idx8.append(idx)
    import statistics
    below = sum(1 for x in idx8 if x < 0.9)
    mid = sum(1 for x in idx8 if 0.9 <= x <= 1.1)
    above = sum(1 for x in idx8 if x > 1.1)
    A(f"- M1 的 {len(idx8)} 个 (候选,ρ,seed) 格在 H=8 上的独立步比值："
      f"中位数 **{statistics.median(idx8):.2f}**，范围 {min(idx8):.2f}–{max(idx8):.2f}")
    A(f"- 分布：**{below} 个 < 0.9**（低于独立步基准）、{mid} 个在 0.9–1.1（与基准相容）、"
      f"{above} 个 > 1.1（高于基准；可能是抽样噪声、难度异质或负相关）")
    A("- 只作**分布描述**：{} / {} 个格低于独立步基准。不能据此认定这些格都存在错误聚簇——"
      "该比值不识别偏离的来源。".format(below, len(idx8)))
    A("")

    A("## 5. 证据缺口（明确记录，不追加实验）\n")
    A("| 缺口 | 现有日志为何不够 | 需要什么 |")
    A("|---|---|---|")
    A("| 决策类型分解 | 只记了\"对/错\"，没记预测槽位 | 逐决策记录 `(target_slot, predicted_slot)` |")
    A("| t×state 联合表 | 只有\"按 t\"和\"按状态\"两张边际表 | 记录 `(t, progress, wrong?, done?)` 联合计数 |")
    A("| 逐任务错误相关性 | 只有聚合计数 | 每任务的错误步位置序列 |")
    A("")
    A("这些缺口**不需要新训练**就能补（改日志即可），但按裁定 26 本轮**不做**；"
      "留待确实要重启这条线时再补。\n")

    A("## 6. 结论（对研究问题的一句话回答）\n")
    A("> 在当前环境与规模下，多步瓶颈**主要**来自单步决策能力不足（`∏m_t ≈ 0.09 @ H=8`）；"
      "**其次**是额外的多步损失（A 臂 I=0.72，来源未分离）；状态分布外推解释的是"
      "\"逐 step 从 0.83 掉到 0.63\"那一段，已被 M1b 部分修复（→0.73 平），"
      "但不足以把成功率拉进可测窗口。")

    Path("results/paper/E1L_ERROR_DECOMP.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    Path("results/paper/e1l_error_decomp.json").write_text(json.dumps({
        "A": {"H8": sa, "prod": pa, "index": ia},
        "B": {"H8": sb, "prod": pb, "index": ib},
        "m1_index8_median": sorted(idx8)[len(idx8) // 2],
        "m1_index8_range": [min(idx8), max(idx8)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L[:16]))
    print("\n写出 E1L_ERROR_DECOMP.md / e1l_error_decomp.json")


if __name__ == "__main__":
    main()
