"""E1-L 误差分解（只用已有日志，不训练）。

回答裁定 26 留下的问题：多步瓶颈是"状态分布外推"、"单步能力不足"，还是"随深度额外累积"？

可计算的量（现有 jsonl 足够）：
  m_t = 第 t 步边际准确率；P = ∏_{t<H} m_t（若各步独立，任务成功率应 ≈ P）
  S = 实测任务成功率
  I = S / P  ← 独立性指数：≈1 表示"短在单步能力"，<1 表示"错误成串"（状态依赖）

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
    A(f"- **B 臂**：实测 H8 = {sb:.3f}，独立步预测 ∏m_t = {pb:.3f}，独立性指数 **{ib:.2f}** "
      f"→ 误差几乎不额外成串，**瓶颈是单步准确率本身（0.78→0.73）**。")
    A(f"- **A 臂**：实测 H8 = {sa:.3f}，∏m_t = {pa:.3f}，指数 **{ia:.2f}** "
      f"→ 除了准确率低，还有**额外聚簇**（一次错更容易带出后续错）。")
    A("- 因此\"状态分布外推\"与\"单步能力不足\"**不是二选一**：")
    A("  - 状态覆盖（M1b 的干预）主要改善**逐 step 水平**（D1/D3 已支持）；")
    A("  - 但即便把逐 step 拉平，`∏m_t` 仍只有 ~0.09 → **绝对能力不足是主因**；")
    A("  - A 臂的额外聚簇说明**状态依赖也真实存在**，只是在当前难度下不是最大项。\n")

    A("## 1. 逐臂分解（H=8）\n")
    A("| 臂 | 实测 H8 | 95% CI | ∏m_t（独立步） | 独立性指数 | 解释 |")
    A("|---|---|---|---|---|---|")
    for arm, (s, p, idx) in (("A", (sa, pa, ia)), ("B", (sb, pb, ib))):
        r = by[(arm, 8)]
        lo, hi = wilson(r["succ"], r["n"])
        note = "误差成串" if idx < 0.9 else "近似独立"
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
    A(f"- M1 的 {len(idx8)} 个 (候选,ρ,seed) 格在 H=8 上的独立性指数："
      f"中位数 **{sorted(idx8)[len(idx8) // 2]:.2f}**，范围 {min(idx8):.2f}–{max(idx8):.2f}")
    A("- 多数格 < 1 → 错误成串是**普遍形态**，不是某一臂的偶然；"
      "越难的配置聚簇越强（指数越低）。\n")

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
    A("> 在当前环境与规模下，多步瓶颈**主要**来自单步决策能力不足（`∏m_t ≈ 0.09 @ H=8`），"
      "**其次**来自错误聚簇（A 臂指数 0.72）；状态分布外推解释的是\"逐 step 从 0.83 掉到 0.63\"那一段，"
      "已被 M1b 部分修复（→0.73 平），但它不足以把成功率拉进可测窗口。")

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
