"""H4 干扰密度实验 结果汇总（严格按预注册 v1.1 口径）。

输入：results/paper/h4_density_runs.jsonl（126 行）
输出：results/paper/H4_RESULTS.md + results/paper/h4_density_summary.json

判据（预注册 §4，执行前锁定）：
  ① 3 个 seed 的 Spearman ρ（密度 0/1/2/4 vs params_needed）全部 > 0
  ② Δ(C3−C0) 的 95% CI（t, df=2）下限 > 0
  ③ Δ(C4−C0) 的 95% CI 覆盖 0
  三条同时成立 → H4-G 成立
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

WIDTHS = [4, 6, 8, 12, 16, 24, 32]
CONDS = ["C0", "C1", "C2", "C3", "C4", "C5"]
DENSITY = {"C0": 0, "C1": 1, "C2": 2, "C3": 4}
SEEDS = [0, 1, 2]
T2 = 4.302652729911275  # t(0.975, df=2)


def params_needed(curve, pvals, target=0.80):
    for i in range(len(curve) - 1):
        if curve[i] < target <= curve[i + 1]:
            t = (target - curve[i]) / (curve[i + 1] - curve[i])
            lp = math.log(pvals[i]) + t * (math.log(pvals[i + 1]) - math.log(pvals[i]))
            return float(math.exp(lp)), "interp"
    if curve[0] >= target:
        return float(pvals[0]), "left"
    return None, "none"


def spearman(x, y):
    def rank(a):
        order = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return 0.0 if den == 0 else num / den


def t_ci(vals):
    """3 seed 均值的 95% CI（df=2）。"""
    ok = [v for v in vals if v is not None]
    if len(ok) < 3:
        return None, None, None
    m = mean(ok)
    sd = float(np.std(ok, ddof=1))
    half = T2 * sd / math.sqrt(len(ok))
    return m, m - half, m + half


def boot_ci(vals, rng, n_boot=1000):
    ok = [v for v in vals if v is not None]
    if len(ok) < 2 or len(set(ok)) == 1:
        return (ok[0], ok[0]) if ok else (None, None)
    arr = np.asarray(ok, dtype=float)
    ms = [float(np.median(rng.choice(arr, size=len(arr), replace=True)))
          for _ in range(n_boot)]
    return (float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/paper/h4_density_runs.jsonl")
    ap.add_argument("--hashes", default="results/paper/h4_density_hashes.json")
    ap.add_argument("--out-md", default="results/paper/H4_RESULTS.md")
    ap.add_argument("--out-json", default="results/paper/h4_density_summary.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    hashes = json.loads(Path(args.hashes).read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    acc, params = {}, {}
    for r in rows:
        acc[(r["cond"], r["width"], r["seed"])] = r["acc"]
        params[(r["cond"], r["width"])] = r["params"]
    pv = [params[("C0", w)] for w in WIDTHS]

    per, curves = {}, {}
    for c in CONDS:
        for s in SEEDS:
            curve = [acc[(c, w, s)] for w in WIDTHS]
            curves[(c, s)] = curve
            per.setdefault(c, {})[s] = params_needed(curve, pv)

    def delta(c, base="C0"):
        return [None if (per[c][s][0] is None or per[base][s][0] is None)
                else per[c][s][0] - per[base][s][0] for s in SEEDS]

    rhos = []
    for s in SEEDS:
        xs = [DENSITY[c] for c in ("C0", "C1", "C2", "C3")]
        ys = [per[c][s][0] for c in ("C0", "C1", "C2", "C3")]
        rhos.append(None if any(v is None for v in ys) else spearman(xs, ys))

    d3, d4 = delta("C3"), delta("C4")
    m3, lo3, hi3 = t_ci(d3)
    m4, lo4, hi4 = t_ci(d4)
    crit1 = all(r is not None and r > 0 for r in rhos)
    crit2 = lo3 is not None and lo3 > 0
    crit3 = lo4 is not None and lo4 <= 0 <= hi4
    verdict = "成立" if (crit1 and crit2 and crit3) else (
        "否证（背景熵本身有效）" if (lo4 is not None and lo4 > 0) else "否证（机制不成立/无效应）")

    pn_med = {}
    for c in CONDS:
        ok = [per[c][s][0] for s in SEEDS if per[c][s][0] is not None]
        pn_med[c] = float(np.median(ok)) if ok else None

    summary = {
        "n_rows": len(rows), "config_fp": hashes.get("config_fp"),
        "per_cond_seed": {c: {str(s): {"pn": per[c][s][0], "status": per[c][s][1]}
                              for s in SEEDS} for c in CONDS},
        "median": pn_med, "spearman_rho": rhos,
        "delta_C3_C0": {"vals": d3, "mean": m3, "ci": [lo3, hi3]},
        "delta_C4_C0": {"vals": d4, "mean": m4, "ci": [lo4, hi4]},
        "criteria": {"rho_sign": crit1, "c3_positive": crit2, "c4_null": crit3},
        "verdict": verdict,
        "curves": {f"{c}|{s}": curves[(c, s)] for c in CONDS for s in SEEDS},
    }

    L = []
    A = L.append
    A("# H4 干扰密度 结果\n")
    A(f"> 协议：`H4_GENERALIZATION_PREREG.md` v1.1　`config_fp` = `{hashes.get('config_fp')}`")
    A(f"> 原始数据：`{args.runs}`（{len(rows)} 行 / 应为 126）　主指标：`params_needed(0.80)`")
    A("> P5 只作背景讨论，不参与本表任何数值\n")

    A("## 0. 判定（三条同时成立才算 H4-G 成立）\n")
    A("| 判据 | 观测 | 结论 |")
    A("|---|---|---|")
    A(f"| ① 3 seed 的 ρ 全部 > 0 | {[None if r is None else round(r, 3) for r in rhos]} | "
      f"{'✅' if crit1 else '❌'} |")
    A(f"| ② Δ(C3−C0) 95% CI 下限 > 0 | Δ̅={m3 if m3 is None else format(m3, '+,.0f')}，"
      f"CI=[{lo3 if lo3 is None else format(lo3, '+,.0f')}, {hi3 if hi3 is None else format(hi3, '+,.0f')}] | "
      f"{'✅' if crit2 else '❌'} |")
    A(f"| ③ Δ(C4−C0) 95% CI 覆盖 0 | Δ̅={m4 if m4 is None else format(m4, '+,.0f')}，"
      f"CI=[{lo4 if lo4 is None else format(lo4, '+,.0f')}, {hi4 if hi4 is None else format(hi4, '+,.0f')}] | "
      f"{'✅' if crit3 else '❌'} |")
    A(f"\n**H4-G 判定：{verdict}**\n")

    if pn_med["C0"] and pn_med["C3"]:
        ratio = pn_med["C3"] / pn_med["C0"]
        censored = any(per["C0"][s][1] == "left" for s in SEEDS)
        A(f"- 效应量 `C3/C0` = **{ratio:.2f}×**"
          + ("（**上界**：C0 在 w4 已饱和，真实 C0 需求 ≤ 4,208）" if censored else ""))
    A("")

    A("## 1. 逐条件 `params_needed(0.80)`\n")
    A("| 条件 | 干扰/格 | s0 | s1 | s2 | 中位数 | bootstrap 95% CI |")
    A("|---|---|---|---|---|---|---|")
    for c in CONDS:
        vals = [per[c][s][0] for s in SEEDS]
        st = [per[c][s][1] for s in SEEDS]
        fmt = lambda v, k: "—" if v is None else (f"{v:,.0f}" + ("†" if k else ""))
        lo, hi = boot_ci(vals, rng)
        ci = "—" if lo is None else f"[{lo:,.0f}, {hi:,.0f}]"
        dens = DENSITY.get(c)
        A(f"| {c} | {dens if dens is not None else '0（背景）'} | "
          f"{fmt(vals[0], st[0]=='left')} | {fmt(vals[1], st[1]=='left')} | "
          f"{fmt(vals[2], st[2]=='left')} | "
          f"{'—' if pn_med[c] is None else format(pn_med[c], ',.0f')} | {ci} |")
    A("\n† = 左截断（w4 已达 0.80，真实需求 ≤ 该值）\n")

    A("## 2. 准确率矩阵（3 seed 平均）\n")
    A("| 条件 | " + " | ".join(f"w{w}" for w in WIDTHS) + " |")
    A("|---|" + "---|" * len(WIDTHS))
    for c in CONDS:
        A(f"| {c} | " + " | ".join(f"{mean([acc[(c, w, s)] for s in SEEDS]):.3f}"
                                   for w in WIDTHS) + " |")
    A("")

    A("## 3. 与 C0 的配对差（params_needed，3 seed）\n")
    A("| 对比 | s0 | s1 | s2 | Δ̅ | 95% CI（df=2） |")
    A("|---|---|---|---|---|---|")
    for c in ("C1", "C2", "C3", "C4", "C5"):
        d = delta(c)
        m, lo, hi = t_ci(d)
        A(f"| {c} − C0 | " + " | ".join("—" if v is None else f"{v:+,.0f}" for v in d)
          + f" | {'—' if m is None else format(m, '+,.0f')} | "
          + ("—" if lo is None else f"[{lo:+,.0f}, {hi:+,.0f}]") + " |")
    A("\n> n=3、df=2，区间较宽；只用于判断方向与量级，不做逐条件显著性宣称。\n")

    A("## 4. 曲线（逐 seed 准确率，w 从 4 到 32）\n")
    for c in CONDS:
        A(f"**{c}**")
        for s in SEEDS:
            A("- s{}: ".format(s) + " ".join(f"{curves[(c, s)][i]:.3f}" for i in range(len(WIDTHS))))
    A("")

    A("## 5. 复现与哈希\n")
    A(f"- `config_fp`：`{hashes.get('config_fp')}`")
    A(f"- 设备：{hashes.get('device')}　torch {hashes.get('torch')}　numpy {hashes.get('numpy')}")
    A("- 数据哈希（每条件）：")
    for _, m in (hashes.get("conds") or {}).items():
        A(f"  - {m['name']}（干扰 {m['decoys']}）：img_train `{m['img_hash_train']}`，"
          f"base_train `{m['base_hash_train']}`")
    A("\n## 6. 边界\n")
    A("- B-res：本分辨率元素无文字（4–8 px），任务靠颜色区分 → 只覆盖颜色/形状特征空间重叠；")
    A("- B-col：C6 已证实颜色是主导线索 → 本实验测的是颜色维度的干扰密度；")
    A("- B-scale：最大 106K 参数，不外推到十亿级；")
    A("- B-p5：P5 只作背景，其 `params_needed` 数值不与本表并列。")

    Path(args.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print("\n".join(L[:20]))
    print(f"\n写出 {args.out_md} / {args.out_json}")


if __name__ == "__main__":
    main()
