"""E1-L · M1b 分析（判据 D1–D3，见 E1L_M1B_PREREG.md v1.0）。

读 `results/paper/e1l_m1b_diag.jsonl`（A/B 两臂 × H × 聚合计数），
写 `results/paper/E1L_M1B_RESULTS.md` + `e1l_m1b_summary.json`。

判据（预注册 §3；**主判据在 H=8 上算**，H=1/4 仅报告）：
  D1  gap_A ≥ 0.05 且 gap_B ≤ 0.5 × gap_A
  D2  rate_B(H8) − rate_A(H8) ≥ 0.10
  D3  Δacc(progress≥1) − Δacc(progress=0) ≥ 0.05
总体：D1 且 D2 → 初步支持；其一 → 部分支持；都不 → 未获支持。
"""
import argparse
import json
import math
from pathlib import Path

BAND_H = 8


def rate(cell):
    return None if not cell or cell["n"] == 0 else cell["ok"] / cell["n"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - half), min(1.0, c + half))


def summarize(rows):
    out = {}
    for r in rows:
        out.setdefault(r["arm"], {})[r["H"]] = r
    return out


def gap_step0(agg):
    """acc(t=0) − 池化 acc(t≥1)；缺数据返回 None。"""
    a0 = agg["acc_by_t"][0]
    if a0["n"] == 0:
        return None
    rest_n = sum(c["n"] for c in agg["acc_by_t"][1:])
    rest_k = sum(c["ok"] for c in agg["acc_by_t"][1:])
    if rest_n == 0:
        return None
    return a0["ok"] / a0["n"] - rest_k / rest_n


def pooled(cells):
    n = sum(c["n"] for c in cells)
    k = sum(c["ok"] for c in cells)
    return (k, n)


def acc_bucket(aggs, keys):
    cells = [a["by_progress"][k] for a in aggs for k in keys if k in a["by_progress"]]
    k, n = pooled(cells)
    return None if n == 0 else k / n


def judge(rows):
    s = summarize(rows)
    for arm in ("A", "B"):
        if arm not in s or BAND_H not in s[arm]:
            raise SystemExit(f"缺少 arm {arm} 的 H={BAND_H} 数据")

    a8, b8 = s["A"][BAND_H], s["B"][BAND_H]
    gap_a, gap_b = gap_step0(a8), gap_step0(b8)
    d1_ok = gap_a is not None and gap_a >= 0.05 and gap_b <= 0.5 * gap_a

    ra8 = a8["succ"] / a8["n"] if a8["n"] else None
    rb8 = b8["succ"] / b8["n"] if b8["n"] else None
    d2_ok = (rb8 - ra8) >= 0.10

    hs = [h for h in (1, 4, 8) if h in s["A"] and h in s["B"]]
    a_cells = [s["A"][h] for h in hs]
    b_cells = [s["B"][h] for h in hs]
    d0 = (acc_bucket(b_cells, ["0"]) or 0) - (acc_bucket(a_cells, ["0"]) or 0)
    dp = (acc_bucket(b_cells, ["1-3", "4-7"]) or 0) - \
         (acc_bucket(a_cells, ["1-3", "4-7"]) or 0)
    d3_ok = (dp - d0) >= 0.05

    if d1_ok and d2_ok:
        verdict = "机制得到初步支持"
    elif d1_ok or d2_ok:
        verdict = "部分支持"
    else:
        verdict = "机制未获支持"

    return {"gap_A": gap_a, "gap_B": gap_b, "D1": d1_ok,
            "rate_A_H8": ra8, "rate_B_H8": rb8, "D2": d2_ok,
            "delta_prog0": d0, "delta_prog_pos": dp, "D3": d3_ok,
            "D3_reversed": dp < d0, "verdict": verdict,
            "H_used": BAND_H, "H_reported": hs}


def render(rows, j):
    s = summarize(rows)
    L = []
    A = L.append
    A("# E1-L · M1b 机制诊断 结果\n")
    A("> 协议：`E1L_M1B_PREREG.md` v1.0　数据：`e1l_m1b_diag.jsonl`")
    A("> 两臂只差训练数据分布（A=`progress=0`，B=混合 `progress`）；同一任务流、同一 rollout")
    A("> ⚠️ 仅 1 seed / 2 次训练 → **初步机制证据**，不得写成\"机制已证实\"\n")

    A("## 0. 判定\n")
    A(f"**{j['verdict']}**\n")
    A("> **归因限定**：B 与 A 的差异是**联合干预**（`progress` 的取值 + 由 `progress` 推导出的 `done` 灰格），\n"
      "> 不是单变量 `progress`。可以说\"中间状态覆盖\"获得初步支持，**不能**把收益单独归因于进度条或数值。\n")
    A("| 判据 | 观测 | 结论 |")
    A("|---|---|---|")
    A(f"| D1 step0 与后续步的差被缩小一半 | gap_A={j['gap_A']:.3f} → gap_B={j['gap_B']:.3f}"
      f"（H={j['H_used']}） | {'✅' if j['D1'] else '❌'} |")
    A(f"| D2 H8 成功率提升 ≥0.10 | A={j['rate_A_H8']:.3f} → B={j['rate_B_H8']:.3f}"
      f"（Δ={j['rate_B_H8'] - j['rate_A_H8']:+.3f}） | {'✅' if j['D2'] else '❌'} |")
    A(f"| D3 改善落在新增覆盖的状态 | Δ(progress≥1)={j['delta_prog_pos']:+.3f} vs "
      f"Δ(progress=0)={j['delta_prog0']:+.3f} | {'✅' if j['D3'] else '❌'}"
      f"{'（**反向**：改善只在 progress=0）' if j['D3_reversed'] else ''} |")
    A("")

    b8 = s["B"][8]
    rb8 = j["rate_B_H8"]
    last = b8["acc_by_t"][7]["ok"] / max(1, b8["acc_by_t"][7]["n"])
    A("### 0.1 离 M1 的门槛还差多少\n")
    A(f"- B 的 H8 = **{rb8:.3f}**；M1 要求 H8 ≥ **0.22** → 还差 **{0.22 / rb8:.1f}×**。")
    A(f"- 参考（**独立同分布示意，不是预测/上界**）：若末步准确率 {last:.3f} 从第 1 步起保持，"
      f"`{last:.3f}^8 = {last ** 8:.3f}`，仍 < 0.22。")
    A("  实际 rollout 存在状态依赖与错误累积，因此这个数字只能说明"
      "\"单步水平本身不足\"，**不能**当作 H8 的真实预测。")
    A("  → 仅靠抹平衰减不够；绝对准确率是主因（见 `E1L_ERROR_DECOMP.md` 的独立性分解）。")
    A("")

    A("## 1. 成功率\n")
    A("| H | A | B | Δ | A 的 95% CI | B 的 95% CI |")
    A("|---|---|---|---|---|---|")
    for h in j["H_reported"]:
        a, b = s["A"][h], s["B"][h]
        ra, rb = a["succ"] / a["n"], b["succ"] / b["n"]
        ca, cb = wilson(a["succ"], a["n"]), wilson(b["succ"], b["n"])
        A(f"| {h} | {ra:.3f} | {rb:.3f} | {rb - ra:+.3f} | "
          f"[{ca[0]:.3f}, {ca[1]:.3f}] | [{cb[0]:.3f}, {cb[1]:.3f}] |")
    A("")

    A("## 2. 逐 step 准确率（H=8）\n")
    A("| t | A | B | Δ |")
    A("|---|---|---|---|")
    a8, b8 = s["A"][8], s["B"][8]
    for t in range(8):
        ca, cb = a8["acc_by_t"][t], b8["acc_by_t"][t]
        ra, rb = rate(ca), rate(cb)
        A(f"| {t} | {ra:.3f} | {rb:.3f} | {rb - ra:+.3f} |")
    A("")

    A("## 3. 按 progress 分组（H=1/4/8 池化）\n")
    A("| progress | A | B | Δ |")
    A("|---|---|---|---|")
    for b in ("0", "1-3", "4-7"):
        ra = acc_bucket([s["A"][h] for h in j["H_reported"]], [b])
        rb = acc_bucket([s["B"][h] for h in j["H_reported"]], [b])
        if ra is None or rb is None:
            A(f"| {b} | — | — | — |")
        else:
            A(f"| {b} | {ra:.3f} | {rb:.3f} | {rb - ra:+.3f} |")
    A("")

    A("## 4. 按标记状态分组（H=1/4/8 池化）\n")
    A("| 状态 | A | B | Δ |")
    A("|---|---|---|---|")
    for mk in ("none", "done_only", "wrong"):
        ca = [s["A"][h]["by_marks"][mk] for h in j["H_reported"] if mk in s["A"][h]["by_marks"]]
        cb = [s["B"][h]["by_marks"][mk] for h in j["H_reported"] if mk in s["B"][h]["by_marks"]]
        ka, na = pooled(ca)
        kb, nb = pooled(cb)
        A(f"| {mk} | {'—' if na == 0 else f'{ka / na:.3f}'} | "
          f"{'—' if nb == 0 else f'{kb / nb:.3f}'} | "
          f"{'—' if na == 0 or nb == 0 else f'{kb / nb - ka / na:+.3f}'} |")
    A("")
    A("> 两臂都**没有**用 `wrong`/`done` 标记训练；若 B 在这些状态下仍差，说明还有未覆盖的状态。")

    A("\n## 5. 与预注册的偏差（执行后补记）\n")
    A("| # | 预注册写的 | 实际执行 | 原因与影响 |")
    A("|---|---|---|---|")
    A("| 1 | B 臂\"仍无标记\" | B 的 `done` 掩码由 progress 推导：进度条显示 k 时，"
      "恰有 k 个非目标格子为灰色 | 裁定 25 要求 `progress`/进度条/`done` **一致**，"
      "不允许只改数字。`done` 由 `progress` 唯一决定，因此自变量仍只有 `progress`；"
      "但 B 的覆盖从\"仅进度条\"变成\"进度条 + 对应灰格\" |")
    A("| 2 | B 覆盖 `progress ∈ [0, t]` | 同；未包含 `progress=H` | 与预注册一致（终止态不可达） |")
    A("| 3 | — | B **未覆盖**\"progress>0 但标记为空\"（漂移清标记）的状态 | "
      "训练数据不模拟漂移；这是已知缺口，§4 的 `wrong` 分组间接反映 |")
    A("")

    A("\n## 6. 边界\n")
    A("- 1 seed / 2 次训练 → 初步证据；不宣称机制普适；")
    A("- 单配置（w24、d1、ρ0.3、6×6、160px），不外推；")
    A("- ρ 观察（M1 里 ρ0.3 略优于 ρ0.0）**不升级**为机制结论：本实验也不分离布局与标记；")
    A("- 本结果**不**等于 M1 通过：M1 的判据是 96 格窗口，与此无关。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/paper/e1l_m1b_diag.jsonl")
    ap.add_argument("--out-md", default="results/paper/E1L_M1B_RESULTS.md")
    ap.add_argument("--out-json", default="results/paper/e1l_m1b_summary.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.runs).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    j = judge(rows)
    md = render(rows, j)
    Path(args.out_md).write_text(md, encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md[:1200])
    print(f"\n写出 {args.out_md} / {args.out_json}")


if __name__ == "__main__":
    main()
