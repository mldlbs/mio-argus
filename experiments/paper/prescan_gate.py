"""阈值型指标 预扫描门禁（对应 docs/PRESCAN_PROTOCOL.md v1.0 §6）。

输入：预扫描 jsonl，一行一个 (条件, 横轴档位, seed) 的结果。

    {"cond": "C0", "x": 6768, "seed": 0, "value": 0.961}

门禁（任一 FAIL → 退出码 1，正式矩阵不得启动）：
    G0 结构：每条件 × 每 seed 的横轴档位一致
    G1 严格内部：min < threshold < max
    G2 左端余量：value(x_min) < threshold - margin
    G3 右端余量：value(x_max) > threshold + margin
    G4 分辨率：跨越点唯一，且相邻档 x 比值 <= max_gap_ratio
    G5 单调性：声明 --monotone 时，曲线必须非递减

用法：
    python experiments/paper/prescan_gate.py --runs results/paper/h4_density_runs.jsonl `
        --x-col params --value-col acc --threshold 0.80 --monotone
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path, ccol, xcol, ycol, scol):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({"cond": r[ccol], "x": float(r[xcol]),
                     "seed": r[scol], "value": float(r[ycol])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--cond-col", default="cond")
    ap.add_argument("--x-col", default="x")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--seed-col", default="seed")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--max-gap-ratio", type=float, default=2.5)
    ap.add_argument("--monotone", action="store_true",
                    help="声明将使用单调插值（见协议 §5）")
    ap.add_argument("--monotone-tol", type=float, default=0.05,
                    help="单调性容差：允许的回落幅度，默认 0.05 ≈ 3×SE（n_test=1000 时）")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    rows = load(args.runs, args.cond_col, args.x_col, args.value_col, args.seed_col)
    curves = defaultdict(dict)   # (cond, seed) -> {x: value}
    for r in rows:
        curves[(r["cond"], r["seed"])][r["x"]] = r["value"]

    conds = sorted({c for c, _ in curves}, key=str)
    seeds = sorted({s for _, s in curves})
    results = []   # (gate, cond, seed, ok, detail)

    # G0：结构一致（同 seed 各条件的 x 集合相同；同条件各 seed 的 x 集合相同）
    xs_by = {k: tuple(sorted(v)) for k, v in curves.items()}
    uniq = set(xs_by.values())
    results.append(("G0", "ALL", "-", len(uniq) == 1,
                    f"横轴档位集合数量={len(uniq)}（应为 1）"))
    for c in conds:
        got = {xs_by[(c, s)] for s in seeds if (c, s) in xs_by}
        results.append(("G0", c, "-", len(got) == 1, f"该条件下档位集合数量={len(got)}"))

    thr = args.threshold
    for c in conds:
        for s in seeds:
            key = (c, s)
            if key not in curves:
                results.append(("G0", c, s, False, "缺少该 (条件,seed)"))
                continue
            kv = curves[key]
            xs = sorted(kv)
            ys = [kv[x] for x in xs]

            # G1 严格内部
            g1 = min(ys) < thr < max(ys)
            results.append(("G1", c, s, g1,
                            f"[{min(ys):.3f}, {max(ys):.3f}] vs thr={thr:.2f}"))

            # G2 左端余量
            g2 = ys[0] < thr - args.margin
            results.append(("G2", c, s, g2,
                            f"x_min={xs[0]:g} value={ys[0]:.3f} 需 < {thr - args.margin:.3f}"))

            # G3 右端余量
            g3 = ys[-1] > thr + args.margin
            results.append(("G3", c, s, g3,
                            f"x_max={xs[-1]:g} value={ys[-1]:.3f} 需 > {thr + args.margin:.3f}"))

            # G4 分辨率：跨越点唯一 + 相邻档比值（无跨越点时不适用，由 G1/G2/G3 阻断）
            cross = [i for i in range(len(ys) - 1) if ys[i] < thr <= ys[i + 1]]
            if len(cross) == 1:
                i = cross[0]
                ratio = xs[i + 1] / xs[i] if xs[i] else float("inf")
                g4 = ratio <= args.max_gap_ratio
                detail = f"唯一跨越 [{xs[i]:g}, {xs[i+1]:g}] ratio={ratio:.2f}"
            elif len(cross) == 0:
                g4, detail = True, "无跨越点（左/右截断，G1–G3 已阻断；本关不适用）"
            else:
                g4, detail = False, f"跨越点不唯一 n={len(cross)}（曲线非单调）"
            results.append(("G4", c, s, g4, detail))

            # G5 单调性（仅在声明单调插值时强制；容差内回落不算违反）
            if args.monotone:
                bad = [f"{xs[i]:g}({ys[i]:.3f})>{xs[i+1]:g}({ys[i+1]:.3f})"
                       for i in range(len(ys) - 1)
                       if ys[i + 1] < ys[i] - args.monotone_tol]
                results.append(("G5", c, s, not bad,
                                "非递减" if not bad else "超容差回落 " + ", ".join(bad[:3])))
            else:
                results.append(("G5", c, s, True, "未声明单调插值（跳过）"))

    fails = [r for r in results if not r[3]]
    ok = not fails

    L = []
    A = L.append
    A("# 预扫描门禁报告\n")
    A(f"> 输入：`{args.runs}`　阈值={thr:.2f}　margin={args.margin}　"
      f"max_gap_ratio={args.max_gap_ratio}　单调插值={'是' if args.monotone else '未声明'}")
    A(f"> 条件 {len(conds)} 个，seed {len(seeds)} 个，检查 {len(results)} 项\n")
    A(f"## 结论：{'PASS — 允许启动正式矩阵' if ok else 'FAIL — 正式矩阵不得启动'}\n")
    A("| 门 | 条件 | seed | 结果 | 详情 |")
    A("|---|---|---|---|---|")
    for g, c, s, good, d in results:
        A(f"| {g} | {c} | {s} | {'✅' if good else '❌'} | {d} |")
    if fails:
        A("\n### 阻断项（按条件汇总）\n")
        by = defaultdict(set)
        for g, c, s, good, d in fails:
            by[c].add(g)
        for c in sorted(by, key=str):
            A(f"- **{c}**：{[g for g in ('G1','G2','G3','G4','G5') if g in by[c]]}")
        A("\n按 `docs/PRESCAN_PROTOCOL.md` §5 处理这些形态（分别标 `≤`/`>`/`null`），"
          "或另开预注册扩大扫描范围；**不得**在原协议上追加后直接混入主分析。")

    text = "\n".join(L) + "\n"
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"[report] {args.report}")
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
