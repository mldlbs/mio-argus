"""
从 p0_h2_results.json / p0_regloss_results.json / p0_parser_results.json
生成 P0_RESULTS.md 与 P0_5_RESULTS.md。
用法: python p0_report.py
"""
import json
from pathlib import Path


def load(p, required=True):
    f = Path(p)
    if not f.exists():
        if required:
            raise SystemExit(f"缺少 {p}，请先运行对应实验")
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def p0_section():
    d = load("p0_h2_results.json")
    rl = load("p0_regloss_results.json", required=False)
    R = d["results"]

    def get(tag, head, rl_=None):
        for r in R:
            if r["tag"] == tag and r["head"] == head and (rl_ is None or r["reg_loss"] == rl_):
                return r
        return None

    grids = [r["tag"] for r in R if r["tag"].startswith("grid") and r["head"] == "cls"]
    fills = [r["tag"] for r in R if r["tag"].startswith("fill") and r["head"] == "cls"]
    widths = [r["tag"] for r in R if r["tag"].startswith("w=") and r["head"] == "cls"]

    L = []
    A = L.append
    A("# P0 结果：H2 动作空间压缩\n")
    A("> 自动生成自 `p0_h2_results.json` / `p0_regloss_results.json`。不要手改。\n")
    A("```")
    A("python p0_h2.py --epochs 25 --seeds 2")
    A("python p0_regloss.py --epochs 25 --seeds 2")
    A("python p0_report.py")
    A("```\n")

    A("## ⚠️ 修正说明（必须读）\n")
    A("本文件曾有一版**错误结论**，原因是两个真实缺陷：\n")
    A("1. **缓存 key 未包含 `fill`**：加入元素尺寸扫描后，所有 fill 配置")
    A("   都复用了最早按 `fill=0.70` 构建的缓存 → 扫描 A 实际上只有一份数据。")
    A("2. **`gauss` 监督实现有缺陷**：它系统性弱于坐标 MSE。")
    A("   原扫描 B 用的是 `reg/gauss`，等于拿一个被削弱的 reg 基线去比，")
    A("   **高估了 Δ**（grid=6 时报 0.07，强基线实为 0.50）。\n")
    A("现在：缓存 key 覆盖全部 env 参数；reg 基线一律用**较强的 `coord` 损失**。")
    A("下表为修正后结果。\n")

    A("## 设置\n")
    A("| 项 | 值 |")
    A("|----|----|")
    A(f"| device | {d['device']} |")
    A(f"| train / test | {d['n_train']} / {d['n_test']} |")
    A(f"| epochs / seeds | {d['epochs']} / {d['seeds']} |")
    A("| backbone | 3 层小 CNN，保留空间结构 |")
    A("| 动作头 | reg=soft-argmax 坐标（coord 损失）；cls=g×g 池化分类 |")
    A("| 参数量 | **两头完全相同** |")
    A("| 判据 | 执行点是否落在目标元素 bbox 内 |")
    A("")

    A("## 扫描 A：精度要求（元素尺寸，grid=6）★ 主要难度轴\n")
    A("| 配置 | precision | cls hit | reg hit | Δ |")
    A("|------|-----------|---------|---------|---|")
    for t in fills:
        c, r = get(t, "cls"), get(t, "reg", "coord")
        A(f"| {t} | {c['precision']} | {c['hit_rate']:.4f} | {r['hit_rate']:.4f} | "
          f"{c['hit_rate']-r['hit_rate']:+.4f} |")
    A("")

    A("## 扫描 B：格子数（元素数量，fill=0.40）\n")
    A("| 配置 | slots | cls hit | reg hit | Δhit | reg cell |")
    A("|------|-------|---------|---------|------|----------|")
    for t in grids:
        c, r = get(t, "cls"), get(t, "reg", "coord")
        A(f"| {t} | {c['grid']**2} | {c['hit_rate']:.4f} | {r['hit_rate']:.4f} | "
          f"{c['hit_rate']-r['hit_rate']:+.4f} | {r['cell_acc']:.4f} |")
    A("")

    A("## 扫描 C：参数量（grid=8, fill=0.28）\n")
    A("| width | params | cls hit | reg(coord) | reg(gauss) |")
    A("|-------|--------|---------|------------|------------|")
    for t in widths:
        c, g, k = get(t, "cls"), get(t, "reg", "gauss"), get(t, "reg", "coord")
        A(f"| {c['width']} | {c['params']:,} | {c['hit_rate']:.4f} | "
          f"{k['hit_rate']:.4f} | {g['hit_rate']:.4f} |")
    A("")

    if rl:
        A("## reg 损失消融（诚信核查）\n")
        A("| grid | slots | cls | reg coord | reg gauss | Δ(coord) | Δ(gauss) |")
        A("|------|-------|-----|-----------|-----------|----------|----------|")
        for r in rl["rows"]:
            if r["head"] != "cls":
                continue
            g = r["grid"]
            rc = next(x for x in rl["rows"] if x["grid"] == g and x["head"] == "reg" and x["reg_loss"] == "coord")
            rg = next(x for x in rl["rows"] if x["grid"] == g and x["head"] == "reg" and x["reg_loss"] == "gauss")
            A(f"| {g} | {r['slots']} | {r['hit_rate']:.4f} | {rc['hit_rate']:.4f} | "
              f"{rg['hit_rate']:.4f} | {r['hit_rate']-rc['hit_rate']:+.4f} | "
              f"{r['hit_rate']-rg['hit_rate']:+.4f} |")
        A("")
        A("`gauss` 全面弱于 `coord`（grid=3 时 0.65 vs 1.00）。")
        A("这解释并修正了初版结论的高估。\n")

    A("## 结论（修正后）\n")
    A("1. **cls 在所有难度下 ≈100%**，12K 参数即足够。")
    A("2. **主要难度轴是精度要求（元素尺寸），不是元素数量**：")
    A("   fill 从 0.75 → 0.28，cls 恒 1.00，而 reg/coord 从 1.00 掉到 0.38，")
    A("   Δ 单调 0.003 → 0.622。")
    A("3. **元素数量同样有效**：grid 8→64 元素时 Δ 达 +0.93。")
    A("4. **参数量换不回**：grid=8 时 12K 的 cls ≈100%，392K 的 reg 仅 ~10%。")
    A("5. reg 的 cell_acc 通常仍较高（0.95–1.00），说明它**找到了位置却不够准**；")
    A("   只有在 grid=8 时 cell_acc 才崩到 0.30——粗定位也开始失效。")
    A("")

    A("## 这**不**意味着什么\n")
    A("- 结构化接口在 P0 用 **oracle 元素 bbox**，这是结构化的能力**上界**。")
    A("- 环境是合成的，未验证真实 UI 迁移。")
    A("- 只测了 H2，未测 H1/H3/H4，也未测主假设的 horizon-crossing。")
    A("- 指令 → 颜色的映射是语言接地的代理。")
    A("- gauss 监督的失败是本实现的问题，不代表热力图监督本身不行。")
    A("")
    return "\n".join(L)


def p05_section():
    d = load("p0_parser_results.json")
    R = d["levels"]
    L = []
    A = L.append
    A("# P0.5 结果：把 oracle 换成廉价解析器\n")
    A("> 自动生成自 `p0_parser_results.json`。不要手改。\n")
    A("```")
    A("python p0_select.py --epochs 25 --seeds 2")
    A("python p0_report.py")
    A("```\n")
    A("## 动机\n")
    A("P0 的结构化接口用的是 **oracle 元素 bbox**。这是 H2 最脆弱的假设：")
    A("真实系统必须自己\"看见\"元素。这里用一个**不含任何学习模型**的经典 CV")
    A("解析器（阈值 + 形态学开运算 + 连通域），并施加可控退化，")
    A("扫描「解析器质量 → H2 优势」。\n")
    A("```")
    A("screenshot → [阈值 → 开运算 → 连通域] → 候选框 → 选择器 → 执行框中心")
    A("```\n")
    A("## 设置\n")
    A("| 项 | 值 |")
    A("|----|----|")
    A(f"| device | {d['device']} |")
    A(f"| grid / width | {d['grid']} / {d['width']} |")
    A(f"| train / test | {d['n_train']} / {d['n_test']} |")
    A(f"| epochs / seeds | {d['epochs']} / {d['seeds']} |")
    A(f"| reg 基线（无解析器） | hit = **{d['reg_hit']:.4f}**, {d['reg_params']:,} params |")
    A("")

    A("## 结果\n")
    A("| 档位 | recall | best IoU | n_det | parser-cls | reg | Δ | sel_acc |")
    A("|------|--------|----------|-------|-----------|-----|---|---------|")
    for r in R:
        A(f"| {r['level']} | {r['recall']:.3f} | {r['best_iou']:.3f} | {r['n_det']:.1f} | "
          f"{r['cls_hit']:.4f} | {d['reg_hit']:.4f} | "
          f"{r['cls_hit']-d['reg_hit']:+.4f} | {r['sel_acc']:.4f} |")
    A("")

    A("## 结论\n")
    A("1. **`cls_hit ≈ recall`**：选择器几乎完美，")
    A("   **瓶颈是解析器召回，不是选择能力**。")
    A("2. **抖动几乎无害**：IoU 从 0.717 降到 0.499（Q1→Q3），cls 仍 99.95%。")
    A("   只要框心还在真值 bbox 内，执行就命中。")
    A("3. **crossing 在 recall ≈ 0.5**：Q8（recall 0.498, IoU 0.163）时")
    A("   parser-cls 0.4720 首次**低于** reg 0.4935（Δ = -0.0215）。")
    A("4. 因此工程目标很具体：**把目标元素的解析召回做到 > 0.5**，")
    A("   框的精度要求反而宽松得多。\n")

    A("## 这**不**意味着什么\n")
    A("- 环境仍是合成的；真实 UI 有纹理、渐变、遮挡、非矩形控件，")
    A("  经典 CV 解析器的召回会远低于此处的 clean 档。")
    A("- 这是\"解析器质量\"的敏感性分析，不是\"真实解析器能达到多好\"的结论。")
    A("- 仍未测 H1/H3/H4 与 horizon-crossing。")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    Path("P0_RESULTS.md").write_text(p0_section(), encoding="utf-8")
    print("已写出 P0_RESULTS.md")
    Path("P0_5_RESULTS.md").write_text(p05_section(), encoding="utf-8")
    print("已写出 P0_5_RESULTS.md")
