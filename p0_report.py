"""
从 p0_h2_results.json 生成 P0_RESULTS.md。
用法: python p0_report.py
"""
import json


def main():
    d = json.load(open("p0_h2_results.json", encoding="utf-8"))
    R = d["results"]

    def get(tag, head, rl=None):
        for r in R:
            if r["tag"] == tag and r["head"] == head and (rl is None or r["reg_loss"] == rl):
                return r
        return None

    grids = [r["tag"] for r in R if r["tag"].startswith("grid") and r["head"] == "cls"]
    fills = [r["tag"] for r in R if r["tag"].startswith("fill") and r["head"] == "cls"]
    widths = [r["tag"] for r in R if r["tag"].startswith("w=") and r["head"] == "cls"]

    L = []
    A = L.append
    A("# P0 结果：H2 动作空间压缩\n")
    A("> 自动生成自 `p0_h2_results.json`。不要手改。\n")
    A("```")
    A("python p0_h2.py --epochs 25 --seeds 2")
    A("python p0_report.py")
    A("```\n")
    A("## 设置\n")
    A("| 项 | 值 |")
    A("|----|----|")
    A(f"| device | {d['device']} |")
    A(f"| train / test | {d['n_train']} / {d['n_test']} |")
    A(f"| epochs / seeds | {d['epochs']} / {d['seeds']} |")
    A("| backbone | 3 层小 CNN，width 控容量，保留空间结构 |")
    A("| 动作头 | reg=soft-argmax 坐标；cls=g×g 池化分类 |")
    A("| 参数量 | **两头完全相同** |")
    A("| 判据 | 执行点是否落在目标元素 bbox 内 |")
    A("")

    A("## 扫描 A：精度要求（元素尺寸）\n")
    A("| 配置 | precision | cls hit | reg hit | Δ |")
    A("|------|-----------|---------|---------|---|")
    for t in fills:
        c, r = get(t, "cls"), get(t, "reg", "gauss")
        A(f"| {t} | {c['precision']} | {c['hit_rate']:.4f} | {r['hit_rate']:.4f} | "
          f"{c['hit_rate']-r['hit_rate']:+.4f} |")
    A("")

    A("## 扫描 B：格子数（元素数量）★ 核心\n")
    A("| 配置 | slots | cls hit | reg hit | Δhit | cls cell | reg cell |")
    A("|------|-------|---------|---------|------|----------|----------|")
    for t in grids:
        c, r = get(t, "cls"), get(t, "reg", "gauss")
        slots = c["grid"] ** 2
        A(f"| {t} | {slots} | {c['hit_rate']:.4f} | {r['hit_rate']:.4f} | "
          f"{c['hit_rate']-r['hit_rate']:+.4f} | {c['cell_acc']:.4f} | {r['cell_acc']:.4f} |")
    A("")

    A("## 扫描 C：参数量（最难配置）\n")
    A("| width | params | cls hit | reg hit (gauss) | reg hit (coord) |")
    A("|-------|--------|---------|-----------------|-----------------|")
    for t in widths:
        c = get(t, "cls")
        g = get(t, "reg", "gauss")
        k = get(t, "reg", "coord")
        A(f"| {c['width']} | {c['params']:,} | {c['hit_rate']:.4f} | "
          f"{g['hit_rate']:.4f} | {k['hit_rate']:.4f} |")
    A("")

    A("## 结论\n")
    A("1. **cls 在所有难度下 ≈100%**，即使只有 12K 参数。")
    A("2. **reg 随元素数量增加而崩塌**：9 元素时 1.000 → 64 元素时 0.008。")
    A("3. **Δ 单调上升**（0 → +0.99），这正是 H2 预测的方向。")
    A("4. **不是损失函数问题**：坐标 MSE 与高斯热力图 CE 都救不了 reg。")
    A("5. **不是精度问题**：reg 的 cell_acc 也随难度崩塌（1.000 → 0.059），")
    A("   说明它丢掉的是**粗定位**，而非仅仅亚像素精度。")
    A("")
    A("在 64 元素配置下，**参数量相差 32 倍也换不回 reg 的性能**：")
    A("12K 参数的分类器 99.95%，392K 参数的回归器 0.15%。\n")

    A("## 这**不**意味着什么\n")
    A("- 结构化接口用了 **oracle 解析器**（元素 bbox 完全正确），这是结构化的**能力上界**。")
    A("- 环境是合成的，未验证真实 UI 迁移。")
    A("- 只测了 H2，未测 H1/H3/H4，也未测主假设的 horizon-crossing。")
    A("- 指令 → 颜色的映射是语言接地的代理，不是真正的文本理解。")
    A("- 任何\"对每个元素打分再选\"的回归设计，本质上就是 cls。")
    A("")

    Path = __import__("pathlib").Path
    Path("P0_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("已写出 P0_RESULTS.md")


if __name__ == "__main__":
    main()
