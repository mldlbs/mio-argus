# headless Chromium vs headed：差别有多大，是否影响训练

> 复现：
> ```
> python experiments/web/web_gap.py            # 渲染发散量化（headed 为参考系）
> python experiments/web/web_policy.py         # 训练迁移：headless 训练 → headless/headed 评测
> python experiments/web/web_policy.py --tasks glyph --aug   # 加增强
> ```
> 环境：Windows，系统 Chrome（`channel="chrome"`），Playwright 1.63，`SESSIONNAME=Console`（可跑真 headed）。

---

## 1. 渲染发散（`web_gap.py`，headed 为参考）

| 组件 | headless 像素差(mean) | >8 占比 | bbox 漂移 | 文本度量差 |
|---|---|---|---|---|
| border / gradient / shadow / svg / canvas / transform | **0.00**（`--disable-gpu` 时 ≤0.04，极少数像素） | 0.000 | **0.00 px** | — |
| text（小字） | 0.56（grayscale 模式 0.96） | 0.004 | **0.00 px** | canvas `measureText` Δ=6.57 |
| emoji | 0.66 | 0.005 | 1.00 px | — |

**结论**：
- **布局/几何完全一致**（所有页面 bbox Δ = 0）——CSS 布局引擎是同一个 Blink。
- 唯一真实的像素差在**文字/字形的光栅化**（抗锯齿/字体提示），影响 ~1% 像素，**肉眼不可辨**。
- GPU 开/关只造成极少数像素 ≤70 的差；`--disable-lcd-text`（灰度 AA）反而让文字差更大（0.96 vs 0.56），说明默认 headless 已接近 headed 的 subpixel AA。
- 本机是 Windows，系统字体齐全；**注意 Linux 容器缺字体时此项会显著恶化**（未测）。

---

## 2. 是否影响训练（`web_policy.py`）

任务：N×N 网格，给线索 id，指出目标格。同一批页面：headless 截图训练 → headless 测试 vs **headed 测试**。

| 任务 | headless_acc | headed_acc | gap | 预测翻转率 |
|---|---|---|---|---|
| 颜色（3×3 大色块） | 1.000 | 1.000 | **0.000** | **0.000** |
| 字形（3×3，18px 小字形） | 1.000 | 0.268 | **−0.733** | **0.733** |
| 字形 + 增强（平移/亮度/噪声） | 0.975 | 0.718 | −0.258 | 0.280 |

关键对照：**字形页的 headless/headed 两版截图肉眼完全一致**（mean diff 1.60，仅 1.3% 像素 >8）。
所以那 73% 的翻转**不是环境变了，是模型脆**——一个在 headless 上打到 100% 的紧像素分类器，
把抗锯齿的微差当成了判别依据。

**增强把 gap 从 0.73 压到 0.26（翻转 0.74→0.28）**，代价是分布内 accuracy 1.00→0.975。
残差 0.26 来自"只喂 headless"；**同时喂 headed（或干脆用 headed 训练）可进一步闭合**。

---

## 3. 回答"是否影响训练"

- **不改任务、不改几何**：颜色/几何类策略 **gap=0**。对正则良好或结构化的策略，headless 没问题。
- **对依赖细小字形/文字的紧像素模型**：可能产生**大幅迁移失败**（翻转 0.73）——
  但这是**正则化/域随机化问题**（增强即大幅缓解），不是"headless 环境坏了"。
- **真正该担心的 sim-to-real 因素不是 headless-vs-headed**，而是：
  观测通道（DOM vs 像素）、动作语义（`element.click` vs 真光标）、缺失的 OS/浏览器 chrome、
  Linux 容器缺字体、真实世界的非确定性与站点多样性。

---

## 4. 一句话

> 在 Windows + 系统 Chrome 上，headless 与 headed **布局完全相同、像素差肉眼不可辨**，
> 只在字形光栅化上有 ~1% 像素的差。它对**颜色/几何任务零影响**；
> 会让**脆弱的细小字形分类器**崩，但那是模型缺乏不变性——加增强就把 0.73 的坑填掉 2/3。
> **headless 不是 sim-to-real 的主要矛盾。**

---

## 5. 动作语义（`locator.click` vs 坐标点击）—— 一次部分自我否定

`web_action_gap.py`：三种动作作用在会暴露差异的 widget 上
（A=元素感知 `locator.click`；B=坐标 `mouse.click(cx,cy)`；C=hover+等待+重定位+坐标点击）。

| widget | A locator | B coord | C hover+reloc |
|---|---|---|---|
| simple | ✅ | ✅ | ✅ |
| shifting（hover 位移） | ✅ | ✅ | ✅ |
| overlay（透明层遮挡） | ❌ | ❌ | ❌ |
| tiny（6px） | ✅ | ✅ | ✅ |
| animating（持续动画） | **❌（TimeoutError）** | **✅** | **✅** |
| scroll_off（视口外） | ✅ | ✅ | ✅ |

**结论（与我上一轮的判断相反，如实记录）**：
- 特权通道与坐标通道**不等价**，但方向**不是**我预期的"特权通道藏起难点"：
  本套件里**没有出现 A 成功而 B 失败**的情况；
- 反而在**动画元素**上，`locator.click` 因 Playwright 的**稳定性检查**而超时失败，而坐标点击成功——
  即特权通道在某些情况下**更脆**。
- `overlay` 两通道都失败（真实难点，与通道无关）。
- 因此我此前"`element.click` vs 真光标影响很大"的判断，**在这套受控 widget 上未获支持，且出现反例**。
  要定论需要更真实的套件（iframe / shadow DOM / hover flyout / 滚动容器 / 真实拖拽）——**未测**。

> 与 §2–§4 合起来：headless-vs-headed 是**像素噪声**；
> 动作语义确有差异，但它是**双向**的，不能简单说"用 DOM 训练会作弊"。
