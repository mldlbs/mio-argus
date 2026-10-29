# W30 结果：LoRA 微调 + 留出链重测（FR-5）

> 复现：
> ```
> python experiments/computer_use/train_inloop.py --apps qgis \
>     --splits artifacts/computer_use_inloop/qgis_splits.json --epochs 8 \
>     --out artifacts/checkpoints/w30_full.pth
> python experiments/computer_use/train_inloop.py --apps qgis \
>     --splits artifacts/computer_use_inloop/qgis_splits.json --dedup --epochs 8 \
>     --out artifacts/checkpoints/w30_dedup.pth
> python experiments/computer_use/eval_transfer.py --ckpt artifacts/checkpoints/w30_full.pth \
>     --app qgis --ignore-eval-frames \
>     --train-meta artifacts/checkpoints/w30_full.json --split val --tag W30_full_heldout_chains
> ```
> 数据：`results/computer_use/transfer.json`　基线：[W29_RESULTS.md](W29_RESULTS.md)、[W28B_RESULTS.md](W28B_RESULTS.md)
> 需求：FR-5（LoRA 微调 + 重跑同题对照）

---

## 1. 结论

> **动作策略跨场景迁移成立，坐标精度没有改善。**
> 在两条**完全未见过的链**（`w26_v2`、`w27`）上：动作准确率 **1.000**，多数类基线 0.415；
> 但坐标中位误差 **358px**、命中率 **0.0**（341 个带 rect 的 click 全未命中）。
> 按 plan §4 的 kill 条件——**微任务步成功率 = 0 → 触发归因分析，不扩数据**。

## 2. 设置

| 项 | 值 |
|---|---|
| 训练集 | 14 任务 / **1394 条**（在已验证链上录制，见 W29） |
| 留出集 | 4 任务 / **626 条**，两条链整体未见（`w26_v2`、`w27`） |
| 模型 | `ComputerUseModelV3`，ViT/GPT-2 冻结 + LoRA r=16，30.8M 可训练 |
| 坐标目标 | `norm_coord`（屏无关）；梯度裁剪 1.0 |
| 训练 | 8 epoch，Adam 1e-4，batch 8，约 4.6 分钟/epoch |

## 3. 结果

| 臂 | 训练条数 | 动作准确率 | 多数类基线 | 坐标中位误差 | 命中率 |
|---|---|---|---|---|---|
| 全量 | 1394 | **1.000** | 0.415 | **358 px** | **0/341** |
| 去重消融 | 58 | 1.000 | 0.415 | 785 px | 0/341 |
| 基线（1 epoch） | 1394 | 1.000 | 0.415 | 593 px | 0/341 |

- **动作**：1 epoch 就已经 1.000；8 epoch 没有变化。动作标签高度可学（4 类、模板化指令），
  这条指标已经饱和，**不能再用来证明"微调有用"**。
- **坐标**：1 epoch 593px → 8 epoch 358px，方向对但幅度远不够；按钮约 32px，
  命中需要 <16px，**差 20 倍**。
- **去重消融**：把 1394 条压到 58 条唯一行为后，坐标误差反而升到 785px。
  说明重复样本对坐标是有用的（提供了位置抖动），但**上限被精度卡住，不是被数据量卡住**。

## 4. 归因（为什么动作满分、坐标全错）

1. 动作分类是**离散、少量、模板化**的：47 条指令、4 类动作，1 个 epoch 就能记住映射。
2. 坐标是**连续回归**，且分辨率要求比特征图粒度更细：ViT patch 16，224 输入下每 token
   覆盖约 11px 原图；要稳定落在 32px 按钮内，需要亚 token 级定位。
3. 冻结主干 + LoRA 只调注意力增量，坐标头能拿到的位置信息有限；
   之前 Notepad 实验（W28b §6）也是同样的形态：训练集内 188px、150 epoch 不改善。

## 5. 判定与下一步

按 plan §4：

- 动作准确率不再是有效指标（已饱和，且 W28b 已证明它会被标签分布买通）。
- 步成功率（坐标命中）= 0 → **触发 kill 分支**：不继续扩数据，先改坐标目标。

三个候选，成本从低到高：

| 方案 | 做法 | 预期 |
|---|---|---|
| A 热力图/网格分类 | 把坐标回归换成 32×32 或 64×64 网格分类 + 局部细化 | 直接提升分辨率，改动只在 head |
| B 候选框打分 | 复用 W29 数据里的 `observation.affordances`（2020/2020 都有），把"点哪里"变成"选哪个控件" | 精度问题变成排序问题；但 W28b §8–§10 已证明该路线**换了题**，仅作对照 |
| C 降任务要求 | 只在大目标（>200px）上评测，先跑通闭环 | 能出 W31，但不能解决工具栏按钮 |

推荐 **A**：它是唯一在"像素→坐标"这个问题内部提高分辨率、且不动任务的方案。
B 明确不推荐作为主线（见 W28B_RESULTS §11 的范围更正）。
