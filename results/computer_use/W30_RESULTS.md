# W30 结果：LoRA 微调 + 坐标头改造（FR-5）

> 复现：
> ```
> python experiments/computer_use/train_inloop.py --apps qgis \
>     --splits artifacts/computer_use_inloop/qgis_splits.json --epochs 8 \
>     --out artifacts/checkpoints/w30_full.pth
> python experiments/computer_use/train_inloop.py --apps qgis --head grid --grid 128 \
>     --splits artifacts/computer_use_inloop/qgis_splits.json --epochs 8 \
>     --out artifacts/checkpoints/w30_grid.pth
> python experiments/computer_use/eval_transfer.py --ckpt artifacts/checkpoints/w30_grid.pth \
>     --head grid --grid 128 --app qgis --ignore-eval-frames \
>     --train-meta artifacts/checkpoints/w30_grid.json --split val --tag W30_A_grid128
> ```
> 数据：`results/computer_use/transfer.json`　前序：[W29_RESULTS.md](W29_RESULTS.md)、[W28B_RESULTS.md](W28B_RESULTS.md)

---

## 1. 结论

> **动作策略跨链迁移成立；坐标定位没有解决，而且换头也没用。**
> 留出链（`w26_v2` + `w27`，完全未训练）上：动作 1.000（多数类 0.415）。
> 坐标命中：回归头 **6/216 (2.8%)**、网格头 **12/216 (5.6%)**，
> 而"永远猜最常见格子"这个平凡基线**恰好也是 12/216 (5.6%)**。
> 原因不在解码方式：224 输入下 **一个 ViT patch 覆盖原图 ~183px**，目标按钮 32px，
> 结构上就分辨不出来。**下一步要动输入分辨率/裁剪，不是动头。**

## 2. 设置

| 项 | 值 |
|---|---|
| 训练 | 14 任务 / 1394 条（W29 采集，链级划分） |
| 留出 | 4 任务 / 626 条（`w26_v2`、`w27` 整条未见） |
| 模型 | ViT/GPT-2 冻结 + LoRA r=16，30.8M 可训练 |
| 训练量 | 8 epoch，Adam 1e-4，batch 8，梯度裁剪 1.0 |

## 3. 结果（命中率已按修正后的口径重算）

| 臂 | 动作准确率 | 坐标中位误差 | 命中率 |
|---|---|---|---|
| 回归头（全量 1394） | 1.000 | 358 px | **6/216 = 2.8%** |
| 回归头（去重 58） | 1.000 | 785 px | **0/216** |
| **网格头 G=128** | 0.987 | 306 px | **12/216 = 5.6%** |
| 平凡基线：永远猜最常见格子 | — | — | **12/216 = 5.6%** |
| 上限：分类全对（bin 中心） | — | — | **216/216 = 100%** |

三条读数：

1. **动作已饱和**：1 epoch 就 1.000，8 epoch 不变；这条指标不能再用来证明微调有用。
2. **回归头在留出链上退到 2.8%**，低于平凡基线——它学到的是训练链的位置分布。
3. **网格头正好等于平凡基线**（12/216）。它没有学到"该点哪个格子"，只是输出众数格子。

## 4. 量具自纠：命中率算错过

第一版 `evaluate()` 里，`coord_pred` 已按 `coord_mask` 压缩过长度，却仍与整批
`batch_rows` 做 `zip`，导致只比对了每批的前若干个样本：216 个带 rect 的样本被算成 106 个。
本文所有命中率均已用 `mask.nonzero()` 取下标重算；`train_inloop.evaluate()` 已修复。
**W28b 的命中率同样是这个口径算的，但那里是"全 6 条都没进任何框"，修正后仍为 0，
结论不受影响；不过数字口径以此处为准。**

## 5. 为什么换头等于没换

- 输入被缩放到 224×224。ViT patch 16 → 原图上 **每 token 覆盖 2560/224×16 ≈ 183px**。
- 目标按钮约 32px，即 **0.17 个 token**。任何 head 都无法从这么粗的特征里恢复 32px 定位。
- 这解释了三个一直复现的现象：域内 188px 下不去（W28b §6）、150 epoch 不改善、
  网格头退化成众数。

## 6. 判定与下一步

按 plan §4 的 kill 条件（步成功率 = 0）：本次触发，但归因已经从"数据量/目标函数"
收敛到**表征分辨率**。三个候选：

| 方案 | 做法 | 代价 |
|---|---|---|
| **A′ 提高输入分辨率** | 448 或 896 输入 + patch 对齐，必要时只保留窗口区域 | 显存/耗时上升；改动一行 resize + 位置解码 |
| **B 裁剪局部** | 先粗定位到窗口/面板，再对该区域裁剪放大二次定位 | 两阶段，评测口径要改成两级 |
| C 候选框打分 | 用 affordance 选控件 | 已定性为换题（W28B §11），不做主线 |

推荐 **A′**：它直接对着"每 token 183px"这个数字改，且不改任务定义。
最小验证：把 `IMG_SIZE` 从 224 提到 768（patch 覆盖 ~53px），跑 3 epoch 看命中率是否离开 5.6% 的平地线。
