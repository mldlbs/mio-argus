# Computer Use Model v3 评测报告

> **模型版本**：v3 (ViT-B/16 + GPT-2, LoRA r=16)  
> **评测日期**：{{DATE}}  
> **评测人**：{{AUTHOR}}  
> **Git Commit**：{{COMMIT_HASH}}

---

## 1. 实验配置

| 项目 | 设置 |
|------|------|
| **模型架构** | ViT-B/16 (frozen) + GPT-2 (frozen) + LoRA (r=16) + Cross-Attention Fusion |
| **训练数据** | `computer_use_data_real_balanced/balanced_training_data.json` (90 样本, 6 类各 15) |
| **划分比例** | Train 70% / Val 20% / Test 10% (stratified, seed=42) |
| **测试样本数** | {{N_TEST}} |
| **训练轮数** | 30 epochs |
| **学习率** | 2e-4 (Adam) |
| **Batch Size** | 8 |
| **设备** | {{DEVICE}} (RTX 3060 12GB) |
| **检查点** | `computer_use_model_v3_real.pth` |

---

## 2. 核心指标

| 指标 | 数值 | 基线 | 状态 |
|------|------|------|------|
| **Action Accuracy** | {{ACTION_ACC:.4f}} | ≥ 1.0000 | {{ACTION_STATUS}} |
| **Action Macro-F1** | {{ACTION_F1:.4f}} | ≥ 1.0000 | {{F1_STATUS}} |
| **Coord MAE** | {{COORD_MAE:.6f}} | ≤ 0.1000 | {{COORD_STATUS}} |
| **Scroll Accuracy** | {{SCROLL_ACC:.4f}} | ≥ 0.9000 | {{SCROLL_STATUS}} |

> **判定规则**：Action Accuracy/F1/Scroll Accuracy 不低于基线；Coord MAE 不高于基线。

---

## 3. 类别级指标

| 动作类别 | 测试样本数 | 准确率 | 备注 |
|----------|------------|--------|------|
| click    | {{CLICK_N}} | {{CLICK_ACC:.4f}} |      |
| type     | {{TYPE_N}}  | {{TYPE_ACC:.4f}}  |      |
| scroll   | {{SCROLL_N}}| {{SCROLL_CLS_ACC:.4f}}|      |
| move     | {{MOVE_N}}  | {{MOVE_ACC:.4f}}  |      |
| hotkey   | {{HOTKEY_N}}| {{HOTKEY_ACC:.4f}}|      |
| press    | {{PRESS_N}} | {{PRESS_ACC:.4f}} |      |

---

## 4. 混淆矩阵

```
           click   type  scroll   move hotkey  press
click       {{CM_00}}   {{CM_01}}   {{CM_02}}   {{CM_03}}   {{CM_04}}   {{CM_05}}
type        {{CM_10}}   {{CM_11}}   {{CM_12}}   {{CM_13}}   {{CM_14}}   {{CM_15}}
scroll      {{CM_20}}   {{CM_21}}   {{CM_22}}   {{CM_23}}   {{CM_24}}   {{CM_25}}
move        {{CM_30}}   {{CM_31}}   {{CM_32}}   {{CM_33}}   {{CM_34}}   {{CM_35}}
hotkey      {{CM_40}}   {{CM_41}}   {{CM_42}}   {{CM_43}}   {{CM_44}}   {{CM_45}}
press       {{CM_50}}   {{CM_51}}   {{CM_52}}   {{CM_53}}   {{CM_54}}   {{CM_55}}
```

> 行 = Ground Truth，列 = Prediction。对角线越深越好。

---

## 5. 定性分析 Checklist

- [ ] **坐标误差分布**：是否有系统性偏移（如整体向右上漂移）？
- [ ] **困难样本**：`confidence < 0.7` 或 `coord_MAE > 0.15` 的样本已人工复核？
- [ ] **注意力可视化**：ViT CLS token attention / Grad-CAM 是否聚焦于操作目标区域？
- [ ] **类别混淆**：是否存在 click↔press、scroll↔move 等高混淆对？
- [ ] **边界情况**：极坐标 (0,0) / (1,1) 附近的预测是否合理？

---

## 6. 端到端 / 真实环境验证（可选）

| 任务 | 成功率 | 平均步数 | 备注 |
|------|--------|----------|------|
| 打开记事本并输入 "hello" |      |          |      |
| 网页点击按钮并滚动 |      |          |      |
| 组合键操作 |      |          |      |

> 建议在虚拟机/真实桌面跑 10-20 条完整指令，记录任务完成率。

---

## 7. 结论与下一步

- **整体表现**：{{OVERALL_ASSESSMENT}}
- **主要短板**：
  1.
  2.
  3.
- **改进建议**：
  - 数据增强（旋转、缩放、遮挡）
  - 解冻 ViT/GPT-2 最后 1-2 层
  - 引入更多真实数据（目标 500+ 样本）
  - 尝试更大 LoRA rank (r=32/64) 或全量微调对比

---

## 附件

- `eval_results.json` — 完整数值结果
- `eval_model.py` — 评测脚本
- `test_eval_metrics.py` — 回归测试
- `split_data.py` / `split.json` — 数据划分

---

*报告模板版本：v1.0 | 生成脚本：`scripts/gen_report.py` (待实现)*