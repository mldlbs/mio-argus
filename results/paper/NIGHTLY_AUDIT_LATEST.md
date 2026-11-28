# 夜间体检 2026-11-28 18:43

检查 5 项，FAIL 1 项，新增事实 0 条

| 检查 | 结果 | 详情 |
|---|---|---|
| git_sync | PASS | 未提交=59 未推送=0 |
| data_leak | ACK | leaked=['expanded_training_data.json', 'balanced_training_data.json']（已确认：历史数据集（v3_real/expanded 训练用），泄露已在 VISION_） |
| g6_coverage | FAIL | 5/19 有 G6 记录 |
| ledger_fresh | PASS | 最后一条事实 0.0 天前 |
| tests | SKIP | 建立基线 {'passed': 43, 'failed': 0, 'errors': 0} |

## 需要人看

- **g6_coverage**: 5/19 有 G6 记录
