# 夜间体检 2026-11-28 19:29

检查 4 项，FAIL 0 项，新增事实 1 条

| 检查 | 结果 | 详情 |
|---|---|---|
| git_sync | PASS | 未提交=62 未推送=0 |
| data_leak | ACK | leaked=['expanded_training_data.json', 'balanced_training_data.json']（已确认：历史数据集（v3_real/expanded 训练用），泄露已在 VISION_） |
| g6_coverage | PASS | 10 tested / 9 skipped / 0 missing（共 19）；IMAGE_IGNORED 4 个 |
| ledger_fresh | PASS | 最后一条事实 0.0 天前 |
