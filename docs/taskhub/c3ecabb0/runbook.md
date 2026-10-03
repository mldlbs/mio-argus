> 软件项目文档链 · 7 件套  
> 上游：[测试验收](docs/taskhub/c3ecabb0/test.md) | 下游：—  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 部署运维（Runbook）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.1 |
| 最后更新 | 2026-10-03 19:40 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 本机运行接环实验（QGIS + bridge + 模型推理）。不含云端部署、不含多机 |

## 1. 环境与依赖
- 运行环境：Windows 10/11；Python 3.12（`C:\Users\gf191\AppData\Local\Programs\Python\Python312`）。
- 版本要求：QGIS 3.44 LTR（`D:\QGIS\bin\qgis-ltr.bat`）；torch + CUDA（可用时）；transformers 离线缓存（`TRANSFORMERS_OFFLINE=1`）。
- 外部服务依赖：
  - QGIS 进程 + qgis_bridge（由 `PYQGIS_STARTUP` 注入启动）。
  - checkpoint：`artifacts/checkpoints/computer_use_model_v3_real.pth`（881 MB，必须存在）。
  - ViT/GPT-2 权重：HF 本地缓存（离线可用）。

## 2. 部署步骤
1. 启动 QGIS（带桥）：
   ```powershell
   $env:PYQGIS_STARTUP = "D:\work\code\nanoGPT\experiments\qgis\qgis_bridge\_startup.py"
   Start-Process "D:\QGIS\bin\qgis-ltr.bat" -WindowStyle Hidden
   # 等 ~16 s，验证：python -c "from client import call; print(call({'op':'ping'}))"
   ```
2. 运行 W28 测距（干跑 + 安全闭环）：
   ```powershell
   python experiments/computer_use/loop_w28.py --trials 3
   # 只干跑（不执行点击）：加 --dry-only
   ```
3. 查看结果：`results/computer_use/w28_results.json`（summary 字段）。
4. （W29 起）数据采集 / 微调 / 闭环脚本按 plan.md 逐步追加。

## 3. 配置项
| 配置 | 默认值 | 说明 |
|------|--------|------|
| `--trials` | 3 | 干跑与闭环的试次数 |
| `--dry-only` | false | 只跑干跑（零点击） |
| `TRANSFORMERS_OFFLINE` | 1（脚本内 setdefault） | 离线加载 HF 缓存 |
| 安全窗口 | QGIS 主窗口矩形 | 越界预测点不执行（硬约束） |
| 动作白名单 | click（+ scroll 计划中） | 非白名单动作不执行 |
| 期望容差 | extent 1% span；scale 1% | truth 判定容差 |

## 4. 回滚
- 触发条件：脚本异常、结果异常（如误点/越界点击）、QGIS 崩溃。
- 回滚步骤：
  1. `git checkout -- <改动文件>` 或 `git revert <commit>` 回退脚本与文档。
  2. 重跑 `python -m pytest -q` 确认仓库基线（离线：14 passed / 1 failed）。
  3. 示范数据（artifacts/）删除目录即可重录；结果 json 随 git 回滚。

## 5. 监控与告警
- 关键指标：`summary.dry_action_acc / dry_hit_rate / live_success_rate / oracle_ok`；每步 `fail` 分类计数。
- 健康检查端点：`{"op":"ping"}`（bridge）；进程存在性（`qgis-ltr-bin`）。
- 告警阈值：越界点击 > 0（应立即停跑排查）；oracle_ok=false（机械问题，先修协议再跑模型）。

## 6. 常见故障排查
- QGIS 窗口 -32000（最小化幽灵）→ `ensure_visible` 自动兜底；若仍失败，手动恢复窗口后重跑。
- bridge 连接拒绝 → QGIS 未带 `PYQGIS_STARTUP` 启动；按 §2 重启。
- 模型全 predict scroll / 坐标≈原点 → 域差距（预期行为，W28 已量化）；不是 bug，见 test.md 自查记录。
- QGIS 关设计器后异步崩溃（W27 已知）→ W31 计划接 `ensure_qgis()` 自动重启；当前手动重启。
- CUDA OOM → 关掉其它占显存进程，或临时 CPU 推理（慢 ~10×）。
