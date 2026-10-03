> 软件项目文档链 · 7 件套  
> 上游：[接口契约](docs/taskhub/c3ecabb0/api.md) | 下游：[部署运维](docs/taskhub/c3ecabb0/runbook.md)  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 测试与验收（Test & Acceptance）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.1 |
| 最后更新 | 2026-10-03 19:40 |
| 状态 | planned |
| 负责人 | opencode |
| 适用范围 | W28 已执行协议 + W29–W31 待执行协议。不含模型内部单测（沿用既有 pytest 基线） |

## 1. 验收标准
| FR 编号 | 验收标准 | 验证方式 |
|---------|---------|---------|
| FR-1 | 3 试 0 次越界点击；每步落盘 pred/clicked/truth/fail | `w28_results.json` 检查 `clicked⇒in_window` 不变量 |
| FR-2 | ≥6 指令 × 3 试；输出 action_acc / hit_rate / dist_px 中位 | `summary.dry_*` 字段检查 |
| FR-3 | oracle 自检 PASS（协议 bug 修复后重测） | `summary.oracle_ok == true`（W28 首测 false→定位期望值协议 bug→修复后复测 true） |
| FR-4 | ≥2k 步示范数据；格式符合 data-model；train/val 划分落盘 | W29 数据检查脚本 + 字段校验 |
| FR-5 | 增益表落盘；原 504 条评测集无回归（±0.02） | W30 重测 `w30_results.json` + `test_in_vm_v3.py` 对照 |
| FR-6 | 任务成功率与正确产物/分钟落盘；≥3 试；失败归因 | W31 `w31_results.json` + 对照脚本基线 |

## 2. 测试范围
- 单元测试：
  - 坐标反投影一致性（coord×screen == px）。
  - 安全门逻辑（action≠click / 越界 → 不点击）。
  - 容差函数（ext_close / scale 相对容差）。
- 集成测试：
  - oracle 通路自检（真点击 + bridge truth）。
  - 干跑测距（UIA rect 真值）。
- E2E / 冒烟：
  - 安全闭环三任务（zoom_full / zoom_last / zoom_native）× 3 试。
- 明确不测的：
  - 模型训练正确性（沿用既有 504 条评测集）。
  - 多应用泛化、VM 合成桌面。

## 3. 用例清单
| 用例编号 | 覆盖需求 | 前置 | 步骤 | 预期 |
|---------|---------|------|------|------|
| TC-1 | FR-1 | QGIS 运行 + bridge 在线 + real 场景 | 1. reset real；2. 跑闭环 3 任务 ×3 试；3. 读 json | `clicked==true ⇒ in_window==true`；`success⇒clicked`；fail 分类合法 |
| TC-2 | FR-2 | 同上 | 1. 干跑 6 指令 ×3 试；2. 读 `summary.dry_*` | 字段齐全；hit 判定与 rect 一致（抽查 3 条手算） |
| TC-3 | FR-3 | 同上 | 1. 特权校准期望；2. UIA 真值坐标点击 Zoom Full；3. 读 truth | oracle PASS（W28 为 truth_fail，根因：期望用单层 extent，实际为全图层并集） |
| TC-4 | FR-4 | W23–W27 链可运行 | 1. 带录制器跑链；2. 校验 demo 字段与数量 | ≥2k 步；字段/枚举合法；划分落盘 |
| TC-5 | FR-5 | W29 数据 + 训练环境 | 1. LoRA 微调；2. 重跑 TC-2/TC-3 同题；3. 跑 504 评测集 | 增益表；原评测无回归（±0.02） |
| TC-6 | FR-6 | W30 checkpoint + 验证器机械 | 1. 任务集 ×3 试；2. 对照脚本基线；3. 计成本 | 成功率/成本落盘；失败归因完整 |

## 4. 验收流程
- 谁验收：任务发起人（用户）+ taskhub 任务记录（review 阶段）。
- 何时验收：每个 W 任务提交结果时（`taskhub_submit_result` 附产出物）。
- 通过条件：对应 FR 验收标准全过；不变量无违反；结果 json 与文档字段一致。
- 不通过的回退路径：记录失败分类 → 按 plan 的 kill 条件决定修复/降级/终止。

## 5. W28 实测记录（2026-10-03，run 43731559）
- TC-2（干跑，18 步）：动作准确率 **0.0**（18/18 预测 scroll）；命中率 **0.0**；中位偏差 **475.6 px**。
- TC-1（安全闭环，9 步）：成功率 **0.0**，全部 `action_wrong`（无点击发生，安全门未触发越界）。
- TC-3（oracle）：首测 **FAIL（truth_fail）**——点击真实发生（UIA 真值坐标），根因定位为**期望值协议 bug**：Zoom Full 后 canvas extent 为全图层并集（cities 比 provinces 更宽），不能用 provinces 单层 extent 作期望。修复：特权 `act` 触发一次 Zoom Full 读 extent 作期望（与 zoom_native 的 scale 校准同模式）→ **复测 PASS（oracle_ok=true）**。
- 自查：同一推理代码在模型自家 balanced 数据上 20/20、MAE 0.085 ⇒ 0% 为真实域差距而非代码缺陷。
- 结论：W28 验收（FR-1/FR-2/FR-3）全部达成。
