> 软件项目文档链 · 7 件套  
> 上游：[模块 Spec](docs/taskhub/c3ecabb0/spec.md) | 下游：[接口契约](docs/taskhub/c3ecabb0/api.md)  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 状态模型 / 数据模型（Data & State Model）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.1 |
| 最后更新 | 2026-10-03 19:40 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 闭环结果记录（W28 实测格式）+ 示范数据格式（W29 契约）。不含模型权重文件格式、QGIS 工程文件 |

## 1. 实体与字段

### 1.1 步骤记录 StepRecord（`results/computer_use/w28_results.json` 的 `dry[]` / `live[]`）

| 实体.字段 | 类型 | 约束 | 说明 |
|-----------|------|------|------|
| step.task | string | 非空（live） | 任务名：zoom_full / zoom_last / zoom_native / oracle_zoom_full |
| step.trial | int | ≥0；0=oracle 自检 | 试次 |
| step.instruction | string | 非空 | 模型指令（英文模板短句） |
| step.pred | object | 非空 | `{action, coord:[x,y], scroll, ms}`，coord∈[0,1]² |
| step.px | [float,float] | 非空 | 反投影屏幕坐标（px） |
| step.rect | [int,int,int,int] | dry 非空 | 目标 UIA 矩形（l,t,r,b，屏幕坐标） |
| step.hit | bool | dry 非空 | 预测点是否落在目标 rect 内 |
| step.dist_px | float | ≥0（dry） | 预测点到目标 rect 的欧氏距离 |
| step.action_ok | bool | dry 非空 | 预测动作是否 == click |
| step.in_window | bool | live 非空 | 预测点是否在 QGIS 主窗口内 |
| step.clicked | bool | live 非空 | 是否真执行了点击 |
| step.success | bool | live 非空 | truth 判定是否通过 |
| step.fail | string\|null | 枚举见 §2 | 失败分类；成功为 null |
| step.scale | float | live 可空 | 点击后 canvas scale（truth） |
| step.extent | [float×4] | live 可空 | 点击后 canvas extent（truth） |

### 1.2 示范样本 DemoSample（W29 契约，`artifacts/computer_use_inloop/demo_*.json` 元素）

| 实体.字段 | 类型 | 约束 | 说明 |
|-----------|------|------|------|
| demo.id | string | 唯一（文件内） | 稳定 id（app+phase+seq） |
| demo.screenshot | string | 存在（相对仓库根） | 截图路径（1920×1080 或 2560×1440） |
| demo.instruction | string | 非空 | 英文模板短句（由链步骤语义生成） |
| demo.action | string | ∈ {click,type,scroll,move,hotkey,press} | 脚本实际执行的动作类型 |
| demo.params | object | 非空 | click/move: `{x,y}`（屏幕像素）；scroll: `{direction}`；type/hotkey/press: `{text}` 或空 |
| demo.post_state | object | 可空 | 执行后真值（bridge truth 摘要：extent/scale/tool/renderer 变化） |
| demo.meta.app | string | 非空 | 应用标识（qgis） |
| demo.meta.phase | string | 非空 | 链阶段（symbology/layout/export/…） |
| demo.meta.chain | string | 非空 | 来源脚本（run_w25 / run_w27 / …） |

### 1.3 结果汇总 Summary

| 实体.字段 | 类型 | 约束 | 说明 |
|-----------|------|------|------|
| summary.dry_n | int | ≥0 | 干跑步数 |
| summary.dry_action_acc | float | [0,1] | 动作准确率（预测 click 比例） |
| summary.dry_hit_rate | float | [0,1] | 坐标命中率（落在目标 rect） |
| summary.dry_dist_px_median | float | ≥0 | 距离中位（px） |
| summary.live_n | int | ≥0 | 模型闭环步数（不含 oracle） |
| summary.live_success_rate | float | [0,1] | 步成功率 |
| summary.live_fails | object | 计数 | 失败分类计数 |
| summary.oracle_ok | bool | 非空 | oracle 通路自检是否通过 |

## 2. 状态机

### 2.1 单步执行状态机

| 当前状态 | 事件 | 下一状态 | 副作用 |
|---------|------|---------|--------|
| pending | 截图+推理完成 | predicted | 记录 pred/px |
| predicted | action≠click | skipped_action | 记 fail=action_wrong |
| predicted | 点在窗口外 | skipped_bounds | 记 fail=out_of_window |
| predicted | click 且点在窗口内 | clicked | 真点击 + sleep 1s |
| clicked | truth 达期望 | success | 记 success=true |
| clicked | truth 未达期望 | failed | 记 fail=truth_fail |

- 非法转移处理：状态机在脚本内线性实现；未达终态（异常）→ 脚本终止并保留已落盘记录。

### 2.2 任务级状态（taskhub 侧，任务 c3ecabb0）

| 当前状态 | 事件 | 下一状态 | 副作用 |
|---------|------|---------|--------|
| ready | agent 领取 | implementing | run=43731559，分支 task-c3ecabb0 |
| implementing | 结果提交 success | done | 关联 gitref（commit） |
| implementing | 结果提交 fail | retrying/failed | 按重试策略 |

## 3. 完整性约束
- 唯一键：demo.id 文件内唯一；StepRecord 以 (task, trial) 唯一。
- 外键：demo.meta.chain 指向来源脚本路径；StepRecord 无外键。
- 业务不变量（invariant）：
  - `clicked==true` ⇒ `in_window==true`（安全门不允许越界点击）。
  - `success==true` ⇒ `clicked==true`（未执行不可能成功）。
  - `fail==null` ⇔ `success==true`（live 记录）。
  - coord 分量 ∈ [0,1]；px = coord × 屏幕尺寸（反投影一致）。

## 4. 迁移策略
- 存量数据迁移：不适用（W28 为首版格式；W29 新增 DemoSample 为独立文件，不改 StepRecord）。
- 回滚方式：结果 json 随 git 回滚；示范数据在 artifacts/（gitignored）可重录。
