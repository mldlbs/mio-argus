> 软件项目文档链 · 7 件套  
> 上游：[模块 Spec](docs/taskhub/c3ecabb0/spec.md) | 下游：[接口契约](docs/taskhub/c3ecabb0/api.md)  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 状态模型 / 数据模型（Data & State Model）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.2 |
| 最后更新 | 2026-10-03 20:05 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 闭环结果记录（W28 实测格式）+ 示范数据格式（W29 契约 v2：观察/意图/动作/结果/上下文/失败 六要素）。不含模型权重文件格式、QGIS 工程文件 |

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

### 1.2 示范样本 DemoSample（W29 契约 v2，`artifacts/computer_use_inloop/<task_id>/demo.json` 元素）

**六要素**：观察（observation）/ 意图（intent）/ 动作（action）/ 结果（result）/ 上下文（context）/ 失败（failure）。
前 6 个字段是训练加载兼容子集（W30 loader 直接读），其余是研究元数据。

| 实体.字段 | 类型 | 约束 | 说明 |
|-----------|------|------|------|
| demo.id | string | 文件内唯一 | `<task_id>:<seq>` |
| demo.screenshot | string | 存在（相对仓库根） | 动作前截图（训练 loader 用；采集时按 `--shot-scale` 缩放落盘） |
| demo.instruction | string | 非空 | 英文指令模板（由 phase+事件+控件文本生成，训练 loader 用） |
| demo.action | string | ∈ ACTION_MAP | 动作类型（训练 loader 用） |
| demo.params | object | 非空 | click/move: `{x,y}` 屏幕像素；scroll: `{direction}`；type/hotkey/press: `{text}`（训练 loader 用） |
| demo.norm_coord | [float,float] | click/move 非空 | 归一坐标 [0,1]²（= x/W, y/H；**屏无关**，W30 loader 用） |
| demo.screen | [int,int] | 非空 | 采集时屏幕尺寸 [W,H] |
| demo.observation | object | 非空 | 观察：`{truth, affordances, windows, canvas_rect}` |
| demo.intent | object | 非空 | 意图：`{task, phase, goal}` |
| demo.result | object | 非空 | 结果：`{screenshot(可空), truth, ok}` |
| demo.context | object | 非空 | 上下文：`{app, scene, active_layer, window_rect, perturbation, chain}` |
| demo.failure | object\|null | — | 失败样本标记：`{kind, injected, note}`；正常样本为 null |

- observation.truth：动作前 bridge truth 摘要（layers/extent/scale/tool/renderer）。
- observation.affordances：bridge structure 通道，仅可见且带文本的控件：`{text, rect:[l,t,w,h], enabled, toolbar, cls}`。
- observation.windows：可见顶级窗口清单（`{class, title, rect}`）——遮挡/焦点错误的证据。
- intent.task：任务名（thematic_map 等）；intent.phase：链阶段（symbology/layout_add_map/…）；intent.goal：任务级目标句。
- result.ok：教师动作是否按预期生效（truth 前后对比 / 链步骤返回值）。

**扰动 taxonomy**（`context.perturbation.kind`，随机种子驱动）：

| kind | 注入方式 | 教师行为 |
|------|---------|---------|
| nominal | 无 | 正常链 |
| drift | 随机化场景（real/scenario/cities）、活动图层、画布缩放（随机子范围）、窗口尺寸/最大化 | 正常链（同动作不同视觉状态） |
| occlusion | 外来窗口（记事本）遮挡画布一角（不挡目标控件） | 正常链（局部遮挡下定位） |
| focus_error | 链步骤间特权开一个模态对话框（如 Options…） | 教师先按 Esc 关闭（该步 action=hotkey, text 含 esc），再续链 |

**失败样本不变量**（追加于 §3）：
- `failure.kind=="focus_error"` ⇒ 该步 `action=="hotkey"` 且 `params.text` 含 `"esc"`（恢复动作示范）。
- `failure.injected==true` ⇒ `observation.windows` 含注入窗口/对话框证据（可审计）。
- 失败样本**只增不替**：每个失败样本旁邻的同任务正常样本保留（对照学习）。

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
- 唯一键：demo.id 文件内唯一（`<task_id>:<seq>`）；StepRecord 以 (task, trial) 唯一。
- 外键：demo.context.chain 指向来源脚本路径；StepRecord 无外键。
- 业务不变量（invariant）：
  - `clicked==true` ⇒ `in_window==true`（安全门不允许越界点击）。
  - `success==true` ⇒ `clicked==true`（未执行不可能成功）。
  - `fail==null` ⇔ `success==true`（live 记录）。
  - coord 分量 ∈ [0,1]；px = coord × 屏幕尺寸（反投影一致）。
  - `norm_coord` 分量 ∈ [0,1] 且 = `params.x/W`、`params.y/H`（click/move；W,H = screen）。
  - 训练兼容子集（screenshot/instruction/action/params）对 W30 loader 恒可加载。
- 迁移策略：W29 v2 为新增独立文件；W28 StepRecord 格式不变。

## 4. 迁移策略
- 存量数据迁移：不适用（W28 为首版格式；W29 新增 DemoSample 为独立文件，不改 StepRecord）。
- 回滚方式：结果 json 随 git 回滚；示范数据在 artifacts/（gitignored）可重录。
