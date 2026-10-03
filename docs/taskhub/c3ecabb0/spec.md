> 软件项目文档链 · 7 件套  
> 上游：[架构设计](docs/taskhub/c3ecabb0/architecture.md) | 下游：[状态模型](docs/taskhub/c3ecabb0/data-model.md)  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 模块 Spec（Module Specification）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.2 |
| 最后更新 | 2026-10-03 18:57 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 闭环执行器 + 测距协议（W28 已实现）；示范数据录制器与验证器闭环给出接口约定（W29–W31 实现）。不含模型内部结构与 QGIS 本身 |

## 1. 模块职责与边界
- 做什么：
  - 闭环执行器：截图、模型推理、动作解析、安全门、真点击、truth 判定、逐条落盘。
  - 测距协议：干跑（UIA rect 真值）与安全闭环（bridge truth）两段协议；oracle 通路自检。
  - 示范数据录制（W29 约定）：确定性链运行中记录 `(截图, 指令, 动作, 坐标, 后状态)`。
- 不做什么：
  - 不训练模型（W30 复用 `train_computer_use_v3_real.py`）。
  - 不做多应用/通用桌面（单应用 QGIS）。
  - 不改 qgis_bridge 既有 ops（只消费）。

## 2. 输入 / 输出
- 输入：
  - 全屏截图 `PIL.Image`（2560×1440，16:9）
  - 指令 `str`（英文模板短句，如 `click the Zoom Full button`）
  - bridge truth（JSON：layers/extent/scale/tool）
  - UIA 按钮表（pywinauto：名称 → 屏幕 rect）
- 输出：
  - 结果 JSON：`results/computer_use/w28_results.json`（dry/live 逐条 + summary）
  - 示范数据（W29）：`artifacts/computer_use_inloop/demo_*.json`（见 data-model）

## 3. 依赖
- 上游模块：`mio/mio_argus.py`（模型定义与预处理）；`experiments/qgis/qgis_bridge/client.py`（call）。
- 外部库：torch、transformers（离线缓存）、pyautogui、pywinauto、win32gui、PIL。

## 4. 详细设计
- Brain.predict（`loop_w28.py`）：
  - 图像：`AutoImageProcessor('google/vit-base-patch16-224')` → pixel_values
  - 指令：GPT-2 tokenizer，`max_length=64`，padding=max_length
  - 前向：`model(pixel_values, input_ids, attention_mask)` → `action_logits / coord_pred / scroll_logits`
  - 解析：argmax(action)、`coord_pred`（已归一化 [0,1]）、argmax(scroll) → `{click/type/scroll/move/hotkey/press}`
- 安全执行器 do_step：
  - `ensure_visible(qg)`：bridge `win maximize` + 外部 `ShowWindow(SW_RESTORE→SW_MAXIMIZE)` + `SetForegroundWindow`；按"在屏内+面积最大"选择 QgisApp 句柄（规避 -32000 幽灵）
  - 门：`pred.action == 'click'` 且 `(x,y) ∈ QGIS 主窗口矩形`
  - 判定：`expect(truth)` 回调（extent 容差 1% span；scale 相对容差 1%）
- 测距协议（W28 版）：
  - A 干跑：6 条指令 × 3 试；目标=UIA 可见安全动作（Zoom Full / Zoom to Selection / Zoom Last / Zoom to Native Resolution (100%) / Refresh / Undo）；判定=预测点是否落在目标 rect
  - B 安全闭环：3 任务 × 3 试（zoom_full / zoom_last / zoom_native）；真值=extent/scale
  - oracle 自检：用 UIA rect 中心点一次 Zoom Full，验证"真点击+truth"机械可用
- 协议缺陷（W28 oracle 暴露，**已修复**）：
  - Zoom Full 的期望 extent 不能用单层（provinces）extent——canvas 全量范围是**全部图层并集**（cities 比 provinces 更宽）。
  - 修法（已实现）：期望值改为特权校准（bridge `act` 触发一次 Zoom Full → 读 extent 作期望），与 zoom_native 的 S* 校准同模式；复测 oracle PASS。
- 预期新增 / 修改的文件：
  - 已有：`experiments/computer_use/loop_w28.py`（本轮）
  - 待建：`experiments/computer_use/collect_inloop.py`（W29）、`run_w30`/`run_w31` 系列脚本（命名待定）

## 5. 异常与边界情况
| 场景 | 处理方式 |
|------|---------|
| QGIS 窗口最小化（Qt 自认最大化、Win32 仍 -32000） | `ensure_visible` 外部 ShowWindow 兜底；选窗口时排除屏外句柄 |
| QGIS 进程不在 | 脚本早退并记 `no_qgis_window`（W31 计划接 `ensure_qgis` 自动重启） |
| 目标按钮不在 UIA | 记 `target_not_in_uia`，不参与命中统计 |
| 模型预测非 click | 不执行，记 `action_wrong` |
| 预测点在窗口外 | 不执行，记 `out_of_window`（安全硬约束） |
| 点击后 truth 不达期望 | 记 `truth_fail`（协议 bug 或模型点错） |
| bridge 调用超时/断开 | 抛异常 → 脚本终止并保留已落盘结果（json 增量写出） |
