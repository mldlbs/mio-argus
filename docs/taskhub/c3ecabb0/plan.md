> 软件项目文档链 · 7 件套  
> 上游：[需求规格](docs/taskhub/c3ecabb0/requirement.md) | 下游：—  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）。

# 执行计划（Plan）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.2 |
| 最后更新 | 2026-10-03 18:57 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 模型接环线 W28–W31 的执行计划与任务映射。不含模型研究线（P 系列）与 QGIS 线（W1–W27）本身的排期 |

## 1. 现状（W28 已完成冒烟）
- W28（taskhub `c3ecabb0`，run `43731559`）：闭环执行器 + 两段测距协议已实现并跑完 3 试。
- 结果：干跑动作准确率 0.0（18/18 预测 scroll）、命中率 0.0、中位偏差 475.6 px；安全闭环 0/9（全部 action_wrong，0 次越界点击）。
- 自查：同代码在模型自家数据 20/20 ⇒ 0% 是真实域差距。
- 遗留（已修复）：oracle 自检首测 FAIL——期望值协议 bug（Zoom Full=全图层并集 ≠ 单层 extent）→ 特权校准修复后复测 **PASS**（oracle_ok=true）。
- 产出物：`experiments/computer_use/loop_w28.py`、`results/computer_use/w28_results.json`、本文件所在文档链。

## 2. 阶段计划
### W28b 跨软件迁移实验（A' 路线门控，先于 W29）
- 背景（路线升级 A→A'）：不训练"QGIS 专用模型"，训练**软件无关的桌面操作能力**；QGIS 只是训练环境之一。决定性问题是**迁移**：在软件 A 上微调的能力，能否零样本迁移到从未训练的软件 B？
- 假设：通用操作能力存在 ⇔ 跨软件迁移显著 > 0。若迁移 ≈ 0，则模型在记忆软件特异性（像素/控件布局），A' 不成立，需先改表征再谈数据规模。
- 实验矩阵（三个臂）：
  1. **E1 学习曲线（域内对照）**：QGIS 任务 N∈{10,50,100} 微调 → 留出 QGIS 任务评测。回答"教师数据在域内多快起效"。
  2. **E2 迁移 QGIS→Notepad**：仅 QGIS 数据微调 → Notepad 零样本评测。
  3. **E3 迁移 Notepad→QGIS（最便宜、决定性）**：仅 Notepad 数据微调（采集秒级/任务）→ QGIS 零样本评测（离线，W28 干跑协议复用）。
- 协议：
  1. 录制器 `experiments/computer_use/inloop_rec.py`（**应用无关**）：双层 patch（pyautogui + pywinauto BaseWrapper/UIA 基类/EditWrapper/ButtonWrapper）；观察 = 全屏截图 + UIA affordances（structure 通道，QGIS 走 bridge）；验证 = UIA 状态（Notepad 无特权 bridge，真值即 UIA 文本）。
  2. Notepad 教师链 `collect_notepad.py`：写文本 → 全选 → 另存（剪贴板路径）→ UIA 验证内容；扰动 drift（文本/窗口位置/尺寸随机）/ occlusion / focus_error。
  3. QGIS 教师链 `collect_inloop.py`：复用 W25 链（分类设色 → 布局 → 地图/标签/图例 → 导出 PNG）。
  4. 微调 `train_inloop.py`：多应用 manifest，从 v3 checkpoint 迁移学习（冻结 ViT/GPT-2），`norm_coord` 屏无关归一。
  5. 离线评测 `eval_transfer.py`：动作准确率 / 坐标命中率 / scroll 准确率，按应用分组（不依赖软件运行）。
- 判定规则：
  - 迁移高（E3 QGIS 零样本动作准确率 ≥ 30%）→ 通用能力涌现 → 投 W29 多应用混合数据。
  - 迁移 ≈ 0（< 10%）→ 先改表征（affordance 中心指令、控件相对坐标、UI 树文本进指令），复测后再定数据规模。
  - 域内曲线平坦（E1 100 任务 ≤ 15%）→ 数据质量/模型容量问题，先修数据再扩容。
- 验收：E1/E2/E3 三组数据点 + 迁移结论落盘（`results/computer_use/transfer.json`）。
- 依赖：W28（协议已修）。
- 资源：RTX 3060 12.9GB；Notepad 采集秒级/任务，QGIS 采集分钟级/任务；**两采集器共用真实键鼠，必须串行**。

### W29 示范数据采集（脚本当老师）—— 门控于 W28b 结论
- 步骤：
  1. 写录制器 `experiments/computer_use/collect_inloop.py`：包装 W23–W27 链的关键动作点（渲染器设置、布局加项、导出、场景重置），在每个脚本动作前后截屏并落盘。
  2. 指令生成：按链步骤语义生成英文模板短句（如 `click the Zoom Full button`、`select graduated renderer`）。
  3. 数据格式：符合 data-model §1.2 DemoSample v2（六要素）；输出 `artifacts/computer_use_inloop/<task_id>/demo.json`（gitignored）。
  4. 划分：按链（场景）划分 train/val，避免同屏泄漏。
  5. **多应用混合**（若 W28b 迁移成立）：QGIS + Notepad（+ 第三应用）混合采集，软件本身不进指令（软件无关模板）。
- 验收：≥2k 步；字段/枚举合法；划分落盘（对应 FR-4 / TC-4）。
- 依赖：W28b 迁移结论（决定是否值得全量采集）。
- 风险：录制密度不足 → 增加动作点（对话框交互步骤优先）。

### W30 微调 + 重测 —— 门控于 W29
- 步骤：
  1. 复用 `train_computer_use_v3_real.py` 数据加载与训练循环，换 W29 数据（多应用 manifest）。
  2. 微调（冻结 ViT/GPT-2 主干；`norm_coord` 加载器）；checkpoint 写 `artifacts/checkpoints/`。
  3. 重跑 W28 同题（干跑 6 指令 + 安全闭环 3 任务 ×3 试），修复后的 oracle 自检同时复验。
  4. 回归检查：跑既有 504 条评测（`test_in_vm_v3.py`），指标不得回退 ±0.02。
- 验收：增益表（W28→W30 的 action_acc / hit_rate / live_success）；原评测无回归（对应 FR-5 / TC-5）。
- 依赖：W29 数据 + W28b 迁移结论。
- kill 条件：若微调后微任务步成功率仍 = 0 → 停止扩数据，转为分析（域/指令/坐标三类归因）并在任务中记录负结果。

### W31 真任务闭环（验证器在环）
- 步骤：
  1. 任务集：W23 场景子集（选择导出 / 缓冲 / 出图）改造成"模型驱动 + 真值验证"的单/多步任务。
  2. 验证器：复用 W5/W10 机械（终局状态判定：文件产物 / extent / renderer），失败重试按 W11 成本账。
  3. 对照：脚本基线（成功率 100% 上界）vs 模型驱动（成功率 + 正确产物/分钟）。
- 验收：任务成功率、成本、失败归因落盘；≥3 试（对应 FR-6 / TC-6）。
- 依赖：W30 checkpoint。
- kill 条件：任务成功率 = 0 且重试不改善 → 记录负结果，回退到"单步可用性报告"。

## 3. 依赖与并行
- 关键路径：W28 修复（协议，已完成）→ **W28b 迁移实验（门控）** → W29 采集 → W30 微调重测 → W31 真任务闭环。
- 任务映射（taskhub）：
  - `c3ecabb0`：W28（completed；本文档链为其设计/计划产出）。
  - `1f5a83e4` / `eebcd50a` / `da00b3ba`：W29 / W30 / W31（已建，depends_on 链 + W28b 门控）。
  - 待建：W28b 跨软件迁移实验（本文件 §2 W28b）。
- 资源：单机单 GPU（RTX 3060 12.9GB）；Notepad 采集与 QGIS 采集共用真实键鼠 → 串行；采集与训练不同时进行。

## 4. 检查点与停止条件（汇总）
| 检查点 | 判定 | 动作 |
|--------|------|------|
| W28 oracle 修复后重测 | PASS | 继续 W28b（已完成） |
| W28 oracle 修复后重测 | FAIL | 先修机械（不训练），再复测 |
| W28b E3 迁移 Notepad→QGIS | ≥ 30% 动作准确率 | 通用能力成立 → 投 W29 多应用混合 |
| W28b E3 迁移 Notepad→QGIS | < 10% | 改表征（affordance 中心/控件相对坐标），复测后再定 |
| W28b E1 域内曲线（100 任务） | ≤ 15% | 修数据质量/容量，不扩容 |
| W30 微任务成功率 | > 0 | 进入 W31 |
| W30 微任务成功率 | = 0 | 触发 kill：归因分析 + 负结果归档 |
| W31 任务成功率 | > 0 | 出"可用性 + 成本"报告 |
| W31 任务成功率 | = 0 | 负结果归档，回到单步可用性结论 |
