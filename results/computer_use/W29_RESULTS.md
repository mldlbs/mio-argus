# W29 结果：接环示范数据采集（FR-4）

> 复现：
> ```
> # 采集（每条链约 5–12 分钟，占用真实键鼠）
> python experiments/computer_use/record_chain.py --rounds 1
> # 只重算划分
> python experiments/computer_use/record_chain.py --write-splits
> ```
> 脚本：`experiments/computer_use/record_chain.py`（底层 `inloop_rec.Recorder`）
> 数据：`artifacts/computer_use_inloop/qgis/`（gitignored）、`qgis_splits.json`
> 需求：[requirement.md](../../docs/taskhub/c3ecabb0/requirement.md) FR-4

---

## 1. 结论

- **2020 条示范**（FR-4 要求 ≥2k，达标）
- **125 个唯一行为**（按 action+instruction+rect+坐标去重，重复系数 16.2×）
- 链级划分：train 14 任务 / 1394 条，val 4 任务 / 626 条，**val 的两条链整体未见**（`w26_v2`、`w27`）
- 采集方式改过一次，见 §2——这是本轮唯一值得记的工程决策

## 2. 采集方式：不再重写 UI

| 方案 | 结果 |
|---|---|
| `collect_inloop.py`（自实现菜单/对话框导航） | 首版只产出 32 条退化样本（1 click + 31 条重复 press）；修掉 4 个真实 bug 后最好一轮 207 步，仍 truth 失败，且**失败点每轮漂移**（style_provinces → style_cities → style_bg） |
| **`record_chain.py`（在已验证链上挂录制器）** | **13/13 次执行全 PASS**，单轮 98–203 步 |

做法：`run_w25` / `run_w26(THEMES)` / `run_w27` 本来就有结果，直接在其上挂
`inloop_rec.Recorder`（pyautogui + pywinauto 双层 patch），链本身一行不改。
`collect_inloop.py` 里修到的 4 个 bug（双击开属性、`fill_rect` 下标、ramp 箭头重试、
颜色未提交）仍然保留，供以后走"自实现导航"时复用，但不再是采集主路径。

## 3. 数据

| 链 | 轮次 | 每轮步数 |
|---|---|---|
| w25（单主题地图） | 6（含早期 5 次重复验证） | 98 |
| w26_v1 / v2 / v3 / v4 | 各 2 | 98 / 110 / 110 / 97 |
| w27（深色重绘） | 2 | 203 |

- 动作分布：press 868 / click 728 / type 302 / hotkey 122（**无 scroll / move**）
- 覆盖：截图 2020/2020 存在、affordance 2020/2020 非空、`target_rect` 910/2020（click 类）
- 六要素（observation / intent / action / result / context / failure）齐全

## 4. FR-4 对照

| 验收项 | 结果 |
|---|---|
| ≥2k 步 | ✅ 2020 |
| 格式符合 data-model | ⚠️ 见 §5（`intent.phase` 全为 `idle`） |
| train/val 划分落盘 | ✅ 链级划分，`qgis_splits.json` |

## 5. 已知缺陷（W30 前要决定是否修）

1. **`intent.phase` 全为 `idle`**：`Recorder` 的相位由采集脚本注入，`run_w25/26/27` 不提供，
   所以 2020 条全是 `idle`。对训练影响有限（loader 主要用 instruction/action/坐标），
   但不符合 data-model 的 phase 枚举，属于**已知偏差**，已在此记录，不假装合规。
2. **无 scroll / move 样本**：这三条链不用滚动/拖拽，动作空间只覆盖 4 类；
   W30 的 scroll 头拿不到训练信号（评测时仍会报 scroll 指标，可作负对照）。
3. **唯一行为 125 / 样本 2020**：6 条链共享约 90% 的 UI 步骤（菜单、对话框、导出），
   换链只多出主题相关的 21 步。继续加轮次只增加重复度。

## 6. 给 W30 的输入

- 训练集：`qgis_splits.json` 的 `train`（14 任务 / 1394 条）；留出 `val`（`w26_v2`+`w27`，未见过）
- **建议做一次消融**：全量 1394 vs 按唯一签名去重后的子集——直接回答"重复样本有没有用"，
  这比再加 1k 数据便宜，也能解释 §5.3 的重复系数
- **评测口径必须带坐标命中率**：W28b 已经证明动作准确率会被标签分布买通；
  域内坐标误差 188px vs 按钮 32px 的教训直接适用于本数据集
- kill 条件沿用 plan §4：微调后微任务步成功率 = 0 → 归因分析，不扩数据
