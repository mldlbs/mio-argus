# Bigemap 无特权执行基准（BIGEMAP_EXEC_BENCH）

> 状态：预注册草案 v1（2026-11-06）。指标、注入矩阵、判据先定，再写 runner。
> 相关：`results/computer_use/CANDIDATE_RECAP.md` §8、`results/qgis/QGIS_FINDINGS.md` playbook、
> `results/computer_use/W31_RESULTS.md`、`experiments/qgis/qgis_bridge/ingest_bigemap.py`。

## 0. 一句话

在**只有真实用户拥有的通道**（UIA + Win32 窗口管理 + 真实输入 + 像素）的第三方桌面应用上，
量化「结构化信息 + 确定性控制」相对「像素定位 + 坐标点击」能把**失败率、重试次数、执行耗时**降低多少；
并把每一次降级（画布、无 pattern 控件、状态残留）单独记账，而不是平均掉。

## 1. 两条 claim 必须分开

| claim | 内容 | 本基准能否证明 |
|---|---|---|
| **S（系统）** | 执行层可靠：窗口、定位、动作、状态、恢复 | ✅ 这是本基准的唯一主张 |
| **M（模型）** | 小模型策略在窄域内够用 | ❌ 不证；模型臂只作对照 |

原因：W28–W31 已给出 M 的负结果（动作可学 1.000，坐标不可学；0/9，success@1 = success@5）。
若 B 臂跑出高成功率，不能据此回收 M 的结论。**报告必须同时写"系统可靠性 X；其中模型贡献 Y"。**

## 2. 范围

**做**：单应用（Bigemap）、单机、单个 AOI 的石景山区任务集；A/B/C 三臂对照；故障注入下的恢复。
**不做**：六类数据全量下载、跨应用泛化、模型训练、GUI 框架横评、多机/多分辨率。

## 3. 被测系统边界（无特权定义）

| 允许 | 禁止（会让结果不可外推） |
|---|---|
| UIA patterns（Invoke/Select/SetValue/WindowPattern.Close） | 进程内注入、DLL、私有 IPC、读 Bigemap 配置/数据库 |
| Win32：EnumWindows / GetWindowRect / ShowWindow / SetWindowPos / GetForegroundWindow | `PostMessage`/`SendMessage` 直投控件（绕过真实输入语义） |
| 真实输入：SendInput / pyautogui / 剪贴板粘贴 | 用 oracle 计算点击坐标（C 臂除外） |
| 截屏、模板匹配、OCR | 修改应用、预置注册表、注入代理 |

判定口径：**一条能力如果真实用户做不到，就不能进 A/B 臂。**

## 4. 任务集

| ID | 任务 | 控制类 | 主层 | 成功判据（可机械判定） |
|---|---|---|---|---|
| T1 | 窗口恢复 | window | L0 | 最小化 → 恢复后：`hwnd` 可枚举、rect 与 canonical `(1344,16,2544,916)` 差 ≤2px、`IsWindowVisible`、UIA 主窗节点 ≥200。**门控只看 restored**；`foreground_ok` 单独报告（受 OS 前台锁影响，见 F2） |
| T2 | 标准控件 | standard | L1–L2 | UIA `Invoke` 打开「导入边界」→ 3s 内出现「导入边界数据」且节点 ≥10、主窗变 disabled（模态）；再 `Close` → 2s 内消失、主窗恢复 enabled |
| T3 | 无 pattern 表格行 | table_row | L2 | 滚动 ≤14 次找到「第16级」→ 截取单元格左侧勾选框 → 确保未勾选（残留则先清）→ 点击 → **像素差分**（暗像素占比 ≤0.02 → >0.02）。UIA 点击前后**无任何变化**（103 节点逐字段相同），尺寸/理论大小是静态估算，**不能**作为选中判据 |
| T4 | 自绘画布 | canvas | L1–L2 | 视觉检测 AOI 蓝色边框 bbox → 双击内部 → 3s 内出现「下载对话框」（103±5 节点） |
| T5 | 端到端下载 | 混合 | L0–L4 | T4 → 任务名 `bench_<trial>` → 勾「第16级」→ `确定` → `BMDownload\bench_<trial>\*.tif` 出现且大小稳定（≤180s）→ `gdalinfo -json`：可读、WGS84 角点与 AOI `(116.2098,39.9048,116.2201,39.9118)` 偏差 ≤2e-5°（≈1 px）、尺寸 >0 → 记录 SHA256 |
| T6 | 行政区边界下载 | 混合 | L0–L4 | 区域路径（省/市/区县）选择 → 面板下载图标 → 行政边界页 → SHP/WGS84/省+区县 → 确定 → `BMDownload/<task>/*.shp`（≤90s）→ `ogr2ogr` 转 GeoJSON：2 个面、其中一个面的 bounds 与**面板读数**（`西,北 >> 东,南`）差 ≤2e-3° |

> **T5 的两种范围来源**：`--source aoi`（双击画布 AOI，默认）与
> `--source region --region 省/市/区县 --level N`（右侧区域面板直接取 extent，**不需要画 AOI**）。
> 后者是通用路径；AOI 只用于非行政区/自定义范围。

T5 的 AOI 与级别固定（石景山 AOI、第16级），变体另设：更换 AOI（T5v）用于检验是否只是"路径写死"。

## 5. 两层正交的记账

能力层（每一步只归一层）：

| 层 | 含义 | 判定 |
|---|---|---|
| L0 | 窗口/几何 | 窗口状态与 rect |
| L1 | 定位 | 找到的目标是否就是预期控件（名字或 bbox） |
| L2 | 动作生效 | 动作后状态是否按预期改变 |
| L3 | 状态连续 | 跨步字段/句柄/对话框是否保持 |
| L4 | 任务 | 产物真值 |

控制类（结果必须分类报告，禁止混算）：`window` / `standard` / `table_row` / `canvas`。

> 依据：`CANDIDATE_RECAP.md` §8.5 的三层能力差（可读可点可验 / 可读不可设 / 无名只能几何）。
> 混算会让 100% 的标准控件类掩盖 30% 的画布类。

## 6. 指标定义

| 指标 | 定义 | 口径 |
|---|---|---|
| **SFR** 静默失败率 | 环内判定为 pass、但事后 oracle 判 fail 的步数 / 有判定的步数 | 最关键；"动作发出≠发生"的量化 |
| **FER** 首点被吞率 | 未聚焦窗口上第一次鼠标点击后状态无变化 / 该类点击总数 | 对应 W19 根因 |
| **HR** 命中率 | L2 `pass` / L2 尝试数，**按控制类分列** | 不含 vacuous |
| **SPR** 状态保持率 | 跨步后字段值仍符合预期的字段数 / 应保持字段数 | 表单、对话框、句柄 |
| **RR(f)** 恢复率 | 注入故障 f 后恢复到预期状态的次数 / 注入次数 | 只统计注入，不统计自然失败 |
| **SR** 任务成功率 | 成功 trial / 有效 trial（reset 失败 = 无效，不计入分母） | 主终点：T5 |
| **success@1** | 首次尝试即成功 / 有效 trial | |
| 重试次数 | 每个成功 trial 的 attempt 数 − 1 | 中位数 + IQR |
| 墙钟 | 每个**成功** trial 的端到端秒数 | 中位数 + IQR |
| 人工接管 | 人改变了应用状态或脚本参数的次数 / trial | 必须为 0 才叫自主 |
| reset 耗时 | R1–R6 总耗时 | 单独报告，不摊进任务耗时 |

**时间闭合断言**：`sum(segment_ms) == wall_ms`（容差 50ms），否则该记录作废。
分段：`reset / perceive / decide / act / verify / recover`。这条直接回应"两小时花在哪"不可归因的问题。

**置信区间**：所有比例用 Wilson 95% CI；A vs B 用 Fisher 精确检验（n=30/臂）。
禁止只报点估计。

## 7. 故障注入矩阵

自然失败太稀疏，恢复率必须**注入**才能测。

| ID | 注入方式 | 期望 | 计入 |
|---|---|---|---|
| F1 | 在 perceive 与 act 之间 `ShowWindow(SW_MINIMIZE)` | 自动恢复并完成该步 | RR(F1)、墙钟 +恢复段 |
| F2 | act 前把全屏 Chrome/记事本置前 | 不点错、不丢步 | RR(F2)、FER |
| F3 | 触发模态提示（未选区域点确定 → 「请选择一个要导入的区域」） | 关提示、回到原状态、继续 | RR(F3)、SPR |
| F4 | 两步之间关闭目标对话框 | 重新打开并**重填**已填字段 | RR(F4)、SPR |
| F5 | 预置同名下载任务 | 改名重试一次成功（`文件已存在…`） | RR(F5) |
| F6 | 快照后移动/缩放窗口，再按旧 rect 动作 | 取 rect 前校验，拒绝 stale | RR(F6)、stale_rect 计数 |
| F7 | 下载中断网 10s 后恢复（或代理故障） | 可续传/可重试，且不产生半文件 | RR(F7)（行政边界分支可选） |
| F8 | 置顶不透明窗遮挡目标控件（T2） | 像素通道无输入；UIA 仍可定位 + `Invoke` | RR(F8)、失败性质（诚实失败 vs 静默失败） |

注入必须**可编程、可复现、可关闭**；同一 trial 最多注入 1 个故障，避免叠加归因不清。

## 8. reset 协议

W31 §6 的教训：上一次尝试残留的对话框/状态会让 oracle 误判（曾 2/3 假失败）。

| # | 步骤 | 失败处理 |
|---|---|---|
| R1 | 新建 trial 目录（T1–T4：`artifacts/bench_dl/<trial>`；T5：`BMDownload/bench_<trial>`），记录**文件清单**（路径/大小/mtime） | 作废 |
| R2 | 任务名唯一：`bench_<trial>_<epoch>` | 作废 |
| R3 | 枚举顶层窗口，`WindowPattern.Close` 关闭 {下载对话框, 导入边界, 选择文件, 定位, 提示}，验证剩余 = 0 | 作废 |
| R4 | 主窗恢复并摆到 canonical rect（差 ≤2px），验证前台 | 作废 |
| R5 | 主窗 `IsEnabled`，无模态；下载列表记录基线条数 | 作废 |
| R6 | 记录 pre-state（AOI 存在、级别未选、目录为空） | 作废 |
| R7 | 连续 3 次 reset 失败 → 中止整轮，**不降级继续** | 中止 |

**oracle 用增量**：只认 trial 目录里新增的文件，避免历史产物冒充本次成果。

## 9. 三臂与随机化

| 臂 | 定位 | 动作 | 验证 |
|---|---|---|---|
| **A** 现状 | 像素：模板/OCR/固定偏移 | 坐标点击（真实输入） | 事后截图 diff |
| **B** 提案 | UIA 枚举 + rect 双向校验（A2） | pattern 优先（Invoke/SetValue/Select），鼠标仅 `canvas`/`table_row` | 差分后验（pre/post 必须变化） |
| **C** 恢复对照 | 像素（OCR） | 坐标点击 | 故障后**重新截图 + 重新 OCR**；用于分离 UIA 与恢复循环 |
| **S** 共用 | — | 窗口层（T1）两臂共用，单独归 S，避免重复计数 | — |

- 每臂每任务 **30 次**（T1–T4 可 20；先跑 5 次 pilot 验证 reset 与时间闭合）。
- 顺序用交替块 `A B B A`，抵消网络/热漂移；记录 order。
- A 臂**不得**读 UIA 名字（否则不是 A）；B 臂**不得**对标准控件用坐标直点（否则不是 B）。
- T4 两臂都只能用鼠标——这是设计上的"共同下限"，不是缺陷。

## 10. 日志 schema（JSONL，一行一步）

```json
{"kind":"step","trial":"T5-B-0007","arm":"B","task":"T5","attempt":1,"step":3,
 "layer":"L2","control_class":"table_row","t0":1762300000.12,"t1":1762300000.51,
 "window":{"hwnd":132456,"rect":[1344,16,2544,916],"minimized":false,"foreground":true},
 "control":{"name":"第16级","type":"DataItem","rect":[1712,540,1790,562],"patterns":[]},
 "action":{"kind":"click","source":"uia_rect","coords":[1720,551]},
 "expect":{"pred":"row_size_nonzero","pre":"0x0"},
 "observed":{"size":"480x425","theory":"1.2 M"},
 "verdict":"pass","silent":false,"failure_class":"none",
 "recovery":{"injected":"F1","action":"restore+refetch","ok":true,"ms":4200},
 "segment_ms":{"perceive":120,"decide":0,"act":180,"verify":80,"recover":0},
 "wall_ms":380}
```

必备字段：`kind, trial, arm, task, attempt, step, layer, control_class, t0, t1,
window, control, action, expect, observed, verdict, silent, failure_class, segment_ms, wall_ms`。
trial 级记录：`kind:"trial"` + `success, attempts, takeovers, reset_ok, reset_ms, artifact{path,sha256,bytes,gdal}`。

`failure_class` 枚举：`none / focus_eaten / stale_rect / wrong_control / no_pattern /
synthetic_rejected / state_lost / env / timeout / vacuous`。

## 11. 反空过与反作弊

1. **差分前置**：每个 `expect` 必须绑定 pre-state 并**要求变化**。pre-state 已满足 → `vacuous`，不计入 HR（W31 `zoom_native` 3/3 假通过）。
2. **oracle 不入口**：A/B 臂的 oracle 只在 trial 结束后判定，绝不参与控制。
3. **静默失败专职审计**：trial 结束后用 oracle 回放每步 `expect`，与环内 `verdict` 比对，差异记 `silent`。
4. **失败样本不得剔除**：只允许因 reset 失败标 invalid，且必须报告 invalid 率。

## 12. 环境前置

| 项 | 要求 |
|---|---|
| 会话 | 真实交互控制台会话（非 RDP 断开、非 Session 0）；`SendInput` 语义才成立 |
| 显示 | 2560×1440，DPI 100%，任务栏不自动隐藏，锁屏/屏保关闭 |
| 应用 | Bigemap 版本/build 记录在案；窗口 canonical `(1344,16,2544,916)` |
| 网络 | 基线 VPN **关**（F7 单独注入）；Chrome 关闭（曾致 UIA 判 offscreen） |
| 干扰 | 通知/专注助手关闭；电源计划高性能 |
| 环境记录 | 每轮写 `env.json`：分辨率、DPI、版本、VPN 状态、时间戳 |

## 13. 预注册判据（先写死，跑完不改）

**主终点**：T5 任务成功率，`SR_B − SR_A ≥ 20pp` 且差异 95% CI 下界 > 0（Fisher p<0.05）。

**支持性终点（全部满足才叫"显著改善"）**：

| 指标 | 判据 |
|---|---|
| SFR | `SFR_B ≤ 0.5 × SFR_A` |
| 重试 | 中位重试次数 B < A |
| 墙钟 | 每次成功的墙钟 B ≤ 1.2 × A |
| 人工接管 | A、B 均为 0（否则该臂不算自主，单独报告） |
| reset | invalid 率 ≤10% |

**结论分级**：

- 主终点 + 支持性全过 → 「系统性改善，可复现」
- 仅主终点过 → 「成功率提升，代价未降」
- 主终点不过 → **负结果归档**，不调参重跑；追加实验须标注为事后探索

## 14. 产物与目录

| 产物 | 位置 |
|---|---|
| 本协议 | `docs/BIGEMAP_EXEC_BENCH.md` |
| 汇总/校验器 | `experiments/computer_use/exec_bench.py` |
| runner（T1–T5 已接，A 臂待接） | `experiments/computer_use/bigemap_runner.py` |
| 原始日志 | `results/computer_use/bigemap_bench_<arm>_<date>.jsonl` |
| 汇总 | `results/computer_use/BIGEMAP_BENCH_RESULTS.md` + `bigemap_bench_summary.json` |
| 下载产物 | `artifacts/bench_dl/<trial>/`（gitignored） |

## 15. 实施清单

```python
reset(task) -> ResetReport            # R1–R6，返回 ok/pre_state/ms
snapshot() -> {window, controls}      # UIA 枚举 + Win32 窗口
act(action) -> ActResult              # invoke/click/type/key/scroll
expect(pred, pre) -> {verdict, observed, vacuous}
inject(fault_id) -> None              # F1–F7
run_trial(arm, task, faults) -> TrialRecord
```

CLI：`python experiments/computer_use/exec_bench.py --arm B --task T5 --trials 30 [--inject F1] [--dry]`

> 当前状态：runner 已接 T1–T5（A/B/C/S 臂）、T3 故障注入（F1/F4/F6）、T2 遮挡（F8）、T5 故障（F1/F4/F6）。
> 结果：无故障 A/B 均 100%（天花板），B 定位快 9–15×；T3 故障 **A 0/90、B/C 90/90**；
> 遮挡 **A 0/30、B 30/30**；T5 端到端 **A 0/30（仅 F4）、B 90/90（4.4 s）、C 89/90（8.0 s）**；
> **T6 行政区边界：B 基线 10/10（11.4 s）、F1 10/10、F4 9/10、F6 10/10，合计 39/40（97.5%）**；
> 区域三级选择占 63%；**跨区域 3 个区县 15/15（14.4 s）**。**F7（网络/代理）本轮不做**（无管理员/以太网，需手工断网），列为已知缺口。
> **可靠 = 恢复循环 + 独立 oracle；UIA = 速度（2.8×/8.7×）+ 遮挡可达性；像素 = 画布/无 pattern 回退。**

**顺序**：pilot 5 次/臂（只验 reset + 时间闭合 + oracle 非空过）→ 修口径 → 正式 30 次/臂 → 汇总。

## 16. 已知风险

| 风险 | 处理 |
|---|---|
| 画布类上限低，可能主导失败 | 分类报告；不因 T4 差就否掉整体 |
| T5 依赖外部服务/VPN | F7 单独归因；失败记 `env` 而非能力 |
| 状态残留污染 | reset 失败即作废；连续 3 次中止 |
| 单机单版本，外推有限 | 结论限定在"同版本同分辨率"；不写通用结论 |
| B 的收益来自路径写死 | T5v 换 AOI/级别复验；只看 T5 主结果不作为泛化证据 |
| 模型贡献被系统收益掩盖 | 报告强制两句话：系统 X / 模型 Y（见 §1） |

## 17. 与既有结论的接口

| 既有 | 本基准怎么用 |
|---|---|
| W19：首点被吞 → 改 `Invoke` 4/4、9.1s vs 15.45s（playbook B1/C6） | B 臂的动作策略；FER 单独计量 |
| W4：过期坐标点中隔壁（静默错误） | F6 注入；`stale_rect` 计数 |
| CANDIDATE_RECAP §8.5：可读/可点/可验三层 | 控制类分层与 HR 分列 |
| W31 §2：`zoom_native` 假通过 | 差分前置 + `vacuous` 判定 |
| W31 §6：残留状态致 oracle 假失败 | reset 协议 R1–R7 |
| `ingest_bigemap.py` | T5 外部 oracle（GDAL 角点/CRS/要素） |
