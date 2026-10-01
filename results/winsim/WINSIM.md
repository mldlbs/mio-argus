# WinSim：Windows 桌面 Agent 沙箱（设计）

> **状态**：v1.0 —— **W0–W8 已全部完成**；汇总见 [results/winsim/WINSIM_RESULTS.md](WINSIM_RESULTS.md)
> **日期**：2026-09-27
> **前置**：[results/web/WEB_CHANNEL.md](..\web\WEB_CHANNEL.md) / [results/web/WEB_GAP.md](..\web\WEB_GAP.md) / [docs/PLASTICITY.md](..\..\docs\PLASTICITY.md) / [docs/MIO_ENV.md](..\..\docs\MIO_ENV.md)
> **约束**：这不是"仿 Windows UI"，而是**可控、可重放、有外部真值的 Agent 训练/评测沙箱**。
> **验收目标**：支持 `train on N 应用/布局 → test on unseen 应用/布局` 的站点级 held-out，并能量出像素与结构通道的差距。

---

## 0. 一句话

> 把 Windows 桌面做成一个 **Web 前端模拟器**，但它的价值不在"像 Windows"，
> 而在它同时给出了四样 CUA 训练缺的东西：
> **真渲染的 GUI + 外部真值 + 确定性状态 + 可注入故障**。

---

## 1. 为什么值得做（相对现有环境的增量）

| 现有 | 缺什么 | WinSim 补什么 |
|---|---|---|
| `mio_env`（网格合成） | 太简单、自证 | 真桌面复杂度、多窗口、多应用 |
| 真实站点（Web） | 不可控、不可重放、难故障注入、慢 | 全可控、确定性、毫秒级、可注入 |
| 真实 Windows VM | 慢（1–3 step/s）、不可复现、难并行 | 无头、可并行、可重放、快 |
| 静态截图数据集 | 无交互、无状态 | 完整开放/关闭/拖动/多窗口状态 |

**它填的是"高保真 + 全可控"这个空档**，正好是 sim-to-real 与 RL 都需要的中间层。

---

## 2. 硬约束（来自本项目已有结论，不可违背）

| 约束 | 来源 | 在 WinSim 里怎么落地 |
|---|---|---|
| **外部真值** | `REPORT`/`ROADMAP C1` | 成功由**应用状态机**判定（文件真的被移动、设置真的生效），不由"点击了就算" |
| **真渲染** | WEB_GAP | 用真实浏览器引擎渲染（DOM/CSS 真布局真字体），不用 canvas 手绘 |
| **确定性 + 可重放** | P1/P3 纪律 | 全局 seed；记录 `(t, action)` 事件流即可回放；时间由虚拟时钟驱动 |
| **成本模型** | P9 教训 | 每次观察/切换窗口/等待都计费；否则元认知退化 |
| **站点级 held-out** | WEB_CHANNEL | 应用、主题、布局、语言可独立采样；评测留出未见组合 |
| **双通道可切换** | H2 / WEB_CHANNEL | 同一次操作可只暴露 `pixels`，或额外暴露 UIA/DOM 树（教师/评委用） |
| **故障注入** | 参考你给的清单 | UI 漂移 / 弹窗 / 右键菜单 / Loading / Disabled 都在状态机里是**一等事件** |

---

## 3. 结构

```
WinSim
├── Shell（桌面外壳）
│   ├── Desktop（壁纸/图标/右键菜单）
│   ├── WindowManager（开关/最小化/最大化/拖动/resize/z/focus）
│   ├── Taskbar（Start/搜索/应用/托盘/网络/音量/时钟）
│   └── StartMenu（用户/固定/推荐/全部应用）
├── Apps（应用注册表，可插拔）
│   ├── FileExplorer / Settings / Notepad / Terminal / TaskManager / Browser
├── System
│   ├── Clipboard / Notifications / ContextMenu / Dialog / Shortcuts
├── FaultInjector（UI漂移/弹窗/加载/禁用/遮挡/动画/焦点抢占）
└── AgentAPI（双通道观测 + 动作 + 真值查询 + 回放）
```

**关键架构决定：窗口状态不归应用管，统一在 WindowManager 里。**
（你给的状态模型正确，直接采用。）

---

## 4. 技术选型（务实版）

你建议 React+TS+Zustand+Tailwind。我基本同意，但**加一条我们的纪律**：

| 层 | 选择 | 理由 |
|---|---|---|
| 框架 | React + TypeScript | 组件化、可插拔应用 |
| 状态 | Zustand | 窗口/系统状态集中，便于快照与回放 |
| 样式 | Tailwind | 快速做主题/布局变体（held-out 需要） |
| 渲染 | **真实 Chromium**（Playwright 驱动），不做 canvas 自绘 | 真渲染 = 真布局/真字体（WEB_GAP 已证可用） |
| 时间 | **虚拟时钟**（可 step/加速），禁用真实 `setTimeout` 驱动逻辑 | 确定性重放的前提 |
| 随机 | 全局 seeded PRNG，禁止 `Math.random` | 同 seed 同世界 |
| 真值 | **状态机 + 事件日志**，不是 DOM 快照 | 外部真值 |

> 纯 web 前端（HTML/CSS/JS）**够用且更好**：不需要 Electron/Tauri，
> 因为我们要的是"GUI 的正确观测"，不是"真的 Windows 进程"。

---

## 5. 状态模型

```ts
// 窗口（你给的模型，采纳）
type WindowState = {
  id: string; appId: string; title: string
  x: number; y: number; width: number; height: number
  minimized: boolean; maximized: boolean; focused: boolean; zIndex: number
  modalFor?: string | null          // 弹窗归属
}

// 系统（新增：把"外部真值"和"故障"变成一等状态）
type SystemState = {
  clock: number                     // 虚拟时钟 ms
  clipboard: string
  openDialogs: DialogState[]
  notifications: Notif[]
  faults: FaultState[]              // 激活中的故障
  files: FileTree                   // 外部真值：文件系统
  settings: Record<string, unknown> // 外部真值：设置项
}

type DialogState = { id: string; kind: 'confirm'|'file'|'error'; owner: string }
type FaultState  = { kind: FaultKind; target: string; until: number; seed: number }
```

命令（你给的 API 正确，扩展两条）：

```ts
openApp(appId) / closeWindow(id) / minimizeWindow(id) / maximizeWindow(id)
focusWindow(id) / moveWindow(id,x,y) / resizeWindow(id,w,h)
// 新增（外部真值与故障）
queryTruth(): {files, settings, windows}   // 评委/教师用，策略看不到
injectFault(fault) / clearFault(id)
```

---

## 6. Agent 接口（双通道 + 成本）

```ts
// 观测：默认只给像素；结构是特权通道（可选，用于教师/评委）
observe({channel: 'pixels' | 'uia' | 'both'}): {
  pixels: Uint8Array,          // 截图
  uia?: UIANode[],             // 可交互树（name/role/bbox/state）
  activeWindow: string,
  clock: number
}

// 动作：全部走真实事件（坐标），不走 element.click 特权直调
act(a:
  | {type:'click', x, y, button?}
  | {type:'dblclick', x, y} | {type:'rightclick', x, y}
  | {type:'type', text} | {type:'hotkey', keys:string[]}
  | {type:'scroll', x, y, dy} | {type:'drag', x1,y1,x2,y2}
): void

// 成本：观察/等待/切换都计费（元认知需要）
```

**判据**：`observe(pixels)` 与 `observe(uia)` 在**同一任务**上给出不同成功率——
差距即"结构补贴"，与 WEB_CHANNEL 的 5/6 结论对齐。

---

## 7. 故障注入（一等公民）

| 故障 | 作用 | 检验的架构能力 |
|---|---|---|
| UI 漂移（元素移动/重排） | 打断过期的接地 | H1 重观察 |
| 弹窗 / 模态 Dialog | 遮挡 + 焦点抢占 | H1 + H3 恢复 |
| 右键菜单 | 二级路径 | 多步规划 |
| Loading / 延迟 | 时间不确定 | 等待决策（成本） |
| Disabled / 权限 | 非法动作 | 验证与拒绝 |
| 动画中的元素 | 稳定性 | 见 WEB_GAP §5：坐标点击反而不脆 |

**要求**：每个故障都是 `SystemState.faults` 的显式条目，可 seed、可撤销、可回放。

---

## 8. 训练与评测协议

```
数据：程序化生成任务（改文件 / 改设置 / 打开应用 / 移动窗口 …）
真值：由 files/settings/windows 状态判定（不是"点了就算"）
通道：pixel 训练（uia 只做教师生成标签）；uia 通道并列报告
held-out：
  - 应用级：train on {Explorer, Notepad} → test on {Settings, TaskManager}
  - 主题/布局/语言级：同应用不同皮肤与 RTL
  - 故障级：train 无故障 → test 注入故障
指标：成功率、恢复率、端到端成本(含观察)、pixel-vs-uia 差距
```

---

## 9. 分阶段落地（每阶段都可独立验证）

| 阶段 | 内容 | 判据 |
|---|---|---|
| **W0** ✅ | Shell：Desktop + WindowManager + Taskbar（真 Chromium 渲染，确定性时钟） | 同一 seed 两次运行像素**逐位相同** |
| **W1** ✅ | FileExplorer（真实文件树）+ Settings（真实设置项）；成功由 `truth()` 判定 | oracle 全成功、noop 全失败、确定性成立 |
| **W2** ✅ | AgentAPI 双通道 + 成本模型 | `observe(pixels)` vs `uia` 出现可测差距 |
| **W3** ✅ | 故障注入器：UI 漂移 / 模态弹窗 / Loading | naive 失败、robust 成功、可清除、可复现 |
| **W4** ✅ | held-out 协议（故障级） | in-domain 满分、OOD 掉的量化 |
| **W5** ✅ | 学习型 next-action 选择器 + 应用/主题级 held-out | 结构对主题不变（drop 0）；未见应用掉 0.5 |
| **W6** ✅ | 像素通道（候选框裁剪 CNN）vs 结构，主题/应用 OOD 对照 | 结构主题掉 0；像素主题掉 0.41、应用掉 0.84 |
| **W7** ✅ | 端到端：自建检测器（去掉"完美候选框"） | 检测 recall in 0.31 vs 结构 1.00；OOD 掉到 0.08/0.00 |
| **W8** ✅ | 端到端联合（检测+选择）→ next-action 命中 | in 0.26 vs 结构 1.00；OOD 0.08/0.00 |
| **W9** ✅ | 更强检测头（stride 2）+ 修尺寸监督 | in e2e 0.290 / recall 0.412；提升有限 → 方法极限 |
| **W10** ✅ | 检测+选择联合（选择器训在检测框上） | in e2e 0.426 → **0.790**（+0.365）；主因是分布不匹配 |
| **W11** ✅ | 索引化动作空间（op+target+兼容掩码；新增 TYPE_TEXT/rename） | masked 0.934 vs unmasked 0.868（+0.066） |
| **W12** ✅ | 混合臂：结构可用→索引化动作空间，不可用→回退像素（新增 canvas 应用） | hybrid 0.804 > struct 0.750 > pixel 0.505 |

**W0 就能证伪**：如果同 seed 两次渲染做不到逐位相同，确定性重放的前提不成立，项目要重做。

### 已实现（证据）

- **W0**（`winsim/index.html` + `winsim_w0.py`）：同 seed 两次独立加载截图
  **逐字节相同**（sha `8240b8d4…`），`observe()` 一致，不同 seed 画面不同（对照）。
- **W1**（`winsim_w1.py`）：`move_notes` / `wifi_off` 两个任务——
  oracle（用 `elements()` 教师通道 + 坐标点击）**全成功**，noop **全失败**，
  且成功由 `truth()`（`fs`/`settings`）判定，不是"点了就算"；确定性仍成立。
- **W2**（`winsim_w2.py`）：`observe({channel})` 双通道 + 计费
  （观察 `pixels=1.0 / uia=0.2`、动作 `0.1`、等待 `0.1/s`）：

  | 任务 | oracle/uia | oracle/pixels | blind（不观察） |
  |---|---|---|---|
  | move_notes | ✅ cost 1.25 | ✅ cost **4.45** | ❌ |
  | wifi_off | ✅ cost 0.35 | ✅ cost **1.15** | ❌ |

  成本差 **3.2 / 0.8**；盲操作失败（窗口位置随 seed 变 → 必须观察）；成本可确定性复现。
- **W3**（`winsim_w3.py`）：故障注入（drift/dialog/loading，全部进 `S.faults`，可 seed/清除）：

  | 场景 | naive（一次观测） | robust（每步重观察+关弹窗+等待） | robust cost |
  |---|---|---|---|
  | none | ✅ | ✅ | 1.60 |
  | drift | ❌ | ✅ | 1.60 |
  | dialog | ❌ | ✅ | 1.90 |
  | loading | ❌ | ✅ | 1.60 |
  | all | ❌ | ✅ | 1.90 |

  确定性成立、`clearFault` 生效。（修过一个真 bug：`render()` 只清 `.win`，
  弹窗遮罩残留 → OK 反复出现。）
- **W4**（`winsim_w4.py`）：故障级 held-out。in-domain 故障 `{none, drift}`，
  OOD 故障 `{dialog, loading}`：

  | 策略 | in-domain | OOD | drop |
  |---|---|---|---|
  | naive | 0.50 | 0.00 | +0.50 |
  | **inductive**（只学会应对 drift） | **1.00** | **0.00** | **+1.00** |
  | robust（重观察+关弹窗+等待） | 1.00 | 1.00 | +0.00 |

  → 在 in-domain 上做到满分（1.00），换个没见过的故障就**掉到 0**；
  只有含"重观察/恢复/等待"的通用策略迁移得过去。
- **W5**（`winsim_w5.py`）：**学习型** next-action 选择器（结构通道特征 + 任务/状态特征，
  softmax over 候选）在 oracle 轨迹上模仿；新增主题系统（a/b/c in-domain，d OOD）
  与终端应用（`run` 任务作 OOD-app）：

  | split | acc |
  |---|---|
  | in-domain | **1.000** |
  | OOD-theme | **1.000**（drop +0.000） |
  | OOD-app（未见应用/任务） | **0.500**（drop +0.500） |
  | chance | 0.183 |

  → 结构特征对**主题完全不变**（与 site_holdout 的像素通道形成对照：那边主题 OOD 掉 0.63）；
  但**未见应用/任务**仍掉一半——覆盖问题对学习型策略同样成立。
- **W6**（`winsim_w6.py`）：**像素通道**（对每个候选元素从其 bbox 裁剪 → 小 CNN → 特征）
  与结构通道在**同一批样本**上对比：

  | split | struct | pixel |
  |---|---|---|
  | in-domain | 1.000 | 1.000 |
  | OOD-theme (d) | 1.000 | 0.587 |
  | OOD-app (run) | 0.500 | 0.159 |

  theme drop：struct **+0.000** / pixel **+0.413**；
  app drop：struct **+0.500** / pixel **+0.841**。

  → 与 `site_holdout` 完全一致：**结构对主题不变，像素扛不住主题/应用外移**；
  且这次是在**可交互桌面沙箱**里测出来的。
- **W7**（`winsim_w7.py`）：**端到端**——训练一个小检测器（CenterNet-lite）
  从像素里自己产出元素框，量它对**目标元素**的召回：

  | split | 检测器 recall |
  |---|---|
  | in-domain | **0.307** |
  | OOD-theme | 0.075 |
  | OOD-app | 0.000 |

  对照：结构通道（uia 白送框）= **1.000**；给定框的像素选择（W6）in-domain = **1.000**。

  → **"把结构从像素里恢复出来"这一步才是瓶颈**：白送框时像素选择满分，
  自己检测就只剩 0.31，且 OOD 直接崩（0.08 / 0.00）。与 P7/P8、`site_holdout` 一致。
  （限定：检测器是小型 CenterNet-lite；更大/更高分辨率头可能提高 recall，未测。）
- **W8**（`winsim_w8.py`）：把整条像素链路接起来——截图 → 检测器出候选框 →
  选择器选一个 → 点其中心；度量 next-action **命中率**（点中框中心落在 oracle 目标内）：

  | split | end-to-end hit | 对照 |
  |---|---|---|
  | in-domain | **0.260** | 结构 1.000 / 给定框像素选择 1.000 / 纯检测 recall 0.307 |
  | OOD-theme | 0.075 | |
  | OOD-app | 0.000 | |

  → **端到端像素 agent 在 WinSim 上 in-domain 只有 0.26**，OOD 直接崩（0.08 / 0.00）。
  即"白送结构"的通道 1.00 vs 自己从像素做起 0.26——**这就是 sim-to-real 的真实落差**。
- 环境：纯 HTML/JS，无框架；seeded RNG + 虚拟时钟；无 `Math.random`/`Date`/定时器/动画。

**尚未做**：更强检测头（能否把 0.26 抬起来）；检测/选择**联合训练**（当前各自独立训练）；
更大任务集 / 更长 horizon；把 WinSim 作为评测床回接 MioArgus 像素策略。

---

## 10. 与现有资产的关系

- 复用 `p0_env` 的**难度轴**（尺寸/熵/变化率）作为生成参数；
- 复用 `Q1` 的"探针"结论作为**故障与成本**的设计依据；
- 复用 `mio_env` 的 `reset/step`、成本、`stale` 观测纪律；
- 复用 `web_gap.py` / `web_channel.py` 的**测量方法**（这正是 held-out 要用到的）。

---

## 11. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 又一次自欺 | 任务与真值都是我们写的 | 真值尽量落在**文件/设置/进程状态**这类"世界算出来"的量上 |
| 保真度陷阱 | 追求像 Windows 而不是可测 | 指标是 held-out 掉幅，不是截图相似度 |
| 故障即答案 | 故障模式写死 → 策略只学会那几种 | 故障参数连续采样 + held-out 故障类型 |
| 工程量 | React 全套很大 | W0 可先用**纯 HTML/JS + Playwright**，够用即止 |

---

## 附：一句话

> 真正值钱的不是"仿了一个 Windows"，而是造出了一个**真渲染、外部真值、确定性、可注入故障**
> 的桌面沙箱——它能同时回答我们项目的三个悬而未决的问题：
> 结构 vs 像素的差距、故障恢复、以及站点/应用级 held-out 的泛化。
