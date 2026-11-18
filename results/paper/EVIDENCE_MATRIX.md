# P0–P9 证据矩阵（只读审计）

> 审计日期：2026-11-17　审计方式：**只读**（未改动实验代码、未启动实验、未安排人工审核）
> 审计原则：**不把项目报告里的结论当作实验事实**。每条主张必须回到原始 JSON，
> 并把脚本与命令写清楚；对不上的，一律降级并在 `EVIDENCE_GAPS.md` 记录冲突。

## 0. 证据等级

| 等级 | 定义 |
|---|---|
| **A** | 原始数据、脚本、复现命令完整，且 JSON 数值与主张一致 |
| **B** | 有实测数据，但存在复现或实验设计缺口（缺对照 / 缺规模 / 缺协变量） |
| **C** | 只有局部结果或间接证据 |
| **D** | 只有报告描述，无可追溯实验依据 |
| **否证** | 实验结果与预注册假设相反 |

复现命令均为**脚本默认参数**（`--quick` 关闭）。本次审计**未执行**这些命令，
因此「复现」一列表示"可执行入口"，不代表已复现。

---

## 1. 矩阵

### H1 — 重观察替代参数（外部化）

| 项 | 内容 |
|---|---|
| 实验 | P1 `p1_horizon_results.json` |
| 原始结果 | ρ=0.3, H=8：`cls_task` k=1 **1.000** vs k=8 **0.0767**（gap +0.923）；ρ=0 时各 k **逐位相同**（0.0700/0.0700/0.0700）；ρ=1, H=8：k=1 0.9967 vs k=8 0.0000 |
| 脚本 | `experiments/paper/p1_horizon.py` |
| 复现 | `python experiments/paper/p1_horizon.py` |
| 等级 | **B（测量降权）** |
| 理由 | 数据扎实、与 `P1_RESULTS.md` 一致；但**没有大模型基线**（作者在文末也承认"同等参数下的比较"），因此"替代参数"这一句未被检验，只证明了"重观察是长程任务的必要条件" |

### H2 — 动作空间压缩（结构替代感知容量）

| 项 | 内容 |
|---|---|
| 实验 | P0 `p0_h2_results.json` / `p0_regloss_results.json` |
| 原始结果 | 同为 106,112 参数：fill=0.75 `cls` 1.000 / `reg` 0.997；fill=0.55 `cls` 0.9995 / `reg` **0.8085**；fill=0.40 `cls` 1.000 / `reg` **0.511**；grid=4 时 `cls` 1.000 / `reg(coord)` 0.962 / `reg(gauss)` 0.3195 |
| 脚本 | `experiments/paper/p0_h2.py`、`p0_regloss.py` |
| 复现 | `python experiments/paper/p0_h2.py`、`python experiments/paper/p0_regloss.py` |
| 等级 | **B（cls 分支作废，仅回归/解析器分支保留）** |
| 理由 | P0 的 `cls` 优势已被 M3 证实来自颜色捷径（C6），**不得**再作为结构化动作有效的证据；剩余有效证据是 P7/P8 的解析器瓶颈与 `reg` 对比，以及 P0 的 `reg` 内部对比。假设原文"用少得多的参数达到同成功率"仍未做参数扫描 |

### H2-边界 — 真实解析器下的反转

| 项 | 内容 |
|---|---|
| 实验 | P7 `p7_realistic_results.json`、P8 `p8_learned_parser_results.json` |
| 原始结果 | P7 extreme+dense：`parser_recall` 0.845，`cls_parser` **0.833**，`reg` **0.977**，`parser_cost` 0.167；P8 学习式解析器：`recall` 0.518/0.515，`iou` 0.213/0.236 |
| 脚本 | `experiments/paper/p7_realistic_ui.py`、`p8_learned_parser.py` |
| 复现 | `python experiments/paper/p7_realistic_ui.py`、`python experiments/paper/p8_learned_parser.py` |
| 等级 | **否证（对 H2 的适用边界）** |
| 理由 | 一旦元素解析不完美，`cls` 的成功率被 `parser_recall` 直接封顶，而**坐标回归反超**（0.977 vs 0.833）；学习式解析器 recall 仅 0.52 → H2 的收益完全取决于外部解析器质量。另见 C6：`cls` 与 `reg` 的差距部分来自颜色查找 |

### H3a — 重新接地式 verifier

| 项 | 内容 |
|---|---|
| 实验 | P2 `p2_verifier_results.json` |
| 原始结果 | JSON：`single_step` `pi_w32` 1.0 / `pi_w64` 1.0 / `v_alone` 1.0；`diagnostics`: `ceiling_topk` 1.0、`v_pick_given_in_topk` 1.0、`joint` 1.0 |
| 报告主张 | `P2_RESULTS.md`：score 最大绝对差 **0.007153**；"联合 0.5960 < π top-1 0.8600"；"0.86→0.60" |
| 脚本 | `experiments/paper/p2_verifier.py`（`p2_equiv_check.py` 无 `__main__`，需 import 调用） |
| 复现 | `python experiments/paper/p2_verifier.py` |
| 等级 | **C（数据冲突）** |
| 理由 | **报告的 0.86/0.60 在本 JSON 中无法复现**（JSON 全部饱和为 1.0）。结构性论证（V 与 π 共享 backbone ⇒ 同构）是分析性的、成立；但关键实证数字缺可追溯来源 → 按"缺可追溯实验依据"降级 |

### H3b — 结果验证式 verifier

| 项 | 内容 |
|---|---|
| 实验 | P3 `p3_verifier_results.json` |
| 原始结果 | ρ=0.3：H1 0.980→0.990、H2 0.9367→0.9767、H4 0.8367→**0.900**、H8 0.5567→**0.7033**；`pi_V` 与 `pi_oracle` **逐格相等**；V 参数 6,393（π 的 1/17），`v_metrics.acc` = 1.0 |
| 脚本 | `experiments/paper/p3_verifier.py` |
| 复现 | `python experiments/paper/p3_verifier.py` |
| 等级 | **A（已支持，跨条件泛化，存在明确边界）** |
| 理由 | P3：数值与 `P3_RESULTS.md` 一致，增益随 H 单调上升且达到 oracle 上界；`V.acc=1.0`（任务可完美判定）是保留意见。**H3b 泛化（2026-11-18）**：`grid8` ✅ [+0.075,+0.114]、`delay1` ✅ [+0.049,+0.226]、`eps15` ❌ [−0.527,−0.198] → **2/3 通过**。边界见 `EVIDENCE_GAPS.md` §7 |

### H4 — 窄域低熵 / 同类干扰项

| 项 | 内容 |
|---|---|
| 实验 | **主**：`h4_density.py`（126 次，2026-11-18）；**背景**：P5 `p5_h4_results.json`（不参与主分析） |
| 原始结果 | 准确率（3 seed 均值，w4→w32）：C0 `0.938→0.999`、C1 `0.533→0.896`、C2 `0.362→0.778`、C3 `0.302→0.602`、C4（噪声，零干扰）`0.840→0.611`（**反降**）、C5（自然背景，零干扰）`0.615→0.921`。`params_needed(0.80)`：C0 ≤**6,768**（左截断）、C1 **22,114**、C2 **null**、C3 **null**（106,112 参数仍未达 0.80）、C4 ≤6,768、C5 11,298 |
| 脚本 | `experiments/paper/h4_density.py`（分析 `h4_analyze.py`） |
| 复现 | `python experiments/paper/h4_density.py` → `python experiments/paper/h4_analyze.py` |
| 等级 | **B（描述性支持；主指标不可估）** |
| 理由 | 预注册判据 ② 需要 `params_needed(C3)`，而 C2/C3 在 7 档 width 内**从未达到 0.80** → 判据**不可计算**，按预注册 §4.7 记 `null`，**既不判成立也不判否证**。描述性证据方向一致：固定 width 上 `C0>C1>C2>C3` 在 **5/7** 档成立（w12–w32 全部成立），`C3/C0 ≥ 15.7×`（下界）；零干扰的高熵条件 C4 与 C0 无差（Δ=+258，CI [−853,+1,369] 覆盖 0）。P5 只能作背景：`inst_dim=32`、单次运行、任务流不可复现（见 `H4_REUSE_AUDIT.md`） |

### Q1–Q4 — 可塑性 / 门控机制

| 项 | 内容 |
|---|---|
| 实验 | `q1_q2_q3_q4_*.json` |
| 原始结果 | Q4 `verdict`：noise0.0 `B-A = 0.00076`、`O-A = 0.018`；noise0.4 `B-A = 0.00006`、`O-A = 0.093`、`KvOracle = 0.386`；Q3 学习门控与无门控差 < 0.001；Q1 `B_kv` 的 `d_ge1 = 0.426` 但 `d_mid = 0.0026` |
| 脚本 | `experiments/plasticity/q1_q2_q3_q4_*.py` |
| 复现 | `python experiments/plasticity/q1_plasticity.py` 等（参数见各自 argparse） |
| 等级 | **否证（学习式门控无增益）** |
| 理由 | 学习到的门控与"无门控"在多个种子/噪声水平下差异 ≈ 0；只有 oracle 门控有小幅增益（+0.018/+0.093），说明"该不该更新"这件事**可被价值化，但当前学不出来** |

### P4a–P4d — 群体分工（主线外延）

| 项 | 内容 |
|---|---|
| 原始结果 | P4d：population 参照 `acc 0.6887 @109,350`；最强通才 `0.3708 @382,208`（3.5× 参数仍追不上）；P4c：OOD `router_acc 0.0`、`routed 0.058` vs `oracle 0.195` |
| 脚本 | `experiments/paper/p4_population.py`、`p4b_capacity.py`、`p4c_ood.py`、`p4d_capacity_scaling.py` |
| 复现 | 对应脚本默认参数 |
| 等级 | **B** |
| 理由 | 域内"专精 > 等参数通才"证据强；但 router 依赖域标签、OOD 完全失效 → 不能外推 |

### P9 — 主动缩放

| 项 | 内容 |
|---|---|
| 原始结果 | extreme：T=2 `mono 0.6014` / `glimpse 0.7014` / `random 0.7043`；T=3 `glimpse 0.7443` / `random 0.7529`；12 格里 **9 格 random ≥ glimpse** |
| 脚本 | `experiments/paper/p9_active_zoom.py` |
| 复现 | `python experiments/paper/p9_active_zoom.py` |
| 等级 | **否证（"主动"部分）** |
| 理由 | 增益来自"多裁剪 = 更高分辨率/覆盖"，与"选哪里"无关；`P9_RESULTS.md` 自己已写明，但 `REPORT.md` 摘要未提 |

---

## 2. 一句话汇总

| 假设 | 等级 | 当前能说的话 |
|---|---|---|
| **H（核心：crossing 曲线）** | **D** | **从未被检验**：没有大模型臂，`S(θ_small)/S(θ_large)` 与成本 `C` 都没有数据 |
| H1 | B | 重观察是长程任务必需；**未证明**能替代参数 |
| H2 | B | 合成域内结构化更鲁棒；**未做参数扫描**，且被解析器质量一票否决 |
| H3a | C | 结构上不可能（同构），但关键实证数字不可追溯 |
| H3b | A（+2/3 泛化） | 结果验证有效；grid8/delay1 仍为正增益，eps15(ε=0.15) 反转为负 |
| H4 | B（描述性支持，主指标不可估） | 干扰密度↑ → 成本↑（C3/C0 ≥15.7×）；零干扰的噪声背景无额外成本 |
| Q1–Q4 | 否证 | 学习式门控 ≈ 无门控 |
| P9 主动缩放 | 否证 | 随机裁剪同等或更好 |
| P4 群体分工 | B | 域内强，OOD 崩 |
