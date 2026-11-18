# H4 复用审计：P5 能否作为 H4 的对照

> 日期：2026-11-18　性质：**只读审计**（未跑训练矩阵、未改指标定义、未改 P5 代码）
> 对象：`experiments/paper/p5_h4_entropy.py` + `results/paper/p5_h4_results.json`
> 用途：判断 P5 的 E0/E1/E3 能否直接作为 H4 预注册中 C0/C4/C5 的对照。
> 关联：`H4_GENERALIZATION_PREREG.md`（草案 v0.1，未执行）

---

## 0. 裁定：**B — 有限可比**

P5 只能作**背景参考**（量级、趋势方向），**不得进入 H4 主分析**。

四条硬理由（每条都经过实测，不靠读注释推断）：

1. **P5 是单次运行**：每个 `level × width` 只有一个数，json 里没有 seed / repeat 字段；
   无法与 H4 的 3-seed 均值做任何统计比较。
2. **模型配置不同**：P5 用 `inst_dim=32`，本项目其余实验（P0/M3/H3b）用 64 →
   同一 width 的参数量差 **3,072（w8）/ 6,144（w32）**，`params_needed` 的绝对值不可跨文件比较。
3. **任务流不可复现**：同一 seed、同一 `E0-flat`，P5 与 `p0_env` 生成的图像**不相等**
   （实测 `max|Δ| = 0.878`，目标 slot 25 vs 19）——采样顺序与 rng 消耗不同，
   "同 seed 即同任务流"不成立。
4. **过程不可追溯**：无 checkpoint、无 config 哈希、无任务流哈希、无 git commit 记录。

因此：**H4 的 C4/C5 必须在同一 harness 内重跑**，不能拿 P5 的数当对照。

---

## 1. 证据与哈希

| 文件 | sha256(前16) | 大小 |
|---|---|---|
| `experiments/paper/p5_h4_entropy.py` | `22cf7aef07a8797b` | 10,841 B |
| `results/paper/p5_h4_results.json` | `7a5f9fa0ad22461e` | 1,227 B |
| `results/paper/P5_RESULTS.md` | `9934b9e1d7e8e5cd` | 4,397 B |
| `experiments/paper/p0_env.py`（环境基准） | `1dc39ebfe7f987a0` | 8,526 B |
| `experiments/paper/p0_models.py`（模型基准） | `c9485f2946d7af36` | 4,158 B |

P5 常量：`GRID=6, IMG_SIZE=160, S=36, EPOCHS=20, N_TR=3000, N_TE=1000`，
`WIDTHS=[4,6,8,12,16,24,32]`，训练数据 `seed=100`，测试数据 `seed=999`，
`train()` 的 `torch.manual_seed` 恒为 **0**（所有 width 共用同一初始化种子，无逐次扰动）。

---

## 2. 五项一致性检查

| 审计项 | 结论 | 依据（可复核） |
|---|---|---|
| **模型架构** | ⚠️ **不一致** | 都是 `P0Net(head=cls, grid=6, stages=3, in_res=160)`，但 P5 固定 `inst_dim=32`（`p5_h4_entropy.py:159`），其余实验为 64。实测参数量：w8 `9,248` vs `12,320`（差 3,072）；w32 `99,968` vs `106,112`（差 6,144） |
| **评估口径** | ⚠️ **方法同、精度不足** | 主指标同为"单步 argmax 命中率"；`params_needed` 同为 log 参数轴线性插值。但 P5 的 `params_needed` 是**四元组** `[p@0.70, p@0.80, p@0.90, p@0.95]`，引用时必须写明取哪个下标，否则同名不同义；且每点**仅一次运行**，无区间 |
| **测试集** | ❌ **不可比（同一 seed 不同任务）** | P5 自己实现 `build_data`：先 `rng.choice` 选标签、再取目标 slot、再在 `render_level` 里先画背景后画元素；`p0_env.make_sample` 是"先标签→逐槽几何→最后取目标"。实测 seed=999 / E0-flat：图像不相等（`max|Δ|=0.878`），目标 slot 25 vs 19。分布族相近，**任务流不同** |
| **环境** | ⚠️ **绘制常量一致，采样不一致** | 网格线 `(214,218,224)`、元素描边 `(25,25,25)`、flat 底 `(246,247,249)`、fill=0.40、jitter=4、颜色 `PALETTE[LABEL_TO_ID[label] % len(PALETTE)]` 与 `p0_env.render` 完全一致；但元素几何公式是**复制**而非调用，且背景绘制在元素之前消耗 rng → 同 seed 不能逐位复现 |
| **实验过程可追溯** | ❌ **不足** | 不存 checkpoint；json 仅 `device/grid/epochs/n_train/n_test/widths/params/curves/params_needed`，**无 seed、无重复、无配置哈希、无任务流哈希、无 commit** |

---

## 3. 不一致清单（按影响排序）

| # | 不一致 | 影响 | 能否接受 |
|---|---|---|---|
| 1 | `inst_dim=32`（P5）vs 64（其余） | `params` 与 `params_needed` 的绝对数值不可跨文件比较；"同一 width"不是同一模型 | ❌ 不能接受（除非 H4 也用 32，但那会让 H4 与自身其余实验不可比） |
| 2 | 单次运行、无重复 | 无法分离"条件效应"与"运行噪声"（本项目实测同配置重跑极差可达 0.146） | ❌ 不能接受 |
| 3 | 任务流不可复现（同 seed ≠ 同图像） | 无法声称"用同一测试集"，也无法做配对比较 | ❌ 不能接受 |
| 4 | 无配置/任务流哈希 | 事后无法证明复现的是同一环境 | ⚠️ 可补（H4 重跑时记录 `config_fp` + 数据哈希） |
| 5 | 无 checkpoint | 无法复查"某 width 是否学坏" | ⚠️ 可补（保存 config_fp + 逐 width 曲线即可，不必存权重） |
| 6 | `params_needed` 是四元组 | 论文里写 `params_needed(0.80)` 时必须明确是下标 1，否则与 P5_RESULTS 的默认展示（p@0.90 用于判定）混淆 | ⚠️ 口径写明即可 |

> 注：P5 的**判定逻辑**用 `p@0.90`（`p5_h4_entropy.py:260`），而 H4 预注册的主指标写的是 `params_needed(0.80)`。
> 这不是错误，但**两者不是同一个数**，跨文件混引会直接产生错误结论。H4 必须只认自己的 0.80 口径。

---

## 4. 可复用范围

| 用途 | 是否可用 | 说明 |
|---|---|---|
| 引用"blobs 最差、iid 噪声优于 blobs" | ✅ 可以（**定性**） | 趋势方向明确（w=4：0.127 vs 0.743） |
| 作为 H4 C4/C5 的**数值对照** | ❌ 不可以 | 见 §3-1/3-2/3-3 |
| 作为 H4 的**背景参考**（动机/讨论） | ✅ 可以 | 必须标注"不同 harness、单次运行、`inst_dim=32`，仅作参考" |
| 作为 `params_needed` 的基线表 | ❌ 不可以 | 参数量口径不同、阈值口径不同 |
| 复用其**背景生成代码**（`make_bg`） | ✅ 可以（需移植并哈希） | 四类背景的定义可直接搬到 H4 harness，但必须在同一次运行内生成 |

### 4.1 额外发现：`E1-noise` 曲线本身形态异常

P5 的四条曲线（w=4→32）：

```
E0-flat    0.938 0.986 0.992 0.996 0.996 0.995 0.998   ← 单调、饱和
E1-noise   0.743 0.872 0.840 0.753 0.676 0.591 0.539   ← 到 w=8 见顶后**反降**
E2-blobs   0.127 0.197 0.317 0.461 0.443 0.557 0.615   ← 单调上升（最差）
E3-natural 0.677 0.704 0.813 0.877 0.892 0.877 0.889   ← 见顶后持平
```

`E1-noise` 在更大模型上**变差**（0.743 → 0.539），说明该条件不是"参数不够"，而是
噪声背景诱导了记忆/过拟合（或该 seed 下训练不稳）。副作用是它在**任何** width 下都
达不到 0.90（最高 0.872）→ json 中 `E1-noise` 的 `p@0.90` / `p@0.95` 为 `null`，
而 P5 自己的判定逻辑恰好用的是 `p@0.90`（`p5_h4_entropy.py:260`）。
即：**P5 对 E1 这条对照根本没算出判定量**，它只是被画进了曲线。

结论不变，但理由加强：**C4（iid 噪声）必须由 H4 自己重跑并给 3 seed 区间**，
否则连"噪声成本是否随规模下降"都无法判断。

---

## 5. 对预注册的影响：**84 次方案作废**

`H4_GENERALIZATION_PREREG.md` §3 原写：

> 若时间受限，先跑 C0–C3（84 次），C4/C5 用 P5 的既有结果对照（**同口径**）。

本次审计否掉了括号里的"同口径"：模型宽度口径、运行次数、任务流三者都不同。
因此 **84 次方案不能保留**，否则 H4 的核心对比（熵 vs 干扰）会退化成跨 harness 拼接。

剩下两个可接受方案：

| 方案 | 条件 | 运行数 | 代价 | 对照有效性 |
|---|---|---|---|---|
| **A（推荐）** | C0–C5 全部同 harness | 6 × 7 × 3 = **126** | ~4–6 h | 全部同批次，最强；可直接做 Spearman 与 C3/C0 比值 |
| B | C0–C5，width 缩到 5 档（4/8/16/24/32） | 6 × 5 × 3 = **90** | ~3–4 h | 同批次；插值点变稀，`params_needed` 区间变宽 |

**推荐 A**：H4 的假设本质是"两个因素在**同一坐标系**里比较"，126 次是买到这个坐标系的成本；
若必须压缩，选 B 并在报告里写明插值区间变宽，**不要**退回 C（84）。

---

## 6. 审计方法（可复核）

```powershell
# 1) 文件哈希
python -c "import hashlib,pathlib; [print(f, hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()[:16]) for f in ['experiments/paper/p5_h4_entropy.py','results/paper/p5_h4_results.json']]"

# 2) 参数量口径差（inst_dim 32 vs 64）
python -c "import sys; sys.path.insert(0,'experiments/paper'); from p0_models import P0Net; [print(w, P0Net(width=w,head='cls',num_slots=36,grid=6,num_labels=64,inst_dim=32,in_res=160,stages=3).n_params(), P0Net(width=w,head='cls',num_slots=36,grid=6,num_labels=64,inst_dim=64,in_res=160,stages=3).n_params()) for w in (8,32)]"

# 3) 同 seed 任务流一致性（结果：不相等）
python -c "import sys; sys.path.insert(0,'experiments/paper'); import numpy as np; from p0_env import EnvCfg, make_sample; import p5_h4_entropy as P5; cfg=EnvCfg(grid=6,img_size=160,label_vocab=36,fill=0.40,jitter=4,color_by_label=True); s=make_sample(np.random.default_rng(999),cfg); d=P5.build_data(1,999,'E0-flat'); print(s.target_slot, int(d['slots'][0]), np.array_equal(s.image, d['images'][0].astype(np.float32)/255.0))"
```

三条命令的输出已写入 §2/§3，任何人可重跑核对。

---

## 7. 明确不做

- 不修改 H4 主指标（仍是 `params_needed(0.80)`，即四元组下标 1）；
- 不为复用旧数据去改 P5 的 `inst_dim`；
- 不在审计阶段跑任何训练（本次 0 次训练）；
- 不用 P5 的数直接填 H4 的 C4/C5 单元格；
- 不因"P5 已有点"而缩小 H4 条件集。

---

## 8. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-18 | v1.0 | 首次审计：裁定 B；五项检查 + 4 项不一致；三种矩阵方案 |
| 2026-11-18 | v1.1 | 补实测证据：同 seed 任务流不等（`max|Δ|=0.878`）；`params_needed` 四元组口径；P5 判定用 p@0.90 而 H4 用 0.80；**84 次方案作废**，只保留 126 / 90 |
