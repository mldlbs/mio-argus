# MioGUI 训练环境（任务环境）

> 目的：Q1–Q3 是"探针"，无法训练可塑/元认知/世界模型。本目录提供真正可训练的
> **任务环境**：`reset()/step()`、ground-truth 奖励、可非平稳、可部分观测。
> 用法：`python mio/mio_env.py --demo`　基线：`python mio/mio_bc.py`

---

## 1. 任务

多步 GUI grounding（复用 `p0_env` 的网格渲染）：

```
每个 step 给指令 "click <label>"，agent 点击真实当前布局中该 label 的元素。
点对 → 进度 +1；点错 → 不推进；H 步后 episode 结束。
每步之后布局以概率 ρ 重随机。
```

`success = 完成 H 步`；`return = 累计奖励`（含 reobserve 成本）。

---

## 2. 为训练而设计的四个开关

| 开关 | 作用 | 服务的架构能力 |
|---|---|---|
| `observe_mode ∈ {fresh, stale}` | stale 下观察只在 reset/reobserve 时刷新 | **部分可观测**；让"重观察"成为决策 |
| `reobserve_cost` | reobserve 的即时代价（负 reward） | **元认知的成本模型**（P9 教训：无成本则 σ 退化） |
| `rule_permute` | 每 episode 重排 label→color | **跨 episode 非平稳**（可塑性） |
| 双接口 | obs 同时给 `pixels` 与 `elements` | 像素策略 vs 结构策略（H2） |

`max_attempts` 对**任何动作**计数，防止"永远 reobserve"的活锁（已踩过）。

---

## 3. 接口

```python
from mio_env import MioCfg, MioGUIEnv
env = MioGUIEnv(MioCfg(observe_mode="stale", rho=0.3))
obs = env.reset(seed=0)
obs, reward, done, info = env.step({"type": "click", "slot": 3})
env.step({"type": "click_xy", "x": 0.4, "y": 0.6})
env.step({"type": "reobserve"})
env.expert_action()                      # oracle，供模仿
sample_expert_dataset(n, cfg, seed)      # 逐 step (obs, action, reward, done)
```

`obs = {pixels:(3,S,S) float, elements:[...], instruction:str, step:int, H:int, stale:bool}`

---

## 4. 监督基线（序列模仿第一档）

`mio_bc.py`：结构化接口 + 指令匹配 + `stale→reobserve`。
**两处解耦**（都是踩坑后加上的）：
- 选 slot 用归一化嵌入点积 softmax；是否 reobserve 用**独立 sigmoid 门**，且门**只吃 `stale`**
  （早期把两者塞进同一 softmax、或让门共享指令嵌入 → 训练塌缩，train acc 卡 0.23）；
- 数据配对必须是 **动作前观察 ↔ 动作**（早期 off-by-one，116/150 标签错位）。

### 实测（600 episodes 专家数据，stale，ρ=0.3，100 epochs）

| | success | return | reobserves/ep |
|---|---|---|---|
| expert | 1.000 | 4.945 | 1.10 |
| BC（`mio_bc.py`） | **1.000** | 4.942 | 1.15 |

```
复现：python mio/mio_bc.py
```

⇒ 环境可被监督训练，模仿能复现 oracle 的成功率与成本。

---

## 5. 局限与下一步

- BC 目前用**结构化接口**（labels/stale）。像素策略（CNN over `pixels`）未接——下一步。
- `stale` 标志当前对策略**可见**，因此"何时重观察"偏易。要真的考元认知，应把它藏起来
  （策略需从历史推断漂移）。
- 无 RL；`rule_permute` 的跨 episode 非平稳只实现未启用。
- 像素渲染的文字较小，像素策略能否读 label 未验证。

**下一步**：像素 BC → 隐藏 stale（历史推断）→ RL。

---

## 6. 与整个项目的关系

- **H1/元认知**：`stale` + `reobserve_cost` 把"什么时候再看一眼"变成一个**有价决策**。
- **可塑性**：episode 重置 + `rule_permute` 提供跨 episode 非平稳。
- **可验证奖励**：成功由 ground-truth 进度判定（ROADMAP 的 C1 原则）。
- 这是把 Q1–Q3 的机制结论**放进一个能训练的闭环**的第一步。
