# 复用审计：现成 GUI 模型能不能在我们的任务上工作

> 复现：
> ```
> python experiments/computer_use/audit_eval.py --build-manifest
> python experiments/computer_use/audit_eval.py --predict structure
> python experiments/computer_use/audit_eval.py --predict self
> HF_ENDPOINT=https://hf-mirror.com python experiments/computer_use/audit_eval.py --predict uground
> # GUI-Actor（隔离 venv，transformers 4.51.3）
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/audit_predict_gui_actor.py
> python experiments/computer_use/audit_eval.py --score
> ```
> 结果：`results/computer_use/audit_results.json`　清单：`artifacts/computer_use_inloop/audit_manifest.json`

---

## 1. 结论

> **现成模型能用，但只在我们能提供文字标签的那部分任务上。**
> 同一批 216 条冻结样本：自研 5.6%，UGround-V1-2B **49.1%**，GUI-Actor-2B **43.5%**（top-5 49.5%）。
> 但按「指令是否含控件可见文字」拆开：含标签 **56.4% / 50.0%**，不含标签 **0.0%（0/28）**。
> W31 的六个工具栏图标任务全部属于后者，闭环 **0/9**——与 0/28 完全一致。
> 所以 W28–W31 证伪的是自研组合，而这次审计证伪的是「现成模型能解决图标控件」。
> 真正稳定的那一层是 **结构通道**（可用率 89.8%）。

## 2. 冻结条件（审计规格）

| 项 | 值 |
|---|---|
| 样本 | val 两条链（`w26_v2`、`w27`）中带 rect 的 click 样本，**216 条，冻结** |
| 坐标映射 | 所有 provider 输出归一化 [0,1]，由同一脚本换算到截图像素 |
| 判据 | 点落在 target_rect 内即命中；top-k = 前 k 个点任一命中 |
| 失败分类 | no_output / invalid / wrong |
| 尺寸分桶 | 目标短边 <50 / 50–100 / ≥100 px（**212/216 是 <50px**） |

四个 provider 全部走同一 manifest + 同一评分器，未做任何按模型的调参或后处理。

## 3. 环境核验（不是抄榜单）

| | UGround-V1-2B | GUI-Actor-2B |
|---|---|---|
| 权重 | `osunlp/UGround-V1-2B`（**Apache-2.0**） | `microsoft/GUI-Actor-2B-Qwen2-VL`（**MIT**） |
| 接口 | 标准 `Qwen2VLForConditionalGeneration`，文本输出 `(x,y)`∈[0,1000) | 需 `gui_actor` 代码包（HF 仓库不含代码）+ `Qwen2VLForConditionalGenerationWithPointer` |
| 本地阻塞 | 无 | 官方要求 `flash_attention_2`；**Windows 装不上**，改用 `sdpa` 验证可跑 |
| 环境 | 主环境（transformers 5.15.1） | **隔离 venv**：transformers 4.51.3，`--system-site-packages` 复用 torch |
| 网络 | `huggingface.co` 直连超时，全部经 `hf-mirror.com` 下载 | 同左；代码包从 `codeload.github.com` 取 |

## 4. 结果（216 条，本地实测）

| 方案 | top-1 | top-5 | 小目标(<50px) | 延迟 p50 | 显存 | no_out / invalid |
|---|---|---|---|---|---|---|
| 自研 grid@224（W30） | 5.6% | 5.6% | 5.7% | **40 ms** | **901 MB** | 0 / 0 |
| UIA/Bridge 词面匹配 | 14.4% | 15.3% | 14.6% | ~0 | 0 | 0 / 0 |
| **UGround-V1-2B** | **49.1%** | 49.1%¹ | **50.0%** | 22.9 s | 4.4 GB | 0 / 0 |
| **GUI-Actor-2B** | **43.5%** | **49.5%** | 44.3% | **1.95 s** | 4.9 GB | 0 / 0 |

¹ UGround 只输出单点，top-5 与 top-1 同值（`meta.topk=1`），不是缺数据。

补充：
- **结构通道可用率 89.8%**（216 条里 194 条的目标控件本来就在 UIA/bridge 列表里）。
  词面匹配只拿到 14.4% ⇒ 结构信息的瓶颈是「指令→控件」匹配，不是信息缺失。
- 四条都没有出现解析失败或越界；失败全部是"有效但没点中"。
- `large(≥100px)` 桶只有 4 条且全部 0%，样本太少，不解读。

## 6. 关键拆分：现成模型只在"有文字标签"的控件上有效

把 216 条按「指令是否包含控件可见文字」拆开：

| 模型 | 含标签（188 条） | 不含标签（28 条） |
|---|---|---|
| UGround-V1-2B | **56.4%** | **0.0%** |
| GUI-Actor-2B | **50.0%** | **0.0%** |

也就是说：**两个模型本质都是"读字定位"**——指令里有 `'cities'`、`'5'`、`'Mode'` 这类可见文字时命中率 50–56%，
一旦目标是纯图标（例如 QGIS 工具栏的 Zoom Full / Last / Native），命中率是 **0/28**。

## 7. 闭环验证（W31 + GUI-Actor-2B）

把 GUI-Actor 直接接进 W31 的验证器闭环（同任务、同判定、同重试协议）：

| 定位器 | 任务成功 | success@1 | 失败归因 |
|---|---|---|---|
| 自研 v3（W31） | 0/9 | 0/9 | truth_fail 45 |
| **GUI-Actor-2B** | **0/9** | 0/9 | truth_fail 45（点击点集中在屏幕右下角 2340,1385） |

Oracle 基线两次都是 3/3，机械通路正常。
离线诊断（同一帧、同一指令）显示它的 top-5 全落在屏幕右侧（1861,56 / 2493,60…），
离 Zoom Full 按钮（359,49）很远 —— 与 §6 的 0/28 一致。

## 8. 这次审计回答了什么

1. **分层结论**：文字标签类控件（188/216）现成模型 50–56%，可用；**图标类控件（28/216）两者都是 0%**。
   自研 5.6% 的问题不是"没调好"，而是"整条像素路线在图标控件上都不成立"。
2. **之前"像素定位不可行"的结论下早了一半**：对文字标签控件，现成模型证明可行；对图标控件仍然不可行。
3. **延迟差 12 倍**：UGround 23s 不可交互；GUI-Actor 2s 可用，且原生给 top-5。
4. **结构通道是唯一覆盖全体的那一层**：可用率 89.8%（含图标控件），
   因此产品形态应是「结构通道为主 + 文字标签类交给现成 grounder」。

**组合覆盖率（216 条，任一路径命中即算）**：

| 组合 | 覆盖 |
|---|---|
| 结构词面 | 14.4% |
| UGround | 49.1% |
| GUI-Actor | 43.5% |
| UGround ∪ GUI-Actor | 56.5% |
| **三者并集** | **62.5%**（81 条全miss） |

即使把三条通道全叠加，仍有 37.5% 点不中——其中 28 条是纯图标，其余 53 条是文字标签但三者都失手。
**这说明"堆模型"的边际收益有限，还差的 37.5% 需要结构通道的精确 rect（89.8% 可用）来补，而不是再加一个 grounder。**

## 9. 建议的下一步（不训练）

1. ~~把 GUI-Actor 接进 W31 闭环~~ —— **已做，0/9**（因为它全是图标控件，见 §7）
2. 测「结构通道 → 精确 rect」与「grounder → 文字控件」的组合覆盖率：两者并集应该接近 100%
3. 图标控件若必须用视觉：只能靠更大预算（GUI 预训练 + 高分辨率 tiling）或专用图标数据集；不在当前资源内
4. 仍不做任何微调，直到上面给出"组合是否够用"的答案

## 10. 局限

- n=216，单一应用（QGIS），单一截图尺度（0.6 缩放存盘）。
- 模型版本固定为审计时的 HF 版本；UGround 用 5.x 主环境、GUI-Actor 用 4.51.3，两者预处理实现不同但输入图像与判据一致。
- `sdpa` 与官方 `flash_attention_2` 可能有微小数值差异，未做等价性对照。
