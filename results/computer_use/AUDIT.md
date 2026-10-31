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

> **现成 2B GUI 模型在我们的真实任务上可用，且比自研强 8–9 倍。**
> 同一批 216 条留出样本、同一套判据：自研 5.6%，UGround-V1-2B **49.1%**，
> GUI-Actor-2B **43.5%**（top-5 49.5%）。
> 也就是说，W28–W31 证明的是「**自研那套组合不可行**」，不是「这个任务不可行」。

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

## 5. 这次审计回答了什么

1. **能工作**。49.1% / 43.5% 是在 32px 级工具栏按钮上、零训练、零标注、离线跑的本地数字。
2. **之前"像素定位不可行"的结论下早了**：它只对「2020 条示范 + 冻结 ImageNet ViT + 224 输入」这个组合成立。
3. **延迟差 12 倍**：UGround 23s 不可交互；GUI-Actor 2s 可用，且原生给 top-5（多候选对验证器有用）。
4. **结构通道仍是独立的一层**：89.8% 的可用率意味着"grounder 出候选 + UIA 精确化"的组合最值得测。

## 6. 建议的下一步（不训练）

1. 把 GUI-Actor-2B 接进 W31 闭环，量**任务成功率**（而不是像素命中）；
2. 测「GUI-Actor 候选 + 结构通道校验」的组合增益；
3. 仍不做任何微调，直到上面两条给出"现成方案够不够"的答案。

## 7. 局限

- n=216，单一应用（QGIS），单一截图尺度（0.6 缩放存盘）。
- 模型版本固定为审计时的 HF 版本；UGround 用 5.x 主环境、GUI-Actor 用 4.51.3，两者预处理实现不同但输入图像与判据一致。
- `sdpa` 与官方 `flash_attention_2` 可能有微小数值差异，未做等价性对照。
