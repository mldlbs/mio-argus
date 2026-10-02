# MioArgus

**MioArgus** 是一个 computer-use 智能体：看屏幕，替你操作。

> *Mio*（妙）+ *Argus* — Argus Panoptes（百眼巨人），希腊神话中不眠的看守者。MioArgus 给模型一双看屏幕的眼、一只操作鼠标的手。

构建于 [nanoGPT](https://github.com/karpathy/nanoGPT) 之上：冻结的 **ViT-B/16** 视觉编码器 + 冻结的 **GPT-2** 语言模型，各用 **LoRA (r=16)** 适配，经**跨注意力（cross-attention）**融合模块输出 GUI 动作。

## 它做什么

给定一张截图和一句自然语言指令，MioArgus 预测：

| 输出 | 空间 |
|------|------|
| 动作 | `click`、`type`、`scroll`、`move`、`hotkey`、`press` |
| 坐标 | `[0, 1]` 归一化 `(x, y)` |
| 滚动方向 | `up`、`down`、`none` |

## 架构

```
screenshot ──▶ ViT-B/16 (frozen + LoRA)  ─┐
                                          ├─▶ Cross-Attention Fusion ─▶ heads
instruction ─▶ GPT-2    (frozen + LoRA)  ─┘        │
                                          ┌────────┼────────┐
                                     action    coord    scroll
                                      head      head     head
```

- 总参数 **220,389,131**，可训练 **30,821,387**
- 坐标回归用 **Huber loss**，动作/滚动用交叉熵
- checkpoint 与旧版 `nn.Linear` state dict 兼容

## 结果

在 504 条均衡评测集（每动作 84 条）上：

| 指标 | 数值 |
|------|------|
| 动作准确率 | **100.00%** |
| 坐标 MAE | **0.0667** |
| 滚动准确率 | **93.06%** |

## 快速开始

训练：

```sh
python experiments/computer_use/collect_synthetic.py          # 生成均衡数据（无需显示器）
python experiments/computer_use/train_computer_use_v3_real.py # 微调 LoRA adapters
python experiments/computer_use/test_in_vm_v3.py              # 评测
```

起服务：

```sh
docker compose -f infra/docker-compose.yml up --build
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "instruction": "click the submit button"}'
```

API 端点：`/health`、`/model/info`、`/predict`、`/predict/batch`、`/predict/file`。

## 测试

```sh
pytest experiments/computer_use/test_simple.py experiments/computer_use/test_eval_metrics.py experiments/computer_use/test_vm_e2e.py -q   # 12 passed
```

全量基线：`python -m pytest -q` → **13 passed / 1 failed / 1 error**（`test_model` 为预存 state_dict 的已知失败，`test_vm_e2e` 需网络；设 `TRANSFORMERS_OFFLINE=1` 后为 14 passed / 1 failed）——见 [AGENTS.md](AGENTS.md)。

## 研究

MioArgus 不追求与大 VLM 比规模。赌注是：**computer use 是一个窄域**——一个带结构化 UI 状态、压缩动作空间和验证器的小型接地策略，能以更低算力匹配整体大模型，且差距**随任务时距（horizon）拉大**。

可证伪假设、子假设（H1–H4）、实验协议与杀死条件见 **[docs/HYPOTHESIS.md](docs/HYPOTHESIS.md)**；仓库结构与文件纪律见 **[STRUCTURE.md](STRUCTURE.md)**。

### 综合报告

**[docs/REPORT.md](docs/REPORT.md)** 把所有线收拢在一起。已测内容摘要：

| 假设 | 状态 | 关键数字 |
|------|------|----------|
| H1 重观察替代参数 | ✅ | 差距随 H 增大：0.29 → 0.65 → 0.92 |
| H2 动作空间压缩 | ⚠️ 有条件 | grid：12K 分类 ≈100% vs 392K 回归 ≈10%。自由布局 UI：当且仅当 `parser_recall > reg_accuracy` 时胜（tiny/micro +0.13/+0.22；texture −0.63）。小型学习型检测器 recall 上限 ~0.52 → 不够 |
| H3 验证替代精度 | ⚠️ 分裂 | 重新接地 ❌ / 结果验证 ✅（V 小 17 倍） |
| H3 验证器精度要求 | ✅ | FNR < ~0.8/H；FNR 远比 FPR 代价高 |
| H4 窄域低熵 | ✅ | 4,208 参数 → 0.938 vs 100K 参数 → 0.889（自然背景） |
| 种群自进化 | ❌ | 混杂；无能力证据 |

四条线收敛到同一结论：**瓶颈不在模型多大，而在必须从高熵输入里恢复哪些信息。**
P7 把它写成可计算的不等式：结构化动作选择胜过裸坐标回归**当且仅当 `parser_recall > reg_accuracy`**（12/12 配置）——平坦背景上的小目标它赢（+0.13/+0.22），纹理背景上解析器崩掉就输得重（−0.55..−0.63）。P8 说明小型学习型检测器补不上这个缺口：recall 饱和在 ~0.52 而门槛 >0.78，零参数的经典解析器在干净背景上反而更强（0.847 vs 0.518）。结构化动作空间的实用区间是 `{高精度要求} ∩ {可靠解析器}`——真实桌面/网页应用只有无障碍/DOM 树满足它，像素不满足。

单项结果：[P0](results/paper/P0_RESULTS.md) · [P0.5](results/paper/P0_5_RESULTS.md) · [P1](results/paper/P1_RESULTS.md) · [P2](results/paper/P2_RESULTS.md) · [P3](results/paper/P3_RESULTS.md) · [P4a](results/paper/P4A_RESULTS.md) · [P4a'](results/paper/P4B_RESULTS.md) · [P4c](results/paper/P4C_RESULTS.md) · [P4d](results/paper/P4D_RESULTS.md) · [P5](results/paper/P5_RESULTS.md) · [P6](results/paper/P6_RESULTS.md) · [P7](results/paper/P7_RESULTS.md) · [P8](results/paper/P8_RESULTS.md) · [P9](results/paper/P9_RESULTS.md)

其他实验线：[WinSim 桌面沙箱](results/winsim/WINSIM_RESULTS.md) · [QGIS 真桌面](results/qgis/QGIS_FINDINGS.md) · [可塑性四问](results/plasticity/Q4_RESULTS.md) · [真实站点结构通道](results/web/REAL_SITE.md) · [路线图](docs/ROADMAP.md)

---

# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

*以下为上游 [nanoGPT](https://github.com/karpathy/nanoGPT) README 的中文翻译（MioArgus 基于其上构建；英文原文见上游仓库）。*

---

**2025 年 11 月更新** nanoGPT 有个更新更好的表亲叫 [nanochat](https://github.com/karpathy/nanochat)。你多半本想用/找的就是 nanochat。nanoGPT（本仓库）已经很老、已弃用，但为存档我把它留着。

---

训练/微调中等规模 GPT 最简单、最快的仓库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重写，重实战（teeth）轻教学。仍在活跃开发中，目前 `train.py` 能在单节点 8×A100 40GB 上用约 4 天，在 OpenWebText 上复现 GPT-2 (124M)。代码朴素可读：`train.py` 是约 300 行的样板训练循环，`model.py` 是约 300 行的 GPT 模型定义，可选择性地加载 OpenAI 的 GPT-2 权重。就这些。

![repro124m](assets/gpt2_124M_loss.png)

因为代码非常简单，按需魔改、从头训练新模型、或微调预训练 checkpoint 都很容易（例如目前能拿来当起点的最大模型是 OpenAI 的 GPT-2 1.3B）。

## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers`（huggingface transformers，用于加载 GPT-2 checkpoint）<3
- `datasets`（huggingface datasets，用于下载 + 预处理 OpenWebText）<3
- `tiktoken`（OpenAI 的快速 BPE 代码）<3
- `wandb`（可选日志）<3
- `tqdm`（进度条）<3

## 快速开始

如果你不是深度学习专业人士、只是想感受一下魔法、趟趟水，最快的入门是拿莎士比亚全集训练一个字符级 GPT。第一步，把它下载成单个（1MB）文件，再从原始文本变成一长串整数：

```sh
python data/shakespeare_char/prepare.py
```

这会在该数据目录里生成 `train.bin` 和 `val.bin`。现在开始训练你的 GPT。规模大小很大程度上取决于你的算力：

**我有 GPU**。很好，用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 里的配置快速训一个迷你 GPT：

```sh
python train.py config/train_shakespeare_char.py
```

瞄一眼配置就会看到：我们训的 GPT 上下文最长 256 字符、384 特征通道、6 层 Transformer 每层 6 头。单张 A100 上这次训练约 3 分钟，最优验证 loss 为 1.4697。按配置，checkpoint 写入 `--out_dir` 目录 `out-shakespeare-char`。训练完成后，把采样脚本指向该目录、用最优模型采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`。对一个在 GPU 上训了 3 分钟的字符级模型来说还不错。在这个数据集上微调一个预训练 GPT-2 模型多半能得到更好的结果（见后面的微调一节）。

**我只有一台 macbook**（或其他性能平平的电脑）。别慌，我们照样能训 GPT，只是得把配置调低一档。我建议装 PyTorch nightly 抢鲜版（安装时[在这里选](https://pytorch.org/get-started/locally/)），它目前很可能让代码更高效。即使不用它，一次简单的训练可以长这样：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

这里因为跑在 CPU 而不是 GPU 上，必须同时设 `--device=cpu`，并用 `--compile=False` 关掉 PyTorch 2.0 编译。评估时用更嘈杂但更快的估计（`--eval_iters=20`，从 200 降下来）、上下文只有 64 字符而不是 256、每步 batch 只有 12 条而不是 64。我们还用小得多的 Transformer（4 层、4 头、128 维嵌入），迭代次数降到 2000（相应地通常用 `--lr_decay_iters` 把学习率衰减到 max_iters 附近）。网络这么小，正则化也得放松（`--dropout=0.0`）。这仍然大约跑 3 分钟，但 loss 只到 1.88、样本也因此更差，不过仍然很好玩：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
生成的样本像这样：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

在 CPU 上约 3 分钟就能有这种字符级模型的"神韵"雏形，还不错。如果你愿意多等，随意调超参、加大网络、加长上下文（`--block_size`）、加长训练等。

最后，在 Apple Silicon MacBook 上用较新 PyTorch 时记得加 `--device=mps`（Metal Performance Shaders 的缩写）；PyTorch 会用片上 GPU，训练能*显著*加速（2-3 倍）、也允许用更大网络。详见 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## 复现 GPT-2

更认真的深度学习从业者可能更想复现 GPT-2 的结果。那就开始——先对数据集分词，这里是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，对 OpenAI（私有的）WebText 的开放复现：

```sh
python data/openwebtext/prepare.py
```

这会下载并分词 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集，生成 `train.bin` 和 `val.bin`，把 GPT2 的 BPE token id 序列以原始 uint16 字节存成一条。然后就能开训。要复现 GPT-2 (124M)，你至少需要 8×A100 40GB 节点并运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这会用 PyTorch 分布式数据并行（DDP）跑约 4 天，loss 降到约 2.85。一个刚在 OWT 上评测的 GPT-2 模型 val loss 约 3.11，但微调后会降到 ~2.85 一带（因为有明显的领域差距），两个模型就大致持平了。

如果你在集群环境、运气好有多个 GPU 节点，可以让 GPU 嗡嗡转起来，例如跨 2 个节点：

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

最好先基准测一下互联带宽（例如 iperf3）。特别地，如果没有 Infiniband，给上面的启动命令前缀加 `NCCL_IB_DISABLE=1`。多节点训练能跑，但大概率会*慢如爬*。checkpoint 默认定期写入 `--out_dir`。采样只需 `python sample.py`。

最后，单 GPU 训练直接跑 `python train.py`。看看它的全部参数——脚本尽量做到可读、可魔改、透明。多半你需要按需求调其中不少变量。

## 基线

OpenAI 的 GPT-2 checkpoint 让我们能在 openwebtext 上定几条基线。数字如下：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

观察到 train/val 上如下 loss：

| 模型 | 参数量 | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

不过要注意，GPT-2 训练在（封闭、从未发布的）WebText 上，而 OpenWebText 只是对该数据集尽力而为的开放复现。这意味着数据集之间有领域差距。事实上，拿 GPT-2 (124M) checkpoint 直接在 OWT 上微调一段时间就能把 loss 压到 ~2.85。就"复现"而言，这才是更合适的基线。

## 微调

微调与训练没有本质不同，只是确保从预训练模型初始化、并用更小的学习率训练。想看在新文本上微调 GPT 的例子，去 `data/shakespeare` 跑 `prepare.py`，下载 tiny shakespeare 数据集并用 GPT-2 的 OpenAI BPE 分词器把它渲染成 `train.bin` 和 `val.bin`。与 OpenWebText 不同，这几秒就跑完。微调可能非常快，例如单 GPU 上只要几分钟。跑一个微调例子：

```sh
python train.py config/finetune_shakespeare.py
```

这会加载 `config/finetune_shakespeare.py` 里的配置覆盖（不过我没怎么调它们）。基本上，我们用 `init_from` 从 GPT2 checkpoint 初始化、照常训练，只是更短、学习率更小。如果显存不够，试着减小模型尺寸（可选 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）或减小 `block_size`（上下文长度）。最优 checkpoint（最低验证 loss）会在 `out_dir` 目录里，例如按配置默认在 `out-shakespeare`。然后可以跑 `sample.py --out_dir=out-shakespeare`：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇哦 GPT，你这是走进了什么黑暗角落。我其实没怎么调配置里的超参，随便试！

## 采样 / 推理

用 `sample.py` 既可以对 OpenAI 发布的预训练 GPT-2 模型采样，也可以对自己训练的模型采样。例如，下面用最大的可用模型 `gpt2-xl` 采样：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

想对自己训练的模型采样，用 `--out_dir` 指向相应目录。也可以用文件里的文本给模型喂提示，例如 ```python sample.py --start=FILE:prompt.txt```。

## 效率笔记

做简单的模型基准与 profiling，`bench.py` 可能有用。它与 `train.py` 训练循环核心做的事完全一样，只是省掉了许多其他复杂度。

注意本仓库默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。写这段时（2022 年 12 月 29 日），nightly 版里已经有 `torch.compile()`。这一行代码带来的提升很明显，例如把每步迭代时间从 ~250ms 砍到 135ms。PyTorch 团队干得漂亮！

## 待办

- 调研并用 FSDP 替代 DDP
- 在标准评测上评估零样本困惑度（LAMBADA？HELM？等）
- 把微调脚本的超参调好，我觉得现在不太好
- 训练中线性增大 batch 的调度
- 换用其他位置编码（rotary、alibi）
- 我觉得应该把 optim buffer 和模型参数在 checkpoint 里分开
- 围绕网络健康的额外日志（例如梯度裁剪事件、幅度）
- 关于更好初始化等的更多调研

## 故障排查

注意本仓库默认用 PyTorch 2.0（即 `torch.compile`）。这相当新且实验性，并非所有平台都支持（例如 Windows）。如果遇到相关报错，加 `--compile=False` 关掉它。代码会变慢，但至少能跑。

想对本仓库、GPT 与语言建模多些背景，看我的 [Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)可能有帮助。特别是 [GPT 那期视频](https://www.youtube.com/watch?v=kCc8FmEb1nY)，如果你已有一些语言建模背景会很对胃口。

更多问题/讨论，欢迎到 Discord 的 **#nanoGPT** 频道：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有 nanoGPT 实验都由我最爱的云 GPU 供应商 [Lambda labs](https://lambdalabs.com) 的 GPU 驱动。感谢 Lambda labs 赞助 nanoGPT！
