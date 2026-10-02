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

*以下为原始 nanoGPT README，保留英文原文以存真（MioArgus 基于其上构建）。*

---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

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

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

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

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
