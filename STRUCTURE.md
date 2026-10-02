# 目录结构规范（STRUCTURE）

> **强制规则**：写任何新代码/新文档之前，必须先在本文件为它定位置（查 §2 归属表）。
> 表里没有的文件类型 → 先补表、再写文件。规则本身见 `AGENTS.md`。

## 0. 阶段状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 1 | 文档 → `docs/`、结果产物 → `results/<line>/`、数据与大文件 → `artifacts/`；脚本留根目录，只改路径字符串 | ✅ 已完成 |
| 阶段 2 | 实验脚本按线迁入 `experiments/`、部署迁入 `infra/`、mio 归组；根 `conftest.py` + 8 处 CLI 路径引导修跨线 import；winsim 结果改指 `results/winsim/` | ✅ 已完成 |
| 阶段 3 | 收尾：`W1_RESULTS.md` 归位 `results/qgis/`、qgis_bridge 启动引导旧绝对路径改 HERE 优先、桥内草稿不入库、本文件树同步 | ✅ 已完成 |

## 1. 当前布局（阶段 2 完成态）

```
nanoGPT/
├── AGENTS.md  STRUCTURE.md  README.md  LICENSE
├── conftest.py                             # pytest 路径引导（根 + mio/ + infra/）
│
├── model.py  train.py  sample.py  bench.py  configurator.py   # nanoGPT 核心（唯一可留根的 py）
├── scaling_laws.ipynb  transformer_sizing.ipynb
├── config/  data/  assets/                 # 核心配置/数据/素材
├── out-*/                                  # 训练输出（gitignored，运行时产生）
│
├── docs/                                   # 手写文档（规范、架构、报告）
│   ├── ARCHITECTURE.md  REPORT.md  HYPOTHESIS.md  ROADMAP.md
│   ├── PLASTICITY.md  MIO_ENV.md  VM_TESTING.md
│   └── EVAL_REPORT_TEMPLATE.md  EVAL_REPORT.md
│
├── results/                                # 实验产物（json 由脚本生成，勿手改）
│   ├── paper/          # P0–P9：*_results.json + P*_RESULTS.md
│   ├── plasticity/     # Q1–Q4：q*_results.json + Q*_RESULTS.md
│   ├── web/            # web_*/site_*/real_site_* 结果 + WEB_*.md + SITE_HOLDOUT.md + REAL_SITE.md
│   ├── computer_use/   # eval_results.json
│   ├── winsim/         # WINSIM.md + WINSIM_RESULTS.md + winsim_w*_results.json
│   └── qgis/           # QGIS_FINDINGS.md + W1_RESULTS.md（qgis_bridge 运行产物就地，见 §3 例外）
│
├── experiments/                            # 实验脚本按线分目录（线内扁平 import）
│   ├── paper/          # p0–p9 脚本 + p0_env.py/p0_models.py/p3_env.py（线内共享）+ *_report.py
│   ├── plasticity/     # q1_plasticity / q2_plasticity_query / q3_learned_gate / q4_importance_gate
│   ├── computer_use/   # collect_* / make_synthetic_data / split_data / model_computer_use
│   │                   # train_computer_use* / eval_model / gen_report / vlm_adapter / test_*
│   ├── winsim/         # winsim_w0..w12.py + winsim/index.html（结果写 results/winsim/）
│   ├── web/            # web_*.py  site_holdout.py  real_site_*.py  SITES.json
│   └── qgis/           # qgis_bridge/（自包含：脚本 + 数据 + 运行产物就地）
│
├── artifacts/                              # 大文件与数据集（全部 gitignored）
│   ├── computer_use_data*/  synthetic_data/
│   ├── checkpoints/     # computer_use_model*.pth
│   ├── p0_cache/  hf_cache/
│
├── infra/                                  # 部署与环境
│   ├── Dockerfile  docker-compose.yml  requirements*.txt
│   ├── api_server.py  run_vm_test.bat  setup_virtualbox.sh  collect_in_vm/
│
└── mio/                                    # mio_env.py  mio_bc.py  mio_argus.py
```

## 2. 文件归属表（新文件先查这里）

| 文件类型 / 产物 | 位置 | 约定 |
|---|---|---|
| nanoGPT 核心模型/训练/采样 | 根目录 | 只有核心五件套（model/train/sample/bench/configurator）可留根 |
| 新实验脚本 | `experiments/<line>/` | 先确认所属线；线内模块扁平 import，跨线依赖必须加路径引导并在本表登记 |
| 实验结果 json | `results/<line>/` | 脚本写出路径必须指向此处，勿写根目录 |
| 实验结果/发现文档（`*_RESULTS.md`、`*_FINDINGS.md`） | `results/<line>/` | 与其 json 同目录 |
| 规范/架构/路线/总报告类手写文档 | `docs/` | 禁止放根目录 |
| 数据集、checkpoint、缓存 | `artifacts/` | 不入 git；`.gitignore` 以 `artifacts/` 整体登记 |
| 部署文件（Docker、api、依赖清单、bat/sh） | `infra/` | compose 相对路径以 `infra/` 为基准，build context 指回仓库根 |
| mio 相关脚本 | `mio/` | 跨线用 `p0_env` 时在文件头加路径引导 |
| 临时探索脚本 | 不入仓库 / 用后即删 | 不允许在根目录堆积未登记脚本 |

## 3. 例外与遗留

- **qgis_bridge 自包含**：`experiments/qgis/qgis_bridge/` 内的 `w*_results.json`、`*_actor_data.npz`、`.pt`、临时截图等运行产物**就地保存**（HERE 链读写、波次间 npz 交接），不迁 `results/qgis/`；发现/细目文档（`QGIS_FINDINGS.md`、`W1_RESULTS.md`）在 `results/qgis/`。
- **桥内草稿不入库**：`qgis_bridge/` 的 `_*.png`、`_*.log` 为运行时草稿（`.gitignore` 登记、`git rm --cached` 反跟踪），脚本运行时再生。
- **桥启动引导路径**：`_boot.py`/`_startup.py`/`bridge.py` 用 `__file__` 解析目录、绝对路径回退（QGIS `--code`/`PYQGIS_STARTUP` 执行时可能无 `__file__`）。
- `SITES.json` 实际随线数据放在 `experiments/web/`（代码内嵌 SITES 字典，该文件为原始输入数据）。
- `out-shakespeare-char/` 等 `out-*` 为 nanoGPT 训练输出，保持根级（README 有示例命令）。

## 4. 路径书写约定

- **一律从仓库根运行**（`python experiments/<line>/xxx.py`、`pytest`）：脚本读写仓库内文件用**相对仓库根**的路径字符串。
- 从任意 cwd 运行的脚本用 `Path(__file__)` HERE 风格（winsim、qgis_bridge）；HERE 系脚本写出结果时显式指到 `results/<line>/`（qgis_bridge 例外见 §3）。
- **跨线 import 约定**：pytest 走根 `conftest.py`（登记根、`mio/`、`infra/`）；CLI 直跑在文件头加 3 行路径引导（参考 `experiments/computer_use/eval_model.py`、`mio/mio_env.py`）。
- `artifacts/` 下目录名保持原名（如 `artifacts/computer_use_data/`），只加前缀，避免语义漂移。
- 容器内路径（如 `HF_HUB_CACHE=/app/hf_cache`）与宿主机路径分开，改宿主机路径时不动容器内路径。
