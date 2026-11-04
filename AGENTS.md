## 文件纪律（强制）

1. **写任何新代码/新文档之前，必须先在 `STRUCTURE.md` §2 归属表里为文件定位置**；表中没有的文件类型，先补表再写文件。不允许直接往根目录放文件。
2. 实验产物（结果 json、`*_RESULTS.md`）只能写到 `results/<line>/`；数据集与大文件只能放 `artifacts/`（gitignored）；手写文档只能放 `docs/`。
3. 迁移或新建路径时，改完必须验证：`rg` 无旧路径残留、`python -m pytest -q` 与基线一致（在线基线：13 passed / test_model 1 failed / test_vm_e2e error；设 `TRANSFORMERS_OFFLINE=1` 后 test_vm_e2e 可通过，为 14 passed / 1 failed）。
4. 目录结构变更（移动/改名/新增顶层目录）需先改 `STRUCTURE.md` 并在此登记，再动文件。

<!-- MIO_CONTEXT_BEGIN -->
## Mio 最近上下文（自动注入）
- 2026/11/4 11:26:22 | digest(7d): nanoGPT 5 任务 100% 成功
- 2026/9/22 21:09:38 | success | 已创建 PLASTICITY.md （设计草案 v0.1，未写实现代码）。 核心设计 ：把"可塑性"这个模糊概念压成唯一可能成立的命题—— > 快速权重与线性注
- 2026/9/22 21:04:47 | success | 这是一份"宣言式"架构。诊断（Transformer 的六个缺陷）基本成立，但处方需要按项目的纪律审一遍。先说结论，再逐条对证据。 结论先行 | 维度 | 评价
- 2026/9/19 18:31:15 | success | 收尾完成。 per-R 复训尝试（为归因 mono-R@576 崩溃） ：未成功，已如实记录。
- 从零训练：train recall 仅 0.025（size
- 2026/9/19 16:36:30 | success | 路线 b 已闭环，结论翻转。 实现 ：给 p9_active_zoom.py 补 mono-R 臂——用 _at_res(R) 临时切换渲染画布、按 R/192
- 2026/9/19 09:21:03 | success | P9（路线 b）已实现并跑完。新增 p9_active_zoom.py 、 P9_RESULTS.md 、 p9_active_zoom_results.jso
<!-- MIO_CONTEXT_END -->
