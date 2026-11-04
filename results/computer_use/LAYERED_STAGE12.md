# 阶段 1–2：基线冻结 + 分层视觉识别消融

> 复现（隔离环境，不训练任何模型）：
> ```
> set PYTHONPATH=<GUI-Actor-main>/src
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A1
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A3 --crop1 512
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A3 --crop1 512 --upscale 1024
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A4 --crop1 512 --crop2 256 --upscale 1024
> artifacts/audit_env/Scripts/python.exe experiments/computer_use/layered_eval.py --variant A5 --grid 2 --upscale 1024
> python experiments/computer_use/layered_report.py
> ```
> 数据：`artifacts/computer_use_inloop/audit_predictions/layered_*.json`

---

## 1. 阶段 1：基线冻结

| 项 | 值 |
|---|---|
| 样本 | 216 条（val 链 `w26_v2` + `w27`，带 rect 的 click） |
| 清单 | `audit_manifest.json`（图像、rect 已换算到截图空间） |
| 坐标 | 所有 provider 输出归一化 [0,1]，由同一脚本映射与判定 |
| 判据 | 点落在 target_rect 内即命中 |

固定基线（同一批样本）：

| 方案 | top-1 |
|---|---|
| 自研 grid@224 | 5.6% |
| 结构词面 | 15.7% |
| UGround-V1-2B | 49.1% |
| **GUI-Actor-2B（本报告的主干）** | **43.5%** |

## 2. 阶段 2：分层识别消融（全部零训练）

| 变体 | 做法 | 目标占 token | top-1 | 中位误差 | p50 |
|---|---|---|---|---|---|
| A1 | 整屏原生分辨率 | ~1.1 | 43.5% | 30 px | 2.1 s |
| A3 | A1 点为中心裁 512，**不放大** | ~1.1 | 41.2% | 26 px | 2.3 s |
| **A3u** | 裁 512 → **放大到 1024** | **~2.3** | **62.5%** | **14 px** | 3.5 s |
| A4 | A3u 后再裁 256 → 放大 1024 | ~4.6 | 62.5% | 26 px | 5.6 s |
| A5 | 2×2 重叠切片 + 放大 1024（不依赖粗定位） | ~2.3 | 52.8% | 27 px | 8.7 s |

token 数按 Qwen2-VL 的口径估算（patch 14，merge 2 → 28 px/token；32px 按钮在原图约 1.1 token）。

## 3. 三条结论

1. **裁剪本身无效（41.2%），放大才有效（62.5%）**。
   因为 Qwen2-VL 按原生像素处理，裁小不改变目标占的 token 数，反而丢了上下文；
   把裁剪图放大 2× 才让目标从 ~1.1 token 涨到 ~2.3 token。

2. **一级放大足够，二级放大净增益为 0**（A4 与 A3u 同为 135/216）：
   A4 相对 A3u 新增 4 条、丢失 4 条，纯属重新洗牌，代价是 1.6 倍延迟。
   拐点在 ~2.3 token，与文献"≈3 token 以上收益递减"一致。

3. **收益是真实的、不是交换**：A3u 救回 A1 失败的 **43/122 (35%)**，
   只打坏 A1 成功的 **2/94**。

4. **扫描式不如粗定位式**：A5（2×2 切片 + 放大）只有 52.8%，比 A3u 低 9.7 点，
   延迟却是 2.5 倍。说明 A1 的粗定位锚点是有价值的——先看一眼全图再放大，
   比盲扫所有切片更有效，也更便宜。

## 4. 对训练决策的意义

按"没有增益就不进入训练阶段"的规矩：

- **方向 A（分层视觉定位）**：已用现成模型验证，**不需要训练**。
  配方是「A1 粗定位 → 裁 512 → 放大 1024 → 二次定位」，成本 +1 次推理、+1.4 s。
- **方向 B（候选语义排序）**：成为下一个瓶颈。A3u 仍漏 81 条，
  其中 28 条是纯图标（所有视觉方案 0%），其余属于"选错控件"而非"点不准"。
- **方向 C（图标）**：仍然 0%，没有变化。

## 5. 局限

- 单应用（QGIS）、单屏尺度（2560×1440，截图 0.6 缩放存盘）。
- 裁剪尺度固定 512/256，未做自适应（按目标尺寸先验选择裁剪大小）。
- A5 只测了 2×2 切片，未测 3×3；但趋势（52.8% vs 62.5%，且慢 2.5 倍）不足以支持继续加切片。
- GUI-Actor 在 216 条上有 4 条 large 目标全部失败，样本不足，不解读。
