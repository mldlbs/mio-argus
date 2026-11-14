# 鸭子守卫：图标 + 红点识别（能力层）

> 代码：`experiments/computer_use/ui_icons.py`　模板：`experiments/computer_use/duck_icons/`
> 接入：`experiments/computer_use/duck_guard.py`（`nav()` / 每轮 `badges` 感知 / `--icons`）
> 复现：`python experiments/computer_use/ui_icons.py --image <截图> --selftest`

## 1. 为什么单独做一层

原来的守卫没有任何图标识别：导航是 5 个写死的归一化点，爱心蛋是 4 个写死点，
红点在代码里根本不存在。OCR 只认字，红点/图标没有字；写死坐标又会在窗口改宽高、
UI 重排、角标数字变化时静默失效。所以补的是**「外观 → 名字」的绑定**，不是再调 VLM：

- 图标：灰度 NCC（`TM_CCOEFF_NORMED`，对亮度线性变化不敏感）+ 多尺度模板；
- 红点：HSV 红色掩码 + 连通域 + 形状过滤 + 最近图标关联；
- 防幻觉：局部标准差闸门（背景水印/纯色块会让 NCC 退化到 1.0）。

## 2. 自测（截图 502×889，2026-11-09）

| 条件 | 5 个导航图标 | 分数 | 位置误差 |
|---|---|---|---|
| 原图 | 5/5 | 1.000 | 0 px |
| 缩到 0.8× | 5/5 | 0.952–0.986 | ≤ 0.9 px |
| 放到 1.2× | 5/5 | 0.974–0.991 | ≤ 0.6 px |
| 压暗 0.6×−20 | 5/5 | 0.848–0.999 | — |

红点（角标）：

| 角标 | 位置 | 大小 | 关联 |
|---|---|---|---|
| 蛋 上的红点 | (203.6, 756.8) | 16×16 | **egg**，16.2 px |
| 任务 上的红点 | (384.5, 756.6) | 16×16 | **task**，27.8 px |

- 未关联的红色块（爱心装饰、DAILY PASS、右侧 “+2”）在 JSON 里保留，但调试图不画。
- 维护页（无导航栏）：**0 图标、0 红点**——不把水印/按钮误认成图标。

> 调试图不入库（整窗截图含账号信息）：运行
> `python experiments/computer_use/ui_icons.py --debug` 会在本地生成
> `results/computer_use/duck_icons_debug.png`。

## 3. 接入后的行为

- `Game.nav(key)`：先识别图标再点它的中心；识别不到才回退旧坐标，并在事件里记
  `source=icon/coord`、`score`。
- 每轮开始输出 `红点: egg(...) / task(...)`，写进 status 的 `badges` 和
  `duck_guard_events.jsonl`。
- `python experiments/computer_use/duck_guard.py --icons`：只识别、不动鼠标，打印
  JSON + 写调试图。

## 4. 局限

- 模板是**单主题单尺度**采集的；换主题/换分辨率后要重新 `--harvest`（缩放 ±20% 内已验证）。
- 只识别**位置**，不读角标数字（65/4）；数字对策略没用，需要时再上小区域 OCR。
- 爱心蛋还没有模板（当前账号没出现该入口），`LOVE_ICONS` 仍是回退路径。
- 维护页期间整条导航栏不存在，守卫会走回退坐标空点——这是旧行为，未改策略。
