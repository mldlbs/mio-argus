# Bigemap 执行基准 Runbook

> 从零到第一次跑通。协议见 [BIGEMAP_EXEC_BENCH.md](BIGEMAP_EXEC_BENCH.md)，
> 结果见 `results/computer_use/BIGEMAP_BENCH_RESULTS.md`。

## 1. 前置条件

| 项 | 要求 | 检查方式 |
|---|---|---|
| Bigemap | 已安装、已授权、**正在运行** | 主窗口标题含 `Bigemap GIS Office` |
| 显示 | 2560×1440，DPI 100%，任务栏不自动隐藏 | `--preflight` 的 screen |
| 窗口 | 主窗摆在 `(1344,16,2544,916)`（runner 会自动摆） | `--preflight` 的 window_rect |
| GDAL/OGR | `gdalinfo.exe` / `ogr2ogr.exe` 可执行 | `--preflight` 的 gdalinfo/ogr2ogr |
| AOI | T4/T5 需要画布上**有一个蓝色 AOI 矩形** | `--preflight` 的 aoi |
| Python | `pywinauto`、`pyautogui`、`pywin32`、`numpy`；A/C 臂另需 `rapidocr_onnxruntime` | `--preflight` 的 dep:* |
| 会话 | 真实交互控制台（非 RDP 断开、非 Session 0） | `SendInput` 才有效 |

AOI 不在时：用 `import_shp_verified.py` 导入 `artifacts/qgis_real/bigemap/vector/sjs_admin_2026_wgs84_aoi.geojson`
对应的 SHP，或直接在画布上画一个矩形并确认它有蓝色边框。

## 2. 配置

`experiments/computer_use/bench_config.json`：

| 字段 | 含义 |
|---|---|
| `app_title_key` | 主窗标题关键字 |
| `screen` | 期望分辨率 `[w,h]` |
| `window_rect` | canonical 窗口 `[L,T,R,B]` |
| `download_root` | Bigemap 下载根（保存路径在应用内改不了） |
| `gdalinfo` / `ogr2ogr` | GDAL 二进制路径 |
| `aoi_bounds` | 基准 AOI（T5 校验用） |
| `t6_default_region` | T6 默认区域路径 |

换机器只改这个文件，不要改脚本。

## 3. 预检

```bash
python experiments/computer_use/bigemap_runner.py --preflight
```

输出示例：

```
CHECK                      TAG  DETAIL
dep:pywinauto              PASS
screen                     PASS 2560x1440（期望 2560x1440）
app                        PASS Bigemap GIS Office
window_rect                PASS [1344, 16, 2544, 916]
aoi                        WARN size=None（T4/T5 需要）
gdalinfo                   PASS D:\QGIS\bin\gdalinfo.exe GDAL 3.13.3
...
15/15 ok, fatal=0
```

- `FAIL`：必须修，否则不要开跑；
- `WARN`：只影响对应任务（如 `aoi` 只挡 T4/T5，不挡 T6）。

## 4. 运行

```bash
# 基线
python experiments/computer_use/bigemap_runner.py --task T1 --trials 5
python experiments/computer_use/bigemap_runner.py --task T5 --trials 30 --arm B
python experiments/computer_use/bigemap_runner.py --task T6 --trials 30 --arm B --region "河北省/石家庄市/长安区"

# 故障（T3/T5/T6 支持 F1/F4/F6；T2 支持 F8）
python experiments/computer_use/bigemap_runner.py --task T3 --trials 30 --arm B --inject F4
python experiments/computer_use/bigemap_runner.py --task T2 --trials 30 --arm A --inject F8

# 汇总
python experiments/computer_use/exec_bench.py --log results/computer_use/xxx.jsonl \
  --json results/computer_use/xxx_summary.json
```

臂：`A` 像素无恢复 / `B` UIA+恢复 / `C` 像素+恢复 / `S` 共用（窗口、画布）。
每次运行都会写 `results/computer_use/bigemap_bench_<arm>_<时间>.jsonl`。

## 5. 读结果

- 一行一步 + 一行一 trial；`exec_bench.py` 会校验字段、时间闭合、枚举；
- 关键指标：`SR`（任务成功率）、`SFR`（静默失败率）、`HR`（按控制类）、`RR`（故障恢复率）、中位墙钟；
- 校验失败（`[warn]`）说明记录不合法，先修再汇总。

## 6. 故障排查（都是实际踩过的）

| 现象 | 原因 | 处理 |
|---|---|---|
| 找不到主窗口 | 应用没起 / 标题变了 | 先 `--preflight` |
| T4/T5 卡在 `aoi` | 画布上没有蓝色 AOI 矩形 | 重新导入边界；`--preflight` 应显示 PASS |
| 关窗后下一次 AOI 检测假失败 | 地图重绘有 100–200 ms 空窗 | 已内置 3×120 ms 重试 |
| OCR 读到空白 | 对话框首帧未画完 | 已内置 0.35 s 绘制等待 |
| 任务名粘贴失败 | 剪贴板被别的进程占用 | `set_clipboard` 已重试 6 次 |
| 点击「确定」没反应 | 未聚焦窗口的第一下点击被吃掉 | 用 UIA `Invoke`；鼠标点击前先 `restore_window` |
| 产物 8 字节 | Bigemap 先建占位文件 | 已要求 >10 KB + `gdalinfo` 重试 |
| 面板下载图标点不到 | 面板随组合框行上下移动 | 已改为锚定面板标题文字 |
| Esc 后区域面板消失 | 面板不响应 `WM_CLOSE`，Esc 会关掉 | 不要用 Esc 清理；用 `--reset` 逻辑 |
| 保存路径改不了 | 应用禁止 UIA 写该字段 | 保持默认，用唯一任务名子目录 |

## 7. 已知缺口

- **F7（网络/代理）未覆盖**：需管理员或人工断网；
- T6 乡镇级（第 5 级）未接；
- 影像的“任意 AOI 导入”未接进 runner；
- T6 的 A/C 臂、T5v（换 AOI/级别）、部分遮挡未测；
- 结论限定：单机、单版本、2560×1440、30 次/臂。
