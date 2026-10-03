> 软件项目文档链 · 7 件套  
> 上游：[状态模型](docs/taskhub/c3ecabb0/data-model.md) | 下游：[测试验收](docs/taskhub/c3ecabb0/test.md)  
> 约定：正文**分点 + 换行**（每点一行，`-` 列表）；指引注释处填内容，未用章节删除。

# 接口契约（API Contract）

## 文档信息

| 项 | 值 |
|---|---|
| 文档版本 | v0.1 |
| 最后更新 | 2026-10-03 19:40 |
| 状态 | approved |
| 负责人 | opencode |
| 适用范围 | 接环线全部进程内接口：qgis_bridge 调用（truth/win/set_extent/act/reset）、Brain.predict 契约、结果 JSON 输出。明确不含：公网 HTTP 服务、模型权重文件格式、QGIS 内部 API |

## 1. 文档信息与范围
| 项 | 值 |
|----|----|
| 契约版本 | v1 |
| 适用范围 | loop_w28.py 及后续 W29–W31 脚本；qgis_bridge 的既有 ops（只消费不改） |
| 明确不含 | 公网 HTTP 接口、数据库直连、模型训练接口 |

## 2. 环境与基础地址
| 环境 | Base URL | 说明 |
|------|----------|------|
| dev | `127.0.0.1:4848`（qgis_bridge，TCP+JSON） | 本机 QGIS 进程内桥；端口以 `client.py` 为准 |
| staging | 不适用 | 无多环境 |
| prod | 不适用 | 本地实验 |

- 路径前缀：无（单 socket 端口，JSON `{"op": ...}` 请求）。
- 协议与端口：TCP + UTF-8 JSON；qgis_bridge 默认端口见 `experiments/qgis/qgis_bridge/client.py`。
- 网关 / 反向代理差异：无。
- 字符编码：UTF-8
- 内容类型：`application/json; charset=utf-8`

## 3. 版本策略
- 版本位置：无版本号（本机开发接口；ops 向后兼容追加）。
- 当前版本：v1（W28 使用集）。
- 版本号规则：新增 op=兼容；改既有 op 语义=破坏（需同步本文件）。
- 多版本并存策略：不适用。
- 旧版本弃用流程：不适用。

## 4. 认证与鉴权
| 项 | 约定 |
|----|------|
| 认证方式 | 无（本机回环地址） |
| 凭证来源 | — |
| 传递位置 | — |
| 有效期 | — |
| 刷新方式 | — |
| 权限模型 | 无 |
| 未认证响应 | 不适用 |
| 越权响应 | 不适用 |

## 5. 通用请求头
| Header | 必填 | 类型 | 默认 | 说明 |
|--------|------|------|------|------|
| 不适用 | — | — | — | 进程内 socket JSON，无 HTTP 头 |

## 6. 通用响应结构
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ok` | boolean | 是 | 是否成功 |
| `err` | string | 失败时 | 错误描述（如 `no canvas` / `unknown op`） |
| 其余字段 | any | 否 | 各 op 的返回值（见 §18） |

## 7. 统一错误码
| HTTP | 业务码 | 含义 | 触发条件 | 处理建议 |
|------|--------|------|---------|---------|
| — | `no canvas` | 画布不可用 | QGIS 未就绪/工程异常 | 重启 QGIS 后重试 |
| — | `no main window` | 主窗口不可用 | QGIS 未启动 | 先启动 QGIS + bridge |
| — | `action not found/enabled` | 工具栏动作不可用 | 名称错/被禁用 | 用 `find_actions` 核对名称 |
| — | `unknown op` | op 不存在 | 拼写错 | 对照 §17 清单 |
| — | 连接拒绝 | bridge 未启动 | QGIS 未带 PYQGIS_STARTUP | 按 runbook 启动 QGIS |

## 8. 分页 / 排序 / 过滤
| 项 | 参数 | 类型 | 默认 | 上限 | 说明 |
|----|------|------|------|------|------|
| 页码 | 不适用 | — | — | — | 无分页接口 |
| 每页条数 | 不适用 | — | — | — | — |
| 排序字段 | 不适用 | — | — | — | — |
| 排序方向 | 不适用 | — | — | — | — |
| 过滤条件 | 不适用 | — | — | — | — |

- 分页响应字段：不适用。
- 游标 vs 偏移的选择理由与限制：不适用。
- 超限行为：不适用。
- 多字段排序的优先级规则：不适用。

## 9. 字段命名与类型规范
| 项 | 规范 |
|----|------|
| 命名风格 | snake_case（Python/JSON 一致） |
| ID 类型 | string（8 位 hex，taskhub 任务 id） |
| 布尔 | JSON true/false；Python bool |
| 金额 | 不适用 |
| 数组 | 空数组 `[]` 与 null 区分：坐标等用固定长度数组，不用 null |
| 可选字段 | 字段缺失=未采集；显式 null=采集了但无值（如 `fail: null` 表示成功） |

## 10. 时间 / 数值 / 空值约定
- 时间格式：ISO 8601（如 `2026-10-03T19:40:00`）。
- 时区：本机时间（无时区后缀）或 UTC+0（taskhub 用 UTC）。
- 时间精度：秒。
- 数值范围与溢出处理：坐标 px 为 int（反投影后 round）；float 一律 Python float（IEEE754）。
- 空值语义：见 §9 可选字段。

## 11. 枚举全集
| 枚举 | 取值 | 含义 | 出现在哪些接口 |
|------|------|------|---------------|
| action | click / type / scroll / move / hotkey / press | 模型动作空间（6 类） | Brain.predict 输出、demo.action |
| scroll | up / down / none | 滚动方向（3 类） | Brain.predict 输出 |
| fail | action_wrong / out_of_window / truth_fail / no_qgis_window / target_not_in_uia | 失败分类 | StepRecord.fail |
| op | truth / win / set_extent / act / reset / shot / close_dialogs / select_expr / zoom_to_selection / export / layouts | bridge 操作 | bridge 调用 |
| scene | real / regions / cities | 桥场景重置 | bridge reset |

## 12. 幂等性与重试
| 接口 / 场景 | 幂等 | 幂等键 | 重试建议 |
|------------|------|--------|---------|
| op=truth | 是（只读） | — | 可直接重试 |
| op=win（maximize） | 是 | — | 可直接重试 |
| op=set_extent | 是（设定值） | — | 可直接重试 |
| op=act（trigger action） | 否（如 Undo） | — | 不重试；由脚本按任务语义决定 |
| op=reset | 是（重建场景） | — | 可直接重试 |
| Brain.predict | 是（纯推理） | — | 可直接重试 |

- 安全重试的状态码：不适用（socket 层异常即重连）。
- 绝不可重试的状态码：不适用。
- 重试间隔与退避策略：socket 连接失败重试 3 次 × 1 s（client 侧约定）。
- 幂等键的生成规则、有效期、重复提交的响应：不适用。

## 13. 限流与配额
| 范围 | 阈值 | 窗口 | 超限响应 |
|------|------|------|---------|
| 本机进程 | 无显式限流 | — | — |

## 14. 超时与并发
- 服务端处理超时：bridge 单 op ≤ 5 s（truth 类毫秒级）。
- 建议客户端超时：10 s（`client.call` 默认）。
- 乐观锁 / 版本号字段：无。
- 并发冲突响应：无（单 QGIS 进程串行处理）。
- 长任务的处理方式：同步阻塞（本实验均为短 op）。

## 15. 文件上传与下载
| 项 | 约定 |
|----|------|
| 上传方式 | 不适用 |
| 单文件上限 | 不适用 |
| 允许类型 | 不适用 |
| 下载方式 | 不适用（导出走 bridge op=export 写本地文件） |
| 鉴权方式 | 不适用 |

## 16. 安全与脱敏
- 敏感字段清单及脱敏规则：无（本地实验数据）。
- 传输要求：回环地址，不出本机。
- 日志禁止记录的字段：无。
- 越权与注入防护：安全门（点击仅限 QGIS 窗口内）见 spec。

## 17. 接口清单
| Method | Path | 用途 | 权限 | 幂等 | 限流 | 关联状态 |
|--------|------|------|------|------|------|---------|
| CALL | `{"op":"truth"}` | 读工程/canvas 真值（layers/extent/scale/tool/renderer） | 无 | 是 | 无 | 任意 |
| CALL | `{"op":"win","maximize":true}` | 主窗口最大化/前置 | 无 | 是 | 无 | 任意 |
| CALL | `{"op":"set_extent","box":[x0,y0,x1,y1]}` | 设置画布范围（测试状态预置） | 无 | 是 | 无 | 任意 |
| CALL | `{"op":"act","kind":"action","text":"..."}` | 特权触发工具栏动作（期望值校准） | 无 | 否 | 无 | 任意 |
| CALL | `{"op":"reset","scene":"real"}` | 重置场景（清工程+载真实数据） | 无 | 是 | 无 | 任意 |
| FUNC | `Brain.predict(img, instruction)` | 模型推理（截图+指令→动作+坐标） | 无 | 是 | 无 | 任意 |

## 18. 接口明细

### CALL {"op":"truth"}

- 用途：读真值（判定闭环每步成败）
- 权限：无
- 幂等：是
- 副作用：无
- 关联状态：任意状态

#### 请求参数

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 | 说明 | 示例 |
|------|------|------|------|------|------|------|------|
| op | body | string | 是 | — | == "truth" | 操作名 | truth |

#### 响应字段

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `ok` | bool | 是 | 是否成功 | true |
| `truth.layers[]` | array | 是 | 图层信息（name/crs/features/selected/visible/renderer/extent） | — |
| `truth.extent` | [float×4] | 是 | canvas 可见范围 | [8.2e6, 2.0e6, 1.5e7, 7.1e6] |
| `truth.scale` | float | 是 | canvas 比例尺 | 1.07e7 |
| `truth.tool` | string | 否 | 当前地图工具名 | Pan Map |
| `truth.bg` | string | 否 | 画布背景色（#rrggbb） | #0a1220 |

#### 错误码

| HTTP | 业务码 | 含义 | 触发条件 | 处理建议 |
|------|--------|------|---------|---------|
| — | no canvas | 画布不可用 | QGIS 工程异常 | 重启 QGIS |
| — | 连接拒绝 | bridge 未启动 | 无 PYQGIS_STARTUP | 按 runbook 启动 |

#### 示例

请求：

```json
{"op": "truth"}
```

响应（200）：

```json
{"ok": true, "truth": {"extent": [8193930, 2063112, 15000562, 7086440], "scale": 10726445.9, "tool": "Pan Map"}}
```

### CALL {"op":"set_extent"}

- 用途：预置画布范围（为闭环任务制造初始状态）
- 权限：无
- 幂等：是
- 副作用：改变 canvas 视图（不改变数据）
- 关联状态：任意状态

#### 请求参数

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 | 说明 | 示例 |
|------|------|------|------|------|------|------|------|
| op | body | string | 是 | — | == "set_extent" | 操作名 | set_extent |
| box | body | [float×4] | 是 | — | x0<x1, y0<y1（图层 CRS 单位） | 目标范围 | [9.0e6,3.3e6,1.38e7,5.8e6] |

#### 响应字段

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `ok` | bool | 是 | 是否成功 | true |

#### 错误码

| HTTP | 业务码 | 含义 | 触发条件 | 处理建议 |
|------|--------|------|---------|---------|
| — | no canvas | 画布不可用 | 工程异常 | 重启 QGIS |
| — | KeyError box | 缺参数 | 未传 box | 补参数 |

#### 示例

请求：

```json
{"op": "set_extent", "box": [9000000, 3300000, 13800000, 5800000]}
```

响应（200）：

```json
{"ok": true}
```

### CALL {"op":"act"}

- 用途：特权触发工具栏动作（期望值校准，如 Zoom Full / Zoom to Native）
- 权限：无
- 幂等：否（如 Undo 类动作）
- 副作用：执行 QGIS 动作（异步触发）
- 关联状态：任意状态

#### 请求参数

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 | 说明 | 示例 |
|------|------|------|------|------|------|------|------|
| op | body | string | 是 | — | == "act" | 操作名 | act |
| kind | body | string | 是 | — | == "action" | 操作类别 | action |
| text | body | string | 是 | — | 与工具栏动作文本完全一致（& 已去除） | 动作名 | Zoom Full |

#### 响应字段

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `ok` | bool | 是 | 是否成功 | true |
| `did` | string | 成功时 | 执行描述 | trigger_async:Zoom Full |

#### 错误码

| HTTP | 业务码 | 含义 | 触发条件 | 处理建议 |
|------|--------|------|---------|---------|
| — | action not found/enabled | 动作不可用 | 名称错/被禁用 | find_actions 核对 |
| — | no main window | 主窗不可用 | QGIS 异常 | 重启 QGIS |

#### 示例

请求：

```json
{"op": "act", "kind": "action", "text": "Zoom to Native Resolution (100%)"}
```

响应（200）：

```json
{"ok": true, "did": "trigger_async:Zoom to Native Resolution (100%)"}
```

### FUNC Brain.predict(img, instruction)

- 用途：模型推理（截图+指令 → 动作/坐标/滚动）
- 权限：无
- 幂等：是（eval 模式）
- 副作用：无
- 关联状态：任意状态

#### 请求参数

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 | 说明 | 示例 |
|------|------|------|------|------|------|------|------|
| img | arg | PIL.Image | 是 | — | RGB；建议 16:9（训练为 1920×1080） | 全屏截图 | — |
| instruction | arg | string | 是 | — | 英文短句 | 任务指令 | click the Zoom Full button |

#### 响应字段

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `action` | string | 是 | 动作（见 §11） | scroll |
| `coord` | [float,float] | 是 | 归一化坐标 ∈[0,1]² | [0.15, 0.04] |
| `scroll` | string | 是 | 滚动方向（见 §11） | none |
| `ms` | float | 是 | 推理耗时（ms） | 42.1 |

#### 错误码

| HTTP | 业务码 | 含义 | 触发条件 | 处理建议 |
|------|--------|------|---------|---------|
| — | CUDA OOM | 显存不足 | 并发/大图 | 释放显存或改 CPU |
| — | checkpoint 缺失 | 权重不存在 | 路径错 | 核对 artifacts/checkpoints/ |

#### 示例

请求：

```python
pred = brain.predict(img, "click the Zoom Full button")
```

响应（200）：

```json
{"action": "scroll", "coord": [0.0005, 0.0], "scroll": "none", "ms": 41.7}
```

## 19. 兼容性与废弃策略
- 什么算破坏性变更：改既有 op 语义、改 StepRecord/DemoSample 字段含义、删枚举值。
- 什么算兼容变更：新增 op、新增可选字段、新增枚举值。
- 变更通知方式与提前期：本文件 §20 变更记录 + 任务讨论。
- 废弃流程：标记 → 双写/双读 → 下线（本实验线从简：一次性切换 + 记录）。
- 契约校验方式：结果 json 与本文档字段对照（W28 实测已核对）。

## 20. 变更记录
| 日期 | 契约版本 | 变更内容 | 类型 | 影响方 |
|------|---------|---------|------|--------|
| 2026-10-03 | v1 | 首版（W28 使用集：truth/win/set_extent/act/reset + Brain.predict + StepRecord） | 兼容 | loop_w28.py 及后续 W29–W31 |

## 21. 附录
- HTTP 状态码全集：不适用（进程内 socket）。
- 业务错误码全集：见 §7（no canvas / no main window / action not found/enabled / unknown op / 连接拒绝）。
- 枚举全集：见 §11（action / scroll / fail / op / scene）。
- 数据字典 / 术语表：StepRecord / DemoSample 见 [状态模型](docs/taskhub/c3ecabb0/data-model.md)；bridge 术语（extent/scale/tool）见 QGIS_FINDINGS。
- 相关文档：状态模型 `docs/taskhub/c3ecabb0/data-model.md`、测试验收 `docs/taskhub/c3ecabb0/test.md`、模块 Spec `docs/taskhub/c3ecabb0/spec.md`。
