# Windows 多逻辑指针 PoC：第一轮结果

> 目标：验证 `CreateSyntheticPointerDevice` / `InjectSyntheticPointerInput` 能否在
> 不改目标应用的前提下提供 N 个独立指针。
> 环境：Windows 10.0.26300，非管理员，Python 3.12 ctypes（本机无 C++ 编译器）。

## 1. 结论

| # | 指标 | 结果 |
|---|---|---|
| M1 | 创建 10 个指针，独立 Pointer ID | **通过**：50 条 `WM_POINTER`，10 个 ID（如 116–125） |
| M2 | 同时移动，坐标互不干扰 | **通过**：每个 ID 的 `ptPixelLocation` 独立变化 |
| M3 | 同时点击，按键状态不串扰 | **通过**：每个 ID 各自 `DOWN → UPDATE → UP` |
| M4 | 两个窗口同时接收各自输入 | **未通过**：10 个触点按 5/5 分到两窗，两窗均 0 条消息 |
| M5 | 传统鼠标窗口 | **未通过**：只收到 5/10 个指针，**0 条 WM_MOUSE**（无 mouse promotion） |

**最重要的负面结果**：`InjectSyntheticPointerInput` 在“没送达”时仍然返回 `TRUE`。
M1 里 10 个触点成功；同样的代码换成 3/4/5/6/8/9 个触点，全部 `TRUE` 但**窗口一条消息都收不到**。
这是一个静默失败接口，必须靠消息面观测来判定，不能看返回值。

## 2. 实测细节

### M1–M3：单窗口 10 触点

```
N=10 points x=200..965 (85px 间距), y=420
inject: down/move/up 全部 ok=True err=0
pointer_msgs=50, distinct_ids=10
每个 ID 序列：WM_POINTERENTER → DOWN → UPDATE → UP → LEAVE
```

- 只有一个指针带 `POINTER_FLAG_PRIMARY`（0x2000），其余没有；
- 坐标按触点独立，移动后 y 从 420 → 510；
- 没有 `WM_TOUCH`，也没有 `WM_MOUSE`（因为窗口处理了 `WM_POINTER` 并返回 0）。

### 数量阈值

| 触点数 | 消息数 |
|---|---|
| 3 / 4 / 5 / 6 / 8 / 9 | **0** |
| 10 | **50** |

10 个触点但间距改成 30px（聚在 270px 内）→ **0**。
目前只在“**10 个触点 + 单窗口 + 85px 间距**”这一个组合下复现成功。

### M4：两窗口各 5 触点

左右两个 topmost 窗口，10 个触点一帧注入（左 5 / 右 5）：

```
left: pointer=0 ids=[]
right: pointer=0 ids=[]
```

分窗帧整体被丢弃；单窗口 10 触点可以，跨窗口不行。

### M5：传统窗口（不处理 WM_POINTER）

10 触点注入后：

```
pointer=25, ids=5 (127–131), 每个 ID 5 条（ENTER/DOWN/UPDATE/UP/LEAVE）
mouse=0
```

传统窗口靠 `DefWindowProc` 只能拿到 **5/10** 个指针，且**没有任何鼠标消息**。
也就是说：纯鼠标应用（大量 Win32/WinForms/老 Electron）**看不到这些触点**。

## 3. 对目标的裁定

| 目标 | 裁定 |
|---|---|
| 同一桌面 N 个独立指针 | **有条件成立**：N=10、单窗口、pointer-aware 应用 |
| 每个指针独立坐标/按键/生命周期 | 成立（M1–M3） |
| 不同指针同时操作不同窗口 | **未成立**（M4） |
| 兼容 Win32/WPF/WinForms/Electron | **不成立**：传统鼠标路径收不到；pointer-aware 路径待测 |
| 不改目标应用源码 | 成立，但代价是“只对已经支持 WM_POINTER 的应用有效” |

## 4. 下一步

1. **搞清 10 触点阈值**：换一台机器/另一个 Windows build 复测；查 `maxCount` 与帧完整性的关系；
2. **测真实 pointer-aware 应用**：WPF 的 `TouchDown`、Chromium 的 `pointerdown`（pointerType=touch），看是否真能拿到 10 个 `pointerId`；
3. 若“跨窗口”是硬需求，合成指针这条路不够，需要 HID/驱动层（本轮明确不做）；
4. 保留本轮的负面结论：**API 成功 ≠ 输入送达**，任何后续实现都必须带消息面 oracle。

## 5. 复现

```bash
python experiments/multipointer/run_m1.py 10   # M1–M3
python experiments/multipointer/run_m4.py      # M4（两窗口）
python experiments/multipointer/run_m5_legacy.py 10  # M5（传统窗口）
```

产物：`results/multipointer/m1_10.jsonl`、`m4_left.jsonl`、`m4_right.jsonl`、`m5_legacy_10.jsonl`。
