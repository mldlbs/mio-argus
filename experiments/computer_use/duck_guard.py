"""DUCK x MY x DUCK 常驻看护脚本（guard）。

一直在跑的动作循环：
  1. 关掉挡路弹窗（番茄节 / 舞台下台 / 废弃南瓜车 / 升级物品说明）
  2. 喂鸭：只点单价 <= 200 的「点击喂食 N」，到 200 就停
  3. 合蛋：只在「同类型 + 同等级」之间拖拽（蛋 / 爱心蛋 不通用）
  4. 敲蛋：只敲 >= 12 级的蛋（用户策略），槽位蛋不敲
  5. 买商人蛋：只买 5 / 10 面具的，留足面具给槽位蛋
  6. 繁殖：鸭子「已准备好交配」且玉米充足时付 1000 玉米

安全约束：
  - 单价硬上限 200（CLICK 前二次校验）
  - 不买星星 / 订阅 / 星标鸭
  - 窗口不在前台就跳过本轮，不抢用户鼠标
  - 出现 results/computer_use/duck_guard.stop 立即退出

用法（仓库根目录）：
    python experiments/computer_use/duck_guard.py --minutes 120
    python experiments/computer_use/duck_guard.py --once        # 跑一轮就退出
    python experiments/computer_use/duck_guard.py --no-focus    # 不抢前台
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pyautogui  # noqa: E402
import win32con  # noqa: E402
import win32gui  # noqa: E402

import bigemap_runner as bench  # noqa: E402
from ui_icons import IconMatcher, associate, find_badges, grab, to_gray  # noqa: E402
from duck_perceive import classify_page  # noqa: E402


def no_failsafe() -> None:
    """关闭 pyautogui failsafe，改用 --stop / --health 作为人工刹车。

    必须放在所有 import 之后调用：bigemap_runner 在导入时会把 FAILSAFE 重置为 True
    （见 bigemap_runner.py 的 `pyautogui.FAILSAFE = True`），先设后导入等于没设。
    窗口钉在屏幕右上角，鼠标动作经常逼近角落，开着 failsafe 会整轮中止。
    """
    pyautogui.FAILSAFE = False


no_failsafe()

# ---------------------------------------------------------------- 风险与页面

# 动作 → 风险等级（docs/DUCK_EXEC_PROTOCOL.md §1）
RISK = {
    "nav": "L1",
    "collect": "L1",
    "feed": "L2",
    "merge": "L2",
    "buy": "L3",
    "breed": "L3",
    "crack": "L3",
    "slot": "L3",     # 花星星解锁槽位（用户已授权：星星只用于解锁槽位）
    "daily": "L2",    # 领 DAILY PASS 奖励（只领，不花）
    "ads": "L2",      # 任务页看广告领玉米（只领，不花）
    "sell": "L3",     # 市场挂单卖鸭换星星（用户规则：已喂满且 ≥UNCOMMON）
}

# 任务 → 必须在哪个页面执行
TASK_PAGE = {
    "collect": "duck",
    "feed": "duck",
    "merge": "egg",
    "crack": "egg",
    "buy": "egg",
    "breed": "duck",
    "slot": "duck",
    "daily": "duck",
    "ads": "task",
    "sell": "duck",
}

OUT_DIR = ROOT / "results" / "computer_use"
LOG_PATH = OUT_DIR / "duck_guard.log"
STATUS_PATH = OUT_DIR / "duck_guard_status.json"
STOP_PATH = OUT_DIR / "duck_guard.stop"
PID_PATH = OUT_DIR / "duck_guard.pid"
EVENTS_PATH = OUT_DIR / "duck_guard_events.jsonl"   # 过程数据（给分析/策略任务消费）
CONFIG_PATH = HERE / "duck_guard_config.json"        # 策略参数（下游可直接改，热加载）
GATE_PATH = OUT_DIR / "duck_gate_pass.json"          # 感知门禁（docs/DUCK_EXEC_PROTOCOL.md §5）

DEFAULT_CONFIG = {
    "tiers": {
        "routine": {"enabled": True, "interval": 3600, "tasks": ["collect", "feed"]},
        "upgrade": {"enabled": True, "interval": 3600, "tasks": ["merge"]},
    },
    "actions": {"feed": True, "collect": True, "merge": True,
                "crack": False, "buy": False, "breed": True},
    "allow_risk3": True,
    "feed_cap": 200,
    "feed_max_clicks": 400,
    "crack_min_level": 12,
    "merge_budget": 2,
    "claim_max_clicks": 3,
    "interval": 60,
}


def load_config() -> dict:
    """读取策略配置；文件不存在就落一份默认的。下游改文件即可改策略。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        if CONFIG_PATH.exists():
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        else:
            CONFIG_PATH.write_text(
                json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
                encoding="utf-8")
    except Exception as exc:  # 配置坏了不能拖垮看护
        log(f"配置读取失败，用默认值：{exc}")
    return cfg


def log_event(**fields) -> None:
    """写一条 JSONL 过程数据。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds")}
    rec.update(fields)
    try:
        with EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# 窗口钉在屏幕右上角（2560x1440，宽 502 高 889）
PIN_RECT = (2058, 0, 502, 889)

FEED_CAP = 200          # 单次喂食硬上限
CRACK_MIN_LEVEL = 12    # 只敲 >= 12 级
MASK_RESERVE = 120      # 面具预留（攒槽位蛋）
BUY_PRICES = (5, 10)    # 只买这个价位的商人蛋

# 喂食气泡：「点击喂食 🌽96」。整条直接 OCR 会把玉米图标和数字连成 `%`（96 → "9%" / "喂食%"），
# 所以气泡文字用来判「有没有气泡」，数字单独裁玉米右侧区域再放大识别。
FEED_BAND = (0.26, 0.545, 0.48, 0.115)
PRICE_BOX = (0.520, 0.595, 0.200, 0.045)   # rx, ry, rw, rh（数字区）
FEED_SCAN_DUCKS = 9     # 轮播遍历最多看几只鸭子
# 进度条上的鸭子槽位图标（窗口相对坐标，实测 8 个一排在 y≈0.696）
SLOT_XS = tuple(0.265 + 0.0558 * i for i in range(8))
SLOT_Y = 0.696

GRID_X = (0.291, 0.394, 0.498, 0.604, 0.709)
# 行间距实测约 52.3px（889 高的窗口里），原先按 48px 写，误差逐行累积到 13px，
# 导致点第 4/5 行时落在格子边缘甚至横幅上（"识别不到"的真正原因）。
GRID_Y = (0.420, 0.478, 0.538, 0.596, 0.655)
CART_CELL = (4, 4)      # 南瓜车，点了会弹说明，扫描时跳过

NAV = {
    "market": (0.261, 0.885),
    "egg": (0.388, 0.877),
    "duck": (0.478, 0.879),
    "gods": (0.620, 0.879),
    "task": (0.727, 0.879),
}

POPUP_CLOSE = (0.916, 0.227)   # 卡片右上角 X
APP_CLOSE = (0.906, 0.163)     # 小程序标题栏 X（不用，仅记录）


# ---------------------------------------------------------------- 基础工具

def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def write_status(data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        STATUS_PATH.write_text(
            json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def gate_open() -> tuple[bool, dict]:
    """感知门禁：没有 duck_gate_pass.json 且 passed=true 就拒绝生产动作。"""
    try:
        data = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        return bool(data.get("passed")), data
    except Exception:
        return False, {}


def stop_requested() -> bool:
    return STOP_PATH.exists()


def read_pid() -> int | None:
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    import subprocess
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return False


def acquire_lock() -> bool:
    """单实例锁：已有存活实例时拒绝启动，避免两个进程抢鼠标。"""
    import os
    pid = read_pid()
    if pid and pid != os.getpid() and pid_alive(pid):
        log(f"已有看护实例在运行（PID={pid}），本次退出")
        return False
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def do_stop() -> int:
    """要求看护退出：写 stop 文件 + 结束进程（最多等 10 秒）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STOP_PATH.write_text("stop", encoding="utf-8")
    pid = read_pid()
    if not pid:
        print("没有找到 PID 文件，已写 stop 标记；如果进程还在，下一轮会自己退出。")
        return 0
    if not pid_alive(pid):
        print(f"进程 {pid} 已经不在了。")
        try:
            PID_PATH.unlink()
        except Exception:
            pass
        return 0
    for _ in range(20):
        if not pid_alive(pid):
            break
        time.sleep(0.5)
    if pid_alive(pid):
        import subprocess
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True)
        print(f"进程 {pid} 已强制结束。")
    else:
        print(f"进程 {pid} 已正常退出。")
    try:
        PID_PATH.unlink()
    except Exception:
        pass
    return 0


def do_status() -> int:
    pid = read_pid()
    alive = bool(pid and pid_alive(pid))
    print(f"看护进程：{'运行中 PID=' + str(pid) if alive else '未运行'}")
    if STATUS_PATH.exists():
        try:
            print(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()[-5:]
        print("最近日志：")
        print("\n".join(lines))
    return 0


def find_game_window() -> int | None:
    hits: list[tuple[int, str]] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        if "DUCK" in title.upper() or "鸭" in title:
            hits.append((hwnd, title))

    win32gui.EnumWindows(cb, None)
    return hits[0][0] if hits else None


class FocusLost(Exception):
    """游戏不在前台时中止本轮，绝不抢焦点、不移动鼠标。"""


class EnvBlocked(Exception):
    """环境不可用（活动浮层/维护），不是脚本能解决的，应长暂停而不是重试。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


ENV_BLOCKED_MARKS = {
    "THROWED TOMATOES": "tomato_overlay",
    "TOMATO FIESTA": "tomato_event",
    "MAINTENANCE": "maintenance",
}


def check_env(g: Game) -> None:
    """检测不可恢复的环境状态；命中就抛 EnvBlocked。"""
    t = g.text(0.0, 0.30, 1.0, 0.65)
    for mark, reason in ENV_BLOCKED_MARKS.items():
        if mark in t:
            raise EnvBlocked(reason)


class Game:
    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self.l = self.t = self.w = self.h = 0
        self.slot_cells: set[tuple[int, int]] = set()   # 已确认的换槽蛋，后续扫描跳过
        self.slot_scans = 0
        self._matcher = None
        self.refresh()

    # -- 窗口 --
    def refresh(self) -> bool:
        try:
            l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        except Exception:
            return False
        self.l, self.t = l, t
        self.w, self.h = r - l, b - t
        return self.w > 100 and self.h > 100

    def pin(self) -> None:
        x, y, w, h = PIN_RECT
        try:
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, x, y, w, h,
                                  win32con.SWP_SHOWWINDOW)
        except Exception:
            pass
        time.sleep(0.25)
        self.refresh()

    def focused(self) -> bool:
        fg = win32gui.GetForegroundWindow()
        return fg == self.hwnd or win32gui.GetAncestor(fg, win32con.GA_ROOT) == self.hwnd

    def usable(self) -> bool:
        """窗口可见，且采样点都落在本窗口上（不只依赖前台状态）。"""
        if not win32gui.IsWindowVisible(self.hwnd):
            return False
        for rx, ry in ((0.5, 0.30), (0.5, 0.60), (0.30, 0.90)):
            x, y = self.P(rx, ry)
            h = win32gui.WindowFromPoint((x, y))
            if not h:
                return False
            root = win32gui.GetAncestor(h, win32con.GA_ROOT)
            if root != self.hwnd and h != self.hwnd:
                return False
        return True

    def focus(self) -> bool:
        """AttachThreadInput 提升置前成功率（后台进程直接 SetForegroundWindow 常被拒）。"""
        fg = win32gui.GetForegroundWindow()
        if fg == self.hwnd:
            return True
        try:
            cur_tid = win32gui.GetWindowThreadProcessId(fg)[0]
            tgt_tid = win32gui.GetWindowThreadProcessId(self.hwnd)[0]
            if cur_tid and tgt_tid and cur_tid != tgt_tid:
                ctypes.windll.user32.AttachThreadInput(cur_tid, tgt_tid, True)
                try:
                    win32gui.SetForegroundWindow(self.hwnd)
                finally:
                    ctypes.windll.user32.AttachThreadInput(cur_tid, tgt_tid, False)
            else:
                win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            pass
        time.sleep(0.2)
        return self.focused()

    def P(self, rx: float, ry: float) -> tuple[int, int]:
        return (self.l + int(rx * self.w), self.t + int(ry * self.h))

    def R(self, rx: float, ry: float, rw: float, rh: float) -> tuple[int, int, int, int]:
        return (self.l + int(rx * self.w), self.t + int(ry * self.h),
                int(rw * self.w), int(rh * self.h))

    def click(self, rx: float, ry: float, wait: float = 0.6) -> None:
        no_failsafe()   # 防止其它模块导入时把 FAILSAFE 重置回 True
        x, y = self.P(rx, ry)
        pyautogui.click(x, y)
        time.sleep(wait)

    def drag(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        """住 0.25s → 快拖 8 步 → 目标停 0.5s → 松手（实测唯一有效手势）。"""
        x1, y1 = self.P(*p1)
        x2, y2 = self.P(*p2)
        ctypes.windll.user32.SetCursorPos(x1, y1)
        time.sleep(0.25)
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)   # LEFTDOWN
        time.sleep(0.25)
        for i in range(1, 9):
            ctypes.windll.user32.SetCursorPos(
                x1 + (x2 - x1) * i // 8, y1 + (y2 - y1) * i // 8)
            time.sleep(0.02)
        time.sleep(0.5)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)   # LEFTUP
        time.sleep(0.6)   # 实测松手后 0.2s 画面就稳定，原来等 1.5s 纯属浪费（单次合并 1/3 耗时）

    # -- OCR --
    def text(self, rx=0.0, ry=0.0, rw=1.0, rh=1.0) -> str:
        region = self.R(rx, ry, rw, rh)
        lines, _ = bench.ocr_lines(region)
        return " ".join(l["text"] for l in lines if l["text"].strip())

    def lines(self, rx=0.0, ry=0.0, rw=1.0, rh=1.0):
        region = self.R(rx, ry, rw, rh)
        lines, _ = bench.ocr_lines(region)
        return lines

    # -- 页面 --
    def matcher(self) -> IconMatcher:
        if self._matcher is None:
            self._matcher = IconMatcher()
        return self._matcher

    def _nav_region(self, bgr) -> tuple[int, int, int, int]:
        h, w = bgr.shape[:2]
        return (0, int(0.78 * h), w, int(0.16 * h))

    def detect_nav(self) -> list:
        """整条导航栏的图标识别（位置不依赖写死坐标）。"""
        bgr = grab(self.hwnd)
        if bgr is None:
            return []
        return self.matcher().find(to_gray(bgr), region=self._nav_region(bgr))

    def find_nav(self, key: str):
        bgr = grab(self.hwnd)
        if bgr is None:
            return None
        dets = self.matcher().find(to_gray(bgr), names=[key], region=self._nav_region(bgr))
        return dets[0] if dets else None

    def nav_badges(self, dets=None) -> list:
        """红点（消息角标）：返回带 near 关联的列表。"""
        bgr = grab(self.hwnd)
        if bgr is None:
            return []
        if dets is None:
            dets = self.matcher().find(to_gray(bgr), region=self._nav_region(bgr))
        return associate(find_badges(bgr), dets,
                         max_dist=max(32.0, 0.10 * bgr.shape[1]))

    def nav(self, key: str, wait: float = 1.3) -> None:
        # 1) 图标匹配
        det = self.find_nav(key)
        if det is not None:
            x, y = self.l + det.center[0], self.t + det.center[1]
            pyautogui.click(x, y)
            log_event(action="nav", nav=key, source="icon", score=det.score, xy=[x, y])
            time.sleep(wait)
            return
        # 2) OCR 找底栏标签（比固定比例稳）
        label = {"duck": "鸭子", "egg": "蛋", "market": "市场",
                 "task": "任务", "gods": "GOOS"}.get(key)
        if label:
            band = self.R(0.0, 0.855, 1.0, 0.09)
            lines, _ = bench.ocr_lines(band)
            for l in lines:
                if l["text"].strip() == label:
                    r = l["rect"]
                    x, y = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
                    pyautogui.click(x, y)
                    log_event(action="nav", nav=key, source="ocr_label", xy=[x, y])
                    time.sleep(wait)
                    return
        # 3) 固定比例兜底
        rx, ry = NAV[key]
        x, y = self.P(rx, ry)
        pyautogui.click(x, y)
        log_event(action="nav", nav=key, source="coord", xy=[x, y])
        time.sleep(wait)

    def close_popup(self) -> bool:
        """关挡路弹窗；返回是否做了动作。"""
        t = self.text(0.02, 0.10, 0.96, 0.75)
        if not t.strip():
            time.sleep(0.4)
            t = self.text(0.02, 0.10, 0.96, 0.75)

        # 通用规则（放在最前面）：页面上只要有「领取/收取」就点它。
        # 领取永远免费，比逐个枚举活动名可靠——实测「喂鸭大赛…领取」这类面板
        # 不在枚举表里，会一直卡住让所有导航失败。
        if "领取" in t or "收取" in t:
            hits = [l for l in self.lines(0.03, 0.10, 0.94, 0.86)
                    if ("领取" in l["text"]) or ("收取" in l["text"])]
            if hits:
                hits.sort(key=lambda l: l["rect"][1], reverse=True)
                r = hits[0]["rect"]
                pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
                time.sleep(1.8)
                log("弹窗内领取奖励")
                log_event(action="claim_modal")
                return True
        # 注意：「准备·」「升级鸭子」是**鸭子页的正常状态**（已准备好质押面板），
        # 不是弹窗。放在这里会让每轮都误判成"弹窗关不掉"并白重载两次（2026-11-14 实测）。
        marks = ("ABANDONED CART", "TOMATO", "KICKED OUT", "获得鸭子的几率提高",
                 "游戏物品", "获得槽位", "SWEET HALLOWEEN",
                 "CANDY PATH", "质押这只鸭子", "星星不够", "再买一些",
                 "鸭子繁殖完成", "物品已绑定", "你获得了", "绑定的物品")
        no_space = ("KICKEDOUT", "THROWEDTOMATOES", "HOORAY", "TOMATOFIESTA",
                    "PUDGSTERS", "ABANDONEDCART")

        def hit(s: str) -> bool:
            sn = s.replace(" ", "")
            return any(m in s for m in marks) or any(m in sn for m in no_space)

        if not hit(t):
            # 不认识的页面：先看有没有「面板级关闭键」（右上角的 X）。
            # 活动/物品面板（如「将物品插到鸭子上」）不在 marks 里，不认识就会一直卡住，
            # 导致后续所有导航失败（2026-11-13 实测踩到）。
            # 注意：不能用窄条裁剪去找 X，OCR 在窄条里会漏字；
            # 必须整幅读，再按相对坐标过滤（面板 X 在 x≈0.90、y≈0.22）。
            # y 必须 ≥0.12：小程序自身的关闭按钮在 y≈0.05，点错会把游戏关掉。
            for l in self.lines(0.0, 0.10, 1.0, 0.55):
                s = l["text"].strip()
                if s not in ("X", "x", "×", "✕", "╳"):
                    continue
                r = l["rect"]
                rx = (r[0] - self.l) / max(1, self.w)
                ry = (r[1] - self.t) / max(1, self.h)
                if rx < 0.72 or ry < 0.12:
                    continue
                pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
                time.sleep(1.2)
                log("关闭未知面板（点面板右上角 X）")
                log_event(action="close_modal", button="X")
                return True
            # 弹窗动画/OCR 抖动：补读一次再判定，避免漏关
            time.sleep(0.5)
            t = t + " " + self.text(0.02, 0.10, 0.96, 0.75)
            if not hit(t):
                return False
        # （领取已在函数开头统一处理）
        reward_modal = any(m in t for m in ("鸭子繁殖完成", "物品已绑定", "你获得了"))
        if reward_modal:
            for l in self.lines(0.15, 0.60, 0.70, 0.30):
                if "领取" in l["text"]:
                    r = l["rect"]
                    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
                    time.sleep(1.6)
                    log("弹窗内领取奖励")
                    log_event(action="claim_modal")
                    return True
        # 收集类弹窗：只有「关闭 / TO THE DUCK」按钮
        for l in self.lines(0.10, 0.55, 0.80, 0.32):
            txt = l["text"].strip()
            if txt in ("关闭", "TOTHEDUCK") or "TO THE DUCK" in txt.replace(" ", "").upper():
                r = l["rect"]
                pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
                time.sleep(1.2)
                log("关闭收集弹窗")
                log_event(action="close_modal", button=txt)
                return True
        # 通用关闭：面板右上角的「X」。
        # 活动/物品类面板（如「将物品插到鸭子上」）不在上面的 marks 里，
        # 不认识就会一直卡住，导致后续所有导航都失败（2026-11-13 实测踩到）。
        for l in self.lines(0.25, 0.08, 0.72, 0.50):
            s = l["text"].strip()
            if s in ("X", "x", "×", "✕", "╳"):
                r = l["rect"]
                pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
                time.sleep(1.2)
                log("关闭未知面板（点右上角 X）")
                log_event(action="close_modal", button="X")
                return True
        # 普通弹窗：直接点卡片右上角 X（不要先按 ESC，ESC 对这类浮层无效且可能重触发）
        closed = False
        for _ in range(2):
            self.click(*POPUP_CLOSE, wait=1.5)
            t2 = self.text(0.02, 0.10, 0.96, 0.75)
            if not any(m in t2 for m in marks):
                closed = True
                break
        if not closed:
            # 浮层可能还在退场动画：再等 2 秒复核一次
            time.sleep(2.0)
            t2 = self.text(0.02, 0.10, 0.96, 0.75)
            closed = not any(m in t2 for m in marks)
        if not closed:
            is_show = any(m in t2 for m in ("TOMATO", "THROWED", "HOORAY", "KICKED OUT"))
            if is_show:
                # 舞台结算浮层由服务端反复推送，重载通常会立刻再触发 → 先不重载，
                # 但**必须逐级升级**，否则会像 2026-11-12 那样连续 11 轮卡死、整夜零产出。
                self.popup_fails = getattr(self, "popup_fails", 0) + 1
                n = self.popup_fails
                log(f"舞台浮层仍在（连续 {n} 次）")
                log_event(action="popup_stuck", kind="stage", fails=n)
                if n % 3 == 0:
                    # 换手段：ESC + 点卡片外侧空白，绕过位置漂移
                    pyautogui.press("esc")
                    time.sleep(0.8)
                    self.click(0.10, 0.20, wait=1.2)
                    t3 = self.text(0.02, 0.10, 0.96, 0.75)
                    if not any(m in t3 for m in marks):
                        self.popup_fails = 0
                        log("舞台浮层已关闭（ESC + 点空白）")
                        return True
                if n >= 5:
                    # 卡了 5 轮以上，重载是唯一剩下的手段（风险：可能再触发一次，但比卡死好）
                    pyautogui.hotkey("ctrl", "r")
                    time.sleep(6.0)
                    log(f"舞台浮层连续 {n} 次未关掉，强制重载一次")
                    log_event(action="reload", reason="stage_stuck", fails=n)
                    self.popup_fails = 0
                return True
            self.popup_fails = 0
            # 其它卡死弹窗才重载
            pyautogui.hotkey("ctrl", "r")
            time.sleep(6.0)
            log(f"重载小程序（弹窗两次未关掉）：{t2.strip()[:90]!r}")
            log_event(action="reload", reason="popup_stuck", text=t2.strip()[:120])
        else:
            self.popup_fails = 0
        return True


# ---------------------------------------------------------------- 读状态

def read_state(g: Game) -> dict:
    t = g.text()
    st = {"raw": t[:400]}
    m = re.search(r"鸭群\s*\|?\s*([\d.]+K?)", t)
    if m:
        st["flock"] = m.group(1)
    m = re.search(r"点击喂食\s*(\d+)", t)
    if m:
        st["feed_price"] = int(m.group(1))
    prices = re.findall(r"\b(\d{2,3})\b", t)
    st["page"] = "unknown"
    for key, word in (("duck", "点击喂食"), ("duck", "繁殖进行中"),
                      ("egg", "鸭群奖励"), ("market", "我的市场"),
                      ("task", "Challenge Week"), ("halloween", "CANDY PATH")):
        if word in t:
            st["page"] = key
            break
    return st


def star_balance(g: Game) -> int | None:
    """顶栏的星星数量（`鸭群 200.0K 30 ★` 里的 30）。"""
    import re as _re
    t = g.text(0.55, 0.150, 0.40, 0.050)
    m = _re.search(r"(\d{1,5})\s*[☆★]", t)
    if m:
        return int(m.group(1))
    t = g.text()
    m = _re.search(r"(\d{1,5})\s*[☆★]", t)
    return int(m.group(1)) if m else None


def mask_balance(g: Game) -> int | None:
    """蛋页顶栏最右侧的面具数量（星星在它左边）。"""
    lines = g.lines(0.885, 0.150, 0.115, 0.042)
    nums = [l["text"].strip() for l in lines
            if re.fullmatch(r"\d{1,5}", l["text"].strip())]
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


_TEMPLATES = None


def _templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        from duck_slot_tm import Templates
        _TEMPLATES = Templates(HERE / "templates")
    return _TEMPLATES


def duck_slots_vision(g: Game) -> list[list[int]]:
    """用槽位分类器找「duck」格子，返回屏幕坐标框。

    带红点（可收心/喂食就绪）的排前面——红点是更可靠的就绪信号。
    """
    try:
        import ui_icons
        from duck_slots import segment_slots
        lib = _templates()
        if not lib.ready():
            return []
        bgr = ui_icons.grab(g.hwnd)
        if bgr is None:
            return []
        badges = ui_icons.find_badges(bgr)
        out, plain = [], []
        for s in segment_slots(bgr):
            r = lib.match(s["crop"])
            pc = r.get("per_class") or {}
            d, p = pc.get("duck", 0.0), pc.get("padlock", 0.0)
            # 用「duck 与 padlock 的差距」判定，而不是绝对分数
            if r["cls"] != "duck" or d < 0.60 or (d - p) < 0.05:
                continue
            x, y, w, h = s["box"]
            cx, cy = x + w / 2, y + h / 2
            has_badge = any(
                abs(bx - cx) < 26 and -6 < (by - cy) < 30
                for bx, by in (b.get("center", (0, 0)) for b in badges))
            rec = [g.l + x, g.t + y, w, h, r["score"]]
            (out if has_badge else plain).append(rec)
        return out + plain
    except Exception:
        return []


def slot_inventory(g: Game) -> list[dict]:
    """**不点击**，纯视觉枚举鸭子页的槽位行，给每个槽位定状态。

    这是执行流程的第一层：先确定槽位（padlock / duck / unknown），
    再对 duck 槽位读鸭子状态。绝不按固定坐标盲点——那等于把点击当探针。

    返回 [{i, kind, score, cx, cy}]，cx/cy 为屏幕绝对坐标。
    """
    from duck_slots import segment_slots
    lib = _templates()
    bgr = grab(g.hwnd)
    if bgr is None or not lib.ready():
        return []
    out = []
    for i, s in enumerate(segment_slots(bgr)):
        r = lib.match(s["crop"])
        pc = r.get("per_class") or {}
        d, p = pc.get("duck", 0.0), pc.get("padlock", 0.0)
        if r["cls"] == "duck" and d >= 0.60 and (d - p) >= 0.05:
            kind = "duck"
        elif r["cls"] == "padlock" and p >= 0.60 and (p - d) >= 0.05:
            kind = "padlock"
        else:
            kind = "unknown"
        x, y, w, h = s["box"]
        out.append({"i": i, "kind": kind, "score": round(float(r["score"]), 3),
                    "cx": g.l + x + w // 2, "cy": g.t + y + h // 2,
                    "box": [x, y, w, h]})
    # 整排都被判成「锁」是可疑的：选中「新鸭子/无槽位」卡片时整排会变暗，
    # 此时分类器会把灰点全当成 padlock。宁可报「视觉失败」，也不要据此跳过所有鸭子。
    if out and all(s["kind"] == "padlock" for s in out):
        return []
    return out


def slot_kind_at(g: Game, rx: float, ry: float = 0.691) -> str:
    """不点击，直接对某个位置的小图做分类（duck/padlock/unknown）。"""
    try:
        import ui_icons
        bgr = ui_icons.grab(g.hwnd)
        if bgr is None:
            return "unknown"
        cx, cy = int(rx * g.w), int(ry * g.h)
        crop = bgr[max(0, cy - 16):cy + 16, max(0, cx - 16):cx + 16]
        lib = _templates()
        if not lib.ready() or crop.size == 0:
            return "unknown"
        r = lib.match(crop)
        pc = r.get("per_class") or {}
        d, p = pc.get("duck", 0.0), pc.get("padlock", 0.0)
        if r["cls"] == "duck" and d >= 0.60 and (d - p) >= 0.05:
            return "duck"
        if r["cls"] == "padlock" and p >= 0.60 and (p - d) >= 0.05:
            return "padlock"
        return "unknown"
    except Exception:
        return "unknown"


DUCK_STATE_WORDS = ("这是一只新鸭子", "已准备好交配", "繁殖进行中",
                   "鸭子正在出售", "已准备好质押", "点击喂食")

SLOT_ROW_Y = 0.699          # 槽位行图标中心（相对高度）；偏 4px 就测不出高亮
SLOT_LEFT, SLOT_RIGHT = 0.20, 0.98


def detect_slot_anchors(g: Game, min_width: int = 7) -> list[float]:
    """从槽位行图像里检测图标的**真实中心 x**（相对坐标）。

    不按固定步长猜：实测点了图标之间的空隙时页面不变，读到的是上一只鸭子。
    """
    import cv2
    import numpy as np

    img = grab(g.hwnd)
    if img is None:
        return []
    H, W = img.shape[:2]
    best: list[float] = []
    for y0 in (0.680, 0.686, 0.690):
        band = img[int(y0 * H):int((y0 + 0.030) * H), :]
        if band.size == 0:
            continue
        v = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[:, :, 2]
        col = (v > 48).mean(axis=0)
        mask = (col > 0.35).astype(np.uint8)
        segs: list[float] = []
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                if j - i >= min_width:
                    segs.append(round(((i + j) / 2) / W, 3))
                i = j + 1
            else:
                i += 1
        if len(segs) > len(best):
            best = segs
    return best


def selected_slot_index(g: Game, anchors: list[float]) -> int | None:
    """返回**被选中（高亮）**图标的索引；检测不到返回 None。

    实测：选中格亮度 V≈144~170，其余 58~88。这是唯一可靠的「选中」信号，
    页面文字不可靠——点空位时卡片不变，文字仍是上一只鸭子的。
    """
    import cv2

    img = grab(g.hwnd)
    if img is None or not anchors:
        return None
    H, W = img.shape[:2]
    vals: list[float] = []
    for x in anchors:
        cx, cy = int(x * W), int(SLOT_ROW_Y * H)
        c = img[max(0, cy - 8):cy + 8, max(0, cx - 8):cx + 8]
        if c.size == 0:
            vals.append(0.0)
            continue
        vals.append(float(cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[:, :, 2].mean()))
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    top = vals[order[0]]
    second = vals[order[1]] if len(order) > 1 else 0.0
    if top < 110 or top - second < 12:
        return None
    return order[0]


def _row_anchors(g: Game) -> list[float]:
    """返回槽位带上所有图标的 x，并**按等间距补齐漏检的格子**。

    实测漏检会直接少算鸭子：0.337 / 0.449 两颗较暗的图标检测不到，
    导致「有 7 只却只数到 5 只」。图标间距是固定的 0.056，用已有锚点
    重建网格就能把漏检补回来。
    """
    raw = [x for x in detect_slot_anchors(g) if SLOT_LEFT < x < SLOT_RIGHT]
    if not raw:
        return []
    if len(raw) < 3:
        return raw
    diffs = [raw[k + 1] - raw[k] for k in range(len(raw) - 1)]
    small = sorted(d for d in diffs if d < 0.09)
    pitch = small[len(small) // 2] if small else 0.0558
    if not (0.040 < pitch < 0.075):
        pitch = 0.0558
    # 只在两个真实锚点之间补齐漏检的格子（不凭空外推，避免整体偏移半格）
    out: list[float] = [raw[0]]
    for k in range(len(raw) - 1):
        gap = raw[k + 1] - raw[k]
        n = max(1, int(round(gap / pitch)))
        for j in range(1, n):
            out.append(round(raw[k] + j * pitch, 3))
        out.append(raw[k + 1])
    return out


def click_slot_verified(g: Game, x: float, anchors: list[float],
                        tries: int = 2) -> tuple[bool, str]:
    """点图标 x，并**用高亮确认真的选中了它**。返回 (ok, 页面文本)。

    没有这个校验，点空位/点偏都会把上一只鸭子的状态当成结论
    （2026-11-14 用户指出：选中会高亮，下一个应该能识别到）。
    """
    t = ""
    for k in range(tries):
        pyautogui.click(g.l + int(x * g.w), g.t + int(SLOT_ROW_Y * g.h))
        time.sleep(0.95 if k == 0 else 0.7)
        t = " ".join(l["text"] for l in g.lines())
        idx = selected_slot_index(g, anchors)
        if idx is not None and abs(anchors[idx] - x) < 0.012:
            return True, t
    return False, t


def exit_assign_mode(g: Game) -> bool:
    """退出「给它安排一个槽位」的分配模式。

    进入该模式后整排会**全部变成挂锁**（连已有鸭子的槽位也是），
    页面只剩「解锁上一个槽位」。不退出会让后续扫描看到「全是锁」的假象。
    实测退出方式：close_popup() + nav('duck')。
    """
    t = " ".join(l["text"] for l in g.lines())
    if "解锁上一个槽位" not in t and "给它安排一个槽位" not in t:
        return False
    g.close_popup()
    time.sleep(0.6)
    g.nav("duck", wait=1.4)
    time.sleep(0.6)
    return True


SLOT_BAND = (0.686, 0.708)     # 槽位带图标实际占据的 20px
SLOT_CY = 0.6975               # 图标中心（相对高度）


def detect_slot_icons(g: Game) -> list[dict]:
    """检测槽位带里的图标：返回 [{x, patch, v}]，x 只在本帧内有效。

    阈值是**自适应**的（背景 V 中位数约 27，取 max(45, 中位+12)）。
    固定阈值会漏掉暗色挂锁（V≈40~50），导致「有 10 只只数到 5 只」。
    """
    import cv2
    import numpy as np

    img = grab(g.hwnd)
    if img is None:
        return []
    H, W = img.shape[:2]
    y0, y1 = int(SLOT_BAND[0] * H), int(SLOT_BAND[1] * H)
    band = img[y0:y1, :]
    v = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.int16)
    thr = max(45, int(np.median(v)) + 12)
    col = (v > thr).mean(axis=0)
    on = col > 0.22

    segs: list[list[int]] = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            segs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= 3:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    cy = int(SLOT_CY * H)
    out = []
    for a, b in merged:
        if (b - a + 1) < 6:
            continue
        x = ((a + b) / 2) / W
        cx = int(x * W)
        patch = img[cy - 7:cy + 7, max(0, cx - 7):cx + 7]
        if patch.size == 0:
            continue
        pv = float(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        out.append({"x": round(x, 3), "patch": patch, "v": pv})
    return out


def icon_mask(patch):
    """图标形状掩码（选中的图标会变成白色，颜色不可比，形状可比）。

    阈值必须相对**背景**（实测背景 V≈27、图标 V≈58~250），
    不能用补丁自身的中位数——补丁里几乎全是图标，中位数一算就把自己滤没了。
    """
    import cv2
    import numpy as np

    if patch is None or patch.size == 0:
        return None
    v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = (v > 45).astype(np.uint8)
    if m.shape != (14, 14):
        m = cv2.resize(m, (14, 14), interpolation=cv2.INTER_NEAREST)
    return m


def same_icon(a, b, thr: float = 0.22) -> bool:
    """两个图标是否同一格：比**形状掩码**（不比颜色、不比坐标）。"""
    import numpy as np

    ma, mb = icon_mask(a), icon_mask(b)
    if ma is None or mb is None or ma.shape != mb.shape:
        return False
    if ma.sum() == 0 or mb.sum() == 0:
        return False
    return float(np.abs(ma.astype(int) - mb.astype(int)).mean()) < thr


def slot_pitch(icons: list[dict]) -> float:
    """相邻图标的间距（默认 0.0558）。用真实检测值，避免面板缩放导致偏差。"""
    if len(icons) < 3:
        return 0.0558
    diffs = sorted(icons[k + 1]["x"] - icons[k]["x"] for k in range(len(icons) - 1))
    good = [d for d in diffs if 0.040 < d < 0.075]
    return good[len(good) // 2] if good else 0.0558


def selected_icon(icons: list[dict]) -> int | None:
    """选中的是哪一个：亮度最高且明显高于第二名（实测 144~170 vs 58~88）。

    必须排除 x < 0.15 的左侧图标：鸭群图标/分隔线那一列亮度能到 152，
    与真正的选中格只差 20 左右，会把"选中"判到索引 0/2 上，
    走查索引就会乱跳（实测表现为永远在前几只循环）。
    """
    if not icons:
        return None
    icons = [i for i in icons if 0.15 < i.get("x", 0) < 0.96]
    if not icons:
        return None
    order = sorted(range(len(icons)), key=lambda i: -icons[i]["v"])
    if len(order) > 1 and icons[order[0]]["v"] - icons[order[1]]["v"] < 25:
        return None
    return order[0]


def _duck_fp(g: Game):
    """当前鸭子卡片的指纹（32x32 灰度），用来判定「有没有换鸭」。"""
    import cv2

    img = grab(g.hwnd)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(0.32 * h):int(0.56 * h), int(0.30 * w):int(0.70 * w)]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))


def focus_page(g: Game) -> None:
    """把键盘焦点交给页面（点标题栏，无副作用）。"""
    pyautogui.click(g.l + int(0.45 * g.w), g.t + int(0.09 * g.h))
    time.sleep(0.45)


def _click_icon(g: Game, icon: dict) -> None:
    pyautogui.click(g.l + int(icon["x"] * g.w), g.t + int(SLOT_CY * g.h))


def _duck_sig(prof: dict) -> tuple:
    """一只鸭子的身份签名（用于去重 / 判断是否绕回起点）。"""
    return (round(float(prof.get("x") or 0), 3), prof.get("state"),
            prof.get("rarity"), prof.get("level"), prof.get("price"))


def _read_one_duck(g: Game, x: float) -> dict:
    t = " ".join(l["text"] for l in g.lines())
    state = next((w for w in DUCK_STATE_WORDS if w in t), None)
    prof = read_duck_profile(g)
    prof["state"] = state or prof.get("state")
    prof["kind"] = "new" if state == "这是一只新鸭子" else "duck"
    m = re.search(r"喂食\s*(\d+)", t)
    prof["price"] = int(m.group(1)) if m else None
    prof["slot_free"] = "无可用的槽位" in t
    prof["x"] = round(x, 3)
    return prof


def _card_sig_quick(g: Game) -> tuple:
    """卡片签名（稀有度/等级/单价/状态）：高亮消失时的第二判据。"""
    t = " ".join(l["text"] for l in g.lines())
    rar = re.search(r"(COMMON|UNCOMMON|RARE|EPIC|LEGENDARY)", t)
    lv = re.search(r"LVL\s*(\d+)", t)
    price = re.search(r"喂食\s*(\d+)", t)
    st = next((w for w in DUCK_STATE_WORDS if w in t), None)
    return (rar.group(1) if rar else "?", lv.group(1) if lv else "?",
            price.group(1) if price else "-", st,
            "无可用的槽位" in t)


def _stable_icons(g: Game, tries: int = 8, delay: float = 0.25) -> list[dict]:
    """等槽位带停止滚动，返回稳定的图标列表。

    这一排有滚动动画（选中靠右的鸭子后整排会缓动左移），
    过渡帧里两次检测的位置能差 3 格；不等待就会把"动画中的位置"当坐标用。
    """
    prev = None
    for _ in range(tries):
        cur = detect_slot_icons(g)
        cur = [i for i in cur if 0.15 < i.get("x", 0) < 0.96]
        if prev is not None and len(cur) == len(prev):
            if all(abs(a["x"] - b["x"]) < 0.006 for a, b in zip(cur, prev)):
                return cur
        prev = cur
        time.sleep(delay)
    return prev or []


def _strip_shift(prev: list[dict], cur: list[dict]) -> float:
    """估计两帧之间整排的位移（正=右移，负=左移）。

    不能用 cur[0].x - prev[0].x：列表首元素会因左侧图标进出筛选而改变。
    这里按"最佳对齐"取中位差。
    """
    if not prev or not cur:
        return 0.0
    best = (1e9, 0.0)
    pa = [i["x"] for i in prev]
    ca = [i["x"] for i in cur]
    for off in range(-4, 5):
        pairs = [(pa[i], ca[i + off]) for i in range(len(pa))
                 if 0 <= i + off < len(ca)]
        if len(pairs) < max(4, len(pa) // 2):
            continue
        diffs = sorted(b - a for a, b in pairs)
        mid = diffs[len(diffs) // 2]
        cost = sum(abs(d - mid) for d in diffs) / len(diffs)
        if cost < best[0]:
            best = (cost, mid)
    return round(best[1], 3)


def _step_duck(g: Game, direction: int, settle_ok: float = 0.95,
               from_x: float | None = None) -> tuple[bool, float]:
    """切到相邻一只：**帧内索引**取右/左邻，再用高亮确认。

    为什么不用「高亮 + 槽距」：这一排会自动滚动重新居中，
    选到靠右的鸭子后整排左移，`sel_x + pitch` 反而会算到左边的格子，
    走查就在前几只里循环（实测 0.338 → 0.226）。
    同一帧里的 `icons[sel ± 1]` 才是"列表里的下一只"，与滚动无关。
    """
    icons = _stable_icons(g)
    sel = selected_icon(icons)
    if sel is not None:
        idx = sel + direction
        if 0 <= idx < len(icons):
            nxt = round(icons[idx]["x"], 3)
        else:
            return False, 0.0
    elif from_x is not None:
        nxt = round(from_x + direction * slot_pitch(icons), 3)
    else:
        return False, 0.0
    if not (SLOT_LEFT <= nxt <= 0.99):
        return False, nxt

    before_sig = _card_sig_quick(g)
    for k in range(2):
        pyautogui.click(g.l + int(nxt * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(settle_ok if k == 0 else 0.8)
        icons2 = _stable_icons(g)
        sel2 = selected_icon(icons2)
        shift = _strip_shift(icons, icons2)
        exp = nxt + shift          # 整排左移(负)时，目标格的当前位置也会左移
        if sel2 is not None and abs(icons2[sel2]["x"] - exp) < 0.030:
            return True, icons2[sel2]["x"]
        # 高亮消失但卡片确实换了（正在出售 / 无槽位新鸭）：允许这一步
        if (sel2 is None and _card_sig_quick(g) != before_sig
                and current_page(g, tries=1)[0] == "duck"):
            return True, nxt
    # 页面可能停在「这是新鸭子/给它安排槽位」面板上：复位后再试一次
    g.nav("duck", wait=1.2)
    time.sleep(0.7)
    icons = _stable_icons(g)
    sel = selected_icon(icons)
    if sel is None:
        return False, nxt
    idx = sel + direction
    if not (0 <= idx < len(icons)):
        return False, nxt
    nxt2 = round(icons[idx]["x"], 3)
    if not (SLOT_LEFT <= nxt2 <= 0.99):
        return False, nxt2
    before2 = _card_sig_quick(g)
    icons_before = icons
    pyautogui.click(g.l + int(nxt2 * g.w), g.t + int(SLOT_CY * g.h))
    time.sleep(0.95)
    icons2 = _stable_icons(g)
    sel2 = selected_icon(icons2)
    shift2 = _strip_shift(icons_before, icons2)
    if sel2 is not None and abs(icons2[sel2]["x"] - (nxt2 + shift2)) < 0.030:
        return True, icons2[sel2]["x"]
    if (sel2 is None and _card_sig_quick(g) != before2
            and current_page(g, tries=1)[0] == "duck"):
        return True, nxt2
    return False, nxt2


def _duck_sig(prof: dict) -> tuple:
    """一只鸭子的身份签名（用于去重 / 判断是否绕回起点）。"""
    return (round(float(prof.get("x") or 0), 3), prof.get("state"),
            prof.get("rarity"), prof.get("level"), prof.get("price"))


def _read_one_duck(g: Game, x: float) -> dict:
    t = " ".join(l["text"] for l in g.lines())
    state = next((w for w in DUCK_STATE_WORDS if w in t), None)
    prof = read_duck_profile(g)
    prof["state"] = state or prof.get("state")
    prof["kind"] = "new" if state == "这是一只新鸭子" else "duck"
    m = re.search(r"喂食\s*(\d+)", t)
    prof["price"] = int(m.group(1)) if m else None
    prof["slot_free"] = "无可用的槽位" in t
    prof["x"] = round(x, 3)
    return prof


def _card_sig_quick(g: Game) -> tuple:
    """卡片签名（稀有度/等级/单价/状态）：高亮消失时的第二判据。"""
    t = " ".join(l["text"] for l in g.lines())
    rar = re.search(r"(COMMON|UNCOMMON|RARE|EPIC|LEGENDARY)", t)
    lv = re.search(r"LVL\s*(\d+)", t)
    price = re.search(r"喂食\s*(\d+)", t)
    st = next((w for w in DUCK_STATE_WORDS if w in t), None)
    return (rar.group(1) if rar else "?", lv.group(1) if lv else "?",
            price.group(1) if price else "-", st,
            "无可用的槽位" in t)


def _step_duck(g: Game, direction: int, settle_ok: float = 0.95,
               from_x: float | None = None) -> tuple[bool, float]:
    """切到相邻一只：**必须**看到高亮落在目标格上，否则算失败。

    为什么不用"卡片变了"当判据：相邻两只鸭子可能完全同款（同稀有度/等级/单价），
    卡片文字本来就不变；而"卡片变了"也可能只是点到了另一只（导致走查打转）。
    高亮是唯一可靠的"我确实切到了这一格"的证据。
    """
    icons = detect_slot_icons(g)
    sel = selected_icon(icons)
    if sel is not None:
        base = icons[sel]["x"]
    elif from_x is not None:
        base = from_x          # 高亮消失（正在出售 / 无槽位新鸭）：用上次记录的位置推
    else:
        return False, 0.0
    nxt = round(base + direction * slot_pitch(icons), 3)
    if not (SLOT_LEFT <= nxt <= 0.99):
        return False, nxt
    before_sig = _card_sig_quick(g)
    for k in range(2):
        pyautogui.click(g.l + int(nxt * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(settle_ok if k == 0 else 0.8)
        icons2 = _stable_icons(g)
        sel2 = selected_icon(icons2)
        shift = _strip_shift(icons, icons2)
        exp = nxt + shift          # 整排左移(负)时，目标格的当前位置也会左移
        if sel2 is not None and abs(icons2[sel2]["x"] - exp) < 0.030:
            return True, icons2[sel2]["x"]
        # 高亮消失但卡片确实换了（正在出售 / 无槽位新鸭）：允许这一步，
        # 否则走查会卡在这类鸭子前面，后面的全走不到。
        if (sel2 is None and _card_sig_quick(g) != before_sig
                and current_page(g, tries=1)[0] == "duck"):
            return True, nxt
    # 页面可能停在「这是新鸭子/给它安排槽位」面板上：复位后再试一次
    g.nav("duck", wait=1.2)
    time.sleep(0.7)
    icons = detect_slot_icons(g)
    sel = selected_icon(icons)
    if sel is None:
        return False, nxt
    nxt2 = round(icons[sel]["x"] + direction * slot_pitch(icons), 3)
    if not (SLOT_LEFT <= nxt2 <= 0.99):
        return False, nxt2
    before2 = _card_sig_quick(g)
    icons_before = icons
    pyautogui.click(g.l + int(nxt2 * g.w), g.t + int(SLOT_CY * g.h))
    time.sleep(0.95)
    icons2 = _stable_icons(g)
    sel2 = selected_icon(icons2)
    shift2 = _strip_shift(icons_before, icons2)
    if sel2 is not None and abs(icons2[sel2]["x"] - (nxt2 + shift2)) < 0.030:
        return True, icons2[sel2]["x"]
    if (sel2 is None and _card_sig_quick(g) != before2
            and current_page(g, tries=1)[0] == "duck"):
        return True, nxt2
    return False, nxt2


def _duck_sig(prof: dict) -> tuple:
    """一只鸭子的身份签名（用于去重 / 判断是否绕回起点）。"""
    return (round(float(prof.get("x") or 0), 3), prof.get("state"),
            prof.get("rarity"), prof.get("level"), prof.get("price"))


def _read_one_duck(g: Game, x: float) -> dict:
    t = " ".join(l["text"] for l in g.lines())
    state = next((w for w in DUCK_STATE_WORDS if w in t), None)
    prof = read_duck_profile(g)
    prof["state"] = state or prof.get("state")
    prof["kind"] = "new" if state == "这是一只新鸭子" else "duck"
    m = re.search(r"喂食\s*(\d+)", t)
    prof["price"] = int(m.group(1)) if m else None
    prof["slot_free"] = "无可用的槽位" in t
    prof["x"] = round(x, 3)
    return prof


def _card_sig_quick(g: Game) -> tuple:
    """卡片签名（稀有度/等级/单价/状态）：高亮消失时的第二判据。"""
    t = " ".join(l["text"] for l in g.lines())
    rar = re.search(r"(COMMON|UNCOMMON|RARE|EPIC|LEGENDARY)", t)
    lv = re.search(r"LVL\s*(\d+)", t)
    price = re.search(r"喂食\s*(\d+)", t)
    st = next((w for w in DUCK_STATE_WORDS if w in t), None)
    return (rar.group(1) if rar else "?", lv.group(1) if lv else "?",
            price.group(1) if price else "-", st,
            "无可用的槽位" in t)


def _step_duck(g: Game, direction: int, settle_ok: float = 0.95,
               from_x: float | None = None) -> tuple[bool, float]:
    """切到相邻一只。返回 (是否切换成功, 当前格 x)。

    判定成功有两条路（任一成立即可）：
      1) 高亮落到了目标位置（有槽位鸭子的正常情况）；
      2) **卡片内容变了**（无槽位新鸭 / 正在出售的鸭子高亮不动，但卡片会换）。
    高亮不可观测时，用「上一步位置 + 槽距」继续推，而不是直接放弃。
    """
    before_sig = _card_sig_quick(g)
    icons = detect_slot_icons(g)
    sel = selected_icon(icons)
    if sel is not None:
        base_x = icons[sel]["x"]
    else:
        base_x = getattr(g, "_step_x", None)
        if base_x is None:
            return False, 0.0
    nxt = round(base_x + direction * slot_pitch(icons or []), 3)
    if not (SLOT_LEFT <= nxt <= 0.99):
        return False, nxt

    for k in range(2):
        pyautogui.click(g.l + int(nxt * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(settle_ok if k == 0 else 0.8)
        icons2 = detect_slot_icons(g)
        sel2 = selected_icon(icons2)
        at_target = (sel2 is not None
                     and abs(icons2[sel2]["x"] - nxt) < 0.022)
        changed = _card_sig_quick(g) != before_sig
        # 高亮可观测时**必须**落在目标上（否则会三格打转）；
        # 只有高亮消失（正在出售 / 无槽位新鸭）时才用卡片变化兜底。
        ok = at_target if sel2 is not None else changed
        if ok:
            g._step_x = nxt
            return True, nxt

    # 页面可能停在「这是新鸭子/给它安排槽位」面板上：复位后再试一次
    g.nav("duck", wait=1.2)
    time.sleep(0.7)
    before_sig = _card_sig_quick(g)
    pyautogui.click(g.l + int(nxt * g.w), g.t + int(SLOT_CY * g.h))
    time.sleep(0.95)
    icons2 = detect_slot_icons(g)
    sel2 = selected_icon(icons2)
    at_target = sel2 is not None and abs(icons2[sel2]["x"] - nxt) < 0.022
    changed = _card_sig_quick(g) != before_sig
    ok = at_target if sel2 is not None else changed
    if ok:
        g._step_x = nxt
        return True, nxt
    return False, nxt


def _duck_sig(prof: dict) -> tuple:
    """一只鸭子的身份签名（用于去重 / 判断是否绕回起点）。"""
    return (round(float(prof.get("x") or 0), 3), prof.get("state"),
            prof.get("rarity"), prof.get("level"), prof.get("price"))


def _read_one_duck(g: Game, x: float) -> dict:
    t = " ".join(l["text"] for l in g.lines())
    state = next((w for w in DUCK_STATE_WORDS if w in t), None)
    prof = read_duck_profile(g)
    prof["state"] = state or prof.get("state")
    prof["kind"] = "new" if state == "这是一只新鸭子" else "duck"
    m = re.search(r"喂食\s*(\d+)", t)
    prof["price"] = int(m.group(1)) if m else None
    prof["slot_free"] = "无可用的槽位" in t
    prof["x"] = round(x, 3)
    return prof


def _step_duck(g: Game, direction: int, settle_ok: float = 0.95,
               from_x: float | None = None) -> tuple[bool, float]:
    """按「高亮 + 槽距 × 方向」切到相邻一只。返回 (是否成功, 新选中 x)。"""
    icons = detect_slot_icons(g)
    sel = selected_icon(icons)
    if sel is not None:
        base = icons[sel]["x"]
    elif from_x is not None:
        base = from_x          # 高亮消失（正在出售 / 无槽位新鸭）：用上次记录的位置推
    else:
        return False, 0.0
    nxt = round(base + direction * slot_pitch(icons), 3)
    if not (SLOT_LEFT <= nxt <= 0.99):
        return False, nxt
    for k in range(2):
        pyautogui.click(g.l + int(nxt * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(settle_ok if k == 0 else 0.8)
        icons2 = detect_slot_icons(g)
        sel2 = selected_icon(icons2)
        if sel2 is not None and abs(icons2[sel2]["x"] - nxt) < 0.022:
            return True, icons2[sel2]["x"]
    # 页面可能停在「这是一只新鸭子/给它安排一个槽位」面板上，此时点图标不生效。
    # 回鸭子页复位后再试一次（实测这一步能把卡死的状态解开）。
    g.nav("duck", wait=1.2)
    time.sleep(0.7)
    icons = detect_slot_icons(g)
    sel = selected_icon(icons)
    if sel is None:
        return False, nxt
    nxt2 = round(icons[sel]["x"] + direction * slot_pitch(icons), 3)
    if not (SLOT_LEFT <= nxt2 <= 0.99):
        return False, nxt2
    before2 = _card_sig_quick(g)
    icons_before = icons
    pyautogui.click(g.l + int(nxt2 * g.w), g.t + int(SLOT_CY * g.h))
    time.sleep(0.95)
    icons2 = _stable_icons(g)
    sel2 = selected_icon(icons2)
    shift2 = _strip_shift(icons_before, icons2)
    if sel2 is not None and abs(icons2[sel2]["x"] - (nxt2 + shift2)) < 0.030:
        return True, icons2[sel2]["x"]
    if (sel2 is None and _card_sig_quick(g) != before2
            and current_page(g, tries=1)[0] == "duck"):
        return True, nxt2
    return False, nxt2


def is_locked_slot(g: Game, x: float, half: int = 9) -> bool:
    """这一格是不是未解锁的挂锁。

    判据来自实测：把图标的亮区填满外接轮廓后，**挂锁会露出锁孔空洞（≈32px）**，
    鸭子格几乎没有空洞（0~2px）。这样不用点就能知道"这里是锁，别点"。
    """
    import cv2
    import numpy as np

    img = grab(g.hwnd)
    if img is None:
        return False
    H, W = img.shape[:2]
    cx, cy = int(x * W), int(SLOT_CY * H)
    patch = img[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    if patch.size == 0:
        return False
    v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
    m = (v > 45).astype(np.uint8)
    if m.sum() < 20:
        return False
    filled = m.copy()
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    holes = int((filled - m).sum())
    return holes >= 20


def walk_ducks(g: Game, handler=None, max_steps: int = 24) -> list[dict]:
    """枚举鸭子：**按有效点击数计**，坐标只用于点击、绝不当作身份。

    规则：
      1) 归位到最左，先读当前这只（计 1）；
      2) 每次：下一点 = 当前高亮 + 槽距；点完重新识别；
         - 卡片变了 或 高亮换了格  → 这是新的一只，计 1，继续
         - 都没变                 → 点不动了，结束（这次点击不计）
      3) 点之前先判挂锁（锁孔空洞），是锁就不点、直接结束。
    同款鸭子（卡片完全一样）靠"高亮换格"区分，不会被合并。
    """
    g.nav("duck", wait=1.2)
    time.sleep(1.2)
    exit_assign_mode(g)

    icons = _stable_icons(g)
    if not icons:
        log("  走查：检测不到槽位带")
        return []

    # 归位到最左
    for _ in range(8):
        icons = _stable_icons(g)
        sel = selected_icon(icons)
        if sel == 0:
            break
        pyautogui.click(g.l + int(icons[0]["x"] * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(1.0)

    icons = _stable_icons(g)
    sel = selected_icon(icons)
    x = round(icons[0]["x"], 3)
    pyautogui.click(g.l + int(x * g.w), g.t + int(SLOT_CY * g.h))
    time.sleep(1.05)

    out: list[dict] = []
    prev_card = None
    prev_hl = None
    clicks = 0
    for _ in range(max_steps):
        icons = _stable_icons(g)
        sel = selected_icon(icons)
        hl = icons[sel]["x"] if sel is not None else None
        prof = _read_one_duck(g, hl if hl is not None else x)
        card = (prof.get("rarity"), prof.get("level"), prof.get("price"),
                prof.get("state"))

        # 与上一次相比是否"换了一只"
        changed = (prev_card is None) or (card != prev_card) or (
            hl is not None and prev_hl is not None and abs(hl - prev_hl) > 0.01)
        if changed and not (prof["kind"] == "new" and hl == prev_hl):
            out.append(prof)
            clicks += 1
            log(f"  第{clicks}只：高亮 {hl if hl is None else round(hl, 3)} "
                f"{prof['rarity']} Lv{prof['level']} {prof['state']} 价={prof['price']}")
            if handler is not None:
                try:
                    handler(prof, " ".join(l["text"] for l in g.lines()))
                except Exception as exc:
                    log(f"  处理这一只时出错：{exc}")
        else:
            # 点不动有两种可能：真的到头了，或者这一排滚动导致点偏。
            # 用一次 Tab+Enter 核实（键盘在这里只当探针，不参与常规推进）。
            log("  点击没有换鸭子 → 用键盘核实一次")
            pyautogui.press("tab")
            time.sleep(0.3)
            pyautogui.press("enter")
            time.sleep(1.0)
            kprof = _read_one_duck(g, 0.0)
            kcard = (kprof.get("rarity"), kprof.get("level"),
                     kprof.get("price"), kprof.get("state"))
            if kcard != card and current_page(g, tries=1)[0] == "duck":
                log(f"  核实：后面还有一只 {kprof['rarity']} Lv{kprof['level']} "
                    f"{kprof['state']} → 继续")
                icons = _stable_icons(g)
                sel = selected_icon(icons)
                if sel is not None:
                    x = round(icons[sel]["x"], 3)
                prev_card, prev_hl = kcard, (
                    icons[sel]["x"] if sel is not None else None)
                clicks += 1
                out.append(kprof)
                if handler is not None:
                    try:
                        handler(kprof, " ".join(l["text"] for l in g.lines()))
                    except Exception as exc:
                        log(f"  处理这一只时出错：{exc}")
                continue
            log("  核实：确实没有下一只 → 结束")
            break
        prev_card, prev_hl = card, hl

        if hl is None:
            log("  高亮不可观测 → 结束")
            break
        nx = round(hl + slot_pitch(icons), 3)
        if not (SLOT_LEFT <= nx <= 0.985):
            log(f"  到最右（高亮 {hl:.3f}）→ 结束")
            break
        if is_locked_slot(g, nx):
            log(f"  下一格 {nx:.3f} 是未解锁挂锁 → 结束（不点）")
            break

        pyautogui.click(g.l + int(nx * g.w), g.t + int(SLOT_CY * g.h))
        time.sleep(1.05)
        if current_page(g, tries=1)[0] != "duck":
            log("  点到了别的页面（挂锁）→ 结束")
            g.close_popup()
            g.nav("duck", wait=1.1)
            break
        x = nx

    ducks = [d for d in out if d["kind"] == "duck"]
    log(f"  走查完成：有效点击 {clicks} 次 → {len(out)} 只"
        f"（其中待分配 {len(out) - len(ducks)}）")
    if out:
        g.slot_scan = (time.time(), out, [])
    return out


def duck_slots_from_inventory(g: Game) -> tuple[list[dict], list[dict], float]:
    """枚举鸭子槽位清单（带 30 分钟缓存），返回 (鸭子槽位, 完整清单, 首个锁定槽 x)。"""
    cache = getattr(g, "slot_scan", None)
    now = time.time()
    if cache and now - cache[0] < 1800 and cache[1] is not None:
        ducks, anchors = cache[1], cache[2]
    else:
        ducks = walk_ducks(g)
        anchors = sorted({round(float(d.get("x") or 0), 3) for d in ducks})

    inv: list[dict] = []
    for i, x in enumerate(anchors):
        hit = next((d for d in ducks if abs(d["x"] - x) < 0.03), None)
        inv.append({"i": i, "x": x,
                    "kind": hit["kind"] if hit else "padlock",
                    "state": hit.get("state") if hit else None,
                    "price": hit.get("price") if hit else None,
                    "score": 1.0,
                    "cx": g.l + int(x * g.w),
                    "cy": g.t + int(SLOT_ROW_Y * g.h)})
    slots = [s for s in inv if s["kind"] in ("duck", "new")]
    pads = [s["x"] for s in inv if s["kind"] == "padlock" and s["x"] > (slots[-1]["x"] if slots else 0)]
    return slots, inv, (min(pads) if pads else 1.0)


def _duck_card_fp(g: Game):
    """当前鸭子卡片的指纹，用来确认「点击真的切换了鸭子」。"""
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(0.30 * h):int(0.60 * h), int(0.28 * w):int(0.72 * w)]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48))


def click_slot(g: Game, s: dict, verify: bool = True) -> str | None:
    """点一个槽位，并用**高亮**确认真的选中；选不中返回 None（调用方跳过）。

    verify=False 只省掉「是不是鸭子页」的额外整页 OCR，高亮校验始终保留。
    """
    rx = s.get("x") or (s["cx"] - g.l) / max(1, g.w)
    anchors = _row_anchors(g) or [rx]
    # 扫描时的坐标可能因整排滚动而整体偏移半格（点锁后页面会复位），
    # 直接点旧坐标会落在缝隙里、读到上一只鸭子 → 先吸附到当前最近的锚点。
    near = min(anchors, key=lambda a: abs(a - rx))
    if abs(near - rx) < 0.035:
        rx = near
    ok, t = click_slot_verified(g, rx, anchors)
    log_event(action="slot_click", slot_x=round(rx, 3), kind=s.get("kind"),
              selected=ok)
    if not ok:
        return None
    if "BUY A SLOT" in t.upper() and not any(w in t for w in DUCK_STATE_WORDS):
        return None
    if not verify:
        return ""
    if current_page(g, tries=1)[0] != "duck":
        g.close_popup()
        g.nav("duck", wait=1.0)
        return None
    return t


def select_slot(g: Game, i: int) -> bool:
    """点进度条上第 i 个鸭子槽位图标来选中该鸭子。返回是否落在鸭子页。

    为什么不用拖拽：横向拖拽会间歇性失效（实测 4/1/4 只），导致漏扫；
    直接点槽位图标每次都能切换（实测 4/4、8/8 命中）。
    """
    if i >= len(SLOT_XS):
        return False
    g.click(SLOT_XS[i], SLOT_Y, wait=1.0)
    if current_page(g, tries=1)[0] != "duck":
        # 可能是锁槽（BUY A SLOT）或收藏页：关掉并回鸭子页
        g.close_popup()
        g.nav("duck", wait=1.0)
        return False
    return True


def duck_sig(g: Game) -> str:
    """当前这只鸭子的身份指纹：抹掉所有数字（计时器/价格都在变），只留稳定文案。"""
    t = g.text()
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t))[:140]


def _card_fp(g: Game):
    """鸭子卡片的灰度指纹，用来判断拖拽是否真的换了鸭子（比文本可靠：文本会重复）。"""
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(0.30 * h):int(0.62 * h), int(0.20 * w):int(0.80 * w)]
    if crop.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))


def carousel_next(g: Game) -> bool:
    """切到下一只鸭子。鸭子页是**横向轮播**，实测只有快拖能切换。返回是否真的换了。

    拖拽会被浮层/提示打断，所以失败时换一个起始高度再试一次，避免「只看到 1 只」的假结论。
    """
    for attempt, (y0, x1) in enumerate(((0.470, 0.25), (0.500, 0.30), (0.470, 0.25))):
        if attempt == 2:
            # 前两次都没动：卡片可能带锁定遮罩/提示浮层把拖拽吞了，先点一下再拖
            g.click(0.500, 0.470, wait=0.8)
        before = _card_fp(g)
        g.drag((0.75, y0), (x1, y0))
        time.sleep(0.7)
        after = _card_fp(g)
        if current_page(g, tries=1)[0] != "duck":
            # 拖拽偶尔会带出提示浮层（如「查看物品」），关掉并回到鸭子页再判断
            if g.close_popup():
                time.sleep(0.5)
            g.nav("duck", wait=1.0)
            after = _card_fp(g)
            if after is None:
                continue
        if before is None or after is None:
            return True
        if float(abs(after.astype(int) - before.astype(int)).mean()) > 3.0:
            return True
    return False


def find_feedable(g: Game, retry: bool = True, skip: set | None = None) -> int | None:
    """① 视觉枚举槽位 → ② 只对 duck 槽位读鸭子状态 → ③ 找单价 <= FEED_CAP 的。

    槽位状态（padlock / duck / unknown）**先于**鸭子状态确定，且锁定槽位一律不点。
    """
    from collections import Counter
    slots, inv, pad_x = duck_slots_from_inventory(g)
    counts = Counter(s["kind"] for s in inv)
    log(f"  槽位清单：{len(slots)} 个鸭子槽位 / {counts.get('padlock', 0)} 个锁定"
        f"（首锁 x={pad_x:.2f}）/ {counts.get('unknown', 0)} 个未知")
    log_event(action="slot_inventory", slots=len(slots),
              padlocks=counts.get("padlock", 0), unknown=counts.get("unknown", 0),
              first_padlock_x=round(pad_x, 3))
    if not slots:
        g.feed_reason = "vision-miss"
        log("  视觉未识别到任何鸭子槽位（可能被浮层挡住或布局变化），跳过本轮")
        return None

    unread = 0
    new_ducks = 0
    read = 0
    over = getattr(g, "over_cap", None)
    if over is None:
        over = {}
        g.over_cap = over
    nobub = getattr(g, "no_bubble", None)
    if nobub is None:
        nobub = {}
        g.no_bubble = nobub
    now = time.time()
    skipped = 0
    for s in slots:
        key = (round((s["cx"] - g.l) / g.w, 3), round((s["cy"] - g.t) / g.h, 3))
        # 已知超上限的鸭子先跳过（价格只涨不跌，直到重置倒计时，30 分钟内不会变）
        ts = over.get(key)
        if ts and now - ts < 1800:
            skipped += 1
            continue
        # 已知「没有喂食气泡」的也跳过（繁殖中/待质押/出售中，10 分钟内不会变）
        ts2 = nobub.get(key)
        if ts2 and now - ts2 < 600:
            skipped += 1
            continue
        t = click_slot(g, s)
        if t is None:
            log(f"  槽位 x={(s['cx'] - g.l) / g.w:.3f} 不是鸭子页，已跳过")
            continue
        read += 1
        has, price = read_feed_state(g)
        if has and price is not None and price <= FEED_CAP:
            g.feed_reason = "ok"
            return price
        if has and price is not None and price > FEED_CAP:
            over[key] = now      # 记下来，半小时内不再点开它
        if not has:
            nobub[key] = now     # 没气泡：10 分钟内不再点开它
        if has and price is None:
            unread += 1
            log("  当前这只鸭子有喂食气泡但单价读不出，跳过")
    if new_ducks:
        g.feed_reason = "no-slot"
        log(f"  读完 {read} 只鸭子（跳过 {skipped} 只已知超上限）："
            f"{new_ducks} 只新鸭子等待分配槽位，没有可喂的")
    else:
        g.feed_reason = "no-feedable"
        log(f"  读完 {read} 只鸭子（跳过 {skipped} 只已知超上限）："
            f"没有可喂的（单价读不出 {unread} 只）")
    return None


_OCR_FAST = None


_FAST_OCR_OK: bool | None = None      # None=未探测；False=当前引擎仅识别不可用


def ocr_fast(img) -> str:
    """尽量用「仅识别」模型读裁剪图，**不可用时自动回退整页识别**。

    2026-11-14 实测：当前引擎 `use_det=False` 直接返回空串（`[['', 0.0]]`），
    而默认模式能读出「点击喂食4」。喂食全靠这个函数读气泡，所以必须回退，
    否则整条喂食链路会静默停摆（表现为 fed=0 / bubble-gone）。
    """
    global _FAST_OCR_OK
    if _FAST_OCR_OK is not False:
        try:
            out = bench.ocr_engine()(img, use_det=False, use_cls=False)
            res = out[0] if isinstance(out, tuple) else out
            parts = []
            for item in (res or []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    parts.append(str(item[0]))
                elif isinstance(item, str):
                    parts.append(item)
            txt = " ".join(p for p in parts if p)
            if txt:
                _FAST_OCR_OK = True
                return txt
            _FAST_OCR_OK = False
            log("  OCR：仅识别模式返回空，已切回整页识别（更慢但正确）")
        except Exception:
            _FAST_OCR_OK = False
    try:
        res, _ = bench.ocr_engine()(img)
        return " ".join(str(t) for t in
                        ((i[1] if len(i) == 3 else i[0]) for i in (res or [])))
    except Exception:
        return ""


def read_price(g: Game, tries: int = 3) -> int | None:
    """只 OCR 玉米图标右侧的数字区，失败重试。返回 None 表示数字读不出。"""
    import cv2
    rx, ry, rw, rh = PRICE_BOX
    for _ in range(max(1, tries)):
        img = grab(g.hwnd)
        if img is None:
            time.sleep(0.2)
            continue
        h, w = img.shape[:2]
        crop = img[int(ry * h):int((ry + rh) * h), int(rx * w):int((rx + rw) * w)]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        txt = ocr_fast(big)
        if not txt:
            res, _ = bench.ocr_engine()(big)      # 兜底：走完整模型
            txt = " ".join(str(t) for t in
                           ((i[1] if len(i) == 3 else i[0]) for i in (res or [])))
        m = re.search(r"(\d{1,5})", txt)
        if m:
            return int(m.group(1))
        time.sleep(0.25)
    return None


def _has_bubble_pixels(g: Game) -> bool:
    """用像素判断有没有喂食气泡：白底圆角框（亮 + 低饱和）。

    OCR 一次约 0.5-1 秒，而 9 只鸭子里多数没有气泡；先花 50ms 看像素，
    没有气泡就直接判空，省掉整轮 OCR。
    区域必须收紧到气泡本身（x 0.34-0.66, y 0.595-0.655）：
    放宽会把黄色鸭身也算进去（亮度同样 >200），导致永远判「有气泡」。
    """
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return False
    h, w = img.shape[:2]
    crop = img[int(0.595 * h):int(0.655 * h), int(0.34 * w):int(0.66 * w)]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 60)
    return float(white.mean()) > 0.06


def read_feed_state(g: Game, tries: int = 3, retry: bool = True) -> tuple[bool, int | None]:
    """返回 (是否有喂食气泡, 单价)。

    retry=True：读不到「喂食」时重读两次（喂食循环里必须，否则动画遮住气泡会被误判喂满）。
    retry=False：只读一次（批量扫描时用，否则 9 只鸭子要花 20 秒以上）。
    """
    t = _feed_band_text(g)
    if "喂食" not in t and retry:
        for _ in range(2):
            time.sleep(0.35)
            t = _feed_band_text(g)
            if "喂食" in t:
                break
    if "喂食" not in t:
        return False, None
    return True, read_price(g, tries if retry else 1)


def _feed_band_text(g: Game) -> str:
    """读气泡带文字（走仅识别模型）。"""
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return ""
    h, w = img.shape[:2]
    rx, ry, rw, rh = FEED_BAND
    crop = img[int(ry * h):int((ry + rh) * h), int(rx * w):int((rx + rw) * w)]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return ocr_fast(big)     # 仅识别模型（~0.1s）；不做完整模型兜底（那一步要 1s）


def read_feed_price(g: Game) -> int | None:
    return read_price(g)



# ---------------------------------------------------------------- 动作

def do_feed(g: Game, cap: int = FEED_CAP, max_clicks: int = 400) -> dict:
    """喂到单价 > cap 或气泡消失。返回 {fed, start_price, end_price, stopped}。

    提速关键：**按剩余额度批量点击**，而不是每点一次就读一次 OCR。
    原来的循环「读价 → 点 1 下 → 再读价」，单次喂食约 1.8 秒，喂满一只 3 分钟；
    现在按「预计还能点几下」批量点，读完再校正，实测快 3-5 倍。

    安全：批量数由**实测增量**推算（inc = 期间涨价 / 点击次数），
    并乘 0.8 留余量，确保不会在价格已超 cap 的情况下继续点。
    """
    fed = 0
    start = None
    last = None
    stopped = "no-duck"
    inc = 1.0          # 每次喂食的涨价估计
    while fed < max_clicks:
        has, price = read_feed_state(g)
        if not has:
            stopped = "bubble-gone"
            break
        if price is None:
            stopped = "price-unread"
            break
        if start is None:
            start = price
        if price > cap:
            stopped = "cap"
            break
        last = price
        gap = cap - price
        # 还能点几下：预留 20% 余量，至少 1 下、最多 8 下
        n = max(1, min(8, int(gap / max(inc, 0.5) * 0.8)))
        for _ in range(n):
            g.click(0.500, 0.623, wait=0.13)
            fed += 1
        if fed >= max_clicks:
            stopped = "max_clicks"
            break
        time.sleep(0.30)     # 让喂食动画走完，否则气泡被遮住会被误判成「喂满」
        has2, price2 = read_feed_state(g)
        if not has2:
            stopped = "bubble-gone"
            break
        if price2 is not None and price2 > price:
            inc = max(0.3, (price2 - price) / n)
        if fed % 10 < n:
            log(f"  喂食 {fed} 次，当前单价 {price2 if price2 is not None else price}")
            log_event(action="feed_progress", fed=fed,
                      price=price2 if price2 is not None else price)
    else:
        stopped = "max_clicks"
    if fed:
        log_event(action="feed", fed=fed, start_price=start,
                  end_price=last, stopped=stopped)
    return {"fed": fed, "start_price": start, "end_price": last, "stopped": stopped}



EGG_INFO_BOX = (0.14, 0.725, 0.55, 0.075)   # 蛋信息行的窗口相对区域


def read_egg_info(g: Game) -> tuple[str, int] | None:
    """读当前选中蛋的信息行（裁剪 + 放大后 OCR，只接受严格格式 `蛋[N]`）。

    为什么不读整页：信息行常和「解锁鸭蛋并获得一个额外槽位」横幅挤在一起，
    整页 OCR 会把两者混在一起，导致把每格都判成 SLOT（2026-11-13 实测）。
    放大 3x/5x 各试一次；仍读不出就返回 None —— **宁可跳过，也不能猜**。
    """
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return None
    H, W = img.shape[:2]
    x0, y0 = int(EGG_INFO_BOX[0] * W), int(EGG_INFO_BOX[1] * H)
    x1 = int((EGG_INFO_BOX[0] + EGG_INFO_BOX[2]) * W)
    y1 = int((EGG_INFO_BOX[1] + EGG_INFO_BOX[3]) * H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    for scale in (5, 3):
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = bench.ocr_engine()(big)
        txt = " ".join(t for _b, t, _s in (res or []))
        m = re.search(r"(\S*?蛋)\s*\[\s*(\d{1,2})\s*\]", txt)
        if m:
            name = "爱心蛋" if "爱心" in m.group(1) else "蛋"
            return (name, int(m.group(2)))
    return None


def egg_info(g: Game, cell: tuple[int, int]) -> tuple[str, int] | None:
    """点开一格读 (类型, 等级)。读不出就返回 None，绝不猜测。

    注意：不再用 "槽位" in 文本 来判断锁格——横幅里就含「槽位」二字，
    那会把整块板误判成锁定。
    """
    i, j = cell
    for _ in range(2):
        g.click(GRID_X[j], GRID_Y[i], wait=1.15)
        info = read_egg_info(g)
        if info:
            return info
        time.sleep(0.3)
    return None


def _cell_sig_color(bgr, cx: int, cy: int, half: int = 24, size: int = 24):
    """蛋格彩色签名（用于排候选，不用于最终判定）。"""
    import cv2
    import numpy as np
    patch = bgr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    if patch.size == 0:
        return None
    p = cv2.resize(patch, (size, size)).astype("float32")
    return cv2.GaussianBlur(p, (3, 3), 0)


def merge_candidates(g: Game, thr: float = 0.90, top: int = 10) -> list[tuple]:
    """按外观相似度排出「值得一试」的配对，返回 [(cellA, cellB, ncc)]。

    只用来**排候选**，不做最终判定：实测相似度分不开真假
    （真正的同一对 0.99，明显不同的两个蛋也有 0.95）。
    真正的判据是拖拽后两格有没有变化。
    """
    from collections import defaultdict
    import numpy as np
    bgr = grab(g.hwnd)
    if bgr is None:
        return []
    H, W = bgr.shape[:2]
    sigs, cells = {}, []
    for i in range(len(GRID_Y)):
        for j in range(len(GRID_X)):
            if (i, j) == CART_CELL or (i, j) in g.slot_cells:
                continue
            s = _cell_sig_color(bgr, int(GRID_X[j] * W), int(GRID_Y[i] * H))
            if s is not None:
                sigs[(i, j)] = s
                cells.append((i, j))
    out = []
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            x = sigs[cells[a]].ravel()
            y = sigs[cells[b]].ravel()
            x = x - x.mean()
            y = y - y.mean()
            denom = (np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
            ncc = float(np.dot(x, y) / denom)
            if ncc >= thr:
                out.append((ncc, cells[a], cells[b]))
    out.sort(reverse=True)
    return [(a, b, round(n, 4)) for n, a, b in out[:top]]


def _cell_sig(bgr, cx: int, cy: int, half: int = 22):
    """蛋格外观签名：12×12 灰度、零均值单位方差（对亮度不敏感）。"""
    import numpy as np
    import cv2
    patch = bgr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (12, 12)).astype("float32")
    g -= g.mean()
    s = float(g.std()) or 1.0
    return g / s


def grid_clusters(g: Game, thr: float = 18.0) -> list[list[tuple[int, int]]]:
    """按外观把 25 格聚成等价类：同一类 = 视觉上同一等级的蛋。"""
    import numpy as np
    import ui_icons
    bgr = ui_icons.grab(g.hwnd)
    if bgr is None:
        return []
    cells = []
    for i, ry in enumerate(GRID_Y):
        for j, rx in enumerate(GRID_X):
            cells.append(((i, j), _cell_sig(bgr, int(rx * g.w), int(ry * g.h))))
    groups, used = [], set()
    for idx, (cell, sig) in enumerate(cells):
        if idx in used:
            continue
        grp, used2 = [cell], {idx}
        for k in range(idx + 1, len(cells)):
            if k in used:
                continue
            if float(np.abs(sig - cells[k][1]).mean()) < thr:
                grp.append(cells[k][0])
                used2.add(k)
        used |= used2
        groups.append(grp)
    groups.sort(key=len, reverse=True)
    return groups


def _merge_assert(g: Game, src: tuple[int, int], dst: tuple[int, int],
                  lvl: int) -> tuple[bool, str, object, object]:
    """合并后置断言：**两格都要读**，任一格变成 lvl+1 才算成功。

    实测约定不统一：有时升级结果落在目标格，有时落在源格（见 2026-11-13 事件流
    `post_level=3` 但 dst 是另一格）。只读一格必然误判。

    返回 (ok, evidence, src_info, dst_info)。
    """
    a = egg_info(g, src)
    b = egg_info(g, dst)
    if (a and a[1] == lvl + 1) or (b and b[1] == lvl + 1):
        return True, "level", a, b
    return False, "none", a, b


def _merge_fast_once(g: Game) -> bool:
    """视觉聚类快路径：找一个配对并拖拽一次。成功返回 True。

    按外观聚类候选组，对候选**核对两格等级**再拖（2 次读图 vs 25 格盲扫）。
    注意：同一簇里要**把所有配对都试一遍**，不能只看前两个——
    第 0、1 个可能恰好等级不同，但 1、2 个其实是可合并的。
    """
    for cluster in [c for c in grid_clusters(g, 0.22) if len(c) >= 2][:3]:
        for x in range(len(cluster)):
            for y in range(x + 1, len(cluster)):
                (i1, j1), (i2, j2) = cluster[x], cluster[y]
                a = egg_info(g, (i1, j1))
                b = egg_info(g, (i2, j2))
                if not a or not b or a[0] == "SLOT" or b[0] == "SLOT":
                    continue
                if a != b:
                    continue
                if a[1] >= int(CRACK_MIN_LEVEL):
                    log(f"  跳过 Lv{a[1]}（达到敲蛋阈值）")
                    continue
                log(f"  合并(视觉) {a[0]}[{a[1]}] ({i1},{j1}) -> ({i2},{j2})")
                g.drag((GRID_X[j1], GRID_Y[i1]), (GRID_X[j2], GRID_Y[i2]))
                time.sleep(0.9)
                ok, ev, sa, sb = _merge_assert(g, (i1, j1), (i2, j2), a[1])
                log_event(action="merge", egg_type=a[0], level=a[1],
                          src=[i1, j1], dst=[i2, j2], via="vision",
                          post_ok=ok, evidence=ev,
                          post_src=sa, post_dst=sb)
                if ok:
                    return True
                log(f"  后置断言失败：源 {sa} 目标 {sb}（期望 Lv{a[1] + 1}）")
    return False


def _merge_slow_once(g: Game) -> bool:
    """25 格全扫：找同级蛋配对并拖拽一次，并做同样的后置断言。"""
    groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for i in range(5):
        for j in range(5):
            if (i, j) == CART_CELL or (i, j) in g.slot_cells:
                continue
            info = egg_info(g, (i, j))
            if not info:
                continue
            if info[0] == "SLOT":
                g.slot_cells.add((i, j))
                continue
            groups.setdefault(info, []).append((i, j))
    pairs = [(k, v) for k, v in groups.items() if len(v) >= 2]
    if not pairs:
        log("  没有可合并的同级蛋")
        return False
    pairs.sort(key=lambda kv: kv[0][1], reverse=True)
    key, cells = pairs[0]
    best = None
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            d = abs(cells[a][0] - cells[b][0]) + abs(cells[a][1] - cells[b][1])
            if best is None or d < best[0]:
                best = (d, cells[a], cells[b])
    _, (i1, j1), (i2, j2) = best
    log(f"  合并 {key[0]}[{key[1]}] ({i1},{j1}) -> ({i2},{j2})")
    g.drag((GRID_X[j1], GRID_Y[i1]), (GRID_X[j2], GRID_Y[i2]))
    time.sleep(0.9)
    ok, ev, sa, sb = _merge_assert(g, (i1, j1), (i2, j2), key[1])
    log_event(action="merge", egg_type=key[0], level=key[1],
              src=[i1, j1], dst=[i2, j2], via="scan",
              post_ok=ok, evidence=ev, post_src=sa, post_dst=sb)
    if not ok:
        log(f"  后置断言失败：源 {sa} 目标 {sb}（期望 Lv{key[1] + 1}）")
    return ok


EGG_LEVEL_MAP = ROOT / "results" / "computer_use" / "egg_level_map.json"


def _load_level_map() -> dict:
    try:
        return json.loads(EGG_LEVEL_MAP.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_level_map(m: dict) -> None:
    try:
        EGG_LEVEL_MAP.parent.mkdir(parents=True, exist_ok=True)
        EGG_LEVEL_MAP.write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception:
        pass


EGG_SIG_SIZE = 36          # 蛋块裁剪边长（像素）
EGG_SIG_OUT = 24           # 签名分辨率
# 同种判定阈值：实测真对 ≤3.71，异类最小 13.15，取中间偏下的 8.0。
# （不要用"最大间隙自适应"：板上没有真对时它会一路涨到 12，逼近异类区，反而危险。）
EGG_SAME_THR = 8.0
# 候选尝试阈值：拖错一对**没有副作用**（只是选中态变化），
# 所以不需要用阈值去"预判能不能合"——放宽到 18，全部交给合并后的计数验证把关。
# 否则 8~18 之间那些（例如带绿底的同款蛋）会被永久漏掉，合并就不彻底。
EGG_TRY_THR = 24.0         # 候选尝试阈值（实测最小非同位对 19.8，次小 28.4）
EGG_LOCK_V = 40.0          # 亮度低于此判为锁定格


EGG_LIB = ROOT / "results" / "computer_use" / "egg_lib.json"


def _load_lib() -> list:
    try:
        return json.loads(EGG_LIB.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_lib(lib: list) -> None:
    try:
        EGG_LIB.parent.mkdir(parents=True, exist_ok=True)
        EGG_LIB.write_text(json.dumps(lib, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def lib_match(sig, lib: list):
    """在蛋种库里找最接近的一档。返回 (距离, 条目) 或 (None, None)。"""
    import numpy as np
    best = None
    for e in lib:
        try:
            ref = np.array(e["sig"], dtype="float32").reshape(sig.shape)
        except Exception:
            continue
        d = float(np.abs(sig - ref).mean())
        if best is None or d < best[0]:
            best = (d, e)
    if best and best[0] < EGG_SAME_THR:
        return best
    return (None, None)


def lib_remember(sig, level, lib: list) -> bool:
    """把「这个外观 = 这个等级」记进库；已存在则只累加次数。"""
    d, e = lib_match(sig, lib)
    if e is not None:
        e["seen"] = int(e.get("seen", 1)) + 1
        if level is not None and e.get("level") is None:
            e["level"] = int(level)
            return True
        return False
    flat = [round(float(v), 1) for v in sig.reshape(-1)]
    lib.append({"sig": flat,
                "shape": list(sig.shape),
                "level": int(level) if level is not None else None,
                "seen": 1})
    return True


def _learn_merged(src_sig, new_sig, lib: list, src_level) -> None:
    """合并后学习：新外观的等级 = 源等级 + 1（源等级已知时）。"""
    d, e = lib_match(src_sig, lib)
    lvl = (e.get("level") if e else None) or src_level
    if lvl is not None:
        if lib_remember(new_sig, int(lvl) + 1, lib):
            _save_lib(lib)
            log(f"  学到新蛋种：{lvl + 1} 级（来自 {lvl} 级合并）")


def _egg_sig(img, cell, win: int = 26):
    """取单元格内蛋块的质心对齐签名；找不到蛋块返回 None。

    先做一步「去外圈绿底」：带可提交标记的格子，圆角底是绿色，
    会把同款蛋的签名距离从 ~8 拉到 ~12-16（超过阈值而被判成不同种、漏合）。
    只替换格子外圈 8px 的绿色，蛋体本身不动——否则绿色系蛋会被误伤。
    """
    import cv2
    import numpy as np
    H, W = img.shape[:2]
    cx, cy = int(GRID_X[cell[1]] * W), int(GRID_Y[cell[0]] * H)

    cell_size = 48
    half_c = cell_size // 2
    x0, y0 = cx - half_c, cy - half_c
    cpatch = img[max(0, y0):y0 + cell_size, max(0, x0):x0 + cell_size]
    if cpatch.shape[0] < cell_size or cpatch.shape[1] < cell_size:
        return None
    cpatch = cpatch.copy()
    hsv_c = cv2.cvtColor(cpatch, cv2.COLOR_BGR2HSV)
    green = ((hsv_c[:, :, 0] > 35) & (hsv_c[:, :, 0] < 100)
             & (hsv_c[:, :, 1] > 60) & (hsv_c[:, :, 2] > 60))
    ring = np.zeros_like(green)
    ring[:8, :] = True
    ring[-8:, :] = True
    ring[:, :8] = True
    ring[:, -8:] = True
    m = green & ring
    if m.any():
        cpatch[m] = (32, 26, 40)          # 普通暗底

    hsv = cv2.cvtColor(cpatch, cv2.COLOR_BGR2HSV)
    # 蛋像素判定不能只看饱和度：**白蛋饱和度极低（S≈10-30）会被整片过滤掉**
    # （2026-11-14 实测：5 个白蛋完全没进扫描，最该合的蛋从没被考虑）。
    mask = (((hsv[:, :, 2] > 110) | (hsv[:, :, 1] > 60)) &
            (hsv[:, :, 2] > 55)).astype(np.uint8)
    if int(mask.sum()) < 50:
        return None
    ys, xs = np.nonzero(mask)
    gx = int(xs.mean())
    gy = int(ys.mean())
    half = EGG_SIG_SIZE // 2
    patch = cpatch[max(0, gy - half):gy + half, max(0, gx - half):gx + half]
    if patch.shape[0] < EGG_SIG_SIZE or patch.shape[1] < EGG_SIG_SIZE:
        return None
    return cv2.resize(patch, (EGG_SIG_OUT, EGG_SIG_OUT),
                      interpolation=cv2.INTER_AREA).astype("float32")


def scan_board_pixels(g: Game, stable: bool = True) -> dict:
    """**纯像素**读板，不点击。

    1. 锁定格判定：锁定格亮度上限被压住（maxV≈115，可用格 252），
       用 k-means 二聚类自动分，不写绝对阈值。
    2. 同种蛋判定：蛋块质心对齐后取签名，距离阈值**自适应**——
       取所有可用格两两距离的最大间隙中点，限制在 [3, 12]。
       实测真对 ≤3.7、异类 ≥13，所以这个区间总是安全的。
    3. 稳定性：连扫两次，格子集合一致才采信（避免扫到合并动画的中间帧）。
    """
    import cv2
    import numpy as np

    def _one():
        img = grab(g.hwnd)
        if img is None:
            return None
        H, W = img.shape[:2]
        vs, sigs, values = {}, {}, {}
        for i in range(5):
            for j in range(5):
                cell = (i, j)
                if cell == CART_CELL:
                    continue
                cx, cy = int(GRID_X[j] * W), int(GRID_Y[i] * H)
                crop = img[cy - 24:cy + 24, cx - 24:cx + 24]
                if crop.size == 0:
                    continue
                vs[cell] = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
                values[cell] = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].max())
                s = _egg_sig(img, cell)
                if s is not None:
                    sigs[cell] = s

        # 锁定格：maxV 二聚类（两簇差异够大才采信）
        locked, thr_v = [], EGG_LOCK_V
        if len(values) >= 6:
            arr = np.array([[v] for v in values.values()], dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
            _, _lab, centers = cv2.kmeans(arr, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
            lo, hi = sorted(float(c[0]) for c in centers)
            if hi > 0 and (hi - lo) / hi > 0.25:
                thr_v = (lo + hi) / 2
                locked = [c for c, mv in values.items() if mv < thr_v]

        free = [c for c in sigs if c not in locked]

        # 同种阈值：固定 8.0（实测真对 ≤3.71、异类 ≥13.15）。
        # 顺手把最大间隙记下来，作为漂移预警。
        thr = EGG_SAME_THR
        gap_note = ""
        if len(free) >= 3:
            ds = sorted(float(np.abs(sigs[a] - sigs[b]).mean())
                        for x, a in enumerate(free) for b in free[x + 1:])
            if len(ds) >= 2:
                gaps = [(ds[k + 1] - ds[k], k) for k in range(len(ds) - 1)]
                gap, k = max(gaps)
                gap_note = f"最大间隙 {gap:.1f}（{ds[k]:.1f}→{ds[k + 1]:.1f}）"

        # 聚类用**候选阈值**（18）：把 8~18 的近似对也纳入候选，
        # 拖错没有副作用，能不能合由后续计数验证决定。
        used, clusters = set(), []
        for c in free:
            if c in used:
                continue
            grp = [c]
            used.add(c)
            for o in free:
                if o in used:
                    continue
                if float(np.abs(sigs[c] - sigs[o]).mean()) < EGG_TRY_THR:
                    grp.append(o)
                    used.add(o)
            clusters.append(grp)
        clusters.sort(key=len, reverse=True)
        # 结合蛋种库给出每格的等级（未知则 None）
        lib = _load_lib()
        lvl_of = {}
        for c, s in sigs.items():
            d, e = lib_match(s, lib)
            lvl_of[c] = (e.get("level") if e else None)
        return {"clusters": clusters, "locked": locked, "sigs": sigs,
                "thr": thr, "thr_v": thr_v, "n_free": len(free),
                "gap_note": gap_note, "stamp": tuple(sorted(free)),
                "lvl_of": lvl_of, "lib": lib}

    first = _one()
    if first is None:
        return {"clusters": [], "locked": [], "sigs": {}, "thr": EGG_SAME_THR,
                "thr_v": EGG_LOCK_V, "n_free": 0, "stamp": ()}
    if stable:
        for _ in range(2):
            time.sleep(0.25)
            second = _one()
            if second and second["stamp"] == first["stamp"]:
                return second
            if second:
                first = second
    return first


def _count_same(sigs: dict, ref, thr: float = EGG_SAME_THR) -> int:
    """统计当前画面里和 ref 签名相同的格子数。"""
    import numpy as np
    n = 0
    for s in sigs.values():
        if float(np.abs(s - ref).mean()) < thr:
            n += 1
    return n


def anchor_levels(g: Game, b: dict, limit: int = 6) -> int:
    """给未知等级的蛋种锚定等级：每组点一个代表格读 `蛋[N]`，记进蛋种库。

    只在库里缺这个外观时才做，所以每个蛋种一生只点一次；
    读不出（例如 7 被读成「切」）就跳过，交给赔率映射或后续合并学习。
    """
    lvl_of = b.get("lvl_of") or {}
    lib = b.get("lib") or _load_lib()
    done = 0
    for grp in b["clusters"]:
        if done >= limit:
            break
        cell = grp[0]
        if lvl_of.get(cell) is not None:
            continue
        sig = b["sigs"].get(cell)
        if sig is None:
            continue
        try:
            g.click(GRID_X[cell[1]], GRID_Y[cell[0]], wait=1.0)
            st, _ty, lvl = read_cell_state(g)
        except Exception:
            continue
        if st == "egg" and lvl:
            if lib_remember(sig, lvl, lib):
                _save_lib(lib)
            lvl_of[cell] = lvl
            done += 1
    if done:
        log(f"  蛋种库锚定 {done} 个新外观（库中共 {len(lib)} 条）")
    return done


def _sig_key(sig) -> tuple:
    """把签名粗量化成可哈希的 key，用于记录「哪种蛋配哪种蛋拖不动」。

    必须按**外观**记而不是按坐标：格子会被补位换掉，
    同一个坐标后面可能是完全不同的蛋（按坐标拉黑会误伤新蛋）。
    """
    import numpy as np
    return tuple((sig.flatten() // 6).astype(int).tolist())


def do_merge(g: Game, budget: int = 20, max_seconds: float = 240.0,
             allow_crack: bool = False, crack_policy: str = "lowest",
             max_cracks: int = 3) -> int:
    """按「像素读板 → 合并 → 重新截图重算」收敛。

    停止条件（任一先到）：
      - 没有未尝试过的可合并对；
      - 合并次数达到 budget；
      - 用时超过 max_seconds。

    为什么要时间上限：每合并一次都会补位一颗新蛋，可能立刻又凑出同款对，
    所以「合到底」不是有限步数，而是"一直合到没有为止"——用时间兜底才不会被预算切断。
    """
    t0 = time.time()
    done = 0
    cracked = 0
    blocked: set = set()
    crack_skip: set = set()
    anchored = False
    b = None                     # 复用上一次合并后的扫描结果，省一次全板读
    while done < budget and time.time() - t0 < max_seconds:
        if b is None:
            b = scan_board_pixels(g)
        lvl_of = b.get("lvl_of") or {}
        # 配对键必须用**外观**（距离阈值内的像素一致），不能用等级：
        # 蛋种库的等级是推断出来的，误差会累积，用等级配对会拖不同的蛋（实测"没生效"频发）。
        # 等级只用于显示和记录。
        pairs = [c for c in b["clusters"] if len(c) >= 2]
        lv_desc = "、".join(
            f"{lvl_of.get(grp[0]) or '?'}级×{len(grp)}" for grp in b["clusters"][:8])
        log(f"  像素读板：可用 {b['n_free']} 格 / 锁定 {len(b['locked'])} 格；{lv_desc}")
        log(f"  可合并组 {[p for p in pairs]}")
        if b["n_free"] < 3:
            log("  可用格过少（可能被浮层遮挡或画面未就绪），本轮跳过合并")
            break

        # 有空缺等级时先锚定几个（每个蛋种只点一次）。
        # 只在本轮第一次读板时做——每次合并都锚定会白点几十次。
        if not anchored and any(v is None for v in (b.get("lvl_of") or {}).values()):
            anchored = True
            if anchor_levels(g, b, limit=6):
                b = scan_board_pixels(g)
                lvl_of = b.get("lvl_of") or {}
                pairs = [c for c in b["clusters"] if len(c) >= 2]
        log_event(action="board_scan_pixels", clusters=b["clusters"],
                  locked=b["locked"], thr=round(b["thr"], 1))
        pick = None
        best = None
        for grp in pairs:
            lv = lvl_of.get(grp[0])
            for x in range(len(grp)):
                for y in range(x + 1, len(grp)):
                    # 拉黑键用**外观对**，不用坐标：坐标会被补位换蛋，
                    # 按坐标拉黑会把「后来换成能合的新蛋」也误杀。
                    ka = _sig_key(b["sigs"][grp[x]])
                    kb = _sig_key(b["sigs"][grp[y]])
                    key = (ka, kb) if ka <= kb else (kb, ka)
                    if key in blocked:
                        continue
                    # 先按**距离最像**排序（最有把握），距离相同再看等级
                    import numpy as _np
                    dist = float(_np.abs(b["sigs"][grp[x]] - b["sigs"][grp[y]]).mean())
                    score = (round(dist, 2), -(lv or 0))
                    if best is None or score < best[0]:
                        best = (score, grp[x], grp[y], grp, key)
        if best is not None:
            pick = (best[1], best[2], best[3], best[4])
        if pick is None:
            # 没有可合并对时：敲掉一颗孤蛋腾格子，让新蛋补位（实测补位后常立刻凑出新对）
            if not allow_crack or cracked >= max_cracks:
                log("  没有（未尝试过的）可合并对")
                break
            tgt = pick_crack_target(b, crack_policy, exclude=crack_skip)
            if tgt is None:
                log("  没有可敲的孤蛋（或都敲不动）")
                break
            lv_t = (b.get("lvl_of") or {}).get(tgt)
            log(f"  无可合并对 → 敲孤蛋 {tgt}（{lv_t} 级，策略 {crack_policy}）腾位")
            if not crack_cell(g, tgt):
                # 敲不动就换一颗——可能是装饰物（南瓜车旁的花篮）或被活动横幅挡住
                crack_skip.add(tgt)
                log(f"  {tgt} 敲不动，换一颗（已跳过 {len(crack_skip)} 颗）")
                if len(crack_skip) >= 4:
                    log("  连续 4 颗都敲不动，本轮停止")
                    break
                continue
            cracked += 1
            blocked.clear()
            b = None          # 敲完必须重扫（补位蛋已落格）
            continue
        a, bb, grp, pair_key = pick
        ref = b["sigs"][a]
        n0 = _count_same(b["sigs"], ref)
        lv_a = lvl_of.get(a)
        tag = f"{lv_a} 级" if lv_a else "同种"
        log(f"  合并 {a} -> {bb}（{tag}共 {n0} 个: {grp}）")
        g.drag((GRID_X[a[1]], GRID_Y[a[0]]), (GRID_X[bb[1]], GRID_Y[bb[0]]))
        time.sleep(0.45)          # 实测 0.2s 画面就稳定，0.7 太保守
        after = scan_board_pixels(g, stable=False)   # 刚合完，不必再双重确认
        n1 = _count_same(after["sigs"], ref)
        ok = n1 <= n0 - 1
        # 等级只做记录：用蛋种库反查结果格外观对应的等级，避免每次合并都点一次读 OCR
        # （那一步约 1.5s，占单次合并耗时的 1/3）。
        lvl_after = None
        try:
            after_lvl_map = after.get("lvl_of") or {}
            for cell in (bb, a):
                if cell in after_lvl_map:
                    lvl_after = after_lvl_map.get(cell)
                    if lvl_after:
                        break
        except Exception:
            pass
        log_event(action="merge", src=list(a), dst=list(bb), via="pixels",
                  level=lv_a, same_before=n0, same_after=n1, post_ok=ok,
                  level_after=lvl_after, clusters=after["clusters"])
        if ok:
            done += 1
            blocked.clear()          # 局面变了，之前的失败记录作废
            # 学习：结果格的外观 = 等级 + 1（等级只做记录，不参与配对）
            try:
                import numpy as _np
                lib = b.get("lib") or _load_lib()
                src_lvl = lv_a
                for cell in (a, bb):
                    s = after["sigs"].get(cell)
                    if s is not None and float(_np.abs(s - ref).mean()) >= EGG_SAME_THR:
                        _learn_merged(ref, s, lib, src_lvl)
                        break
            except Exception as exc:
                log(f"  等级学习失败：{exc}")
            lv = f"，结果等级 {lvl_after}" if lvl_after else ""
            log(f"  合并成功（{tag} {n0} → {n1} 个{lv}）")
            b = after          # 直接复用这次扫描，下一轮不再重扫
            # 每 5 次抽查一次真实等级，纠正蛋种库的漂移
            if done % 5 == 0:
                try:
                    _reconcile_level(g, bb, b.get("lib") or _load_lib(),
                                     (lv_a + 1) if lv_a else None)
                except Exception:
                    pass
        else:
            blocked.add(pair_key)
            log(f"  这一对没生效（{tag} {n0} → {n1} 个），跳过继续试别的")
            b = after          # 同样复用（失败时棋盘没变，扫描结果仍有效）
    if done:
        log(f"  本轮合并 {done} 次")
    # 收尾报告：让「这一轮到底推进到哪」可见
    try:
        bb = scan_board_pixels(g, stable=False)
        lvls = sorted({v for v in (bb.get("lvl_of") or {}).values() if v})
        log(f"  收尾：合并 {done} 次 / 敲蛋 {cracked} 次 / 板面等级 {lvls or '未知'}"
            f" / 最高 {max(lvls) if lvls else '?'} 级")
        log_event(action="merge_summary", merged=done, cracked=cracked,
                  levels=lvls, free=bb["n_free"], locked=len(bb["locked"]))
    except Exception:
        pass
    return done


CRACK_BTN = (0.753, 0.775)   # 「敲破蛋壳」按钮（选中蛋后出现）


def crack_cell(g: Game, cell: tuple[int, int]) -> bool:
    """敲掉某一格的蛋：选中 → 点「敲破蛋壳」→ 用格子外观变化验证。

    返回是否真的敲掉了（以该格签名改变为准，不靠日志自证）。
    """
    b0 = scan_board_pixels(g, stable=False)
    ref = b0["sigs"].get(cell)
    has_btn = False
    for attempt in range(3):
        g.click(GRID_X[cell[1]], GRID_Y[cell[0]], wait=1.3)
        txt = " ".join(l["text"] for l in g.lines(0.03, 0.68, 0.94, 0.18))
        if "敲破蛋壳" in txt:
            has_btn = True
            break
        if attempt == 0:
            log(f"  {cell} 首次没读到敲蛋按钮，重试（面板：{txt.strip()[:50]!r}）")
    if not has_btn:
        log(f"  {cell} 没有「敲破蛋壳」按钮，跳过")
        return False
    g.click(*CRACK_BTN, wait=1.6)
    b1 = scan_board_pixels(g, stable=False)
    new = b1["sigs"].get(cell)
    import numpy as _np
    if ref is not None and new is not None:
        changed = float(_np.abs(ref - new).mean()) > EGG_SAME_THR
    else:
        changed = True
    log_event(action="crack_cell", cell=list(cell), changed=changed)
    if changed:
        log(f"  敲掉 {cell}")
    else:
        log(f"  {cell} 敲了没变化")
    return changed


def pick_crack_target(b: dict, policy: str = "isolated",
                      exclude: set | None = None) -> tuple[int, int] | None:
    """在「没有可合并对」时挑一颗去敲，用来腾格子让新蛋补位。

    只考虑**孤蛋**（所在组只有 1 个）——有伴的应该去合并，不该敲。

    policy:
      "isolated" 敲**最孤立**的那颗（离最近的蛋最远）：最不可能马上配成对，
                 损失最小。同等孤立时优先敲等级低的。（默认）
      "lowest"   敲等级最低的（保住进阶阶梯）
      "highest"  敲等级最高的（玉米/头奖收益最大，牺牲阶梯）
      "weakest"  敲等级不明的（信息最少）
    """
    singles = [grp[0] for grp in b["clusters"] if len(grp) == 1]
    if exclude:
        singles = [c for c in singles if c not in exclude]
    if not singles:
        return None
    lvl_of = b.get("lvl_of") or {}
    sigs = b.get("sigs") or {}

    if policy == "isolated" and len(sigs) > 1:
        import numpy as np
        best = None
        for c in singles:
            s = sigs.get(c)
            if s is None:
                continue
            near = min(float(np.abs(s - o).mean()) for k, o in sigs.items() if k != c)
            # 越孤立越好（near 大），同等时挑等级低的
            score = (-round(near, 1), lvl_of.get(c) or 99)
            if best is None or score < best[0]:
                best = (score, c)
        if best:
            return best[1]

    known = [c for c in singles if lvl_of.get(c)]
    if policy == "lowest" and known:
        return min(known, key=lambda c: lvl_of[c])
    if policy == "highest" and known:
        return max(known, key=lambda c: lvl_of[c])
    if policy == "weakest":
        unknown = [c for c in singles if not lvl_of.get(c)]
        if unknown:
            return unknown[0]
    return singles[0]


def _reconcile_level(g: Game, cell: tuple[int, int], lib: list, expect: int | None) -> None:
    """用一次真实 OCR 校正该外观的等级标签（自愈）。

    蛋种库的等级是沿「合并 = +1」链条推断的，误差会累积（实测出现过
    「两个 3 级合出 2 级」这种物理上不可能的记录）。每合并若干次抽查一次，
    以 OCR 读到的真实等级为准覆盖，链条就不会越漂越远。
    """
    try:
        g.click(GRID_X[cell[1]], GRID_Y[cell[0]], wait=1.1)
        st, _ty, lvl = read_cell_state(g)
        if st != "egg" or not lvl:
            return
        b = scan_board_pixels(g, stable=False)
        sig = b["sigs"].get(cell)
        if sig is None:
            return
        d, e = lib_match(sig, lib)
        if e is not None and e.get("level") != lvl:
            old = e.get("level")
            e["level"] = int(lvl)
            _save_lib(lib)
            if old != lvl:
                log(f"  等级自纠正：外观 {old} → 实测 {lvl}（期望 {expect}）")
    except Exception:
        pass


def do_assign_slot(g: Game) -> bool:
    """把「这是一只新鸭子」放进空槽位（点「给它安排一个槽位」）。

    这是解锁槽位之后的**必要一步**：只解锁不分配，槽位一直空着、
    鸭子一直显示「新鸭子」，鸭子数量永远不涨（2026-11-14 实测确认）。
    """
    go_home(g)
    if not goto(g, "duck", "duck", tries=3):
        return False
    slots, inv, _pad = duck_slots_from_inventory(g)
    # kind=="new"（面板明说「这是一只新鸭子」）优先；
    # 否则任何槽位都点一下看有没有「给它安排一个槽位」按钮——
    # 该按钮只在「有空槽/槽位锁定」时出现，点鸭子不会误触。
    cand = [s for s in inv if s["kind"] == "new"]
    if not cand:
        cand = inv
    for s in cand:
        t = click_slot(g, s)
        if t is None or "给它安排一个槽位" not in t:
            continue
        hit = [l for l in g.lines() if "给它安排一个槽位" in l["text"]]
        if not hit:
            continue
        r = hit[0]["rect"]
        log(f"  安排新鸭子进槽位 x={(s['cx'] - g.l) / g.w:.2f}")
        pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
        time.sleep(1.8)
        t2 = g.text()
        ok = "给它安排一个槽位" not in t2
        log_event(action="assign_slot", ok=ok)
        if ok:
            log("  新鸭子已安排（状态已变）")
            return True
        log("  安排了但状态没变")
        return False
    log("  没有待分配的新鸭子")
    return False


def do_unlock_slot(g: Game, max_price: int = 300) -> bool:
    """解锁一个鸭子槽位（花星星）。

    入口：点右侧任一锁定槽 → 面板出现「获得一个新槽位 N」。
    安全约束：
      - 先读价格，超过 max_price 不买；
      - 星星不足不买；
      - 点完**验证锁定槽数量减少**，不靠日志自证。
    """
    go_home(g)
    if not goto(g, "duck", "duck", tries=3):
        log("  解锁槽位：进不了鸭子页")
        return False
    slots0, inv, _pad = duck_slots_from_inventory(g)
    pads = [s for s in inv if s["kind"] == "padlock"]
    if not pads:
        log("  解锁槽位：没有锁定槽")
        return False
    before_locks = len(pads)

    s = pads[0]
    pyautogui.click(s["cx"], s["cy"])
    time.sleep(1.6)
    txt = g.text()
    m = re.search(r"获得一个新槽位\s*(\d+)", txt)
    if not m:
        log("  解锁槽位：面板里没有价格，跳过")
        g.close_popup()
        return False
    price = int(m.group(1))
    stars = star_balance(g)
    log(f"  解锁槽位：价格 ★{price}，当前 ★{stars}")
    if stars is not None and stars < price:
        log("  星星不足，跳过")
        g.close_popup()
        return False
    if price > max_price:
        log(f"  价格超过上限 {max_price}，跳过")
        g.close_popup()
        return False

    hit = [l for l in g.lines() if "获得一个新槽位" in l["text"]]
    if not hit:
        log("  找不到解锁按钮")
        return False
    r = hit[0]["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    time.sleep(2.0)

    slots1, inv1, _ = duck_slots_from_inventory(g)
    after_locks = len([x for x in inv1 if x["kind"] == "padlock"])
    ok = after_locks < before_locks
    log(f"  解锁结果：锁定槽 {before_locks} → {after_locks}（{'成功' if ok else '未生效'}）")
    log_event(action="unlock_slot", price=price, ok=ok,
              locks_before=before_locks, locks_after=after_locks)
    return ok


SELL_BTN = (0.80, 0.768)     # 「在市场上出售」确认按钮（说明页滚到底部后）


def _drag_scroll(g: Game, times: int = 1) -> None:
    """在弹窗内拖拽滚动（PageDown 在弹窗里不生效，实测拖拽才行）。"""
    x = g.l + int(0.50 * g.w)
    y1 = g.t + int(0.85 * g.h)
    y2 = g.t + int(0.35 * g.h)
    for _ in range(times):
        pyautogui.moveTo(x, y1)
        pyautogui.mouseDown()
        time.sleep(0.15)
        for k in range(1, 11):
            pyautogui.moveTo(x, y1 + (y2 - y1) * k // 10)
            time.sleep(0.03)
        time.sleep(0.2)
        pyautogui.mouseUp()
        time.sleep(0.9)


def _scroll_until(g: Game, keyword: str, tries: int = 5):
    """滚到出现关键字为止，返回命中的行。"""
    for _ in range(tries):
        hits = [l for l in g.lines() if keyword in l["text"]]
        if hits:
            return hits
        _drag_scroll(g, 1)
    return []


def _find_exact(g: Game, label: str, tries: int = 8):
    """滚到出现**精确匹配**的按钮为止。

    必须用精确匹配：说明页正文里也有「…排成单个队列在市场上出售…」这种句子，
    用 `in` 会点到描述文字（点了毫无反应，这是之前卖鸭一直失败的直接原因）。
    """
    for _ in range(tries):
        hits = [l for l in g.lines() if l["text"].strip() == label]
        if hits:
            return hits[0]
        _drag_scroll(g, 1)
    return None


def do_sell_duck(g: Game) -> bool:
    """把「已喂满且 ≥UNCOMMON」的鸭子挂到市场出售（换星星）。

    路径（实测）：选中鸭子 → 面板右侧 ⓘ → 详情页滚到底 → 市场出售 →
    说明页拖拽滚到底 → 「在市场上出售」。
    验证：该鸭子状态变成「鸭子正在出售」才算成功。
    """
    go_home(g)
    if not goto(g, "duck", "duck", tries=3):
        return False
    # 用「高亮 ± 槽距」在整条槽位带上找可卖的鸭子（先向左到边，再向右扫）
    target = None
    prof = _read_one_duck(g, 0.0)
    if prof.get("sellable"):
        target = prof
    if target is None:
        for _ in range(10):
            ok, nx = _step_duck(g, -1)
            if not ok:
                break
            prof = _read_one_duck(g, nx)
            if prof.get("sellable"):
                target = prof
                break
    if target is None:
        for _ in range(16):
            ok, nx = _step_duck(g, +1)
            if not ok:
                break
            prof = _read_one_duck(g, nx)
            if prof.get("sellable"):
                target = prof
                break
    if target is None:
        log("  卖鸭：没有「已喂满且 ≥UNCOMMON」的鸭子")
        return False
    prof = target
    log(f"  卖鸭：{prof['rarity']} Lv{prof['level']} x={prof['x']:.3f}")

    pyautogui.click(g.l + int(0.906 * g.w), g.t + int(0.735 * g.h))   # ⓘ
    time.sleep(1.8)
    link = _find_exact(g, "市场出售", tries=6)
    if link is None:
        log("  卖鸭：详情页里找不到「市场出售」")
        g.close_popup()
        return False
    r = link["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    time.sleep(2.0)

    btn = _find_exact(g, "在市场上出售", tries=10)
    if btn is None:
        log("  卖鸭：说明页里找不到「在市场上出售」（可能被描述文字干扰）")
        g.close_popup()
        return False
    r2 = btn["rect"]
    pyautogui.click((r2[0] + r2[2]) // 2, (r2[1] + r2[3]) // 2)
    time.sleep(2.5)

    # 验证：出现 LOT CREATED / 鸭子正在出售
    txt = " ".join(l["text"] for l in g.lines()).upper()
    ok = ("LOT CREATED" in txt) or ("鸭子正在出售" in txt)
    if ok:
        log("  卖鸭：已挂到市场（THE DUCK MARKET LOT CREATED / 鸭子正在出售）")
        log_event(action="sell_duck", ok=True, rarity=prof["rarity"],
                  level=prof["level"])
        return True
    log("  卖鸭：没确认到出售状态")
    log_event(action="sell_duck", ok=False, rarity=prof["rarity"],
              level=prof["level"])
    return False


def read_cell_state(g: Game) -> tuple[str, str | None, int | None]:
    """读当前选中格，返回 (state, 类型, 等级)。

    state: "egg" 可用 / "locked" 锁定 / "unknown" 读不出

    等级有两条路：
    1. 直接读 `蛋[7]` 的括号数字；
    2. 括号读不出时（实测会被 OCR 读成 `蛋切`/`蛋[n]`），用**赔率区间**反查。
       同等级的赔率是固定的（1级 2-3、4级 27-36、7级 289-390），
       而长数字 OCR 很稳。映射表在 results/computer_use/egg_level_map.json，
       每读到一次「括号+赔率都清晰」的蛋就自动补一条，越用越准。
    """
    import cv2
    img = grab(g.hwnd)
    if img is None:
        return ("unknown", None, None)
    H, W = img.shape[:2]
    x0, y0 = int(EGG_INFO_BOX[0] * W), int(EGG_INFO_BOX[1] * H)
    x1 = int((EGG_INFO_BOX[0] + EGG_INFO_BOX[2]) * W)
    y1 = int((EGG_INFO_BOX[1] + EGG_INFO_BOX[3]) * H)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return ("unknown", None, None)

    lmap = _load_level_map()
    changed = False
    for scale in (5, 3):
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = bench.ocr_engine()(big)
        txt = " ".join(t for _b, t, _s in (res or []))
        kind = "爱心蛋" if "爱心" in txt else "蛋"
        locked = ("解锁" in txt) or ("槽位" in txt) or ("🔒" in txt)
        odds = re.search(r"(\d{1,4})\s*-\s*(\d{1,4})", txt)
        odds_key = f"{odds.group(1)}-{odds.group(2)}" if odds else None
        m = re.search(r"蛋\s*[\[【]\s*(\d{1,2})\s*[\]】]", txt)
        lvl = int(m.group(1)) if m else None
        if lvl is None and odds_key:
            got = lmap.get(odds_key)
            if got is not None:
                lvl = int(got) if not isinstance(got, dict) else None
        if lvl is not None and odds_key and odds_key not in lmap:
            lmap[odds_key] = lvl
            changed = True
        if changed:
            _save_level_map(lmap)
            changed = False
        if locked and lvl is not None:
            return ("locked", kind, lvl)
        if lvl is not None:
            return ("egg", kind, lvl)
        if locked:
            return ("locked", kind, None)
    return ("unknown", None, None)


def scan_board(g: Game) -> dict:
    """逐格扫描蛋板，返回 {(i,j): (state, type, level)}。合并前必须先做这一步。"""
    board = {}
    for i in range(5):
        for j in range(5):
            if (i, j) == CART_CELL:
                board[(i, j)] = ("cart", None, None)
                continue
            g.click(GRID_X[j], GRID_Y[i], wait=1.15)
            board[(i, j)] = read_cell_state(g)
    return board


def egg_queue_count(g: Game) -> int | None:
    """「下一颗蛋及后续 N 颗」的 N。用它判断蛋板有没有进新蛋。"""
    t = g.text(0.15, 0.700, 0.85, 0.140)
    m = re.search(r"后续\s*(\d+)", t)
    return int(m.group(1)) if m else None


def do_crack(g: Game, min_level: int = CRACK_MIN_LEVEL) -> bool:
    """只敲 >= min_level 的蛋。"""
    for i in range(5):
        for j in range(5):
            if (i, j) == CART_CELL:
                continue
            info = egg_info(g, (i, j))
            if not info or info[0] == "SLOT" or info[1] < min_level:
                continue
            log(f"  敲蛋 {info[0]}[{info[1]}] @({i},{j})")
            g.click(0.777, 0.777, wait=2.2)
            return True
    return False


def do_buy(g: Game) -> bool:
    """买 5 / 10 面具的商人蛋（面具优先花掉，别囤着过期）。"""
    masks = mask_balance(g)
    for rx, price in ((0.30, 5), (0.46, 10)):
        if masks is not None and masks < price:
            continue
        g.click(rx, 0.298, wait=1.2)          # 商人
        if "CONFIRM" not in g.text(0.10, 0.28, 0.60, 0.12):
            continue
        g.click(0.285, 0.352, wait=2.0)       # CONFIRM
        log(f"  买蛋 {price} 面具（余额 {masks}）")
        return True
    return False


def do_claim(g: Game, max_clicks: int = 3) -> int:
    """点当前页所有「领取」（每日奖励 / 活动 / 排名奖励）。只认领取，不点「打开」。

    扫描区间必须覆盖到卡片底部：每日奖励的「领取」在 0.83 附近，
    原来的 0.42–0.92 起点太靠下反而漏掉，现在整块卡片都扫。
    """
    n = 0
    for _ in range(max_clicks):
        hit = [l for l in g.lines(0.03, 0.10, 0.94, 0.86)
               if "领取" in l["text"] or "收取" in l["text"]]
        if not hit:
            break
        # 优先点最靠下的那个（弹窗的主按钮），避免点到标题旁的说明文字
        hit.sort(key=lambda l: l["rect"][1], reverse=True)
        r = hit[0]["rect"]
        pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
        time.sleep(1.9)
        n += 1
    return n


def do_breed(g: Game) -> bool:
    """鸭子已准备好交配 → 付玉米开始繁殖。

    面板结构（实测）：标题「已准备好交配」在上方，按钮「支付费用2000」在 y≈0.34，
    而原实现只在 0.55–0.85 找，永远找不到。现在扫整块卡片，并优先匹配带数字的按钮行
    （避免点到说明文字「的鸭子。支付费用开始繁殖」）。
    """
    if "已准备好交配" not in g.text(0.03, 0.10, 0.94, 0.60):
        return False
    btn = None
    for l in g.lines(0.03, 0.10, 0.94, 0.75):
        s = l["text"].strip()
        if re.fullmatch(r"支付费用\s*\d+", s) or s.startswith("开始繁殖"):
            btn = l
            break
    if btn is None:
        log("  已准备好交配，但找不到「支付费用」按钮")
        return False
    r = btn["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    time.sleep(2.2)
    log(f"  已支付繁殖费用（{btn['text'].strip()}）")
    # 之后弹「匿名/社群/好友」面板：点「寻找」走匿名繁殖，并**校验真的进入了繁殖**
    for attempt in range(2):
        for l2 in g.lines():
            s = l2["text"].strip()
            if "寻找" in s:
                r2 = l2["rect"]
                pyautogui.click((r2[0] + r2[2]) // 2, (r2[1] + r2[3]) // 2)
                time.sleep(2.5)
                break
        t2 = g.text()
        if "鸭子正在繁殖" in t2 or "PREENING" in t2.upper() or "费用已支付" in t2:
            log("  开始繁殖（匿名）")
            log_event(action="breed", mode="anon", paid=btn["text"].strip())
            return True
        time.sleep(0.8)
    log("  已付款但没确认进入繁殖状态")
    log_event(action="breed_failed", paid=btn["text"].strip())
    return False


# ---------------------------------------------------------------- 主循环

LOVE_ICONS = ((0.319, 0.371), (0.319, 0.439), (0.319, 0.500), (0.319, 0.573))


RARITIES = ("COMMON", "UNCOMMON", "RARE", "EPIC", "LEGENDARY")
DUCK_STATES = ("已准备好交配", "这是一只新鸭子", "繁殖进行中", "鸭子正在出售",
               "已准备好质押", "点击喂食")


def read_duck_profile(g: Game) -> dict:
    """读当前这只鸭子的稀有度 / 等级 / 状态 / 是否已喂满 / 是否可卖（只读）。"""
    t = g.text()
    rarity = None
    m = re.search(r"DUCKER\s+(COMMON|UNCOMMON|RARE|EPIC|LEGENDARY)", t)
    if m:
        rarity = m.group(1)
    level = None
    m = re.search(r"LVL\s*(\d+)", t)
    if m:
        level = int(m.group(1))
    state = next((s for s in DUCK_STATES if s in t), None)
    # 喂满判定：头顶显示 FED DUCK，或还有气泡但单价已超上限
    maxed = ("FED DUCK" in t)
    if not maxed and "点击喂食" in t:
        m2 = re.search(r"喂食\s*(\d+)", t)
        if m2 and int(m2.group(1)) > FEED_CAP:
            maxed = True
    # 出售门槛：稀有度 ≥ UNCOMMON（绿色及以上）**且** 已喂满
    sellable = bool(maxed and rarity in ("UNCOMMON", "RARE", "EPIC", "LEGENDARY"))
    return {"rarity": rarity, "level": level, "state": state,
            "maxed": maxed, "sellable": sellable}


def collect_love(g: Game, limit: int = FEED_SCAN_DUCKS) -> dict:
    """逐槽位枚举鸭子，收爱心蛋。

    机制（2026-11-13 实测）：
    - 爱心蛋**只产在「繁殖进行中」的鸭子身上**（面板文案：在繁殖期间收集爱心蛋）；
      点一下卡片本体就收，收完再点会提示「暂无爱心蛋。稍后回来即可获得爱心蛋。」
    - 其它状态的鸭子点下去没有爱心蛋提示，所以**不能用「没看到暂无」推断收到了**。

    枚举用槽位图标（`select_slot`），比拖拽可靠。
    """
    clicked = 0
    claimed = 0
    eggs = 0
    ducks = 0
    roster = []
    slots, _inv, _pad = duck_slots_from_inventory(g)
    for s in slots:
        t = click_slot(g, s)
        if t is None:
            continue
        ducks += 1
        prof = read_duck_profile(g)
        prof["x"] = round((s["cx"] - g.l) / g.w, 3)
        roster.append(prof)
        g.click(0.500, 0.470, wait=1.0)
        clicked += 1
        t2 = g.text()
        if "领取" in t2 or "收取" in t2:
            n = do_claim(g, 2)
            if n:
                claimed += n
                log(f"  鸭子面板领取 {n} 次")
                log_event(action="collect_panel", claimed=n)
        if "繁殖进行中" in t:
            for _ in range(2):
                if "暂无爱心蛋" in g.text():
                    break
                g.click(0.500, 0.470, wait=1.0)
                eggs += 1
        if current_page(g, tries=1)[0] != "duck":
            g.close_popup()
            g.nav("duck", wait=1.0)
    log(f"  爱心蛋：枚举 {ducks} 只鸭子，收到约 {eggs} 个，面板领取 {claimed} 次")
    log_event(action="collect_love", ducks=ducks, eggs=eggs, claimed=claimed,
              roster=[{k: d[k] for k in ("rarity", "level", "state")} for d in roster])
    return {"ducks": ducks, "eggs": eggs, "claimed": claimed, "roster": roster}


# ---------------------------------------------------------------- 分级任务

class PageMismatch(Exception):
    def __init__(self, expected: str, actual: str):
        self.expected, self.actual = expected, actual
        super().__init__(f"page {actual} != {expected}")


def current_page(g: Game, tries: int = 5, gap: float = 0.5) -> tuple[str, float]:
    """页面检测带重试：导航动画期间 OCR 可能读不到锚点，连续 3 次 unknown 才算失败。"""
    page, conf = "unknown", 0.0
    for _ in range(max(1, tries)):
        page, conf, _ev = classify_page(g.text())
        if page != "unknown":
            return page, conf
        time.sleep(gap)
    return page, conf


def goto(g: Game, nav_key: str, page: str, tries: int = 3) -> bool:
    """导航到指定页面：已在该页就直接返回；否则点完确认，不中则重试。

    导航失败最常见的两个原因：底栏被弹层挡住、点击落在浮层上。
    所以每轮失败后先尝试关一层弹层再重试，而不是原地重复点击。
    """
    if current_page(g, tries=1)[0] == page:
        return True
    for i in range(max(1, tries)):
        g.nav(nav_key, wait=1.2)
        if current_page(g, tries=2)[0] == page:
            return True
        if i < tries - 1:
            if g.close_popup():
                log("  导航未到达，已关闭弹层，重试")
            time.sleep(0.5)
    log_event(action="nav_failed", target=page, nav=nav_key,
              actual=current_page(g, tries=1)[0])
    return False


def require_page(g: Game, expected: str) -> None:
    """动作前置断言：页面不符直接抛 PageMismatch，调用方不得继续点击。"""
    page, conf = current_page(g)
    ok = (page == expected)
    log_event(action="page_check", expected=expected, actual=page,
              confidence=conf, ok=ok)
    if not ok:
        raise PageMismatch(expected, page)


def authorize_page(page: str, action: str, cfg: dict) -> tuple[bool, str]:
    """纯函数版授权：给定页面判定，不依赖屏幕，便于自检与回归。"""
    level = RISK.get(action, "L4")
    if level == "L0":
        return True, "ok"
    if page == "unknown":
        return False, "page_unknown"
    need = TASK_PAGE.get(action)
    if need and page != need:
        return False, f"page_not_{need}"
    if level in ("L1", "L2"):
        return True, "ok"
    if level == "L3":
        if cfg.get("allow_risk3") and cfg["actions"].get(action):
            return True, "ok"
        return False, "risk_l3_denied"
    return False, "risk_l4_denied"


def authorize(g: Game, action: str, cfg: dict) -> tuple[bool, str]:
    """统一授权入口：先读页面，再交给纯函数判定。"""
    page, _conf = current_page(g)
    return authorize_page(page, action, cfg)


def go_home(g: Game, tries: int = 3) -> bool:
    """回到带主底栏的页面：子页面点「蛋」标签即可回来。"""
    for _ in range(tries):
        t = g.text(0.0, 0.775, 1.0, 0.13)
        if "市场" in t and "鸭子" in t:
            return True
        pyautogui.press("esc")
        time.sleep(0.4)
        g.click(0.388, 0.877, wait=1.1)   # 点「蛋」标签
    t = g.text(0.0, 0.775, 1.0, 0.13)
    return ("市场" in t and "鸭子" in t)


AD_NO_AD_MARK = "NO ADS"


def _tab_line(g: Game, word: str, max_len: int = 6) -> dict | None:
    """找带红点数字的子标签（OCR 会把 "广告" 读成 "广告1"）。"""
    for l in g.lines():
        t = l["text"].strip()
        if t.startswith(word) and len(t) <= max_len:
            return l
    return None


def _open_ads_tab(g: Game) -> bool:
    """任务页 → 广告 子标签。"""
    l = _tab_line(g, "广告")
    if l is None:
        return False
    r = l["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    time.sleep(1.8)
    return True


def _watch_buttons(g: Game) -> list[dict]:
    """「观看」按钮（排除 "观看广告赚取玉米" 这种说明文字）。"""
    out = []
    for l in g.lines():
        t = l["text"].strip()
        if t.startswith("观看") and len(t) <= 5:
            out.append(l)
    return out


def _page_text(g: Game) -> str:
    return " ".join(l["text"] for l in g.lines())


def _click_text(g: Game, label: str, exact: bool = True) -> bool:
    hit = [l for l in g.lines()
           if (l["text"].strip() == label if exact else label in l["text"])]
    if not hit:
        return False
    r = hit[0]["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    return True


def do_watch_ads(g: Game, max_ads: int = 5, ad_timeout: int = 90) -> dict:
    """任务页「广告」：点「观看」拿玉米。

    实测两种结局（2026-11-14）：
      1) 有广告：播放 → 结束后出现「关闭」/奖励结算
      2) 暂时没广告：「LOOKING FOR AN AD...」→
         「NO ADS RIGHT NOW - YOUR REWARD IS COUNTED ANYWAY!」+「关闭」
         ——奖励照样计入，所以照样算成功
    领完之后那张卡的「观看」按钮会消失（进入冷却/已达上限）。

    安全：只点「观看」和结算弹窗上的「关闭」，不碰任何「即将推出」/付费入口。
    """
    res = {"ads": 0, "no_ad": 0, "closed": 0, "blocked": None}
    go_home(g)
    if not goto(g, "task", TASK_PAGE["ads"]):
        res["blocked"] = "nav_failed"
        return res
    if not _open_ads_tab(g):
        res["blocked"] = "no_ads_tab"
        log("  看广告：找不到「广告」子标签")
        return res

    for _i in range(max_ads + 3):
        # 先领掉挂起的奖励（上一轮看完但没点到/没来得及点的）
        pending = [l for l in g.lines()
                   if l["text"].strip().upper() in ("CLAIM REWARD", "CLAIM")]
        if pending:
            rr = pending[0]["rect"]
            pyautogui.click((rr[0] + rr[2]) // 2, (rr[1] + rr[3]) // 2)
            time.sleep(1.8)
            res["claimed"] = int(res.get("claimed", 0)) + 1
            log(f"  领取广告奖励（累计 {res['claimed']}）")
            log_event(action="ad_claim", total=res["claimed"])
            continue

        btns = _watch_buttons(g)
        if not btns:
            break
        r = btns[0]["rect"]
        pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)

        deadline = time.time() + ad_timeout
        settled = False
        while time.time() < deadline:
            time.sleep(1.0)
            txt = _page_text(g)
            if AD_NO_AD_MARK in txt.upper():
                res["no_ad"] += 1
                if _click_text(g, "关闭"):
                    res["closed"] += 1
                time.sleep(1.4)
                settled = True
                break
            if "关闭" in txt or "领取" in txt:
                if _click_text(g, "关闭") or _click_text(g, "领取"):
                    res["closed"] += 1
                time.sleep(1.4)
                settled = True
                break
        if not settled:
            log(f"  看广告：{ad_timeout}s 内没有结算（可能广告还在播），本轮停止")
            break

        # 看完之后，那张卡的按钮会变成 CLAIM REWARD，**必须再点一次才到账**
        # （实测：点完玉米 +100，卡片又恢复「观看」）
        claimed = False
        for _k in range(6):
            hit = [l for l in g.lines()
                   if l["text"].strip().upper() in ("CLAIM REWARD", "CLAIM")]
            if not hit:
                time.sleep(1.2)
                continue
            rr = hit[0]["rect"]
            pyautogui.click((rr[0] + rr[2]) // 2, (rr[1] + rr[3]) // 2)
            time.sleep(1.8)
            claimed = True
            break
        if claimed:
            res["claimed"] = int(res.get("claimed", 0)) + 1
        res["ads"] += 1
        log(f"  第 {res['ads']} 次观看完成"
            f"（无广告弹窗 {res['no_ad']} 次 / 已领取 {res.get('claimed', 0)}）")
        log_event(action="watch_ad", index=res["ads"], no_ad=res["no_ad"],
                  claimed=bool(claimed))
        # 看完可能回到任务列表，需要重新进广告页
        if not _watch_buttons(g):
            if not _open_ads_tab(g):
                g.nav("task", wait=1.6)
                if not _open_ads_tab(g):
                    break

    log(f"  看广告：完成 {res['ads']} 次（无广告 {res['no_ad']} 次"
        f" / 领取 {res.get('claimed', 0)} 次）")
    return res


DAILY_ROW = (0.03, 0.10, 0.94, 0.86)      # DAILY TASKS 面板的文本扫描范围


def _daily_claim_buttons(g: Game) -> list[dict]:
    """面板里可点的领奖按钮。

    只认两种：
      - "CLAIM REWARD"（每日任务右侧的黄色按钮）
      - 精确等于 "领取"（奖励包）
    明确排除含 PREMIUM / UNLOCK / GO 的按钮——那些是付费或跳转。
    """
    out = []
    for l in g.lines(*DAILY_ROW):
        t = l["text"].strip()
        up = t.upper()
        if "PREMIUM" in up or "UNLOCK" in up:
            continue
        # 只认按钮本身：
        #   - OCR 会把这个按钮拆成两行 "CLAIM" + "REWARD"，所以认 "CLAIM"/"CLAIM REWARD"
        #   - "CLAIM 7 REWARDS IN THE TASKS SECTION" 是**任务描述**，不是按钮（点了没反应）
        if up in ("CLAIM", "CLAIM REWARD") or t == "领取":
            out.append(l)
    return out


def _daily_scroll(g: Game, times: int = 1) -> None:
    """DAILY TASKS 面板只能拖拽滚动（滚轮无效，实测）。"""
    for _ in range(times):
        x = g.l + int(0.50 * g.w)
        y0 = g.t + int(0.82 * g.h)
        y1 = g.t + int(0.58 * g.h)
        pyautogui.moveTo(x, y0)
        pyautogui.mouseDown()
        for k in range(1, 13):
            pyautogui.moveTo(x, y0 + (y1 - y0) * k / 12)
            time.sleep(0.02)
        pyautogui.mouseUp()
        time.sleep(1.0)


def do_daily_pass(g: Game, max_rounds: int = 9) -> dict:
    """打开鸭子页的 DAILY PASS，把能领的奖励全领掉。

    实测结构（2026-11-14）：
      点左侧 DAILY PASS → DAILY TASKS 面板
      面板里：BONUS REWARDS（蓝按钮）+ 若干日常任务（右侧 CLAIM REWARD）
      往下拖拽还有：ERRANDS 任务、「领取」奖励包、PREMIUM（付费，跳过）
    领过的任务会变成黄色 ✓ 并置灰，所以「CLAIM 文案数量减少」就是领到了。
    """
    res = {"opened": False, "claimed": 0, "rounds": 0}
    go_home(g)
    if not goto(g, "duck", TASK_PAGE["daily"]):
        res["blocked"] = "nav_failed"
        return res

    hit = [l for l in g.lines() if "DAILY" in l["text"].upper()]
    if not hit:
        res["blocked"] = "no_daily_button"
        log("  DAILY PASS：找不到入口按钮")
        return res
    r = hit[0]["rect"]
    pyautogui.click((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    time.sleep(2.0)

    txt = " ".join(l["text"] for l in g.lines())
    if "DAILY TASKS" not in txt.upper():
        res["blocked"] = "panel_not_open"
        log("  DAILY PASS：面板没打开")
        return res
    res["opened"] = True

    no_change = 0
    for round_i in range(max_rounds):
        buttons = _daily_claim_buttons(g)
        before = len(buttons)
        got = 0
        for b in buttons:
            rr = b["rect"]
            pyautogui.click((rr[0] + rr[2]) // 2, (rr[1] + rr[3]) // 2)
            time.sleep(1.4)
            after = len(_daily_claim_buttons(g))
            if after < before:
                got += 1
                res["claimed"] += 1
                log(f"  领取成功（剩余可领 {after}）")
                log_event(action="daily_claim", remaining=after)
            before = after
        res["rounds"] += 1

        # 往下拖拽露出更多内容；连续两次既没领到、也滚不动，才算到底
        txt_before = " ".join(l["text"] for l in g.lines())
        _daily_scroll(g, 1)
        txt_after = " ".join(l["text"] for l in g.lines())
        if got == 0 and txt_after == txt_before:
            no_change += 1
            if no_change >= 2:
                break
        else:
            no_change = 0

    # 收尾：关面板，回鸭子页
    g.close_popup()
    time.sleep(0.8)
    if current_page(g, tries=1)[0] != "duck":
        g.nav("duck", wait=1.2)
    log(f"  DAILY PASS：领取 {res['claimed']} 项（{res['rounds']} 轮）")
    return res


def task_collect(g: Game, stats: dict, cfg: dict) -> dict:
    """常规任务：点鸭子收爱心蛋 + 领奖励 + 处理带红点的标签页（只点「领取」）。

    原先这里还有一处写死的 `click(0.886, 0.351)`（右侧活动图标），会误开万圣节活动页；
    现在全部改成「点鸭子本体 + OCR 找领取按钮」，不再有猜测坐标。
    """
    go_home(g)
    if not goto(g, "duck", TASK_PAGE["collect"]):
        return {"blocked": "nav_failed", "target": TASK_PAGE["collect"]}
    love = collect_love(g)        # 点每只鸭子：收爱心蛋 + 领鸭子面板里的奖励
    hearts = int(love["eggs"])
    claimed = int(love["claimed"])

    # 鸭子名册：按 (稀有度, 等级) 分组，凑够 4 只才轮到「合并升级」。
    # 注意：这里只统计并记录，**不做任何升级动作**（用户要求等 4 只再合并）。
    roster = love.get("roster") or []
    from collections import Counter
    groups = Counter((d["rarity"], d["level"]) for d in roster
                     if d.get("rarity") and d.get("level"))
    stats["roster"] = [f"{d['rarity']}Lv{d['level']}:{d['state']}"
                       + ("|满" if d.get("maxed") else "") for d in roster]
    brief = "、".join(f"{k[0]}Lv{k[1]}×{v}" for k, v in sorted(groups.items()))
    log(f"  鸭子名册：{brief or '（空）'}")
    # 可卖候选：稀有度 ≥ UNCOMMON 且已喂满（用户规则：先喂到顶，再考虑卖）
    cands = [(d.get("rarity"), d.get("level"), d.get("x")) for d in roster
             if d.get("sellable")]
    if cands:
        log(f"  可卖候选 {len(cands)} 只（已喂满且 ≥UNCOMMON）：{cands}")
        log_event(action="sell_candidates", candidates=[list(c) for c in cands])
    else:
        log("  可卖候选：0（没有「已喂满且 ≥UNCOMMON」的鸭子）")
    ready = {k: v for k, v in groups.items() if v >= 4}
    if ready:
        msg = "、".join(f"{k[0]}Lv{k[1]}×{v}" for k, v in ready.items())
        log(f"  已凑满 4 只，可以合并升级：{msg}（等你确认后再动手）")
        log_event(action="merge_ready", groups=[{"rarity": k[0], "level": k[1], "n": v}
                                                for k, v in ready.items()])

    # 红点驱动的领取：只在「任务 / 蛋」两页点「领取」，不碰市场与 GOOS（那两页有购买/铸造）
    dets = g.detect_nav()
    dots = {b.get("near") for b in g.nav_badges(dets) if b.get("near")}
    for key in ("task", "egg"):
        if key not in dots:
            continue
        g.nav(key, wait=1.6)
        time.sleep(0.4)
        n = do_claim(g, 2)
        if n:
            claimed += n
            log(f"  {key} 页领取 {n} 次")
        pyautogui.press("esc")
        time.sleep(0.4)
    if dots & {"task", "egg"}:
        go_home(g)
        g.nav("duck")

    stats["claims"] += claimed
    stats["love_eggs"] += hearts
    return {"claimed": claimed, "love_eggs": hearts, "badges": sorted(dots)}


def find_ready_to_breed(g: Game, limit: int = 8) -> bool:
    """先枚举槽位，再逐只读鸭子状态，找「已准备好交配」的发情配种。"""
    slots, _inv, _pad = duck_slots_from_inventory(g)
    for s in slots[:limit]:
        t = click_slot(g, s)
        if t is None:
            continue
        if "已准备好交配" in t:
            return do_breed(g)
    return False

def find_all_feedable(g: Game) -> list[tuple[dict, int]]:
    """枚举所有鸭子，挑出单价 <= FEED_CAP 的，按单价从低到高。"""
    now = time.time()
    over = getattr(g, "over_cap", None) or {}
    g.over_cap = over
    ducks = walk_ducks(g)
    found: list[tuple[dict, int]] = []
    for d in ducks:
        price = d.get("price")
        if price is None:
            continue
        if price > FEED_CAP:
            over[round(d["x"], 3)] = now
            continue
        s = {"cx": g.l + int(d["x"] * g.w), "cy": g.t + int(0.696 * g.h),
             "kind": "duck"}
        found.append((s, price))
    found.sort(key=lambda kv: kv[1])
    log(f"  可喂 {len(found)} 只：{[(round((s['cx'] - g.l) / g.w, 2), p) for s, p in found]}")
    g.feed_reason = "ok" if found else "no-feedable"
    return found


def task_feed(g: Game, stats: dict, cfg: dict) -> dict:
    """常规任务：喂鸭，单价不超过 feed_cap。

    结构改成**走查一次、走到哪只喂哪只**：先扫描再按旧坐标点会点到缝隙
    （实测整排会滚动，旧坐标偏半格），所以扫描和喂食必须在同一遍里完成。
    """
    go_home(g)
    if not goto(g, "duck", TASK_PAGE["feed"]):
        return {"blocked": "nav_failed", "target": TASK_PAGE["feed"]}
    st = read_state(g)
    stats["page"] = st.get("page")

    cap = int(cfg["feed_cap"])
    res = {"cap": cap, "rounds": 0, "fed": 0, "bred": 0, "visited": 0}
    total = 0

    def handle(prof: dict, text: str) -> None:
        nonlocal total
        res["visited"] += 1
        price = prof.get("price")
        stats["feed_price"] = price
        if price is None:
            return
        if price > cap:
            log(f"  单价 {price} 超过上限 {cap}，跳过")
            return
        if res["rounds"] >= int(cfg.get("feed_rounds", 12)):
            return
        log(f"  喂这只（单价 {price}，槽位 x={prof['x']:.2f}）")
        fed = do_feed(g, cap, int(cfg["feed_max_clicks"]))
        total += fed["fed"]
        res["rounds"] += 1
        res.update({k: fed[k] for k in ("fed", "stopped") if k in fed})
        res["fed"] = total
        if fed.get("stopped") == "bubble-gone":
            t_now = g.text()
            if "已准备好交配" in t_now:
                ok3, why3 = authorize(g, "breed", cfg)
                if ok3 and do_breed(g):
                    stats["breeds"] += 1
                    res["bred"] = res.get("bred", 0) + 1
                    log("  喂满了 → 已发起交配")
                elif not ok3:
                    log_event(action="deny", target="breed", reason=why3)

    out = walk_ducks(g, handle)
    res["found"] = bool(out)
    res["candidates"] = sum(1 for d in out if d.get("price") is not None
                            and d["price"] <= cap)
    res["total_fed"] = total
    stats["feeds"] += total
    res["stop_reason"] = "walked" if out else "no-feedable"
    if not out:
        log("  没有可喂的鸭子（已满/出售/质押/繁殖中），本轮喂食结束")

    if cfg["actions"].get("breed"):
        ok3, why3 = authorize(g, "breed", cfg)
        if ok3 and (do_breed(g) or find_ready_to_breed(g)):
            stats["breeds"] += 1
            res["bred"] = True
            log("  已发起配种")
        elif not ok3:
            log_event(action="deny", target="breed", reason=why3)
    return res


def task_merge(g: Game, stats: dict, cfg: dict) -> dict:
    """升级任务：同种蛋两两合并（纯像素读板，不点锁定格）。"""
    go_home(g)
    if not goto(g, "egg", TASK_PAGE["merge"]):
        return {"blocked": "nav_failed", "target": TASK_PAGE["merge"]}
    q = egg_queue_count(g)
    stats["queue"] = q
    # 注意：这里不做「队列没变就跳过」的短路。
    # 「下一颗蛋及后续 N」会不停补货，用它当"没事可做"的依据等于让合并永久停摆
    # （2026-11-14 实测：merge 连续两轮直接 return，一行读板日志都没有）。
    merged = do_merge(g, int(cfg["merge_budget"]),
                      float(cfg.get("merge_seconds", 240)),
                      allow_crack=bool(cfg.get("actions", {}).get("crack")),
                      crack_policy=str(cfg.get("crack_policy", "lowest")),
                      max_cracks=int(cfg.get("crack_max", 3)))
    stats["last_pairs"] = merged
    stats["merges"] += merged
    stats["last_queue"] = q
    res = {"queue": q, "merged": merged}
    if cfg["actions"].get("crack"):
        ok3, why3 = authorize(g, "crack", cfg)
        if ok3 and do_crack(g, int(cfg["crack_min_level"])):
            stats["cracks"] += 1
            res["cracked"] = True
        elif not ok3:
            log_event(action="deny", target="crack", reason=why3)
    if cfg["actions"].get("buy"):
        ok3, why3 = authorize(g, "buy", cfg)
        if ok3 and do_buy(g):
            stats["buys"] += 1
            res["bought"] = True
        elif not ok3:
            log_event(action="deny", target="buy", reason=why3)
    return res


def task_slot(g: Game, stats: dict, cfg: dict) -> dict:
    """花星星解锁槽位（仅在够钱且价格不超上限时）。"""
    ok3, why3 = authorize(g, "slot", cfg)
    if not ok3:
        log_event(action="deny", target="slot", reason=why3)
        return {"blocked": why3}
    done = do_unlock_slot(g, int(cfg.get("slot_max_price", 300)))
    if done:
        stats["slots_unlocked"] = int(stats.get("slots_unlocked", 0)) + 1
    # 解锁之后必须**把新鸭子安排进槽位**，否则槽位空着、鸭子数不涨（实测确认）
    assigned = do_assign_slot(g)
    if assigned:
        stats["slots_assigned"] = int(stats.get("slots_assigned", 0)) + 1
    return {"unlocked": done, "assigned": assigned}


def task_daily(g: Game, stats: dict, cfg: dict) -> dict:
    """常规任务：领 DAILY PASS 上的所有可领奖励。"""
    ok3, why3 = authorize(g, "daily", cfg)
    if not ok3:
        log_event(action="deny", target="daily", reason=why3)
        return {"blocked": why3}
    res = do_daily_pass(g)
    if res.get("claimed"):
        stats["daily_claimed"] = int(stats.get("daily_claimed", 0)) + res["claimed"]
    return res


def task_sell(g: Game, stats: dict, cfg: dict) -> dict:
    """常规任务：把「已喂满且 ≥UNCOMMON」的鸭子挂到市场换星星。

    用户规则（硬约束）：只有**已喂到顶**且**稀有度 ≥ UNCOMMON** 的鸭子才卖；
    卖法用「市场出售」（换星星），不用「立即出售」（换玉米）。
    """
    ok3, why3 = authorize(g, "sell", cfg)
    if not ok3:
        log_event(action="deny", target="sell", reason=why3)
        return {"blocked": why3}
    done = do_sell_duck(g)
    if done:
        stats["ducks_sold"] = int(stats.get("ducks_sold", 0)) + 1
    return {"sold": done}


def task_ads(g: Game, stats: dict, cfg: dict) -> dict:
    """常规任务：任务页看广告领玉米。"""
    ok3, why3 = authorize(g, "ads", cfg)
    if not ok3:
        log_event(action="deny", target="ads", reason=why3)
        return {"blocked": why3}
    res = do_watch_ads(g, int(cfg.get("ad_max", 5)),
                       int(cfg.get("ad_timeout", 90)))
    if res.get("ads"):
        stats["ads_watched"] = int(stats.get("ads_watched", 0)) + res["ads"]
    return res


TIER_TASKS = {"collect": task_collect, "feed": task_feed, "merge": task_merge,
              "slot": task_slot, "daily": task_daily, "ads": task_ads,
              "sell": task_sell}


def run_tier(g: Game, stats: dict, cfg: dict, tier: str, spec: dict) -> None:
    no_failsafe()   # 每个任务批次前重申一次，避免运行时被重置
    for name in spec.get("tasks", []):
        if not cfg["actions"].get(name, True):
            continue
        fn = TIER_TASKS.get(name)
        if not fn:
            continue
        if not g.usable():
            raise FocusLost()
        log(f"[{tier}] {name}")
        try:
            res = fn(g, stats, cfg) or {}
        except FocusLost:
            raise
        except Exception as exc:
            log(f"[{tier}] {name} 失败：{exc}")
            res = {"error": str(exc)[:160]}
        res.update(tier=tier, task=name)
        log_event(action=name, **res)


def tiers_due(cfg: dict, stats: dict) -> bool:
    """有没有到点的层级；没到点就完全不碰窗口（连 OCR 都不做）。"""
    runs = stats.get("_last_run", {})
    now = time.time()
    return any(
        spec.get("enabled", True)
        and now - runs.get(tier, 0) >= float(spec.get("interval", 3600))
        for tier, spec in cfg["tiers"].items())


def one_cycle(g: Game, stats: dict, cfg: dict) -> None:
    """一次调度：感知红点 -> 清弹窗 -> 按 tiers 的间隔跑各自任务。"""
    if not g.refresh():
        raise RuntimeError("window lost")
    if not g.usable():
        raise FocusLost()

    dets = g.detect_nav()
    dots = [b for b in g.nav_badges(dets) if b.get("near")]
    stats["badges"] = [b["near"] for b in dots]
    if dots:
        log_event(action="badges", near=[b["near"] for b in dots],
                  boxes=[b["box"] for b in dots])

    if g.close_popup():
        stats["popups"] += 1
    # 关过之后仍然存在 → 才是真的环境不可用
    check_env(g)

    # 浮层没关掉时不要按小时等：清空计时让下一轮（60 秒后）立刻重试，
    # 否则会像 2026-11-12 那样整夜每轮都撞同一个浮层。
    if getattr(g, "popup_fails", 0) > 0:
        log(f"  浮层未关闭（连续 {g.popup_fails} 次），本轮跳过任务，60 秒后重试")
        stats["_last_run"] = {}
        stats["cycles"] += 1
        log_event(action="cycle_skip", reason="popup_stuck", fails=g.popup_fails)
        return

    runs = stats.setdefault("_last_run", {})
    for tier, spec in cfg["tiers"].items():
        if not spec.get("enabled", True):
            continue
        if time.time() - runs.get(tier, 0) < float(spec.get("interval", 60)):
            continue
        if stats.pop("defer_upgrade", False) and tier != "routine":
            # 喂食被「观测失败」中断：本轮不切页，且让 routine 60 秒后立刻重试，
            # 最多推迟 3 次（否则观测一直失败会把合并永远饿死）。
            n = int(stats.get("defer_count", 0)) + 1
            stats["defer_count"] = n
            if n <= 3:
                runs["routine"] = 0
                runs[tier] = time.time() - float(spec.get("interval", 60)) + 300
                log(f"  本轮推迟 {tier}（第 {n} 次）：喂食中断，60 秒后重试喂食")
                continue
            stats["defer_count"] = 0
            log(f"  已推迟 {n - 1} 次仍未成功喂食，放弃推迟，本轮照常 {tier}")
        run_tier(g, stats, cfg, tier, spec)
        runs[tier] = time.time()

    stats["cycles"] += 1
    log_event(action="cycle_end", cycle=stats["cycles"],
              feeds=stats["feeds"], merges=stats["merges"],
              claims=stats["claims"], queue=stats.get("queue"),
              feed_price=stats.get("feed_price"))
    stats["masks"] = mask_balance(g)
    prev = stats.get("stars")
    now = star_balance(g)
    stats["stars"] = now
    if now is not None and prev != now:
        log(f"  星星：{prev} → {now}")
        log_event(action="stars", before=prev, after=now)


SELFTEST_PAGES = [
    ("duck", "DUCK × MY × DUCK 鸭群 243K 30 点击喂食6 重置倒计时7:00:00 繁殖3/5 质押 市场 蛋 鸭子 GOOS 任务"),
    ("duck", "鸭群 243K 鸭子正在出售 此鸭子已在市场队列中 前往市场 市场 蛋 鸭子 GOOS 任务"),
    ("duck", "因此只鸭子已准备好质押 升级鸭子 市场 蛋 鸭子 GOOS 任务"),
    ("egg", "鸭群 243K 鸭群奖励 下一颗蛋及后续10颗 自动合并 市场 蛋 鸭子 GOOS 任务"),
    ("market", "DUCKS GODS 我的市场 鸭子交易者"),
    ("task", "Challenge Week 广告 Pack Challenges"),
    ("unknown", "这是一段无关的文本"),
]

SELFTEST_AUTH = [
    # (page, action, 期望允许)
    ("duck", "feed", True),
    ("egg", "feed", False),
    ("egg", "merge", True),
    ("duck", "merge", False),
    ("duck", "collect", True),
    ("duck", "buy", False),
    ("egg", "crack", True),    # 敲蛋已按用户策略开放（config.actions.crack=true）
    # 配种已按用户选择开放（config.actions.breed=true），只在鸭子页允许
    ("duck", "breed", True),
    ("egg", "breed", False),
    ("unknown", "feed", False),
]


def selftest() -> int:
    """离线自检：页面分类 + 授权矩阵。不读屏幕、不点任何东西。"""
    ok = True
    print("== 页面分类 ==")
    for expect, text in SELFTEST_PAGES:
        page, conf, ev = classify_page(text)
        good = (page == expect)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  期望 {expect:9s} 实得 {page:9s} conf={conf} ev={ev[:2]}")
    cfg = load_config()
    print("== 授权矩阵 ==")
    for page, action, expect in SELFTEST_AUTH:
        allowed, why = authorize_page(page, action, cfg)
        good = (allowed == expect)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {page:8s} {action:8s} "
              f"{'ALLOW' if allowed else 'DENY ':5s} ({why}) 期望 {'ALLOW' if expect else 'DENY'}")
    print(f"\n自检结果：{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def report(hours: float = 24.0) -> int:
    """汇总事件流：喂食/合并/拒绝/页面异常/错误，按小时分布。只读。"""
    from collections import Counter, defaultdict
    from datetime import datetime as dt, timedelta
    if not EVENTS_PATH.exists():
        print("没有事件流")
        return 2
    cutoff = dt.now() - timedelta(hours=hours)
    rows = []
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            ts = dt.fromisoformat(r["ts"])
        except Exception:
            continue
        if ts >= cutoff:
            rows.append(r)

    acts = Counter(r.get("action") for r in rows)
    feeds = [r for r in rows if r.get("action") == "feed" and r.get("fed")]
    merges = [r for r in rows if r.get("action") == "merge" and r.get("via")]
    denials = Counter(r.get("reason") for r in rows if r.get("action") == "deny")
    blocked = Counter(r.get("blocked") for r in rows if r.get("blocked"))
    page_bad = sum(1 for r in rows if r.get("action") == "page_check" and not r.get("ok"))
    errors = sum(1 for r in rows if r.get("action") == "error")
    per_hour = defaultdict(int)
    for r in rows:
        if r.get("action") == "cycle_end":
            per_hour[r["ts"][11:13]] += 1

    cycles = [r for r in rows if r.get("action") == "cycle_end"]
    productive = [c for c in cycles
                  if (c.get("feeds") or 0) + (c.get("merges") or 0) + (c.get("claims") or 0) > 0]

    out = {
        "window_hours": hours,
        "events": len(rows),
        "cycles": len(cycles),
        "productive_cycles": len(productive),
        "cycle_success_rate": round(len(productive) / max(1, len(cycles)), 3),
        "actions": acts.most_common(10),
        "feed_runs": len(feeds),
        "feeds_total": sum(r["fed"] for r in feeds),
        "feed_price_range": [min(r.get("start_price") or 0 for r in feeds),
                             max(r.get("end_price") or 0 for r in feeds)] if feeds else None,
        "vision_merges": len(merges),
        "vision_merge_ok": sum(1 for r in merges if r.get("post_ok")),
        "merge_levels": Counter(r.get("level") for r in merges).most_common(8),
        "denials": denials.most_common(),
        "blocked": blocked.most_common(),
        "page_check_failed": page_bad,
        "errors": errors,
        "env_blocked": sum(1 for r in rows if r.get("action") == "env_blocked"),
        "env_blocked_reasons": Counter(r.get("reason") for r in rows
                                       if r.get("action") == "env_blocked").most_common(),
        "cycles_by_hour": dict(sorted(per_hour.items())),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    (OUT_DIR / "duck_guard_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


def health(hours: float = 6.0) -> int:
    """一行健康结论：运行中？在干活？卡在哪种原因？"""
    from collections import Counter
    from datetime import datetime as dt, timedelta
    pid = read_pid()
    alive = bool(pid and pid_alive(pid))
    rows = []
    if EVENTS_PATH.exists():
        cutoff = dt.now() - timedelta(hours=hours)
        for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if dt.fromisoformat(r["ts"]) >= cutoff:
                    rows.append(r)
            except Exception:
                continue
    cycles = [r for r in rows if r.get("action") == "cycle_end"]
    prod = [c for c in cycles
            if (c.get("feeds") or 0) + (c.get("merges") or 0) + (c.get("claims") or 0) > 0]
    rate = round(len(prod) / max(1, len(cycles)), 3)
    env = sum(1 for r in rows if r.get("action") == "env_blocked")
    blocked = Counter(r.get("blocked") for r in rows if r.get("blocked"))
    idle = sum(1 for r in rows if r.get("action") == "page_check" and not r.get("ok"))
    errs = sum(1 for r in rows if r.get("action") == "error")

    if not alive:
        verdict = "STOPPED"
    elif env > 0 and rate == 0:
        verdict = "ENV_BLOCKED"
    elif cycles and rate == 0 and blocked.get("nav_failed", 0) > 0:
        verdict = "NAV_BROKEN"
    elif cycles and blocked.get("nav_failed", 0) / max(1, len(cycles)) > 0.3:
        verdict = "NAV_DEGRADED"
    elif errs > 0:
        verdict = "ERRORS"
    elif cycles and rate == 0:
        verdict = "WAITING (无事可做/不在前台)"
    else:
        verdict = "OK"
    print(f"[{verdict}] pid={'%d' % pid if alive else 'none'} "
          f"cycles={len(cycles)} rate={rate} env_blocked={env} "
          f"nav_failed={blocked.get('nav_failed', 0)} page_bad={idle} errors={errs} "
          f"(近 {hours:g}h)")
    return 0


EXP_RULES_PATH = ROOT / "results" / "computer_use" / "exp_loop" / "rules.json"

# 经验规则 → 生产代码里的执行钩子（规则只有在这里找得到实现，才算真正生效）
EXP_HOOKS = {
    "R1_assert_effect": ("click_slot_verified", "_step_duck"),
    "R2_refetch_after_action": ("read_duck_profile", "_page_text"),
    "R3_empty_is_not_absent": ("ocr_fast",),
    "R4_labels_only": ("_find_exact",),
    "R5_hit_icon_center": ("_click_icon",),
}


def report_experience() -> list[str]:
    """启动时核对经验库：只报 active，并检查钩子是否都存在。

    这是"经验不得直接污染生产"的闸门：exp_loop 写的是 proposed/active，
    但代码这边必须自己确认「这条规则的执行点真的存在」，否则不认。
    """
    try:
        data = json.loads(EXP_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        log("  经验库：未找到可用规则（跳过）")
        return []
    active, broken = [], []
    for r in data.get("rules", []):
        if r.get("status") != "active":
            continue
        rid = r.get("rule_id")
        hooks = EXP_HOOKS.get(rid) or ()
        missing = [h for h in hooks if h not in globals()]
        if not hooks or missing:
            broken.append(rid)
            continue
        v = (r.get("metrics") or {}).get("val") or {}
        active.append(f"{rid}(R={v.get('recall')},FPR={v.get('fpr')})")
    if active:
        log("  经验规则已生效：" + "；".join(active))
    if broken:
        log("  经验规则缺少执行钩子，未生效：" + "、".join(broken))
    return active


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="0 = 一直跑")
    ap.add_argument("--interval", type=float, default=0, help="每轮间隔秒；0=用配置里的")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-focus", action="store_true", help="(保留兼容，默认就不抢焦点)")
    ap.add_argument("--focus", action="store_true", help="允许抢前台把游戏提到最上层")
    ap.add_argument("--pin", default="", help="窗口位置 x,y，如 2058,0")
    ap.add_argument("--stop", action="store_true", help="停止正在运行的看护")
    ap.add_argument("--status", action="store_true", help="查看看护状态")
    ap.add_argument("--icons", action="store_true", help="识别当前窗口的图标/红点后退出")
    ap.add_argument("--plan", action="store_true", help="只打印分级调度计划，不动窗口")
    ap.add_argument("--selftest", action="store_true", help="离线自检页面分类与授权矩阵")
    ap.add_argument("--report", type=float, default=-1, help="汇总最近 N 小时事件流")
    ap.add_argument("--health", type=float, default=-1, help="一行健康结论（最近 N 小时）")
    args = ap.parse_args()

    if args.health >= 0:
        return health(args.health or 6.0)

    if args.selftest:
        return selftest()
    if args.report >= 0:
        return report(args.report or 24.0)

    if args.plan:
        cfg = load_config()
        last: dict = {}
        print("分级任务计划（模拟 6 小时）:")
        for tier, spec in cfg["tiers"].items():
            print(f"  {tier}: 每 {spec.get('interval')}s -> {spec.get('tasks')}")
        print("调度轨迹:")
        for t in range(0, 21601, 60):
            due = [k for k, v in cfg["tiers"].items()
                   if v.get("enabled", True)
                   and t - last.get(k, 0) >= float(v.get("interval", 60))]
            if due:
                print(f"  {t//3600}:{(t%3600)//60:02d}  " + ", ".join(due))
                for k in due:
                    last[k] = t
        return 0

    if args.stop:
        return do_stop()
    if args.status:
        return do_status()

    global PIN_RECT
    if args.pin:
        try:
            x, y = (int(v) for v in args.pin.split(","))
            PIN_RECT = (x, y, PIN_RECT[2], PIN_RECT[3])
        except Exception:
            log(f"--pin 解析失败：{args.pin}")

    ok, gate = gate_open()
    if not ok:
        log("感知门禁未通过，拒绝执行生产动作。")
        log(f"  门禁文件：{GATE_PATH}")
        log("  先运行 experiments/computer_use/duck_perceive.py 采集指标，"
            "达标后写入 {\"passed\": true}。")
        return 3

    hwnd = find_game_window()
    if not hwnd:
        log("找不到小程序窗口（标题含 DUCK/鸭）")
        return 2
    if args.icons:
        from ui_icons import cmd_debug
        print(json.dumps(cmd_debug(grab(hwnd)), ensure_ascii=False, indent=2))
        return 0
    g = Game(hwnd)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.once and not acquire_lock():
        return 4
    g.pin()
    log(f"守护启动 hwnd={hwnd} rect={g.l},{g.t},{g.w},{g.h} 上限={FEED_CAP} 敲蛋阈值=Lv{CRACK_MIN_LEVEL}")
    report_experience()

    stats = {"cycles": 0, "feeds": 0, "merges": 0, "cracks": 0,
             "buys": 0, "breeds": 0, "popups": 0, "claims": 0, "love_eggs": 0,
             "errors": 0, "env_blocked": 0,
             "page": None, "feed_price": None, "masks": None, "badges": None}
    started = time.time()
    fails = 0

    while not stop_requested():
        if args.minutes and (time.time() - started) / 60 >= args.minutes:
            log("到达时长上限，退出")
            break
        cfg = load_config()          # 每轮热加载，下游改配置即刻生效
        if not args.once and not tiers_due(cfg, stats):
            time.sleep(20)           # 没到点：不碰窗口，也不做 OCR
            continue
        if args.focus:
            if not g.usable():
                g.pin()
                g.focus()
        else:
            if not g.usable():
                # 默认不抢焦点：游戏不在前台就安静等待，绝不动鼠标
                stats["idle"] = stats.get("idle", 0) + 1
                if stats["idle"] in (1, 10) or stats["idle"] % 60 == 0:
                    log("到点了但游戏不在前台，等待（不抢焦点）")
                write_status(stats)
                if args.once:
                    break
                time.sleep(float(cfg.get("interval", 60)))
                continue
        try:
            one_cycle(g, stats, cfg)
            fails = 0
        except FocusLost:
            stats["focus_lost"] = stats.get("focus_lost", 0) + 1
            log("焦点被切走，本轮中止（不抢焦点）")
            write_status(stats)
            time.sleep(5)
            continue
        except EnvBlocked as exc:
            stats["env_blocked"] = stats.get("env_blocked", 0) + 1
            log(f"环境不可用（{exc.reason}），暂停 10 分钟再试")
            log_event(action="env_blocked", reason=exc.reason)
            write_status(stats)
            time.sleep(600)
            continue
        except Exception:
            fails += 1
            stats["errors"] += 1
            log("异常：" + traceback.format_exc(limit=2).replace("\n", " | "))
            if fails >= 3:
                log("连续失败，重载小程序重试")
                pyautogui.hotkey("ctrl", "r")
                time.sleep(8)
                fails = 0
        stats["elapsed_min"] = round((time.time() - started) / 60, 1)
        write_status(stats)
        if args.once:
            break
        time.sleep(args.interval or cfg.get("interval", 30))

    log(f"守护结束 {json.dumps(stats, ensure_ascii=False)}")
    try:
        import os
        if read_pid() == os.getpid():      # 只清理自己的 PID，别删掉别的实例
            PID_PATH.unlink()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
