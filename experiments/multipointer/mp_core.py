"""Win32 合成指针 PoC 核心：ctypes 定义 + 注入器 + 消息记录窗口。

不写驱动、不改目标应用。只验证 CreateSyntheticPointerDevice / InjectSyntheticPointerInput
在本机是否可用，以及 WM_POINTER / WM_TOUCH / WM_MOUSE 的实际投递行为。
"""
from __future__ import annotations

import ctypes
import json
import threading
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

PT_TOUCH = 2
POINTER_FEEDBACK_DEFAULT = 1
POINTER_FLAG_NEW = 0x1
POINTER_FLAG_INRANGE = 0x2
POINTER_FLAG_INCONTACT = 0x4
POINTER_FLAG_DOWN = 0x10000
POINTER_FLAG_UPDATE = 0x20000
POINTER_FLAG_UP = 0x40000
TOUCH_MASK_CONTACTAREA = 0x1
TOUCH_MASK_PRESSURE = 0x4

WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_TOUCH = 0x0240
WM_POINTERUPDATE = 0x0245
WM_POINTERDOWN = 0x0246
WM_POINTERUP = 0x0247
WM_POINTERENTER = 0x0249
WM_POINTERLEAVE = 0x024A


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class POINTER_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerType", wintypes.UINT),
        ("pointerId", wintypes.UINT),
        ("frameId", wintypes.UINT),
        ("pointerFlags", wintypes.UINT),
        ("sourceDevice", wintypes.HANDLE),
        ("hwndTarget", wintypes.HWND),
        ("ptPixelLocation", POINT),
        ("ptHimetricLocation", POINT),
        ("ptPixelLocationRaw", POINT),
        ("ptHimetricLocationRaw", POINT),
        ("dwTime", wintypes.DWORD),
        ("historyCount", wintypes.UINT),
        ("InputState", wintypes.INT),
        ("dwKeyStates", wintypes.UINT),
        ("PerformanceCount", ctypes.c_ulonglong),
        ("ButtonChangeType", wintypes.INT),
    ]


class POINTER_TOUCH_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerInfo", POINTER_INFO),
        ("touchFlags", wintypes.UINT),
        ("touchMask", wintypes.UINT),
        ("rcContact", RECT),
        ("rcContactRaw", RECT),
        ("orientation", wintypes.UINT),
        ("pressure", wintypes.UINT),
    ]


class POINTER_TYPE_INFO(ctypes.Structure):
    _fields_ = [("type", wintypes.UINT), ("touchInfo", POINTER_TOUCH_INFO)]


user32.CreateSyntheticPointerDevice.restype = wintypes.HANDLE
user32.CreateSyntheticPointerDevice.argtypes = [wintypes.UINT, wintypes.ULONG, wintypes.UINT]
user32.InjectSyntheticPointerInput.restype = wintypes.BOOL
user32.InjectSyntheticPointerInput.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(POINTER_TYPE_INFO),
                                               wintypes.UINT]
user32.DestroySyntheticPointerDevice.argtypes = [wintypes.HANDLE]
user32.GetPointerInfo.argtypes = [wintypes.UINT, ctypes.POINTER(POINTER_INFO)]
user32.GetPointerInfo.restype = wintypes.BOOL
user32.DefWindowProcW.restype = ctypes.c_ssize_t
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                  wintypes.WPARAM, wintypes.LPARAM]


class SyntheticTouch:
    def __init__(self, max_count: int = 10):
        self.dev = user32.CreateSyntheticPointerDevice(PT_TOUCH, max_count,
                                                       POINTER_FEEDBACK_DEFAULT)
        self.err = ctypes.get_last_error()
        if not self.dev:
            raise OSError(f"CreateSyntheticPointerDevice failed: {self.err}")

    def inject(self, contacts: list[dict]) -> tuple[bool, int]:
        arr = (POINTER_TYPE_INFO * len(contacts))()
        for i, c in enumerate(contacts):
            arr[i].type = PT_TOUCH
            ti = arr[i].touchInfo
            ti.pointerInfo.pointerType = PT_TOUCH
            ti.pointerInfo.pointerId = c["id"]
            ti.pointerInfo.pointerFlags = c["flags"]
            ti.pointerInfo.ptPixelLocation = POINT(c["x"], c["y"])
            ti.pointerInfo.ptPixelLocationRaw = POINT(c["x"], c["y"])
            ti.pointerInfo.dwTime = int(time.time() * 1000) & 0xFFFFFFFF
            ti.touchFlags = 0
            ti.touchMask = TOUCH_MASK_CONTACTAREA | TOUCH_MASK_PRESSURE
            ti.rcContact = RECT(c["x"] - 8, c["y"] - 8, c["x"] + 8, c["y"] + 8)
            ti.pressure = c.get("pressure", 32000)
        ok = user32.InjectSyntheticPointerInput(self.dev, arr, len(contacts))
        return bool(ok), ctypes.get_last_error()

    def close(self):
        if self.dev:
            user32.DestroySyntheticPointerDevice(self.dev)
            self.dev = None


WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class PointerWindow:
    """一个记录 WM_POINTER / WM_TOUCH / WM_MOUSE 的顶层窗口。"""

    def __init__(self, title: str, rect=(100, 100, 1200, 900),
                 legacy: bool = False, log: list | None = None):
        self.title = title
        self.rect = rect
        self.legacy = legacy
        self.log = log if log is not None else []
        self.hwnd = None
        self._proc = WNDPROC(self._wndproc)
        self.class_name = f"MP_{title}_{id(self)}"
        self.hinst = kernel32.GetModuleHandleW(None)
        self._registered = False

    def create(self):
        wc = WNDCLASSW()
        wc.style = 0
        wc.lpfnWndProc = self._proc
        wc.hInstance = self.hinst
        wc.hCursor = user32.LoadCursorW(None, 32512)  # IDC_ARROW
        wc.hbrBackground = 6  # COLOR_WINDOW + 1
        wc.lpszClassName = self.class_name
        if not user32.RegisterClassW(ctypes.byref(wc)):
            raise OSError(f"RegisterClass failed: {ctypes.get_last_error()}")
        self._registered = True
        L, T, W, H = self.rect
        self.hwnd = user32.CreateWindowExW(
            0x8,  # WS_EX_TOPMOST
            self.class_name, self.title,
            0x80000000 | 0x10000000,  # WS_POPUP | WS_VISIBLE
            L, T, W, H, None, None, self.hinst, None)
        if not self.hwnd:
            raise OSError(f"CreateWindowEx failed: {ctypes.get_last_error()}")
        user32.SetWindowPos(self.hwnd, -1, L, T, W, H, 0x0040)  # HWND_TOPMOST, SWP_SHOWWINDOW
        user32.UpdateWindow(self.hwnd)
        return self.hwnd

    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if msg in (WM_POINTERDOWN, WM_POINTERUPDATE, WM_POINTERUP,
                       WM_POINTERENTER, WM_POINTERLEAVE):
                pid = wparam & 0xFFFF
                info = POINTER_INFO()
                ok = user32.GetPointerInfo(pid, ctypes.byref(info))
                entry = {
                    "t": round(time.time(), 3),
                    "win": self.title,
                    "msg": "pointer",
                    "wm": hex(msg),
                    "pointer_id": int(info.pointerId) if ok else int(pid),
                    "flags": hex(int(info.pointerFlags)) if ok else None,
                    "x": int(info.ptPixelLocation.x) if ok else None,
                    "y": int(info.ptPixelLocation.y) if ok else None,
                    "hwndTarget": int(info.hwndTarget) if ok else None,
                    "get_ok": bool(ok),
                    "legacy": self.legacy,
                }
                self.log.append(entry)
                if self.legacy:
                    # 传统窗口：交给 DefWindowProc，观察是否被提升为鼠标
                    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)
                return 0
            if msg == WM_TOUCH:
                self.log.append({"t": round(time.time(), 3), "win": self.title,
                                 "msg": "touch", "wm": hex(msg)})
                return 0
            if msg in (WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP):
                x = ctypes.c_short(lparam & 0xFFFF).value
                y = ctypes.c_short((lparam >> 16) & 0xFFFF).value
                self.log.append({"t": round(time.time(), 3), "win": self.title,
                                 "msg": "mouse", "wm": hex(msg), "x": x, "y": y})
                if msg == WM_LBUTTONDOWN:
                    return 0
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
        except Exception as e:  # noqa: BLE001
            self.log.append({"t": round(time.time(), 3), "win": self.title,
                             "msg": "error", "error": f"{type(e).__name__}: {e}"})
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def loop(self):
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def run(self, injector, delay=1.5, after_close=0.6):
        """创建窗口、跑消息循环，injector 在 delay 后于子线程执行。"""
        hwnd = self.create()
        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        holder = {"hwnd": hwnd}

        def fire():
            time.sleep(delay)
            try:
                injector()
            finally:
                time.sleep(after_close)
                user32.PostMessageW(holder["hwnd"], WM_CLOSE, 0, 0)

        th = threading.Thread(target=fire, daemon=True)
        th.start()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        th.join(timeout=2.0)
        return self.log


def touch_seq(points: list[tuple[int, int]], step=0.45):
    """返回一个 injector：10 指 down → move → up。"""
    down = [{"id": i + 1, "x": x, "y": y,
             "flags": POINTER_FLAG_NEW | POINTER_FLAG_INRANGE |
                      POINTER_FLAG_INCONTACT | POINTER_FLAG_DOWN}
            for i, (x, y) in enumerate(points)]
    move = [{"id": i + 1, "x": x, "y": y + 90,
             "flags": POINTER_FLAG_UPDATE | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT}
            for i, (x, y) in enumerate(points)]
    up = [{"id": i + 1, "x": x, "y": y + 90, "flags": POINTER_FLAG_UP}
          for i, (x, y) in enumerate(points)]

    def run(dev: SyntheticTouch):
        out = [("down", dev.inject(down))]
        time.sleep(step)
        out.append(("move", dev.inject(move)))
        time.sleep(step)
        out.append(("up", dev.inject(up)))
        return out

    return run


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
