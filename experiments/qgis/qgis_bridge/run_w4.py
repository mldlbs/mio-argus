"""
QGIS W4：真鼠标臂 —— H1（重观察）在真桌面应用上的第一刀

三种情形，同一目标（工具栏 "Zoom Full"），真值都看 canvas extent：
  A 新鲜坐标：取按钮 rect 后**立刻**真鼠标点击
  B 过期坐标：取 rect 后**改变窗口大小**（UI 漂移），再点旧坐标
  C 盲点    ：没有任何坐标，凭固定点点击

安全：FAILSAFE 开启；结束后指针复位。
用法:
    python experiments/qgis/qgis_bridge/run_w4.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

import pyautogui  # noqa: E402

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def truth():
    return call({"op": "truth"})["truth"]


def btn_rect(text):
    for a in call({"op": "structure"})["actions"]:
        if a["text"] == text and a.get("rect") and a["rect"][2] > 0:
            return a["rect"]
    return None


def click_at(r):
    x, y = r[0] + r[2] // 2, r[1] + r[3] // 2
    pyautogui.moveTo(x, y, duration=0.12)
    pyautogui.click()
    return x, y


def shrink_to_30(reset=True):
    if reset:
        call({"op": "reset"})
    t = truth()
    e = t["layers"][0]["extent"]
    call({"op": "set_extent",
          "box": [e[0], e[1], e[0] + (e[2] - e[0]) * 0.3, e[1] + (e[3] - e[1]) * 0.3]})
    return e


def full_xmin():
    """参考：复位后的 extent[0]（Zoom Full 的期望值）。"""
    call({"op": "reset"})
    return truth()["extent"][0]


def is_full(x, ref, tol=2.0):
    return abs(x - ref) < tol


def main():
    orig = pyautogui.position()
    out = {}
    try:
        call({"op": "win", "maximize": True})
        time.sleep(0.5)
        ref = full_xmin()
        print(f"参考：Zoom Full 后 extent[0] = {ref:.2f}")

        # A 新鲜坐标
        shrink_to_30()
        before = truth()["extent"][0]
        r = btn_rect("Zoom Full")
        t0 = time.perf_counter()
        pos = click_at(r)
        time.sleep(0.5)
        after = truth()["extent"][0]
        dt = (time.perf_counter() - t0) * 1000
        okA = is_full(after, ref)
        print(f"A 新鲜坐标: rect={r} click@{pos} {before:.2f}->{after:.2f} ok={okA} {dt:.0f}ms")
        out["fresh_coords"] = dict(ok=bool(okA), ms=round(dt), rect=r, pos=pos)

        # B 过期坐标（先改窗口大小，再点旧坐标）
        shrink_to_30()
        r = btn_rect("Zoom Full")
        before = truth()["extent"][0]
        call({"op": "win", "w": 1200, "h": 800})       # UI 漂移
        time.sleep(0.4)
        t0 = time.perf_counter()
        pos = click_at(r)                               # 旧坐标
        time.sleep(0.5)
        after = truth()["extent"][0]
        dt = (time.perf_counter() - t0) * 1000
        okB = is_full(after, ref)
        print(f"B 过期坐标: 旧 rect={r} click@{pos} {before:.2f}->{after:.2f} ok={okB} {dt:.0f}ms")
        out["stale_coords"] = dict(ok=bool(okB), ms=round(dt), rect=r, pos=pos)

        # C 盲点
        call({"op": "win", "maximize": True})
        shrink_to_30()
        before = truth()["extent"][0]
        t0 = time.perf_counter()
        pos = click_at([900, 700, 2, 2])
        time.sleep(0.5)
        after = truth()["extent"][0]
        dt = (time.perf_counter() - t0) * 1000
        okC = is_full(after, ref)
        print(f"C 盲点: click@{pos} {before:.2f}->{after:.2f} ok={okC} {dt:.0f}ms")
        out["blind"] = dict(ok=bool(okC), ms=round(dt), pos=pos)

        print(f"\nW4：新鲜坐标 {okA} | 过期坐标 {okB} | 盲点 {okC}")
        print("（预期：True / False / False —— 这就是 H1 在真鼠标上的形式）")
    finally:
        pyautogui.moveTo(orig.x, orig.y, duration=0.15)
        (HERE / "w4_results.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("已写出 experiments/qgis/qgis_bridge/w4_results.json；指针已复位")


if __name__ == "__main__":
    main()
