"""场景执行器：在**真实界面**里构造反例场景。

铁律：只做真实操作并观测，不修改截图、不合成数据。
构造不出来 → 返回 not_constructible 并说明卡在哪，绝不伪造反例。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # experiments/computer_use


def _ensure_duck(g) -> bool:
    import duck_guard as dg
    import time
    for _ in range(5):
        if len(dg.detect_slot_icons(g)) >= 3:
            return True
        g.close_popup()
        dg.go_home(g)
        dg.goto(g, "duck", "duck", tries=2)
        time.sleep(1.2)
    return False


def construct_positional(g, thr: float, max_steps: int = 40) -> dict:
    """把「合法可点对象」送到 x ≥ thr，再点它并验证。

    逐个位置尝试，并记录每个位置的证据：
      缝隙（点了没变化）→ 不算"该位置无合法对象"，继续找
      挂锁（页面跳转）  → 该位置确实不是合法对象
    只有"所有可达位置都试过且都不是合法对象"才判不可构造。
    """
    import time
    import duck_guard as dg

    if not _ensure_duck(g):
        return {"constructible": False, "reason": "鸭子页不可达"}

    tried: list[str] = []
    misses = 0
    for _ in range(max_steps):
        icons = dg.detect_slot_icons(g)
        sel = dg.selected_icon(icons)
        if sel is None:
            misses += 1
            if misses > 3:
                return {"constructible": False,
                        "reason": "选中态持续不可观测（页面未就绪）"}
            g.close_popup()
            dg.go_home(g)
            dg.goto(g, "duck", "duck", tries=2)
            time.sleep(1.2)
            continue
        misses = 0

        pitch = dg.slot_pitch(icons)
        target = round(icons[sel]["x"] + pitch, 3)
        if target > 0.95:
            return {"constructible": False,
                    "reason": f"已到最右；试过的位置：{tried or '无'}"}

        if target >= thr:
            ok, nx = dg._step_duck(g, +1)
            if ok:
                return {"constructible": True, "verified": True, "x": round(nx, 3),
                        "reason": f"在 x={nx:.3f}（≥{thr}）完成了一次合法点击，选中态已改变"}
            page = dg.current_page(g, tries=1)[0]
            if page != "duck":
                tried.append(f"{target:.3f}=挂锁")
                g.close_popup()
                dg.goto(g, "duck", "duck", tries=2)
                time.sleep(1.0)
                return {"constructible": False,
                        "reason": f"可达位置均非合法对象：{'; '.join(tried)}"}
            tried.append(f"{target:.3f}=缝隙")
        ok, _ = dg._step_duck(g, +1)
        if not ok:
            return {"constructible": False,
                    "reason": f"向右无路可走；试过的位置：{tried or '无'}"}
        time.sleep(0.2)

    return {"constructible": False, "reason": f"步数用尽；试过的位置：{tried}"}


def construct_page_identity(g, thr: float) -> dict:
    """在与规则命中相当（文本长度 ≥ thr）的页面上，完成一次合法动作。"""
    import time
    import duck_guard as dg

    def txt() -> str:
        return " ".join(l["text"] for l in g.lines())

    g.nav("task", wait=2.0)
    time.sleep(1.2)
    before = txt()
    if len(before) < thr:
        return {"constructible": False,
                "reason": f"可达页面文本长度 {len(before)} < 阈值 {thr}，无法构造同类场景"}
    if not dg._open_ads_tab(g):
        return {"constructible": False, "reason": "无法定位「广告」子标签"}
    after = txt()
    ok = "观看广告" in after and after != before
    return {"constructible": True, "verified": ok, "x": None,
            "reason": ("长文本页面上完成了一次有效动作（子标签切换成功）" if ok
                       else "动作未产生可观测结果")}
