"""
QGIS W17：真 GUI 工作流的鲁棒性与摊销

W16 证明了纯 GUI 导出（7/7，~9.5 s），但只在一个固定 UI 状态下跑过。
本轮把同一流程放到 4 种 UI 状态（窗口尺寸 × 控制台开关）下各跑 3 次：

  1) 鲁棒性：成功率与耗时是否随布局变化而崩？（W17 前把 W16 里三处
     状态相关的控件定位改成状态无关：Format 按名字、File name 按标签同行、
     OK 取最下行最左按钮。）
  2) 摊销：第 1 次（冷）与稳态（热）的耗时差 —— 发现类信息是否可缓存复用。
  3) 顺带记录：每次重取 UIA 窗口句柄（防过期），并把失败留截图。

用法: python experiments/qgis/qgis_bridge/run_w17.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w16 import export_via_gui, features_of  # noqa: E402

SCENE = "regions"
SEL_EXPR = '"area" > 300'
N_SEL = 2
STATES = [("max", False), ("max", True), ("1000x750", False), ("1000x750", True)]
TRIALS = 3


def prep_state(size, console):
    if size == "max":
        call({"op": "win", "maximize": True})
    else:
        w, h = size.split("x")
        call({"op": "win", "w": int(w), "h": int(h)})
    call({"op": "reset", "scene": SCENE})
    t = call({"op": "truth"})["truth"]
    if bool(t.get("console_visible")) != console:
        call({"op": "act", "kind": "action", "text": "Python Console"})
        time.sleep(0.5)
    call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
    time.sleep(0.4)


def main():
    path = HERE / "data" / "sel.geojson"
    out = {}
    cold = None
    print(f"4 状态 × {TRIALS} 次（Save Selected Features As… → {path.name}）\n")
    print(f"{'state':>18}{'trial':>6}{'成功':>6}{'总ms':>8}{'菜单':>7}{'对话框':>8}{'落盘':>7}{'要素':>5}  备注")
    all_ok = all_n = 0
    for size, console in STATES:
        key = f"{size}{'+console' if console else ''}"
        rows = []
        for i in range(TRIALS):
            prep_state(size, console)
            ok, st, note = export_via_gui(path, selected_only=True)
            names = features_of(path)
            good = ok and len(names) == N_SEL
            ms = st.get("total_ms", 0)
            if cold is None:
                cold = ms
            rows.append(dict(ok=bool(good), ms=ms, note=note, n=len(names)))
            all_ok += int(good); all_n += 1
            print(f"{key:>18}{i+1:>6}{str(good):>6}{ms:>8.0f}{st.get('menu_ms',0):>7.0f}"
                  f"{st.get('dialog_ms',0):>8.0f}{st.get('wait_ms',0):>7.0f}{len(names):>5}  {note or ''}")
        succ = [r for r in rows if r["ok"]]
        out[key] = dict(trials=len(rows), success=len(succ),
                        ms_median=float(np.median([r["ms"] for r in succ])) if succ else None)
    print(f"\n合计 {all_ok}/{all_n} 成功")
    warm = [v["ms_median"] for v in out.values() if v["ms_median"]]
    print(f"冷启动第 1 次 {cold:.0f} ms；各状态稳态中位 {np.median(warm):.0f} ms "
          f"→ 摊销比 {cold/ (np.median(warm) or 1):.2f}×")
    (HERE / "w17_results.json").write_text(json.dumps(
        {"states": out, "cold_ms": cold, "total": f"{all_ok}/{all_n}"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w17_results.json")


if __name__ == "__main__":
    main()
