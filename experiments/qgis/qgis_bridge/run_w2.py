"""
QGIS W2：多步 GUI 任务（全部由 PyQGIS 状态判定）

  T1 select_by_expression ：激活图层 → 触发菜单动作 → 往表达式编辑器写文本
                            → 点 "Select Features" → 真值 selected == 2
  T2 toggle_visibility    ：图层树选中行 + 真键盘 Space → 真值 visible 翻转
  T3 select_all_deselect  ：两个菜单动作 → 真值 5 → 0

用法:
    python experiments/qgis/qgis_bridge/run_w2.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402


def truth():
    return call({"op": "truth"})["truth"]


def sel():
    t = truth()
    return t["layers"][0]["selected"] if t.get("layers") else None


def vis():
    t = truth()
    return t["layers"][0]["visible"] if t.get("layers") else None


def T1():
    call({"op": "activate_layer", "name": "cities"})
    steps = []
    steps.append(call({"op": "act", "kind": "action",
                       "text": "Select Features by Expression…"}))
    steps.append(call({"op": "act", "kind": "text",
                       "widget": {"class": "QgsCodeEditorExpression"},
                       "value": '"pop" > 1000000'}))
    steps.append(call({"op": "act", "kind": "click",
                       "widget": {"class": "QToolButton", "text": "Select Features"}}))
    time.sleep(1)
    ok = sel() == 2
    call({"op": "close_dialogs"})
    return ok, steps


def T2():
    before = vis()
    steps = [call({"op": "layer_tree_click", "name": "cities"})]
    after = vis()
    return (before is not None and after is not None and after != before), steps


def T3():
    steps = []
    call({"op": "activate_layer", "name": "cities"})
    steps.append(call({"op": "act", "kind": "action", "text": "Select All Features"}))
    a = sel()
    steps.append(call({"op": "act", "kind": "action",
                       "text": "Deselect Features from All Layers"}))
    b = sel()
    return (a == 5 and b == 0), steps


def main():
    call({"op": "win", "maximize": True})
    out = {}
    for name, fn in (("select_by_expression", T1),
                     ("toggle_visibility", T2),
                     ("select_all_deselect", T3)):
        call({"op": "reset"})
        t0 = time.perf_counter()
        ok, steps = fn()
        dt = time.perf_counter() - t0
        n_ok = sum(1 for s in steps if s.get("ok"))
        print(f"  [{name:<22}] truth_ok={ok} steps={len(steps)} "
              f"ok_steps={n_ok} {dt*1000:.0f}ms")
        for s in steps:
            if not s.get("ok"):
                print(f"      ! {s}")
        out[name] = dict(truth_ok=ok, steps=len(steps), ok_steps=n_ok,
                         ms=round(dt * 1000))
    all_ok = all(v["truth_ok"] for v in out.values())
    print(f"\nW2 判据：三个多步任务全部由状态判定通过 -> {all_ok}")
    (HERE / "w2_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w2_results.json")


if __name__ == "__main__":
    main()
