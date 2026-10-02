"""
QGIS W16：第一个"真工作流" —— 纯 GUI 产出真 GeoJSON（含模态对话框）

前 15 轮都在缩放/控制台这类无产物动作上打转，且一直绕开**模态对话框**。
本轮把整条真实路径走通（全部靠 UIA + 真实鼠标/键盘，桥只用于环境重置与真值）：

    右键图层 → Export（子菜单）→ Save [Selected] Features As…（模态对话框）
    → 选格式 GeoJSON → 填文件名 → 勾"Save only selected features" → OK
    → 磁盘上出现真文件

关键事实（本轮探明）：
  · 图层上下文菜单是**独立顶层窗口**（UIA 可见其一级项），但 **子菜单 UIA 看不见**
    （像素里能看到）→ 必须用键盘导航（Down×n → Right → Down/Enter）；
  · UIA 的菜单项列表**会漏项**（这里漏了 "Toggle Editing"），按键步数必须按**渲染顺序**数；
  · 导出对话框的控件挂在**主窗口**的 UIA 树里（automation_id 前缀 QgisApp.QgsVectorLayerSaveAsDialogBase），
    不是独立顶层；OK 按钮是系统本地化的（中文"确定"）。

用法: python experiments/qgis/qgis_bridge/run_w16.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from pywinauto import Desktop  # noqa: E402
import pyautogui  # noqa: E402

OUT = HERE / "data"
SCENE = "regions"
SEL_EXPR = '"area" > 300'      # North + South = 2 个
N_SEL, N_ALL = 2, 5


def acquire(retries=8):
    for _ in range(retries):
        try:
            for w in Desktop(backend="uia").windows(visible_only=True):
                if "QGIS" in (w.window_text() or ""):
                    if len(w.descendants(control_type="Button")) > 5:
                        return w
        except Exception:
            pass
        time.sleep(0.4)
    return None


def find_layer_item(qg, layer=SCENE):
    for x in qg.descendants(control_type="TreeItem"):
        try:
            if (x.window_text() or "").strip() == layer:
                return x
        except Exception:
            continue
    return None


def export_via_gui(path, selected_only=True, layer=SCENE):
    """全 GUI 导出。返回 (ok, stages_ms, note)。"""
    st = {}
    t_all = time.time()
    p = Path(path)
    if p.exists():
        p.unlink()
    qg = acquire()
    if qg is None:
        return False, st, "no window"
    item = find_layer_item(qg, layer)
    if item is None:
        return False, st, "no layer item"

    # 1) 上下文菜单 → Export → 子菜单（W23：pyautogui 右键在失焦时不弹 → UIA 右键；
    #    Down 次数不再写死 —— Qt 弹出菜单 UIA 看不见，改由桥 menu_items 读序）
    t0 = time.time()
    items = []
    for attempt in range(3):              # UIA 右键也偶发不弹 → 重试
        qg.set_focus(); time.sleep(0.25)
        item.right_click_input()
        for _ in range(8):
            time.sleep(0.25)
            mi = call({"op": "menu_items"})
            items = mi.get("items") or []
            if items:
                break
        if items:
            break
        pyautogui.press("esc"); time.sleep(0.3)
    if not items:
        pyautogui.press("esc")
        return False, st, "context menu not open"
    # 反馈导航（W23 终解）：Home/环绕计数都不可靠 → 逐步 Down 并核对 currentAction
    n_main = len([it for it in items if not it["sep"]])
    on_target = False
    for _ in range(n_main + 3):
        cur = (call({"op": "menu_items"}).get("current") or "")
        if cur == "Export":
            on_target = True
            break
        pyautogui.press("down"); time.sleep(0.08)
    if not on_target:
        pyautogui.press("esc")
        return False, st, "not on Export"
    pyautogui.press("right"); time.sleep(1.0)   # 打开 Export 子菜单
    # 子菜单：读序 + 反馈步进到目标项
    sub = []
    for _ in range(8):
        time.sleep(0.25)
        sub = (call({"op": "menu_items"}).get("items") or [])
        if sub:
            break
    want = ("Save Selected Features As" if selected_only
            else "Save Features As")
    ssep = [it for it in sub if not it["sep"]]
    if not ssep:                            # 右键没进子菜单 → 再试一次
        pyautogui.press("right"); time.sleep(1.0)
        sub = (call({"op": "menu_items"}).get("items") or sub)
        ssep = [it for it in sub if not it["sep"]]
    on_target = False
    for _ in range(len(ssep) + 3):
        cur = (call({"op": "menu_items"}).get("current") or "")
        if cur and cur.startswith(want):
            on_target = True
            break
        pyautogui.press("down"); time.sleep(0.12)
    if not on_target:
        pyautogui.press("esc")
        return False, st, f"no {want} item"
    pyautogui.press("enter")
    st["menu_ms"] = (time.time() - t0) * 1000

    # 2) 等对话框出现（控件挂在主窗口树里；定位一律用名字/标签，不用绝对坐标）
    t1 = time.time()
    dlg_ok = False
    combo = fname = chk = None
    for _ in range(25):
        time.sleep(0.3)
        qg = acquire()
        if qg is None:
            continue
        combo = None
        for x in qg.descendants(control_type="ComboBox"):
            try:
                if (x.window_text() or "").strip() == "Format":
                    combo = x
            except Exception:
                pass
        if combo is None:
            continue
        # "File name" 标签同一行的 Edit
        lab = None
        for t2 in qg.descendants(control_type="Text"):
            try:
                if (t2.window_text() or "").strip() == "File name":
                    lab = t2
            except Exception:
                pass
        if lab is not None:
            lr = lab.rectangle()
            best = None
            for e in qg.descendants(control_type="Edit"):
                try:
                    rr = e.rectangle()
                    if abs(rr.top - lr.top) < 10 and rr.left > lr.left:
                        if best is None or rr.left < best[0]:
                            best = (rr.left, e)
                except Exception:
                    pass
            fname = best[1] if best else None
        for c in qg.descendants(control_type="CheckBox"):
            try:
                if (c.window_text() or "").strip() == "Save only selected features":
                    chk = c
            except Exception:
                pass
        if fname is not None:
            dlg_ok = True
            break
    if not dlg_ok:
        pyautogui.press("esc")
        return False, st, "dialog not found"

    # 3) 填对话框（每项都读回校验；格式选择必须真的变成 GeoJSON）
    error = ""
    ok_fmt = False
    for attempt in range(3):
        try:
            combo.select("GeoJSON")
        except Exception:
            try:
                combo.click_input(); time.sleep(0.6)
                for x in qg.descendants(control_type="ListItem"):
                    if (x.window_text() or "").strip() == "GeoJSON":
                        x.click_input(); break
                time.sleep(0.5)
            except Exception as e:
                error = f"format:{type(e).__name__}"
        time.sleep(0.4)
        try:
            cur = (combo.selected_text() or "").strip()
        except Exception:
            try:
                cur = (combo.window_text() or "").strip()
            except Exception:
                cur = ""
        if "GeoJSON" in cur:
            ok_fmt = True
            break
        error = f"format stayed {cur!r}"
    if not ok_fmt:
        return False, st, error or "format not set"

    ok_name = False
    for attempt in range(3):
        try:
            fname.set_edit_text(str(p))
        except Exception:
            try:
                fname.click_input(); pyautogui.hotkey("ctrl", "a")
                pyautogui.typewrite(str(p), 0.002)
            except Exception:
                pass
        time.sleep(0.35)
        try:
            cur = (fname.get_value() or "")
        except Exception:
            try:
                cur = fname.window_text() or ""
            except Exception:
                cur = ""
        if Path(cur.strip()).name == p.name:
            ok_name = True
            break
    if not ok_name:
        return False, st, f"filename not set ({cur!r})"

    if chk is not None:                     # 该复选框的状态会在对话框之间持久化 → 显式设定
        try:
            want = 1 if selected_only else 0
            for _ in range(3):
                if chk.get_toggle_state() == want:
                    break
                chk.click_input(); time.sleep(0.3)
        except Exception:
            pass
    for c in qg.descendants(control_type="CheckBox"):      # 别把导出文件加回图层
        try:
            if (c.window_text() or "").strip() == "Add saved file to map":
                if c.get_toggle_state() == 1:
                    c.click_input(); time.sleep(0.3)
        except Exception:
            pass

    # 4) OK：取按钮里最下面一行的最左者（不依赖绝对坐标，换窗口尺寸也成立）
    btns = []
    for b in qg.descendants(control_type="Button"):
        try:
            rr = b.rectangle()
            if rr.width() > 50 and rr.left > 400:
                btns.append((rr.top, rr.left, b))
        except Exception:
            pass
    if not btns:
        return False, st, "no OK button"
    top_max = max(t for t, _, _ in btns)
    row = sorted([x for x in btns if abs(x[0] - top_max) < 12], key=lambda z: z[1])
    okb = row[0][2]
    # W23：真鼠标点击会被"激活窗口"吃掉（W19 同族静默失败）→ UIA invoke 修低层（C6）
    ok_invoked = False
    try:
        okb.invoke()
        ok_invoked = True
    except Exception:
        try:
            okb.click_input()
        except Exception:
            pass
    st["dialog_ms"] = (time.time() - t1) * 1000

    # 5) 等文件落盘；没落盘就再触发一次 OK（最多 2 次）
    t2 = time.time()
    ok = False
    for attempt in range(2):
        for _ in range(40):
            time.sleep(0.25)
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    ok = isinstance(d, dict) and "features" in d
                    if ok:
                        break
                except Exception:
                    pass
        if ok:
            break
        if attempt == 0:                       # 二次触发：invoke → Enter 兜底
            try:
                okb.invoke()
            except Exception:
                pyautogui.press("enter")
            time.sleep(0.5)
    st["wait_ms"] = (time.time() - t2) * 1000
    st["total_ms"] = (time.time() - t_all) * 1000
    st["ok_invoked"] = ok_invoked
    if not ok:
        pyautogui.screenshot().save(str(HERE / f"_w16_fail_{'sel' if selected_only else 'all'}.png"))
        pyautogui.press("enter")
        return False, st, "file not written"
    return True, st, error


def features_of(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return {f["properties"]["name"] for f in d["features"]}
    except Exception:
        return set()


def main():
    call({"op": "win", "maximize": True})
    call({"op": "reset", "scene": SCENE})
    call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})   # 环境：选中 2 个大区
    time.sleep(0.4)
    print(f"环境：{SCENE} 已选中 {len(features_of('/dev/null') if False else [])}"
          f"期望选中 {N_SEL} / 全部 {N_ALL}")

    cases = [("sel.geojson", True, N_SEL), ("all.geojson", False, N_ALL)]
    out = {}
    for fname, sel_only, n_exp in cases:
        path = OUT / fname
        rows = []
        print(f"\n== {'Save Selected Features As…' if sel_only else 'Save Features As…'} → {fname} ==")
        for i in range(4 if sel_only else 3):
            # 每次前重置选择（环境）
            call({"op": "reset", "scene": SCENE})
            call({"op": "select_expr", "expr": SEL_EXPR, "layer": SCENE})
            time.sleep(0.25)
            ok, st, note = export_via_gui(path, sel_only)
            names = features_of(path)
            good = ok and len(names) == n_exp
            rows.append(dict(ok=bool(good), ms=st.get("total_ms"), menu_ms=st.get("menu_ms"),
                             dialog_ms=st.get("dialog_ms"), note=note, n=len(names)))
            print(f"  试 {i+1}: 成功={good} 要素={len(names)} 总耗时={st.get('total_ms',0):.0f} ms "
                  f"(菜单 {st.get('menu_ms',0):.0f} / 对话框 {st.get('dialog_ms',0):.0f} / 落盘 {st.get('wait_ms',0):.0f})"
                  + (f"  [{note}]" if note else ""))
        succ = [r for r in rows if r["ok"]]
        out[fname] = dict(trials=len(rows), success=len(succ),
                          ms_median=float(np.median([r["ms"] for r in succ])) if succ else None,
                          stages={k: float(np.median([r[k] for r in succ])) if succ else None
                                  for k in ("menu_ms", "dialog_ms")},
                          details=rows)
        print(f"  → {len(succ)}/{len(rows)} 成功"
              + (f"，成功中位 {out[fname]['ms_median']:.0f} ms" if succ else ""))

    (HERE / "w16_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print("\n已写出 experiments/qgis/qgis_bridge/w16_results.json")


if __name__ == "__main__":
    main()
