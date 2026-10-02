"""
QGIS 桥（在 QGIS 进程内运行）

提供四条通道：
  truth()      —— QgsProject 状态（图层/CRS/要素/选中/文件）＝ 外部真值
  structure()  —— 特权结构通道：QWidget 树 + QAction 列表（名字/角色/可见/几何）
  act(...)     —— 特权动作：触发 QAction / 点击控件 / 填文本 / 选下拉
  shot(path)   —— 像素通道：主窗口自身渲染（不含桌面其它窗口）

跑法（由 launcher 注入）：
    qgis-ltr-bin.exe --code "exec(open(r'<path>/bridge.py').read())"
端口：127.0.0.1:8765（可用 QGIS_BRIDGE_PORT 覆盖）
"""
import json
import os
import queue
import socket
import threading
import time
import traceback
from pathlib import Path

from qgis.core import Qgis, QgsProject, QgsWkbTypes
from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtGui import QAction
from qgis.PyQt.QtWidgets import (QApplication, QWidget, QAbstractButton,
                                 QLineEdit, QComboBox, QCheckBox)

PORT = int(os.environ.get("QGIS_BRIDGE_PORT", "8765"))
REQ = queue.Queue()


def _geom(w):
    try:
        r = w.geometry()
        p = w.mapToGlobal(r.topLeft())
        return [int(p.x()), int(p.y()), int(r.width()), int(r.height())]
    except Exception:
        return [0, 0, 0, 0]


def _wpath(w):
    parts = []
    node = w
    for _ in range(6):
        if node is None:
            break
        name = node.objectName() or node.metaObject().className()
        parts.append(name)
        node = node.parentWidget()
    return "/".join(reversed(parts))


def _wtext(w):
    for getter in ("text", "currentText", "accessibleName", "windowTitle", "placeholderText"):
        try:
            v = getattr(w, getter)()
            if v:
                return str(v)[:80]
        except Exception:
            pass
    return ""


def structure():
    widgets, actions = [], []
    root = QApplication.instance()
    for w in root.allWidgets():
        try:
            vis = w.isVisible()
        except Exception:
            vis = False
        if not vis:
            continue
        widgets.append({
            "class": w.metaObject().className(),
            "objectName": w.objectName(),
            "text": _wtext(w),
            "path": _wpath(w),
            "enabled": bool(w.isEnabled()),
            "rect": _geom(w),
        })
    win = None
    for w in root.topLevelWidgets():
        if w.isVisible() and w.metaObject().className() in ("QgsMainWindow", "QMainWindow"):
            win = w
            break
    if win is None:
        win = _main_window()
    if win is not None:
        from qgis.PyQt.QtWidgets import QToolBar
        for tb in win.findChildren(QToolBar):
            if not tb.isVisible():
                continue
            for a in tb.actions():
                w = tb.widgetForAction(a)
                if w is None or not w.isVisible() or not a.text():
                    continue
                actions.append({
                    "text": a.text().replace("&", ""),
                    "enabled": bool(a.isEnabled()),
                    "checkable": bool(a.isCheckable()),
                    "checked": bool(a.isChecked()) if a.isCheckable() else None,
                    "shortcut": a.shortcut().toString(),
                    "rect": _geom(w),
                    "toolbar": tb.windowTitle().replace("&", ""),
                })
    return {"widgets": widgets, "actions": actions}


def _bridge_dir():
    f = globals().get("__file__")
    if f:
        return Path(f).resolve().parent
    return Path(r"D:\work\code\nanoGPT\experiments\qgis\qgis_bridge")


DATA_GPKG = os.environ.get(
    "QGIS_BRIDGE_GPKG", str(_bridge_dir() / "data" / "sample.gpkg"))


def reset_env(scene="cities"):
    """环境重置（非 agent 动作）：清空项目、加场景图层、取消选择、缩放到图层范围。"""
    from qgis.core import (QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
                           QgsField, QgsRectangle)
    from qgis.PyQt.QtCore import QVariant
    project = QgsProject.instance()
    project.clear()
    if scene == "regions":
        vl = QgsVectorLayer("Polygon?crs=EPSG:4326", "regions", "memory")
        dp = vl.dataProvider()
        dp.addAttributes([QgsField("name", QVariant.String),
                          QgsField("area", QVariant.Double)])
        vl.updateFields()
        rects = [("North", 6, 44, 14, 50, 500.0), ("South", 12, 36, 20, 44, 400.0),
                 ("Central", 6, 38, 12, 44, 300.0), ("West", 2, 40, 6, 44, 150.0),
                 ("East", 20, 38, 25, 42, 120.0)]
        for name, x0, y0, x1, y1, area in rects:
            f = QgsFeature(vl.fields())
            f.setGeometry(QgsGeometry.fromPolygonXY([[
                QgsPointXY(x0, y0), QgsPointXY(x1, y0),
                QgsPointXY(x1, y1), QgsPointXY(x0, y1)]]))
            f["name"] = name
            f["area"] = area
            dp.addFeature(f)
        vl.updateExtents()
        lyr = vl
    elif scene == "scenario":
        # W23 场景集：投影坐标（米，EPSG:3857）的 regions + data/scenario 下 4 个 GeoJSON
        vl = QgsVectorLayer("Polygon?crs=EPSG:3857", "regions", "memory")
        dp = vl.dataProvider()
        dp.addAttributes([QgsField("name", QVariant.String),
                          QgsField("area", QVariant.Double)])
        vl.updateFields()
        rects = [("North", 600000, 4400000, 1400000, 5000000, 500.0),
                 ("South", 1200000, 3600000, 2000000, 4400000, 400.0),
                 ("Central", 600000, 3800000, 1200000, 4400000, 300.0),
                 ("West", 200000, 4000000, 600000, 4400000, 150.0),
                 ("East", 2000000, 3800000, 2500000, 4200000, 120.0)]
        for name, x0, y0, x1, y1, area in rects:
            f = QgsFeature(vl.fields())
            f.setGeometry(QgsGeometry.fromPolygonXY([[
                QgsPointXY(x0, y0), QgsPointXY(x1, y0),
                QgsPointXY(x1, y1), QgsPointXY(x0, y1)]]))
            f["name"] = name
            f["area"] = area
            dp.addFeature(f)
        vl.updateExtents()
        layers = [vl]
        scn = _bridge_dir() / "data" / "scenario"
        for fn in ("roads", "hospitals", "pois", "floods"):
            p = scn / f"{fn}.geojson"
            lyr2 = QgsVectorLayer(str(p), fn, "ogr")
            if not lyr2.isValid():
                return {"ok": False, "err": f"layer invalid: {p}"}
            layers.append(lyr2)
        for l in layers:
            project.addMapLayer(l)
            l.removeSelection()
        c = _canvas()
        if c is not None:
            xs0, ys0, xs1, ys1 = [], [], [], []
            for l in layers:
                e = l.extent()
                xs0.append(e.xMinimum()); ys0.append(e.yMinimum())
                xs1.append(e.xMaximum()); ys1.append(e.yMaximum())
            from qgis.core import QgsRectangle
            c.setExtent(QgsRectangle(min(xs0), min(ys0), max(xs1), max(ys1)))
            c.refresh()
        return {"ok": True, "scene": scene,
                "features": {l.name(): l.featureCount() for l in layers}}
    elif scene == "real":
        # W24 真实数据场景：Natural Earth 中国 ROI（fetch_real_data.py 产物 → artifacts/qgis_real）
        real_dir = Path(os.environ.get(
            "QGIS_REAL_DIR",
            str(_bridge_dir().parents[2] / "artifacts" / "qgis_real")))
        layers = []
        for fn in ("provinces", "airports", "cities", "ports", "roads"):
            p = real_dir / f"{fn}.geojson"
            if not p.exists():
                return {"ok": False,
                        "err": f"missing {p}（先跑 fetch_real_data.py）"}
            lyr2 = QgsVectorLayer(str(p), fn, "ogr")
            if not lyr2.isValid():
                return {"ok": False, "err": f"layer invalid: {p}"}
            layers.append(lyr2)
        for l in layers:
            project.addMapLayer(l)
            l.removeSelection()
        c = _canvas()
        if c is not None:
            xs0, ys0, xs1, ys1 = [], [], [], []
            for l in layers:
                e = l.extent()
                xs0.append(e.xMinimum()); ys0.append(e.yMinimum())
                xs1.append(e.xMaximum()); ys1.append(e.yMaximum())
            from qgis.core import QgsRectangle
            c.setExtent(QgsRectangle(min(xs0), min(ys0), max(xs1), max(ys1)))
            c.refresh()
        return {"ok": True, "scene": scene,
                "features": {l.name(): l.featureCount() for l in layers}}
    else:
        lyr = QgsVectorLayer(DATA_GPKG + "|layername=cities", "cities", "ogr")
        if not lyr.isValid():
            return {"ok": False, "err": f"layer invalid: {DATA_GPKG}"}
    project.addMapLayer(lyr)
    lyr.removeSelection()
    c = _canvas()
    if c is not None:
        c.setExtent(lyr.extent())
        c.refresh()
    return {"ok": True, "scene": scene, "features": lyr.featureCount()}


def truth():
    project = QgsProject.instance()
    tree = project.layerTreeRoot()
    out = {"crs": project.crs().authid(), "layers": []}
    for lyr in project.mapLayers().values():
        try:
            sel = lyr.selectedFeatureCount()
        except Exception:
            sel = None
        item = tree.findLayer(lyr.id())
        ext = lyr.extent()
        # S1 symbology 真值：渲染器类型 + 分类字段 + 类目（W23）
        rend_info = None
        try:
            rend = lyr.renderer()
            if rend is None:
                rend_info = {"type": None}
            else:
                rend_info = {"type": rend.type()}
                if rend.type() == "categorizedSymbol":
                    try:
                        rend_info["attr"] = rend.classAttribute()
                    except Exception as e:
                        rend_info["attr_err"] = str(e)
                    try:
                        cats = []
                        for c in rend.categories():
                            try:
                                cats.append(c.value())
                            except Exception as e:
                                cats.append(f"<err {e}>")
                        rend_info["cats"] = sorted(cats, key=str)
                    except Exception as e:
                        rend_info["cats_err"] = str(e)
        except Exception as e:
            rend_info = {"type": "error", "msg": str(e)}
        out["layers"].append({
            "name": lyr.name(),
            "crs": lyr.crs().authid(),
            "features": lyr.featureCount(),
            "selected": sel,
            "visible": bool(item.itemVisibilityChecked()) if item else None,
            "renderer": rend_info,
            "extent": [float(ext.xMinimum()), float(ext.yMinimum()),
                       float(ext.xMaximum()), float(ext.yMaximum())],
        })
    try:
        canvas = _canvas()
        if canvas is not None:
            out["scale"] = float(canvas.mapSettings().scale())
            e = canvas.mapSettings().visibleExtent()
            out["extent"] = [float(e.xMinimum()), float(e.yMinimum()),
                             float(e.xMaximum()), float(e.yMaximum())]
            mt = canvas.mapTool()
            if mt is not None:
                try:
                    out["tool"] = mt.action().text().replace("&", "")
                except Exception:
                    out["tool"] = mt.metaObject().className()
    except Exception:
        pass
    out["console_visible"] = any(
        w.objectName() == "PythonConsole" and w.isVisible()
        for w in QApplication.instance().allWidgets())
    return out


def _main_window():
    """优先按标题含 QGIS 的可见顶级窗口；否则取面积最大者。"""
    best, area = None, -1
    for w in QApplication.instance().topLevelWidgets():
        try:
            if not w.isVisible():
                continue
            if "QGIS" in (w.windowTitle() or ""):
                return w
            g = w.geometry()
            a = g.width() * g.height()
            if a > area:
                best, area = w, a
        except Exception:
            pass
    return best


def windows():
    out = []
    for w in QApplication.instance().topLevelWidgets():
        try:
            g = w.geometry()
            out.append({"class": w.metaObject().className(),
                        "title": (w.windowTitle() or "")[:60],
                        "visible": bool(w.isVisible()),
                        "rect": [int(g.x()), int(g.y()), int(g.width()), int(g.height())]})
        except Exception:
            pass
    return out


def layer_tree_click(name, dx=6, req_mode="space"):
    """切换图层可见性：默认用**真键盘**（选中行 + Space）；req_mode='mouse' 用鼠标。"""
    win = _main_window()
    if win is None:
        return {"ok": False, "err": "no main window"}
    try:
        from qgis.gui import QgsLayerTreeView
        from qgis.PyQt.QtCore import Qt, QPoint, QModelIndex
        from qgis.PyQt.QtTest import QTest
    except Exception as e:
        return {"ok": False, "err": f"import: {e}"}
    views = win.findChildren(QgsLayerTreeView)
    if not views:
        return {"ok": False, "err": "no layer tree view"}
    view = views[0]
    model = view.model()
    vp = view.viewport()

    def row_of(name, probe_x=24):
        for y in range(0, vp.height(), 2):
            idx = view.indexAt(QPoint(probe_x, y))
            if not idx.isValid():
                continue
            txt = str(model.data(idx, Qt.DisplayRole) or "")
            if txt == name or txt.startswith(name):
                return y
        return -1

    y = row_of(name)
    if y < 0:
        return {"ok": False, "err": f"layer row not found: {name}",
                "vp": [vp.width(), vp.height()], "rows": model.rowCount(QModelIndex())}
    idx = view.indexAt(QPoint(24, y))
    view.setCurrentIndex(idx)                 # 特权：选中该行
    if req_mode == "mouse":
        pos = QPoint(int(dx), y)
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, pos)
        return {"ok": True, "mode": "mouse", "pos": [int(pos.x()), int(pos.y())]}
    QTest.keyClick(view, Qt.Key_Space)        # 真键盘：空格切换可见性
    return {"ok": True, "mode": "space", "row_y": y}


def activate_layer(name):
    """把某图层设为图层树里的当前图层（特权，之后菜单动作才会启用）。"""
    win = _main_window()
    if win is None:
        return {"ok": False, "err": "no main window"}
    from qgis.gui import QgsLayerTreeView
    from qgis.PyQt.QtCore import Qt, QPoint
    views = win.findChildren(QgsLayerTreeView)
    if not views:
        return {"ok": False, "err": "no layer tree view"}
    view = views[0]
    model = view.model()
    vp = view.viewport()
    for y in range(0, vp.height(), 2):
        idx = view.indexAt(QPoint(24, y))
        if idx.isValid() and str(model.data(idx, Qt.DisplayRole) or "").startswith(name):
            view.setCurrentIndex(idx)
            return {"ok": True, "row_y": y}
    return {"ok": False, "err": f"row not found: {name}"}


def find_actions(substr):
    out = []
    for a in _all_actions():
        t = a.text().replace("&", "")
        if substr.lower() in t.lower():
            out.append({"text": t, "enabled": bool(a.isEnabled())})
    # 去重
    seen, uniq = set(), []
    for d in out:
        if d["text"] in seen:
            continue
        seen.add(d["text"])
        uniq.append(d)
    return {"ok": True, "actions": uniq}


def close_dialogs():
    """关掉所有可见的对话框（privileged 收尾）。"""
    from qgis.PyQt.QtWidgets import QDialog
    n = 0
    for w in QApplication.instance().topLevelWidgets():
        if isinstance(w, QDialog) and w.isVisible():
            w.close()
            n += 1
    QApplication.processEvents()
    return {"ok": True, "closed": n}


def menu_items():
    """读当前打开的弹出菜单（QMenu）的动作序列 + 当前高亮项 —— 结构通道。
    W23：UIA 看不见 Qt 弹出菜单；键盘导航需按 currentAction 反馈走（Home/环绕都不可靠）。"""
    m = QApplication.instance().activePopupWidget()
    menus = []
    if m is not None and m.metaObject().className() in ("QMenu", "QWidgetMenu"):
        menus = [m]
    if not menus:
        for w in QApplication.instance().topLevelWidgets():
            if (w.metaObject().className() in ("QMenu", "QWidgetMenu")
                    and w.isVisible()):
                menus.append(w)
    for m in menus:
        items = []
        for a in m.actions():
            items.append({"text": a.text().replace("&", "").strip(),
                          "sep": bool(a.isSeparator()),
                          "enabled": bool(a.isEnabled()),
                          "submenu": bool(a.menu())})
        if items:
            cur = None
            try:
                a = m.activeAction()        # QMenu API：activeAction()（currentAction 不存在）
                if a is not None:
                    cur = "<sep>" if a.isSeparator() else \
                        a.text().replace("&", "").strip()
            except Exception:
                cur = None
            return {"ok": True, "items": items, "current": cur}
    return {"ok": True, "items": [], "current": None}


def _find_widget(cls=None, text=None, index=0):
    cand = []
    for w in QApplication.instance().allWidgets():
        if not w.isVisible():
            continue
        if cls and w.metaObject().className() != cls:
            continue
        if text is not None and _wtext(w) != text:
            continue
        cand.append(w)
    return cand[min(index, len(cand) - 1)] if cand else None


def _act_by_text(text):
    for x in _all_actions():
        if x.text().replace("&", "") == text and x.isEnabled() and not x.isSeparator():
            x.trigger()
            return True
    return False


def select_expr(expr, layer="cities"):
    """S1：按表达式选择（GUI：菜单动作 + 对话框 + 写表达式 + 点按钮）。"""
    if not activate_layer(layer).get("ok"):
        return {"ok": False, "err": "activate_layer failed"}
    if not _act_by_text("Select Features by Expression…"):
        return {"ok": False, "err": "action missing"}
    for _ in range(40):
        QApplication.processEvents()
        ed = _find_widget("QgsCodeEditorExpression")
        btn = _find_widget("QToolButton", "Select Features")
        if ed is not None and btn is not None:
            ed.setText(expr)
            QApplication.processEvents()
            btn.click()
            QApplication.processEvents()
            close_dialogs()
            return {"ok": True, "expr": expr}
        time.sleep(0.1)
    return {"ok": False, "err": "expression dialog not found"}


def zoom_to_selection():
    """S2：按选择缩放（GUI 动作）。"""
    if _act_by_text("Zoom to Selection"):
        QApplication.processEvents()
        return {"ok": True}
    return {"ok": False, "err": "no Zoom to Selection"}


def export_layer(path, only_selected=True, layer="cities"):
    """S3：导出图层（程序化；GUI 的另存对话框是模态且动作名歧义，记为局限）。"""
    lyr = None
    for l in QgsProject.instance().mapLayers().values():
        if l.name() == layer:
            lyr = l
    if lyr is None:
        return {"ok": False, "err": f"no {layer} layer"}
    from qgis.core import QgsVectorFileWriter, QgsCoordinateTransformContext
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "GeoJSON"
    opts.layerName = layer
    opts.onlySelectedFeatures = bool(only_selected)
    p = Path(path)
    if p.exists():
        p.unlink()
    err, msg, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        lyr, str(p), QgsCoordinateTransformContext(), opts)
    return {"ok": err == QgsVectorFileWriter.NoError, "err": msg, "path": str(p),
            "only_selected": bool(only_selected)}


def _canvas():
    win = _main_window()
    if win is None:
        return None
    try:
        from qgis.gui import QgsMapCanvas
        from qgis.PyQt.QtCore import QPoint
        cands = [c for c in win.findChildren(QgsMapCanvas)
                 if c.isVisible() and c.width() > 50 and c.height() > 50]
        if cands:
            best = max(cands, key=lambda c: c.width() * c.height())
            return best
        allc = win.findChildren(QgsMapCanvas)
        if allc:
            return max(allc, key=lambda c: c.width() * c.height())
    except Exception:
        pass
    for w in QApplication.instance().allWidgets():
        if "MapCanvas" in w.metaObject().className() and w.isVisible():
            return w
    return None


def canvas_rect(c):
    from qgis.PyQt.QtCore import QPoint
    p = c.mapToGlobal(QPoint(0, 0))
    return [int(p.x()), int(p.y()), int(c.width()), int(c.height())]


def _all_actions():
    acts = []
    root = QApplication.instance()
    for w in root.topLevelWidgets():
        acts += w.findChildren(QAction)
    return acts


def act(req):
    kind = req.get("kind")
    if kind == "action":
        want = (req.get("text") or "").replace("&", "")
        for a in _all_actions():
            if a.text().replace("&", "") == want and a.isEnabled() and not a.isSeparator():
                # 关键：异步触发。模态对话框（exec）会阻塞主线程，
                # 同步 trigger 会让泵卡死；singleShot 让我们先返回响应。
                QTimer.singleShot(0, a.trigger)
                return {"ok": True, "did": f"trigger_async:{want}"}
        return {"ok": False, "err": f"action not found/enabled: {want}"}
    # 控件类操作
    spec = req.get("widget") or {}
    cand = []
    for w in QApplication.instance().allWidgets():
        if not w.isVisible():
            continue
        if spec.get("class") and w.metaObject().className() != spec["class"]:
            continue
        if spec.get("objectName") and w.objectName() != spec["objectName"]:
            continue
        if spec.get("text") is not None and _wtext(w) != spec["text"]:
            continue
        cand.append(w)
    if not cand:
        return {"ok": False, "err": f"widget not found: {spec}"}
    w = cand[min(spec.get("index", 0), len(cand) - 1)]
    if kind == "click":
        if isinstance(w, QAbstractButton):
            w.click()
            return {"ok": True, "did": f"click:{_wpath(w)}"}
        return {"ok": False, "err": f"not clickable: {w.metaObject().className()}"}
    if kind == "text":
        val = str(req.get("value", ""))
        for meth in ("setPlainText", "setText", "setValue"):
            if hasattr(w, meth):
                try:
                    getattr(w, meth)(val)
                    return {"ok": True, "did": f"{meth}:{_wpath(w)}"}
                except Exception as e:
                    return {"ok": False, "err": f"{meth}: {e}"}
        return {"ok": False, "err": "not a text widget"}
    if kind == "check":
        if isinstance(w, QCheckBox):
            w.setChecked(bool(req.get("value", True)))
            return {"ok": True, "did": "check"}
        return {"ok": False, "err": "not a QCheckBox"}
    if kind == "combo":
        if isinstance(w, QComboBox):
            w.setCurrentIndex(int(req.get("index", 0)))
            return {"ok": True, "did": "combo"}
        return {"ok": False, "err": "not a QComboBox"}
    return {"ok": False, "err": f"unknown act kind: {kind}"}


def shot(req):
    path = req.get("path") or str(Path(os.environ.get("TEMP", ".")) / "qgis_shot.png")
    win = _main_window()
    if win is None:
        return {"ok": False, "err": "no main window"}
    pm = win.grab()
    ok = pm.save(path, "PNG")
    return {"ok": bool(ok), "path": path, "size": [pm.width(), pm.height()]}


def _handle(req):
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "qgis": Qgis.QGIS_VERSION, "pid": os.getpid()}
    if op == "truth":
        return {"ok": True, "truth": truth()}
    if op == "structure":
        return {"ok": True, **structure()}
    if op == "act":
        return act(req)
    if op == "shot":
        return shot(req)
    if op == "reset":
        return {"ok": True, **reset_env(req.get("scene", "cities"))}
    if op == "layer_tree_click":
        return layer_tree_click(req.get("name"), req.get("dx", 6), req.get("mode", "space"))
    if op == "activate_layer":
        return activate_layer(req.get("name"))
    if op == "find_actions":
        return find_actions(req.get("substr", ""))
    if op == "shortcuts":
        out = {}
        for a in _all_actions():
            try:
                t = a.text().replace("&", "").strip()
                s = a.shortcut().toString() if hasattr(a, "shortcut") else ""
            except Exception:
                continue
            if t and s:
                out.setdefault(t, s)
        return {"ok": True, "shortcuts": out}
    if op == "close_dialogs":
        return close_dialogs()
    if op == "menu_items":
        return menu_items()
    if op == "select_expr":
        return select_expr(req.get("expr", ""), req.get("layer", "cities"))
    if op == "zoom_to_selection":
        return zoom_to_selection()
    if op == "export":
        return export_layer(req.get("path", ""), req.get("only_selected", True),
                            req.get("layer", "cities"))
    if op == "set_extent":
        c = _canvas()
        if c is None:
            return {"ok": False, "err": "no canvas"}
        from qgis.core import QgsRectangle
        b = req["box"]
        c.setExtent(QgsRectangle(b[0], b[1], b[2], b[3]))
        c.refresh()
        return {"ok": True}
    if op == "canvas":
        c = _canvas()
        if c is None:
            return {"ok": False, "err": "no map canvas"}
        ms = c.mapSettings()
        e = ms.visibleExtent()
        return {"ok": True, "class": c.metaObject().className(),
                "rect": canvas_rect(c),
                "scale": float(ms.scale()),
                "extent": [float(e.xMinimum()), float(e.yMinimum()),
                           float(e.xMaximum()), float(e.yMaximum())]}
    if op == "layouts":
        from qgis.core import (QgsProject, QgsLayoutItemLabel,
                               QgsLayoutItemMap)
        out = []
        for l in QgsProject.instance().layoutManager().layouts():
            items = []
            for it in l.items():
                d = {"type": type(it).__name__}
                try:
                    if isinstance(it, QgsLayoutItemLabel):
                        d["text"] = it.text()
                    if isinstance(it, QgsLayoutItemMap):
                        e = it.extent()
                        d["extent"] = [float(e.xMinimum()), float(e.yMinimum()),
                                       float(e.xMaximum()), float(e.yMaximum())]
                except Exception:
                    pass
                items.append(d)
            out.append({"name": l.name(), "items": items})
        return {"ok": True, "layouts": out}
    if op == "windows":
        return {"ok": True, "windows": windows()}
    if op == "win":
        win = _main_window()
        if win is None:
            return {"ok": False, "err": "no main window"}
        dirty = False
        if req.get("maximize") and not win.isMaximized():
            win.showNormal()
            win.showMaximized()
            dirty = True
        if req.get("w") and req.get("h"):
            if win.isMaximized():
                win.showNormal()
            win.resize(int(req["w"]), int(req["h"]))
            dirty = True
        if dirty:
            QApplication.processEvents()
            time.sleep(0.15)
            QApplication.processEvents()
        win.raise_()
        win.activateWindow()
        QApplication.processEvents()
        g = win.geometry()
        return {"ok": True, "rect": [int(g.x()), int(g.y()), int(g.width()), int(g.height())],
                "maximized": bool(win.isMaximized())}
    return {"ok": False, "err": f"unknown op: {op}"}


def _server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(4)
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_serve_one, args=(conn,), daemon=True).start()
        except Exception:
            traceback.print_exc()


def _serve_one(conn):
    with conn:
        f = conn.makefile("rwb")
        for line in f:
            try:
                req = json.loads(line.decode("utf-8"))
            except Exception:
                break
            box = {"done": threading.Event(), "resp": None}
            REQ.put((req, box))
            box["done"].wait(30)
            resp = box["resp"] or {"ok": False, "err": "timeout"}
            f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()


def _pump():
    """Qt 主线程：把队列里的请求执行掉。"""
    while not REQ.empty():
        req, box = REQ.get()
        try:
            box["resp"] = _handle(req)
        except Exception as e:
            box["resp"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
        box["done"].set()


_timer = QTimer()
_timer.timeout.connect(_pump)
_timer.start(30)
threading.Thread(target=_server, daemon=True).start()
print(f"[qgis-bridge] listening on 127.0.0.1:{PORT} pid={os.getpid()}")
