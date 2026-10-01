"""PYQGIS_STARTUP 引导：补 sys.path 兜底后，等主窗口就绪再挂 bridge。"""
import os
import sys
import traceback
from pathlib import Path

LOG = Path(r"D:\work\code\nanoGPT\qgis_bridge\_startup.log")
BRIDGE = r"D:\work\code\nanoGPT\qgis_bridge\bridge.py"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


try:
    try:
        from qgis.PyQt.QtCore import QTimer
    except ImportError:
        prefix = os.environ.get("QGIS_PREFIX_PATH", "D:/QGIS/apps/qgis-ltr")
        root = os.environ.get("OSGEO4W_ROOT", "D:/QGIS")
        for p in [prefix + "/python", prefix + "/python/plugins",
                  root + "/apps/Python312/Lib/site-packages",
                  root + "/apps/Python312/DLLs", root + "/bin", root + "/apps/qt5/bin"]:
            if p not in sys.path:
                sys.path.append(p)
        log("patched sys.path: " + " ; ".join(sys.path[-6:]))
        from qgis.PyQt.QtCore import QTimer

    def _load():
        try:
            src = open(BRIDGE, encoding="utf-8").read()
            exec(compile(src, BRIDGE, "exec"), {"__name__": "__bridge__"})
            log("bridge loaded")
        except Exception:
            log("bridge FAILED:\n" + traceback.format_exc())

    QTimer.singleShot(4000, _load)
    log("boot ok: timer scheduled")
except Exception:
    log("boot FAILED:\n" + traceback.format_exc())
