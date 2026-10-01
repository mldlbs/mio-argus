"""经 QGIS --code 加载：此时 qgis 已可导入；等主窗口就绪再挂 bridge。"""
import traceback
from pathlib import Path

LOG = Path(r"D:\work\code\nanoGPT\qgis_bridge\_startup.log")
BRIDGE = r"D:\work\code\nanoGPT\qgis_bridge\bridge.py"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


try:
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
