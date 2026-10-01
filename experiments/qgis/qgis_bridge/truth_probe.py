"""
QGIS 真值通道冒烟测试（headless PyQGIS）

目的：证明"任务改状态 → PyQGIS 读状态"这条**外部真值**链可用。
内容：程序化造一份 GeoPackage（点图层 + 属性），落盘，再读回并输出 truth JSON。
不依赖任何外部数据下载。

运行（用 QGIS 自带的 Python）：
    D:\\QGIS\\bin\\python-qgis-ltr.bat qgis_bridge\\truth_probe.py
"""
import json
import os
from pathlib import Path

from qgis.core import (QgsApplication, QgsFeature, QgsField, QgsGeometry,
                       QgsPointXY, QgsVectorLayer, QgsProject,
                       QgsCoordinateReferenceSystem, QgsVectorFileWriter,
                       QgsCoordinateTransformContext, QgsWkbTypes)
from qgis.PyQt.QtCore import QVariant

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
GPKG = DATA / "sample.gpkg"


def build_layer():
    lyr = QgsVectorLayer("Point?crs=EPSG:4326", "cities", "memory")
    pr = lyr.dataProvider()
    pr.addAttributes([QgsField("name", QVariant.String),
                      QgsField("pop", QVariant.Int),
                      QgsField("capital", QVariant.Int)])
    lyr.updateFields()
    feats = []
    rows = [("Zurich", 421878, 0), ("London", 8982000, 1), ("Beijing", 21540000, 1),
            ("Bern", 134000, 1), ("Geneva", 201818, 0)]
    for i, (n, p, c) in enumerate(rows):
        f = QgsFeature(lyr.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(8.54 + i * 4.0, 47.37 - i * 3.0)))
        f["name"], f["pop"], f["capital"] = n, p, c
        feats.append(f)
    pr.addFeatures(feats)
    lyr.updateExtents()
    return lyr


def write_gpkg(lyr):
    if GPKG.exists():
        GPKG.unlink()
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "GPKG"
    opts.layerName = "cities"
    err, msg, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
        lyr, str(GPKG), QgsCoordinateTransformContext(), opts)
    return err == QgsVectorFileWriter.NoError, msg


def truth(project):
    out = {"layers": [], "file_exists": GPKG.exists(),
           "file_bytes": GPKG.stat().st_size if GPKG.exists() else 0}
    for lyr in project.mapLayers().values():
        out["layers"].append({
            "name": lyr.name(),
            "type": QgsWkbTypes.displayString(lyr.wkbType()),
            "crs": lyr.crs().authid(),
            "features": lyr.featureCount(),
            "fields": [f.name() for f in lyr.fields()],
        })
    return out


def main():
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", "D:/QGIS/apps/qgis-ltr"), True)
    qgs = QgsApplication([], False)          # False = 无 GUI
    qgs.initQgis()
    try:
        project = QgsProject.instance()
        lyr = build_layer()
        project.addMapLayer(lyr)
        ok, msg = write_gpkg(lyr)
        print(json.dumps({
            "write_ok": ok, "write_msg": msg,
            "truth": truth(project),
        }, indent=2, ensure_ascii=False))
    finally:
        qgs.exitQgis()


if __name__ == "__main__":
    main()
