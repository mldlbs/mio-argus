"""
Bigemap 数据接入校验：扫描下载目录，用 QGIS 自带的 gdalinfo/ogrinfo 核对
CRS / 范围 / 分辨率 / 要素数，写出 ingest_manifest.json。

用法:
    python experiments/qgis/qgis_bridge/ingest_bigemap.py
    python experiments/qgis/qgis_bridge/ingest_bigemap.py --root artifacts/qgis_real/bigemap/sjs

期望 AOI（WGS84，外扩 200m 后）：
    W=116.2098  S=39.9048  E=116.2201  N=39.9118
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GDALINFO = Path(r"D:\QGIS\bin\gdalinfo.exe")
OGRINFO = Path(r"D:\QGIS\bin\ogrinfo.exe")

AOI = {"w": 116.2098, "s": 39.9048, "e": 116.2201, "n": 39.9118}
TOL_DEG = 0.002  # 约 200m，允许导出时再外扩

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def gdal_info(path):
    rc, out = run([str(GDALINFO), "-json", str(path)])
    if rc != 0:
        return {"error": out.strip()[:300]}
    d = json.loads(out)
    info = {
        "size": d.get("size"),
        "crs": ((d.get("coordinateSystem") or {}).get("wkt") or "")[:60],
        "geo_transform": d.get("geoTransform"),
        "corner": d.get("cornerCoordinates"),
        "bands": len(d.get("bands") or []),
        "dtype": (d.get("bands") or [{}])[0].get("type"),
    }
    gt = info["geo_transform"]
    if gt:
        info["pixel_size_deg"] = [abs(gt[1]), abs(gt[5])]
        info["pixel_size_m_approx"] = [round(abs(gt[1]) * 111320 * 0.766, 2),
                                       round(abs(gt[5]) * 110540, 2)]
    return info


def ogr_info(path):
    rc, out = run([str(OGRINFO), "-so", "-al", str(path)])
    if rc != 0:
        return {"error": out.strip()[:300]}
    feats = re.search(r"Feature Count:\s*(\d+)", out)
    crs = re.search(r"ID\[\"EPSG\",(\d+)\]", out)
    ext = re.findall(r"(-?\d+\.\d+),\s*(-?\d+\.\d+)", out)
    return {
        "feature_count": int(feats.group(1)) if feats else None,
        "epsg": int(crs.group(1)) if crs else None,
        "extent_pairs": ext[:4],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "artifacts" / "qgis_real" / "bigemap" / "sjs"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[err] 目录不存在：{root}")
        return 1

    raster = sorted(p for p in (root / "raster").rglob("*") if p.suffix.lower() in (".tif", ".tiff"))
    vector = sorted(p for p in (root / "vector").rglob("*") if p.suffix.lower() in (".shp", ".geojson", ".gpkg"))

    report = {"root": str(root), "aoi": AOI, "raster": [], "vector": [], "checks": []}
    for p in raster:
        info = gdal_info(p)
        info["file"] = str(p.relative_to(root))
        info["size_mb"] = round(p.stat().st_size / 1e6, 1)
        report["raster"].append(info)
        print(f"[raster] {p.name}: {info.get('size')} px {info.get('pixel_size_m_approx')} m/px "
              f"{info.get('size_mb')} MB")
    for p in vector:
        info = ogr_info(p)
        info["file"] = str(p.relative_to(root))
        info["size_kb"] = round(p.stat().st_size / 1e3, 1)
        report["vector"].append(info)
        print(f"[vector] {p.name}: {info.get('feature_count')} features EPSG:{info.get('epsg')}")

    checks = []
    if not raster and not vector:
        checks.append("FAIL: 没有找到任何 .tif / .shp / .geojson")
    if raster:
        checks.append("OK: raster present")
    if vector:
        checks.append("OK: vector present")
    report["checks"] = checks
    for c in checks:
        print("[check]", c)

    out = Path(args.out) if args.out else root / "ingest_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
