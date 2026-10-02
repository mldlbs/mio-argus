"""
QGIS W24：真实地理数据获取与切片（Natural Earth → 中国 ROI，EPSG:3857）

W23 的场景数据是程序化生成的"纸面几何"；本轮换**真实地理数据**：
Natural Earth（公有领域）v5.1.2 静态快照 → 中国国界裁剪 → Web Mercator 投影
（与 W23 同 CRS，米制缓冲语义不变）。

真值：取数时用 shapely 一次算好 —— 与 QGIS 同为 GEOS 引擎，buffer 取
quad_segs=5 对齐 QGIS Processing 的 SEGMENTS=5 默认值。

产物（artifacts/qgis_real/，gitignored；本脚本入库可复现）：
  raw/               原始 GeoJSON（缓存 + sha256）
  china.geojson      中国国界（EPSG:4326，裁剪源）
  provinces.geojson  省级行政区（真实边界；clipsrc 中国，含争议区共 40+）
  cities.geojson     populated places（真实城市点，含 pop_max；bbox 裁剪）
  airports.geojson   机场（真实；clipsrc 中国）
  ports.geojson      港口（真实；bbox 裁剪 → 含境外点，r6 有区分度）
  roads.geojson      道路（真实，含 type 等级；clipsrc 中国）
  real_truth.json    6 场景真值
  meta.json          快照元数据（日期/tag/URLs/sha256/计数）

用法: python experiments/qgis/qgis_bridge/fetch_real_data.py [--force]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[2]                      # nanoGPT 仓库根
OUT = ROOT / "artifacts" / "qgis_real"
RAW = OUT / "raw"

NE_TAG = "v5.1.2"
BASE = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{NE_TAG}/geojson"
SRC = {
    "countries": "ne_50m_admin_0_countries.geojson",
    "provinces": "ne_50m_admin_1_states_provinces.geojson",
    "cities": "ne_10m_populated_places.geojson",
    "airports": "ne_10m_airports.geojson",
    "ports": "ne_10m_ports.geojson",
    "roads": "ne_10m_roads.geojson",
}
BUFFER_M = 300000                            # r3/r4 缓冲半径（米）
BBOX = (72.0, 14.0, 137.0, 55.0)             # 城市/港口取数 bbox（lon/lat, EPSG:4326）


def find_ogr2ogr():
    env = __import__("os").environ.get("QGIS_OGR2OGR")
    for cand in (env, shutil.which("ogr2ogr"), r"D:\QGIS\bin\ogr2ogr.exe"):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("找不到 ogr2ogr（设 QGIS_OGR2OGR 或装 QGIS）")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _openers():
    """直连 + 代理回退（本机实测 raw.githubusercontent 直连会被重置，7890 代理可达）。"""
    import os
    urls = [None, os.environ.get("HTTPS_PROXY"), "http://127.0.0.1:7890"]
    out = []
    for ph in urls:
        if ph:
            out.append(urllib.request.build_opener(
                urllib.request.ProxyHandler({"https": ph, "http": ph})))
        else:
            out.append(urllib.request.build_opener())
    return out


def download(name, force=False):
    dst = RAW / SRC[name]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"  [cache] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return dst
    url = f"{BASE}/{SRC[name]}"
    for attempt in range(3):
        for op in _openers():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "qgis-bridge-w24"})
                t0 = time.time()
                with op.open(req, timeout=180) as r, \
                        open(dst.with_suffix(".tmp"), "wb") as f:
                    shutil.copyfileobj(r, f)
                dst.with_suffix(".tmp").replace(dst)
                print(f"  [get] {dst.name} ({dst.stat().st_size / 1e6:.1f} MB, "
                      f"{time.time() - t0:.0f}s)")
                return dst
            except Exception as e:
                last = e
        print(f"  [retry {attempt + 1}] {SRC[name]}: {last}")
        time.sleep(2)
    raise RuntimeError(f"下载失败: {url}: {last}")


def ogr(args):
    r = subprocess.run([find_ogr2ogr(), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ogr2ogr 失败: {' '.join(args)}\n{r.stderr[-800:]}")


def feature_names(path, field="name"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in d["features"]:
        v = f["properties"].get(field)
        if v is not None:
            out.append(v)
    return out


def normalize_props(path):
    """NE 各图层属性大小写不一（cities 全大写、roads 小写）→ 统一小写键，
    与 W23 机械（features_of 读 "name"、表达式读 "class"/"pop_max"）对齐。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for f in d["features"]:
        f["properties"] = {(k.lower() if isinstance(k, str) else k): v
                           for k, v in (f["properties"] or {}).items()}
    Path(path).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ fetch ---
def fetch(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== 下载 Natural Earth {NE_TAG}（→ {RAW}）==")
    paths = {n: download(n, force) for n in SRC}

    print("== 裁剪中国国界 ==")
    ogr(["-f", "GeoJSON", "-lco", "RFC7946=NO",
         str(OUT / "china.geojson"), str(paths["countries"]),
         "-where", "ADMIN='China'"])
    # provinces/airports/roads：裁到中国国界（语义：中国省份/机场/道路）
    for name in ("provinces", "airports", "roads"):
        ogr(["-f", "GeoJSON", "-lco", "RFC7946=NO",
             "-t_srs", "EPSG:3857",
             "-clipsrc", str(OUT / "china.geojson"),
             str(OUT / f"{name}.geojson"), str(paths[name])])
        n = len(json.loads((OUT / f"{name}.geojson").read_text(encoding="utf-8"))["features"])
        print(f"  {name}.geojson  {n} 要素（clipsrc china）")
    # cities/ports：裁到 bbox（区域含境外点 → r5/r6 空间查询有真实区分度）
    for name in ("cities", "ports"):
        ogr(["-f", "GeoJSON", "-lco", "RFC7946=NO",
             "-t_srs", "EPSG:3857",
             "-spat", *[str(v) for v in BBOX],
             str(OUT / f"{name}.geojson"), str(paths[name])])
        n = len(json.loads((OUT / f"{name}.geojson").read_text(encoding="utf-8"))["features"])
        print(f"  {name}.geojson  {n} 要素（spat bbox）")
    for name in ("provinces", "airports", "cities", "ports", "roads"):
        normalize_props(OUT / f"{name}.geojson")
    print("  属性键已统一小写")
    return paths


# ------------------------------------------------------------------ truth ---
def truth():
    from shapely.geometry import shape
    from shapely.ops import unary_union

    def load(name):
        d = json.loads((OUT / f"{name}.geojson").read_text(encoding="utf-8"))
        return d["features"]

    def props(f):
        return f["properties"] or {}

    def get(p, *keys):
        for k in keys:
            if p.get(k) is not None:
                return p[k]
        return None

    def names(fs):
        return sorted({get(props(f), "name", "NAME") for f in fs}
                      - {None})

    roads = load("roads")
    cities = load("cities")
    airports = load("airports")
    ports = load("ports")
    provinces = load("provinces")

    print("== 真值（shapely，quad_segs=5 对齐 QGIS SEGMENTS=5）==")
    r1_types = sorted({get(props(f), "type", "TYPE") for f in roads if
                       get(props(f), "type", "TYPE")})
    pop_key = None
    for f in cities:
        for k in ("pop_max", "POP_MAX"):
            if props(f).get(k) is not None:
                pop_key = k
                break
        if pop_key:
            break
    if pop_key is None:
        raise RuntimeError("cities 缺少 pop_max 字段")
    mega = [get(props(f), "name", "NAME") for f in cities
            if (get(props(f), pop_key) or 0) > 10_000_000]
    r2_mega = sorted({n for n in mega if n})

    ap_pts = [shape(f["geometry"]) for f in airports]
    buf = unary_union([p.buffer(BUFFER_M, quad_segs=5) for p in ap_pts])
    prov_geom = {get(props(f), "name", "NAME"): shape(f["geometry"]) for f in provinces}
    r4 = sorted(n for n, g in prov_geom.items() if n and g.intersects(buf))

    prov_union = unary_union(list(prov_geom.values()))
    r5 = sorted({n for f in cities
                 if (n := get(props(f), "name", "NAME"))
                 and shape(f["geometry"]).intersects(prov_union)})
    r6 = sorted({n for f in ports
                 if (n := get(props(f), "name", "NAME"))
                 and shape(f["geometry"]).intersects(prov_union)})

    t = {"r1_types": r1_types,
         "r2_megacities": r2_mega,
         "r3_airports": names(airports),
         "r4_provinces": r4,
         "r5_cities": r5,
         "r6_ports": r6,
         "buffer_m": BUFFER_M,
         "pop_field": pop_key,
         "counts": {"roads": len(roads), "cities": len(cities),
                    "airports": len(airports), "ports": len(ports),
                    "provinces": len(provinces)}}
    (OUT / "real_truth.json").write_text(
        json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")
    for k in ("r1_types", "r2_megacities", "r3_airports", "r4_provinces",
              "r5_cities", "r6_ports"):
        print(f"  {k}: {len(t[k])} → {t[k][:8]}{' …' if len(t[k]) > 8 else ''}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="重下（默认缓存）")
    args = ap.parse_args()
    paths = fetch(args.force)
    t = truth()
    meta = {
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": NE_TAG,
        "urls": {k: f"{BASE}/{v}" for k, v in SRC.items()},
        "sha256": {k: sha256(p) for k, p in paths.items()},
        "crs": "EPSG:3857 (Web Mercator, meters)",
        "clip": "ne_50m_admin_0_countries WHERE ADMIN='China'",
        "bbox": BBOX,
        "counts": t["counts"],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"\n已写出 {OUT}\\real_truth.json 与 meta.json")


if __name__ == "__main__":
    main()
