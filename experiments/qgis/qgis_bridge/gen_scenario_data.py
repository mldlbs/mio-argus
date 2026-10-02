"""
W23 场景数据生成（纯 Python，无第三方依赖）

生成 6 个矢量场景（30 场景表首批）配套数据 → data/scenario/*.geojson
坐标系 EPSG:3857（米），使缓冲区半径可直接以米表达。
同时产出 data/scenario_truth.json —— 按构造已知的真值（纯几何计算，
不依赖 QGIS），供执行链的外部核验。

场景（与 run_w23.py 的 SCENARIOS 对应）：
  S1 #1  专题地图   roads 按 class 分类着色（4 类）
  S2 #9  道路分析   选中 class='primary' 的道路并导出
  S3 #11 应急救援   医院 r=150000m 缓冲（Buffer）→ 导出
  S4 #13 医疗资源   缓冲 → 被覆盖的行政区（Extract by location）→ 导出
  S5 #27 POI 分析   落在行政区内的 POI（Extract by location）→ 导出
  S6 #4  内涝分析   位于行政区内的积水点（Extract by location）→ 导出

regions 版图（由 W1 的度坐标 ×1e5 得到，与旧场景同布局）：
  North   (600000, 4400000)–(1400000, 5000000)
  South   (1200000, 3600000)–(2000000, 4400000)
  Central (600000, 3800000)–(1200000, 4400000)
  West    (200000, 4000000)–(600000, 4400000)
  East    (2000000, 3800000)–(2500000, 4200000)

用法: python experiments/qgis/qgis_bridge/gen_scenario_data.py
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
OUT = HERE / "data" / "scenario"
OUT.mkdir(parents=True, exist_ok=True)

R = 150000  # 缓冲半径（米）—— S3/S4 共用

REGIONS = {
    "North":   (600000, 4400000, 1400000, 5000000),
    "South":   (1200000, 3600000, 2000000, 4400000),
    "Central": (600000, 3800000, 1200000, 4400000),
    "West":    (200000, 4000000, 600000, 4400000),
    "East":    (2000000, 3800000, 2500000, 4200000),
}

# 医院：H1 近 West 东缘（缓冲跨入 Central）；H2/H3 各守一区；
# H4/H5 在所有区之外（缓冲存在但不覆盖任何区）
HOSPITALS = {
    "h1": (550000, 4200000),    # West 内，距 Central 西缘 50km < r → 覆盖 West+Central
    "h2": (2250000, 4000000),   # East 内陆，距 South 250km > r → 只覆盖 East
    "h3": (1000000, 4700000),   # North 内陆 → 只覆盖 North
    "h4": (300000, 3000000),    # 全区之外
    "h5": (2600000, 4600000),   # 全区之外（East 东北）
}

# 道路：4 类 × 固定条数，确定性横排网格
ROAD_CLASSES = [("motorway", 4), ("primary", 6), ("secondary", 8), ("residential", 12)]


def road_features():
    feats = []
    i = 0
    y0 = 3100000
    for cls, n in ROAD_CLASSES:
        for k in range(n):
            i += 1
            y = y0 + 130000 * i
            x0 = 150000 + (i % 5) * 210000
            x1 = x0 + 420000 + (i % 3) * 130000
            feats.append(dict(
                name=f"r{i:02d}", cls=cls,
                geom={"type": "LineString",
                      "coordinates": [[x0, y], [x1, y + (40000 if i % 2 else -40000)]]}))
    return feats


# POI：8 个区内 + 4 个区外（缝隙/南缘/北缘之外）
POIS = {
    # 区内
    "po01": ("school", 800000, 4600000),    # North
    "po02": ("school", 1300000, 4500000),   # North
    "po03": ("park", 700000, 4000000),      # Central
    "po04": ("school", 350000, 4200000),    # West
    "po05": ("park", 500000, 4100000),      # West
    "po06": ("school", 1600000, 3800000),   # South
    "po07": ("school", 2300000, 3950000),   # East
    "po08": ("park", 2100000, 4100000),     # East
    # 区外（空隙）
    "po09": ("school", 1700000, 4700000),   # North 与 East 之间以北的空隙
    "po10": ("park", 800000, 3300000),      # South 以南
    "po11": ("school", 300000, 3700000),    # West 以南、Central 以西
    "po12": ("park", 2700000, 3500000),     # East 以南
}

# 积水点：West×4 + East×4 在区内；2 个在区外（none in North/South/Central →
# "位于行政区内" 的 extract 结果 = 低洼两区的 8 个点）
FLOODS = {
    "f01": (250000, 4050000), "f02": (300000, 4300000),
    "f03": (550000, 4100000), "f04": (500000, 4350000),
    "f05": (2050000, 3850000), "f06": (2150000, 3900000),
    "f07": (2400000, 4150000), "f08": (2450000, 3850000),
    "f09": (900000, 3450000),   # 区外
    "f10": (1750000, 4650000),  # 区外（空隙）
}


def gj(name, feats):
    items = feats.items() if isinstance(feats, dict) else [(f["name"], f) for f in feats]
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": [
            {"type": "Feature",
             "properties": {"name": n, **({} if "cls" not in f else {"class": f["cls"]})},
             "geometry": f["geom"]}
            for n, f in items],
    }


def pt_in_rect(p, rect):
    x, y = p
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def pt_rect_dist(p, rect):
    x, y = p
    x0, y0, x1, y1 = rect
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    return math.hypot(dx, dy)


def main():
    roads = road_features()
    (OUT / "roads.geojson").write_text(
        json.dumps(gj("roads", {f["name"]: f for f in roads}),
                   ensure_ascii=False), encoding="utf-8")
    (OUT / "hospitals.geojson").write_text(json.dumps(
        gj("hospitals", {n: dict(geom={"type": "Point", "coordinates": list(p)})
                         for n, p in HOSPITALS.items()}),
        ensure_ascii=False), encoding="utf-8")
    (OUT / "pois.geojson").write_text(json.dumps(
        gj("pois", {n: dict(geom={"type": "Point", "coordinates": [x, y]})
                    for n, (t, x, y) in POIS.items()}), ensure_ascii=False),
        encoding="utf-8")
    (OUT / "floods.geojson").write_text(json.dumps(
        gj("floods", {n: dict(geom={"type": "Point", "coordinates": list(p)})
                      for n, p in FLOODS.items()}), ensure_ascii=False),
        encoding="utf-8")

    # ---- 真值（纯几何） ----
    primary = [f["name"] for f in roads if f["cls"] == "primary"]
    pois_inside = [n for n, (t, x, y) in POIS.items()
                   if any(pt_in_rect((x, y), r) for r in REGIONS.values())]
    floods_inside = [n for n, p in FLOODS.items()
                     if any(pt_in_rect(p, r) for r in REGIONS.values())]
    covered = sorted(n for n, r in REGIONS.items()
                     if min(pt_rect_dist(p, r) for p in HOSPITALS.values()) <= R)
    # 覆盖判定的安全裕度（多边形近似圆的外接误差）：被覆盖区 ≤0.9r，未覆盖区 ≥1.1r
    margins = {n: round(min(pt_rect_dist(p, r) for p in HOSPITALS.values()))
               for n, r in REGIONS.items()}
    assert all(margins[n] <= 0.9 * R for n in covered), margins
    assert all(margins[n] >= 1.1 * R for n in REGIONS if n not in covered), margins

    truth = dict(
        regions=REGIONS, buffer_r=R,
        s1_classes=[c for c, _ in ROAD_CLASSES], s1_field="class",
        s2_primary=sorted(primary),
        s3_hospitals=dict(HOSPITALS), s3_n=len(HOSPITALS),
        s4_covered=covered, s4_margins=margins,
        s5_pois_inside=sorted(pois_inside),
        s6_floods_inside=sorted(floods_inside),
        n_roads=len(roads), n_pois=len(POIS), n_floods=len(FLOODS),
    )
    (HERE / "data" / "scenario_truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"生成 {OUT}: roads={len(roads)} hospitals={len(HOSPITALS)} "
          f"pois={len(POIS)} floods={len(FLOODS)}")
    print(f"真值: S2 primary={len(primary)} S4 covered={covered} "
          f"S5 pois_inside={len(pois_inside)} S6 floods_inside={len(floods_inside)}")
    print("margins(m):", margins)
    print("已写出 data/scenario_truth.json")


if __name__ == "__main__":
    main()
