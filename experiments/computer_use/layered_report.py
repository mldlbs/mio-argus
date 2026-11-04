"""汇总分层消融结果：按目标尺寸分桶、命中率、中位距离、延迟。"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "computer_use_inloop"
PRED = ART / "audit_predictions"
MAN = json.loads((ART / "audit_manifest.json").read_text(encoding="utf-8"))
BY = {r["id"]: r for r in MAN["rows"]}


def bucket(px):
    return "small(<50)" if px < 50 else ("med(50-100)" if px < 100 else "large(>=100)")


def stats(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    agg = defaultdict(lambda: [0, 0])
    dists = []
    lats = []
    for r in rows:
        m = BY.get(r["id"])
        if not m:
            continue
        b = bucket(m["target_min_side_full_px"])
        agg[b][1] += 1
        agg[b][0] += int(bool(r["hit"]))
        if r.get("dist") is not None:
            dists.append(r["dist"])
        lats.append(r.get("latency_ms") or 0)
    n = sum(v[1] for v in agg.values())
    hit = sum(v[0] for v in agg.values())
    dists.sort()
    lats.sort()
    return {
        "name": d["provider"], "n": n, "top1": hit / max(n, 1),
        "by_size": {k: (v[0], v[1], v[0] / max(v[1], 1)) for k, v in sorted(agg.items())},
        "med_dist": dists[len(dists) // 2] if dists else None,
        "p50_ms": lats[len(lats) // 2] if lats else None,
    }


def main():
    files = sorted(PRED.glob("layered_*.json"))
    print(f"{'variant':22s} {'n':>4s} {'top1':>7s} {'med_dist':>9s} {'p50ms':>7s}  by_size")
    for f in files:
        s = stats(f)
        sz = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in s["by_size"].items())
        print(f"{s['name']:22s} {s['n']:4d} {s['top1']*100:6.1f}% "
              f"{(s['med_dist'] or 0):8.0f} {(s['p50_ms'] or 0):7.0f}  {sz}")


if __name__ == "__main__":
    main()
