"""诊断 B1 漏召：目标附近有没有被过滤掉的候选、最近的候选有多远、类别是什么。"""
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = json.loads((ROOT / "artifacts" / "computer_use_inloop" / "cand_manifest.json")
                 .read_text(encoding="utf-8"))

ACTIONABLE = {"QAction", "QToolButton", "QPushButton", "QLineEdit", "QComboBox", "QCheckBox"}
CONTAINER = {"QgisApp", "QMainWindow", "QDialog", "QgsDockWidget", "QWidget"}


def usable(a, screen):
    r = a.get("rect") or [0, 0, 0, 0]
    W, H = screen
    if r[2] <= 0 or r[3] <= 0:
        return False
    if (r[2] * r[3]) / float(W * H) > 0.25:
        return False
    return (a.get("cls") or "") not in CONTAINER


def main():
    misses = []
    reasons = Counter()
    for r in MAN["rows"]:
        cands = [a for a in (r.get("affordances") or []) if usable(a, r["screen"])]
        L, T, R, B = r["rect_full"]
        cx, cy = (L + R) / 2, (T + B) / 2
        hit = [a for a in cands
               if a["rect"][0] <= cx <= a["rect"][0] + a["rect"][2]
               and a["rect"][1] <= cy <= a["rect"][1] + a["rect"][3]]
        if hit:
            continue
        raw = r.get("affordances") or []
        raw_hit = [a for a in raw
                   if a["rect"][0] <= cx <= a["rect"][0] + a["rect"][2]
                   and a["rect"][1] <= cy <= a["rect"][1] + a["rect"][3]]
        if raw_hit:
            cls = (raw_hit[0].get("cls") or "")
            reasons["filtered:" + (cls or "?")] += 1
        else:
            nearest = min((math.dist(((a["rect"][0] + a["rect"][2] / 2),
                                      (a["rect"][1] + a["rect"][3] / 2)), (cx, cy)), a)
                          for a in raw) if raw else (1e9, None)
            if nearest[0] < 40:
                reasons["near_miss<40px"] += 1
            elif nearest[1] is not None and not (nearest[1].get("text") or "").strip():
                reasons["nearest_has_no_text"] += 1
            else:
                reasons["absent"] += 1
        misses.append({"id": r["id"], "instruction": r["instruction"],
                       "target": [round(cx), round(cy)],
                       "min_side": r["target_min_side_full_px"]})

    print("misses", len(misses), "of", len(MAN["rows"]))
    for k, v in reasons.most_common():
        print(f"  {k:28s} {v}")
    print("\n样例：")
    for m in misses[:8]:
        print(f"  {m['instruction'][:50]:52s} target={m['target']} min_side={m['min_side']}")


if __name__ == "__main__":
    main()
