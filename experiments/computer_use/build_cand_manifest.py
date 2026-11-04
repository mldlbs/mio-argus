"""从新采集的 demo 构建候选基准清单（B1/B2 用）。

只取 click 且带 target_rect 的样本；同时保留 observation.affordances（新口径）
与 observation.affordances_legacy（旧口径，前 60 条 text 控件），用于同帧对照。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "computer_use_inloop"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="逗号分隔的任务名；默认取所有 *_r01")
    ap.add_argument("--out", default=str(ART / "cand_manifest.json"))
    args = ap.parse_args()

    want = {t.strip() for t in args.tasks.split(",") if t.strip()}
    rows = []
    for d in sorted((ART / "qgis").glob("*/demo.json")):
        task = d.parent.name
        if want and task not in want:
            continue
        if not want and not task.endswith("_r01"):
            continue
        data = json.loads(d.read_text(encoding="utf-8"))
        for r in data:
            if r.get("action") not in ("click", "move") or not r.get("target_rect"):
                continue
            from PIL import Image
            p = ROOT / r["screenshot"]
            if not p.exists():
                continue
            img = Image.open(p)
            screen = r.get("screen") or [img.size[0], img.size[1]]
            s = img.size[0] / screen[0]
            R = r["target_rect"]
            obs = r.get("observation") or {}
            rows.append({
                "id": r["id"], "task": task, "image": r["screenshot"],
                "image_size": list(img.size), "screen": screen,
                "instruction": r["instruction"],
                "rect_full": R,
                "rect_img": [R[0] * s, R[1] * s, R[2] * s, R[3] * s],
                "target_min_side_full_px": round(min(R[2] - R[0], R[3] - R[1]), 1),
                "affordances": obs.get("affordances") or [],
                "affordances_legacy": obs.get("affordances_legacy") or [],
                "cand_stats": obs.get("cand_stats") or {},
            })

    payload = {"n": len(rows), "source": "re-capture with action+widget candidates",
               "rows": rows}
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[manifest] n={len(rows)} -> {out}")
    if rows:
        import statistics as st
        print("[manifest] new_cands median",
              st.median(len(r["affordances"]) for r in rows),
              "legacy median",
              st.median(len(r["affordances_legacy"]) for r in rows),
              "with_actions", sum(1 for r in rows if r["cand_stats"].get("kinds", {}).get("action")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
