"""经验闭环主流程：采集/载入 → 归因（不看标签）→ 打分 → 晋级 → 落库。

用法：
    python experiments/computer_use/exp_loop/run_loop.py --capture
    python experiments/computer_use/exp_loop/run_loop.py            # 用已有 episode 重跑
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import detect
import rules as R
from episode import read_episodes

OUT = Path("results") / "computer_use" / "exp_loop"

# 阈值网格：由验证器在数据上选，不写死人类判断
GRID = {
    "no_visual_change": [{"thr": v} for v in (0.5, 0.8, 1.2, 2.0, 3.0)],
    "text_unchanged": [{}],
    "empty_recognition": [{"min_energy": v} for v in (6.0, 9.0, 12.0, 16.0)],
    "target_is_sentence": [{"max_chars": v} for v in (6, 8, 10, 12, 16)],
    "click_off_icon": [{"tol": v} for v in (0.008, 0.012, 0.018, 0.025)],
}


def pick_params(eps: list[dict], labels: dict[str, str]) -> dict:
    """在 train 半区上选阈值：最大化 (recall - FPR)。"""
    best: dict[str, dict] = {}
    for det, cands in GRID.items():
        best_score, best_p = -9.0, cands[0]
        for p in cands:
            # 只看 train 半区的样本
            train = [e for e in eps if R.split_of(e["id"]) == "train"
                     and R.applies(det, e)]
            att = detect.attribute(train, OUT, {det: p})
            pos = [a for a in att if labels.get(a["episode"]) == "positive"]
            neg = [a for a in att if labels.get(a["episode"]) == "negative"]
            if not pos or not neg:
                continue
            rec = sum(any(h["detector"] == det for h in a["hypotheses"]) for a in pos) / len(pos)
            fpr = sum(any(h["detector"] == det for h in a["hypotheses"]) for a in neg) / len(neg)
            sc = rec - fpr
            if sc > best_score:
                best_score, best_p = sc, p
        best[det] = best_p
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.capture or not (OUT / "episodes.jsonl").exists():
        sys.path.insert(0, str(HERE))
        import capture as C
        C.capture(OUT)
        # 合并批次：按 id 去重（后采的覆盖同 id）
        def _merge(main: Path, batch: Path) -> None:
            rows = {}
            for p_ in (main, batch):
                if not p_.exists():
                    continue
                for line in p_.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    rows[d.get("id")] = d
            with main.open("w", encoding="utf-8") as fh:
                for d in rows.values():
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            batch.unlink(missing_ok=True)
        _merge(OUT / "episodes.jsonl", OUT / "episodes_batch.jsonl")
        _merge(OUT / "labels.jsonl", OUT / "labels_batch.jsonl")

    eps = read_episodes(OUT / "episodes.jsonl")
    labels = {}
    lpath = OUT / "labels.jsonl"
    if lpath.exists():
        for line in lpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                labels[d["id"]] = d["label"]

    params = pick_params(eps, labels)
    # 归因只拿到 episodes + params，拿不到 labels
    att = detect.attribute(eps, OUT, params)
    (OUT / "hypotheses.json").write_text(
        json.dumps({"params": params, "per_episode": att}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    stats = R.score(att, labels, {e["id"]: e for e in eps})
    built = R.build_rules(stats, params)
    R.save_store(OUT / "rules.json", built,
                 meta={"n_episodes": len(eps),
                       "n_pos": sum(v == "positive" for v in labels.values()),
                       "n_neg": sum(v == "negative" for v in labels.values()),
                       "gate": {"recall_min": R.RECALL_MIN, "fpr_max": R.FPR_MAX}})

    print(f"{'rule':26s} {'status':9s} {'val_recall':>10s} {'val_fpr':>8s}")
    for r in built:
        v = r["metrics"]["val"]
        print(f"{r['rule_id']:26s} {r['status']:9s} "
              f"{str(v['recall']):>10s} {str(v['fpr']):>8s}")
    active = [r["rule_id"] for r in built if r["status"] == "active"]
    print(f"\nactive={len(active)}: {active}")

    # ---------------- 自动合成（人类只给特征族，不给判别关系）
    import features as F
    import synth as S

    feats = F.extract_all(eps, OUT)
    (OUT / "features.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=1), encoding="utf-8")

    ids = [e["id"] for e in eps]
    # 人类规则已覆盖的正样本集合 → 合成只针对漏掉的残差
    active_dets = [r["detector"] for r in built if r["status"] == "active"]
    fired_by = {row["episode"]: {h["detector"] for h in row["hypotheses"]} for row in att}
    covered = {i for i, v in labels.items() if v == "positive"
               and (fired_by.get(i, set()) & set(active_dets))}
    residual = {i for i, v in labels.items() if v == "positive"} - covered
    print(f"\n人类规则覆盖 {len(covered)} 个正样本，剩余 {len(residual)} 个交由自动合成")

    res = S.synthesize(feats, labels, ids, residual_pos=residual)
    (OUT / "auto_rules.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"候选 {res['n_candidates']} 条 → val 选中 {res['n_selected']} 条"
          f"（split={res['split']}）")
    for r in res["rules"]:
        p, te = r["predicate"], r["metrics"]["test"]
        feat = p["feature"] + ("" if not p["all"] else " ∧ " + p["all"][0]["feature"])
        print(f"  {r['status']:8s} {feat:28s} {p['op']}{p['thr']:<10} "
              f"test tp={te['tp']} fp={te['fp']}")

    aut = S.autonomy(feats, labels, res["rules"], eps, OUT, active_dets)
    print(f"\n自主发现率 = {aut['autonomy_rate']}  "
          f"(新 {len(aut['auto_new'])} / (新 {len(aut['auto_new'])} + 人类有效 {len(aut['human_ok'])}))")
    print(f"  人类覆盖 test 正样本 {aut['human_covered_pos']}/{aut['test_pos']}，"
          f"残差 {aut['residual_pos']}")
    print(f"  自动新发现：{aut['auto_new']}")
    print(f"  自动重发现：{aut['auto_rediscovered']}  被 test 否决：{aut['auto_rejected_test']}")
    (OUT / "autonomy.json").write_text(json.dumps(aut, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
