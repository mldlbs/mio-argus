"""覆盖分析：找出规则「没有证据」的那一段条件。

判定"薄弱"的标准是数据层面的，不靠人判断：
  - 规则命中区间内 **没有任何负样本** → 该区间从未出现过"合法动作"，
    规则可能是"这里恰好没有反例"而不是"这里真的危险"。
"""
from __future__ import annotations


def _fires(pred: dict, feats: dict, ep_id: str) -> bool:
    fv = feats.get(ep_id, {}).get(pred["feature"])
    if fv is None or fv < 0:
        return False
    ok = fv <= pred["thr"] if pred["op"] == "<=" else fv >= pred["thr"]
    for extra in pred.get("all", []):
        ev = feats.get(ep_id, {}).get(extra["feature"])
        if ev is None or ev < 0:
            return False
        ok = ok and (ev <= extra["thr"] if extra["op"] == "<=" else ev >= extra["thr"])
    return ok


def analyze(rule: dict, feats: dict, labels: dict, ids: list[str]) -> dict:
    pred = rule["predicate"]
    hit = [i for i in ids if _fires(pred, feats, i)]
    pos = [i for i in hit if labels.get(i) == "positive"]
    neg = [i for i in hit if labels.get(i) == "negative"]
    outside = [i for i in ids if i not in hit]

    gaps = []
    if not neg:
        gaps.append({
            "kind": "no_counterexample_inside_region",
            "detail": f"该规则命中区间内有 {len(pos)} 个失败样本，但**0 个成功样本**",
        })
    if pred["op"] == ">=" and pred["thr"] > 0.4:
        gaps.append({
            "kind": "one_sided_threshold",
            "detail": f"阈值只覆盖一侧（{pred['feature']} ≥ {pred['thr']}），"
                      f"另一侧没有对照",
        })
    return {
        "rule_id": rule["rule_id"],
        "n_hit": len(hit), "n_pos": len(pos), "n_neg": len(neg),
        "n_outside": len(outside),
        "gaps": gaps,
        "weak": bool(gaps),
    }
