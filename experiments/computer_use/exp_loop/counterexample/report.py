"""三个学习指标（按机制去重，不按规则条数）。

  Discovery Rate          独立生成、且通过独立 test 的**机制**数（同机制不同阈值只算 1）
  Falsification Rate      「实际做了反例尝试」的规则里，被反例推翻的比例
  Robust Discovery Rate   通过 反例测试 + 独立 test（+ 生产监督）后仍成立的机制比例

高 Falsification Rate 不是坏事：早期能主动推翻错误假设，比多产规则更有价值。
"""
from __future__ import annotations


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mechanisms(rules: list[dict], firing: dict[str, frozenset]) -> dict[str, list[str]]:
    """把规则按触发向量聚成机制（Jaccard ≥ 0.9 视为同一机制）。"""
    clusters: dict[str, list[str]] = {}
    reps: list[str] = []
    for r in rules:
        rid = r["rule_id"]
        vec = firing.get(rid, frozenset())
        for rep in reps:
            if _jaccard(vec, firing.get(rep, frozenset())) >= 0.9:
                clusters[rep].append(rid)
                break
        else:
            reps.append(rid)
            clusters[rid] = [rid]
    return clusters


def compute(auto_rules: list[dict], verdicts: list[dict],
            firing: dict[str, frozenset]) -> dict:
    active = [r for r in auto_rules if r.get("status") == "active"]
    clusters = mechanisms(active, firing)
    disc = len(clusters)

    tested = [v for v in verdicts if v["verdict"] in ("refuted", "holds")]
    refuted = [v for v in verdicts if v["verdict"] == "refuted"]
    insuff = [v for v in verdicts if v["verdict"] == "insufficient"]

    # 机制级鲁棒性：必须「做过反例尝试且未被推翻」才算稳健。
    # 证据不足（无法构造反例）**不算稳健**——那是"没被检验"，不是"检验通过"。
    refuted_ids = {v["rule_id"] for v in refuted}
    holds_ids = {v["rule_id"] for v in verdicts if v["verdict"] == "holds"}
    robust_clusters = {}
    for rep, members in clusters.items():
        if any(m in refuted_ids for m in members):
            continue
        if not any(m in holds_ids for m in members):
            continue
        robust_clusters[rep] = members

    return {
        "discovery_rate": disc,
        "discovery_detail": {k: v for k, v in clusters.items()},
        "falsification_rate": (round(len(refuted) / len(tested), 3) if tested else None),
        "n_tested": len(tested),
        "n_refuted": len(refuted),
        "n_insufficient": len(insuff),
        "robust_discovery_rate": (round(len(robust_clusters) / disc, 3) if disc else None),
        "robust_mechanisms": list(robust_clusters),
        "note": "Robust 只统计「通过反例测试 + 独立 test」；生产监督验证尚未接入",
    }
