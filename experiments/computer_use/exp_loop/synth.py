"""Detector 自动合成：从「可观测特征」里搜索判别关系，自动生成检测规则。

流程（用户 2026-11-14 指定的方向）：
  原始 episode → 自动提特征 → 自动生成候选（单特征阈值 + 双子句合取）
  → train 定分位点 / val 选规则 / test 独立裁决 → 晋级
人类不再指定"看哪个量"，只提供特征族（像素差分/能量/文本/几何/动作元信息）。
"""
from __future__ import annotations

import itertools
import json
import zlib
from pathlib import Path

import detect
import rules as R

QS = (2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98)
VAL_RECALL_MIN = 0.80
VAL_FPR_MAX = 0.05
TEST_RECALL_MIN = 0.80
TEST_FPR_MAX = 0.05


def split3(ep_id: str) -> str:
    return ("train", "val", "test")[zlib.crc32(ep_id.encode()) % 3]


def _family(name: str) -> str:
    return name.split("_")[0]


def _fires(pred: dict, feats: dict, ep_id: str) -> bool:
    fv = feats.get(ep_id, {}).get(pred["feature"])
    if fv is None or (isinstance(fv, float) and fv < 0):
        return False
    if pred["op"] == "<=":
        ok = fv <= pred["thr"]
    else:
        ok = fv >= pred["thr"]
    for extra in pred.get("all", []):
        ev = feats.get(ep_id, {}).get(extra["feature"])
        if ev is None or ev < 0:
            return False
        if not ((ev <= extra["thr"]) if extra["op"] == "<=" else (ev >= extra["thr"])):
            return False
    return ok


def _score(pred: dict, feats: dict, labels: dict, ids: list[str]) -> dict:
    tp = fp = fn = tn = 0
    for i in ids:
        lab = labels.get(i)
        if lab not in ("positive", "negative"):
            continue
        fired = _fires(pred, feats, i)
        if lab == "positive":
            tp += fired
            fn += (not fired)
        else:
            fp += fired
            tn += (not fired)
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 1.0
    return {"recall": round(rec, 3), "fpr": round(fpr, 3),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def _quantiles(values: list[float], qs=QS) -> list[float]:
    if not values:
        return []
    import statistics
    out = []
    for q in qs:
        k = max(0, min(len(values) - 1, int(round(q / 100 * (len(values) - 1)))))
        out.append(sorted(values)[k])
    return sorted(set(out))


def build_candidates(feats: dict, ids: list[str]) -> list[dict]:
    """自动生成候选（阈值来自 train 分位点，不来自人类判断）。"""
    names = sorted(next(iter(feats.values())).keys()) if feats else []
    # 采集元信息不是失败证据（否则会"发现" A_fast==0 这类假规则）
    drop = {"A_is_click", "A_is_recognize", "A_fast"}
    names = [n for n in names if n not in drop]
    cands: list[dict] = []
    for name in names:
        vals = [feats[i][name] for i in ids if name in feats[i] and feats[i][name] >= 0]
        if len(vals) < 3 or max(vals) == min(vals):
            continue
        for thr in _quantiles(vals):
            for op in ("<=", ">="):
                cands.append({"feature": name, "op": op, "thr": round(thr, 4),
                              "all": [], "family": _family(name)})
    return cands


def _firing_vec(pred: dict, feats: dict, ids: list[str]) -> frozenset:
    return frozenset(i for i in ids if _fires(pred, feats, i))


def _vec_similarity(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def select(cands: list[dict], feats: dict, labels: dict,
           train: list[str], val: list[str],
           residual_pos: set[str] | None = None) -> list[dict]:
    """val 半区选规则：**精度优先 + 边际覆盖**，且只对「人类规则漏掉的正样本」计分。

    residual_pos：已被人类先验规则覆盖的正样本 id 集合；给定时，
    合成只负责发现**人类没发现的那部分**（否则会自动重发现已有的规则）。
    """
    def _tp(p, ids):
        return sum(1 for i in ids if labels.get(i) == "positive" and _fires(p, feats, i))

    def _fp(p, ids):
        return sum(1 for i in ids if labels.get(i) == "negative" and _fires(p, feats, i))

    tr_pos = [i for i in train if labels.get(i) == "positive"
              and (residual_pos is None or i in residual_pos)]
    va_pos = [i for i in val if labels.get(i) == "positive"
              and (residual_pos is None or i in residual_pos)]
    if not va_pos:
        return []

    pool_all = []
    for p in cands:
        va = _score(p, feats, labels, val)
        pool_all.append(dict(p, train=_score(p, feats, labels, train), val=va))

    # 合取：从「命中过残差正样本」的单句里组合（不同特征族）
    base = [p for p in pool_all if _tp(p, tr_pos) + _tp(p, va_pos) >= 1]
    base.sort(key=lambda x: (x["val"]["fp"], -_tp(x, va_pos)))
    conj = []
    for a, b in itertools.combinations(base[:30], 2):
        if a["feature"] == b["feature"]:
            continue
        q = {"feature": a["feature"], "op": a["op"], "thr": a["thr"],
             "all": [{"feature": b["feature"], "op": b["op"], "thr": b["thr"]}],
             "family": f"{a['family']}+{b['family']}"}
        conj.append(dict(q, train=_score(q, feats, labels, train),
                         val=_score(q, feats, labels, val)))

    pool = sorted(pool_all + conj,
                  key=lambda x: (x["val"]["fp"], -_tp(x, va_pos), len(x["all"])))
    covered: set[str] = set()
    kept: list[dict] = []
    for p in pool:
        if p["val"]["fp"] > 0 or p["val"]["fpr"] > VAL_FPR_MAX:
            continue
        new = {i for i in va_pos if _fires(p, feats, i)} - covered
        if not new:
            continue
        vec = _firing_vec(p, feats, val)
        if any(_vec_similarity(vec, _firing_vec(q, feats, val)) > 0.9 for q in kept):
            continue
        covered |= new
        kept.append(p)
    return kept


def synthesize(feats: dict, labels: dict, ids: list[str],
               residual_pos: set[str] | None = None) -> dict:
    train = [i for i in ids if split3(i) == "train"]
    val = [i for i in ids if split3(i) == "val"]
    test = [i for i in ids if split3(i) == "test"]
    cands = build_candidates(feats, train)
    kept = select(cands, feats, labels, train, val, residual_pos)

    out = []
    for p in kept:
        te = _score(p, feats, labels, test)
        promoted = te["tp"] >= 1 and te["fp"] <= max(0, int(0.02 * (te["fp"] + te["tn"])))
        out.append({
            "rule_id": f"A_{p['feature']}_{p['op'].replace('=', 'e')}_{abs(int(p['thr'] * 1000))}",
            "predicate": p,
            "metrics": {"train": p["train"], "val": p["val"], "test": te},
            "status": "active" if promoted else "overfit",
        })
    return {"n_candidates": len(cands), "n_selected": len(kept), "rules": out,
            "split": {"train": len(train), "val": len(val), "test": len(test)}}


def autonomy(feats: dict, labels: dict, auto: list[dict],
             episodes: list[dict], base: Path,
             human_active: list[str] | None = None) -> dict:
    """自主发现率 = 自动发现且**通过 test**、并覆盖「人类规则漏掉的正样本」的新规则数
                  / (该数 + 人类规则中在 test 上也成立的条数)。

    口径说明：
      - 只看 test 半区（从未参与阈值/规则选择）；
      - 成立 = test 上 fp==0 且至少命中 1 个正样本；
      - 自动规则若只是复现人类先验（触发向量 Jaccard ≥ 0.9），不计入"新"。
    """
    human_att = detect.attribute(episodes, base, {})
    human_active = human_active or ["no_visual_change", "empty_recognition",
                                    "click_off_icon"]
    fired_by = {r["episode"]: {h["detector"] for h in r["hypotheses"]} for r in human_att}

    test = [i for i in labels if split3(i) == "test"]
    test_pos = [i for i in test if labels[i] == "positive"]
    test_neg = [i for i in test if labels[i] == "negative"]

    human_cov = {d: {i for i in test_pos if d in fired_by.get(i, set())} for d in human_active}
    covered_by_human = set().union(*human_cov.values()) if human_cov else set()
    residual = [i for i in test_pos if i not in covered_by_human]

    human_ok = []
    for d in human_active:
        fp = sum(1 for i in test_neg if d in fired_by.get(i, set()))
        tp = len(human_cov[d])
        if fp == 0 and tp >= 1:
            human_ok.append(d)

    all_ids = list(labels)
    new_auto, rediscovered, auto_fail = [], [], []
    for r in auto:
        te = r["metrics"]["test"]
        vec = _firing_vec(r["predicate"], feats, all_ids)
        dup = any(_vec_similarity(vec, frozenset(
            i for i in all_ids if d in fired_by.get(i, set()))) > 0.9 for d in human_active)
        if r["status"] != "active":
            auto_fail.append(r["rule_id"])
            continue
        hit_residual = [i for i in residual if i in vec]
        if dup:
            rediscovered.append(r["rule_id"])
        elif hit_residual:
            new_auto.append(r["rule_id"])

    denom = len(new_auto) + len(human_ok)
    return {
        "test_pos": len(test_pos), "test_neg": len(test_neg),
        "human_covered_pos": len(covered_by_human), "residual_pos": len(residual),
        "human_ok": human_ok,
        "auto_new": new_auto, "auto_rediscovered": rediscovered,
        "auto_rejected_test": auto_fail,
        "autonomy_rate": round(len(new_auto) / denom, 3) if denom else 0.0,
    }
