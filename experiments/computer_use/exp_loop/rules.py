"""经验库：假设 → 候选规则 → 独立样本验证 → 晋级/淘汰。

规则不是"模型认为自己懂了"就进系统：
  - 先在 train 半区选阈值，
  - 再在 **val 半区**（不参与阈值选择）打分，
  - recall ≥ 门槛且 FPR ≤ 门槛才置为 active，
  - 只有 active 的规则会被 duck_guard 读取（proposed 永远不生效）。
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

RECALL_MIN = 0.90
FPR_MAX = 0.05

# 假设 → 候选经验的模板（人类只给候选，是否成立由数据决定）
RULE_TEMPLATES = {
    "no_visual_change": {
        "rule_id": "R1_assert_effect",
        "statement": "动作后必须观测到画面/状态变化；观测不到即判动作未生效，"
                     "不得采信动作后的读数。",
        "condition": "action.type==click and expected.kind in {selection_change,state_change}",
        "hook": "click_slot_verified / _step_duck",
        "invalidate_if": "界面存在与动作无关的持续动画，使 frame_diff 长期高于阈值",
    },
    "text_unchanged": {
        "rule_id": "R2_refetch_after_action",
        "statement": "禁止把动作前文本当动作后状态：文本未变时必须重新取证（帧+文本），"
                     "并区分「没生效」与「确实无变化」。",
        "condition": "expected.kind in {selection_change,text_change,state_change}",
        "hook": "read_duck_profile / _page_text",
        "invalidate_if": "动作确实是幂等的（重复执行不产生可观测差异）",
    },
    "empty_recognition": {
        "rule_id": "R3_empty_is_not_absent",
        "statement": "识别返回空串不等于目标不存在：必须回退第二通道（整页识别）后再判，"
                     "且不得据此得出「无气泡 / 无内容」的结论。",
        "condition": "action.type==recognize and ocr_channel==fast",
        "hook": "ocr_fast",
        "invalidate_if": "存在真实空白区域，回退通道同样为空且能量低",
    },
    "target_is_sentence": {
        "rule_id": "R4_labels_only",
        "statement": "可点目标必须是短标签（长度 ≤ 阈值）；长句一律视为说明文本，不可点。",
        "condition": "action.type==click and action.text_target != ''",
        "hook": "_find_exact",
        "invalidate_if": "界面把按钮文案写成整句（实测未出现）",
    },
    "click_off_icon": {
        "rule_id": "R5_hit_icon_center",
        "statement": "点击落点必须落在检测到的图标中心容差内，否则视为缝隙，禁止执行。",
        "condition": "action.type==click and 槽位带可见",
        "hook": "_click_icon",
        "invalidate_if": "图标检测本身不可用（检测数为 0）",
    },
}


def split_of(ep_id: str) -> str:
    """确定性切分：同一 episode 永远落在同一半区。"""
    return "val" if zlib.crc32(ep_id.encode()) % 2 == 0 else "train"


def applies(det: str, ep: dict) -> bool:
    """规则适用范围：只在它关心的动作类型上评分，否则 recall 分母会被污染。"""
    act = ep.get("action") or {}
    kind = act.get("type")
    if det in ("no_visual_change", "text_unchanged", "click_off_icon"):
        return kind == "click"
    if det == "empty_recognition":
        return kind == "recognize"
    if det == "target_is_sentence":
        return kind == "click" and bool((act.get("text_target") or "").strip())
    return True


def score(attribution: list[dict], labels: dict[str, str],
          eps_by_id: dict[str, dict] | None = None) -> dict:
    """按 detector 统计 train/val 上的 recall 与 FPR（**仅在适用样本上**）。

    labels: episode_id -> 'positive' | 'negative'
    """
    stats: dict[str, dict] = {}
    for row in attribution:
        ep_id = row["episode"]
        lab = labels.get(ep_id)
        if lab not in ("positive", "negative"):
            continue
        ep = (eps_by_id or {}).get(ep_id) or {}
        half = split_of(ep_id)
        fired = {h["detector"] for h in row["hypotheses"]}
        for det in list(RULE_TEMPLATES):
            if eps_by_id is not None and not applies(det, ep):
                continue
            s = stats.setdefault(det, {h: {"pos": 0, "pos_hit": 0, "neg": 0, "neg_hit": 0}
                                       for h in ("train", "val")})
            cell = s[half]
            if lab == "positive":
                cell["pos"] += 1
                cell["pos_hit"] += int(det in fired)
            else:
                cell["neg"] += 1
                cell["neg_hit"] += int(det in fired)
    return stats


def build_rules(stats: dict, params: dict | None = None) -> list[dict]:
    """把统计量变成规则对象，并按 val 半区决定 status。

    口径（2026-11-14 修订）：不同规则覆盖**不同失败族**（缝隙 / 识别空串 / 错误导航），
    用统一 recall 门槛会把它们全部否掉。所以改为**精度优先**：
      - val 上误报率必须 ≤ 门槛（fp 必须很少）
      - 至少要命中一个正样本
      - 覆盖广度由「规则集合的边际覆盖」在调用方统计，不写进单条规则
    """
    rules = []
    for det, tpl in RULE_TEMPLATES.items():
        s = stats.get(det)
        if not s:
            continue
        tr, va = s["train"], s["val"]
        precision = (va["pos_hit"] / (va["pos_hit"] + va["neg_hit"])
                     if (va["pos_hit"] + va["neg_hit"]) else 0.0)
        fpr = (va["neg_hit"] / va["neg"]) if va["neg"] else None
        ok = (va["pos_hit"] >= 1 and va["neg"] >= 3
              and fpr is not None and fpr <= FPR_MAX)
        rules.append({
            "detector": det,
            "rule_id": tpl["rule_id"],
            "version": 2,
            "status": "active" if ok else "proposed",
            "statement": tpl["statement"],
            "condition": tpl["condition"],
            "hook": tpl["hook"],
            "invalidate_if": tpl["invalidate_if"],
            "params": (params or {}).get(det) or {},
            "metrics": {
                "train": {"tp": tr["pos_hit"], "n_pos": tr["pos"],
                          "fp": tr["neg_hit"], "n_neg": tr["neg"]},
                "val": {"tp": va["pos_hit"], "n_pos": va["pos"],
                        "fp": va["neg_hit"], "n_neg": va["neg"],
                        "precision": round(precision, 3),
                        "fpr": round(fpr, 3) if fpr is not None else None,
                        "recall": round(va["pos_hit"] / va["pos"], 3) if va["pos"] else None},
            },
            "gate": {"fpr_max": FPR_MAX, "min_tp": 1},
        })
    return rules


def union_coverage(attribution: list[dict], labels: dict[str, str],
                   active_ids: set[str], half_filter: str | None = None) -> dict:
    """规则集合对正样本的覆盖率（这是多规则系统的真实指标）。"""
    pos_ids = [i for i, v in labels.items() if v == "positive"
               and (half_filter is None or split_of(i) == half_filter)]
    fired_by = {r["episode"]: {h["detector"] for h in r["hypotheses"]} for r in attribution}
    covered = [i for i in pos_ids if fired_by.get(i, set()) & active_ids]
    return {"n_pos": len(pos_ids), "covered": len(covered),
            "coverage": round(len(covered) / len(pos_ids), 3) if pos_ids else None,
            "missed": [i for i in pos_ids if i not in covered]}


def save_store(path: Path, rules: list[dict], meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "meta": meta or {}, "rules": rules}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_active(path: Path) -> list[dict]:
    """duck_guard 只读 active 规则。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [r for r in data.get("rules", []) if r.get("status") == "active"]
