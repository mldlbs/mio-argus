"""证伪裁决：把「覆盖分析 + 场景构造」变成对每条规则的三态判决。

三态：
  refuted       找到真实反例 → 规则失效（退出晋级）
  holds         在**有反例可能**的场景里尝试过仍成立（本轮的证据强度）
  insufficient  无法构造反例场景 → 证据不足（不许晋级，也不许当成成立）
"""
from __future__ import annotations

import hypothesis_parser as HP


def judge(rule: dict, cover: dict, premise: dict, attempt: dict | None) -> dict:
    base = {"rule_id": rule["rule_id"], "premise": premise["premise"],
            "gaps": [g["kind"] for g in cover.get("gaps", [])],
            "coverage": {k: cover.get(k) for k in ("n_hit", "n_pos", "n_neg")}}

    if attempt is None:
        return dict(base, verdict="insufficient",
                    reason="未尝试构造反例（无对应场景规格）")
    if not attempt.get("constructible"):
        return dict(base, verdict="insufficient",
                    reason=f"不可构造：{attempt.get('reason')}")
    if attempt.get("verified"):
        return dict(base, verdict="refuted",
                    reason=f"反例成立：{attempt.get('reason')}",
                    counterexample={"x": attempt.get("x")})
    return dict(base, verdict="insufficient",
                reason=f"场景可构造但反例未出现：{attempt.get('reason')}")
