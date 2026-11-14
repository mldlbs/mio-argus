"""场景生成：针对薄弱环节，提出「什么样的真实场景能推翻这条规则」。

重要边界：**只提出场景规格，不修改截图**。能否构造由执行器在真实界面里判断；
构造不出来就报 not_constructible，绝不伪造反例。
"""
from __future__ import annotations

SCENARIOS = {
    # 位置类：需要"同一位置上是合法可点对象"的真实布局
    "positional": {
        "goal": "在规则命中的位置区间内，找到一个**合法可点**对象并成功操作",
        "why": "若该位置存在合法对象且操作成功，则「位置⇒高风险」被推翻",
        "oracle": "操作后必须观测到选中态/页面状态按预期改变",
    },
    # 页面身份类：需要"同一页面特征下的一次成功动作"
    "page_identity": {
        "goal": "在与规则命中时相同（或更强）的页面特征下，完成一次合法动作",
        "why": "若同特征页面上的正常动作成功，则「该特征⇒失败」被推翻",
        "oracle": "动作导致的页面/状态变化必须可观测",
    },
}


def generate(premise: dict) -> dict:
    kind = premise["premise"]
    spec = SCENARIOS.get(kind)
    if not spec:
        return {"constructible": None, "kind": kind,
                "reason": f"前提类型 {kind} 暂无场景规格"}
    return {"constructible": None, "kind": kind, **spec,
            "target_condition": premise["condition"]}
