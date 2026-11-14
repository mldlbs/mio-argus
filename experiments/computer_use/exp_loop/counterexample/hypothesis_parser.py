"""从自动规则里抽出「隐含前提」——规则为什么可能成立，取决于哪个条件。"""
from __future__ import annotations

# 特征族 → 隐含前提的类型（不是结论，只是"这条规则依赖什么"）
PREMISE_OF = {
    "G_x": "positional",        # 依赖「控件在界面上的位置」
    "G_y": "positional",
    "G_icon_dist": "geometry",  # 依赖「落点与图标中心的距离」
    "G_icon_dist_n": "geometry",
    "T_len_b": "page_identity",  # 依赖「这是哪一类页面」
    "T_len_a": "page_identity",
    "T_jaccard": "text_change",
    "T_changed": "text_change",
    "D_whole": "visual_change",
    "D_max": "visual_change",
    "D_mean": "visual_change",
    "D_r": "visual_change",
    "E_": "rendering",
}


def _premise_for(feature: str) -> str:
    if feature.startswith("D_r"):
        return "visual_change"
    for k, v in PREMISE_OF.items():
        if feature == k or feature.startswith(k):
            return v
    return "unknown"


def parse(rule: dict) -> dict:
    """返回 {premise, features, condition, risk}。

    risk 用来提醒"这条前提最容易在什么情况下失效"：
      positional     → 布局滚动 / 状态不同就可能失效
      page_identity  → 只是"这是哪一页"的代理，不是失败机制
    """
    p = rule["predicate"]
    feats = [p["feature"]] + [x["feature"] for x in p.get("all", [])]
    premises = [_premise_for(f) for f in feats]
    premise = premises[0] if len(set(premises)) == 1 else "+".join(sorted(set(premises)))
    risk = {
        "positional": "布局滚动或不同状态下，位置与语义的映射会改变",
        "page_identity": "该量只说明是哪一页，不说明失败机制",
        "geometry": "图标检测失效时距离无意义",
        "text_change": "相邻对象内容相同会导致文本不变",
        "visual_change": "存在与动作无关的动画时画面本就会变",
        "rendering": "渲染差异可能与动作无关",
    }.get(premise, "前提不明")
    return {"rule_id": rule["rule_id"], "premise": premise,
            "features": feats, "condition": f"{p['feature']} {p['op']} {p['thr']}",
            "risk": risk}
