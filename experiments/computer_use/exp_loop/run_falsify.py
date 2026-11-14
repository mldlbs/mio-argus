"""主动证伪主流程：对每条自动规则做「覆盖分析 → 场景构造 → 裁决」。

用法（会操作游戏窗口，先停看护）：
    python experiments/computer_use/exp_loop/run_falsify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "counterexample"))

import coverage_analyzer as CA
import falsification as FA
import hypothesis_parser as HP
import report as REP
import scenario_generator as SG
import scenario_executor as SE
import synth as S

OUT = Path("results") / "computer_use" / "exp_loop"


def main() -> int:
    auto = json.loads((OUT / "auto_rules.json").read_text(encoding="utf-8"))
    feats = json.loads((OUT / "features.json").read_text(encoding="utf-8"))
    labels = {}
    for line in (OUT / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            labels[d["id"]] = d["label"]
    ids = list(labels)

    rules = auto["rules"]
    if not rules:
        print("没有自动规则可证伪")
        return 0

    import duck_guard as dg
    h = dg.find_game_window()
    g = dg.Game(h)
    g.pin()

    verdicts = []
    for r in rules:
        prem = HP.parse(r)
        cov = CA.analyze(r, feats, labels, ids)
        spec = SG.generate(prem)
        print(f"\n=== {r['rule_id']}  ({r['status']}) ===")
        print(f"  前提类型：{prem['premise']}  风险：{prem['risk']}")
        print(f"  覆盖：命中 {cov['n_hit']}（失败 {cov['n_pos']} / 成功 {cov['n_neg']}）")
        for gp in cov["gaps"]:
            print(f"  薄弱环节：{gp['detail']}")

        attempt = None
        if spec.get("constructible") is not False and prem["premise"] == "positional":
            thr = float(r["predicate"]["thr"])
            attempt = SE.construct_positional(g, thr)
        elif prem["premise"] == "page_identity":
            thr = float(r["predicate"]["thr"])
            attempt = SE.construct_page_identity(g, thr)

        v = FA.judge(r, cov, prem, attempt)
        print(f"  裁决：{v['verdict']} — {v['reason']}")
        verdicts.append(v)

    firing = {r["rule_id"]: S._firing_vec(r["predicate"], feats, ids) for r in rules}
    met = REP.compute(rules, verdicts, firing)

    payload = {"verdicts": verdicts, "metrics": met}
    (OUT / "falsification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n---- 三个学习指标 ----")
    print(f"  Discovery Rate        = {met['discovery_rate']}  (机制数, 已按触发向量去重)")
    print(f"  Falsification Rate    = {met['falsification_rate']}  "
          f"({met['n_refuted']}/{met['n_tested']} 次真实反例尝试被推翻)")
    print(f"  Robust Discovery Rate = {met['robust_discovery_rate']}  "
          f"(通过反例+独立 test 的机制比例)")
    print(f"  证据不足（无法构造反例）：{met['n_insufficient']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
