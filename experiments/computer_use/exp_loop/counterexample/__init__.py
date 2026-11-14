"""反例驱动学习（Counterexample-driven Learning）。

分工：
  hypothesis_parser   从规则里抽出「隐含前提」
  coverage_analyzer   找出数据没覆盖的条件（薄弱环节）
  scenario_generator  针对薄弱环节提出反例场景，并判断是否可构造
  scenario_executor   在真实界面里构造场景并采集反例（不允许伪造截图）
  falsification       判定 假设成立 / 失效 / 证据不足
  report              汇总裁决 + 三个学习指标
"""
