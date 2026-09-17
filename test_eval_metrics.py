"""
回归测试：确保评测指标不低于基线
运行方式：pytest test_eval_metrics.py -v
"""
import json
import pytest
from pathlib import Path

# 基线指标（来自首次评测）
BASELINE = {
    "action_accuracy": 1.0,
    "action_macro_f1": 1.0,
    "coord_mae": 0.10,      # 允许一定波动
    "scroll_accuracy": 0.90,
}


def load_results():
    path = Path("eval_results.json")
    if not path.exists():
        pytest.skip("eval_results.json 不存在，请先运行 python eval_model.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_action_accuracy():
    results = load_results()
    assert results["action_accuracy"] >= BASELINE["action_accuracy"] - 1e-6, \
        f"Action accuracy {results['action_accuracy']:.4f} < baseline {BASELINE['action_accuracy']}"


def test_action_macro_f1():
    results = load_results()
    assert results["action_macro_f1"] >= BASELINE["action_macro_f1"] - 1e-6, \
        f"Action macro-F1 {results['action_macro_f1']:.4f} < baseline {BASELINE['action_macro_f1']}"


def test_coord_mae():
    results = load_results()
    assert results["coord_mae"] <= BASELINE["coord_mae"] + 1e-6, \
        f"Coord MAE {results['coord_mae']:.6f} > baseline {BASELINE['coord_mae']}"


def test_scroll_accuracy():
    results = load_results()
    assert results["scroll_accuracy"] >= BASELINE["scroll_accuracy"] - 1e-6, \
        f"Scroll accuracy {results['scroll_accuracy']:.4f} < baseline {BASELINE['scroll_accuracy']}"


def test_per_class_minimum():
    """每个动作类别至少有 1 个测试样本且准确率不为 0"""
    results = load_results()
    for name, metrics in results["per_class_action"].items():
        assert metrics["count"] > 0, f"类别 {name} 无测试样本"
        assert metrics["accuracy"] >= 0.5, f"类别 {name} 准确率过低: {metrics['accuracy']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])