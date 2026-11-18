"""误差分解的数学自检（不读真实数据）。"""
import pytest

import e1l_error_decomp as D


def cells(probs, n=100):
    return [{"n": n, "ok": round(p * n)} for p in probs]


def test_independence_index_one_when_independent():
    """各步独立 → 实测 ≈ ∏m_t → 指数 ≈ 1。"""
    probs = [0.8] * 8
    succ = round(1000 * 0.8 ** 8)          # 独立假设下应有的成功数
    s, p, idx = D.independence_index(cells(probs, 1000), succ, 1000, 8)
    assert abs(idx - 1.0) < 0.02


def test_independence_index_below_one_when_clustered():
    """错误成串 → 实测显著低于 ∏m_t → 指数 < 1。"""
    probs = [0.8] * 8
    succ = round(100 * 0.8 ** 8 * 0.5)     # 只有一半
    s, p, idx = D.independence_index(cells(probs, 100), succ, 100, 8)
    assert idx < 0.6


def test_missing_or_zero_step_returns_none():
    assert D.independence_index(cells([0.8, 0.0]), 0, 100, 2)[2] is None
    assert D.independence_index(cells([0.8]), 10, 100, 4)[2] is None


def test_h_one_is_trivial():
    s, p, idx = D.independence_index(cells([0.7], 100), 70, 100, 1)
    assert abs(idx - 1.0) < 1e-9
