"""Tests for utils/growth.py (Adaptation Controller, Phase 4). Torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from kadhi_cli.utils.growth import (
    ConvergenceMonitor, FrontierPoint, expand_lora_pair, extend_adam_state,
    format_frontier, frontier, plan_growth_step,
)


def test_expansion_preserves_function_exactly():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 16))
    B = rng.standard_normal((8, 4))
    x = rng.standard_normal((16, 5))
    A2, B2 = expand_lora_pair(A, B, 10, rng)
    assert A2.shape == (10, 16) and B2.shape == (8, 10)
    assert np.array_equal(B2 @ (A2 @ x), B @ (A @ x))
    assert not np.allclose(A2[4:], 0)  # new rows are non-trivial, so they can learn


def test_expansion_new_b_columns_get_gradient():
    rng = np.random.default_rng(2)
    A2, B2 = expand_lora_pair(rng.standard_normal((2, 6)), rng.standard_normal((3, 2)), 4, rng)
    x, dy = rng.standard_normal((6, 7)), rng.standard_normal((3, 7))
    grad_B = dy @ (A2 @ x).T
    assert np.abs(grad_B[:, 2:]).max() > 0


def test_expansion_rejects_shrink_and_bad_pair():
    A, B = np.ones((4, 5)), np.ones((6, 4))
    with pytest.raises(ValueError):
        expand_lora_pair(A, B, 2)
    with pytest.raises(ValueError):
        expand_lora_pair(A, np.ones((6, 3)), 8)


def test_extend_adam_state_zero_pads_and_keeps_history():
    s = np.full((2, 5), 3.0)
    out = extend_adam_state(s, factor="A", new_rank=4)
    assert out.shape == (4, 5) and (out[:2] == 3).all() and (out[2:] == 0).all()
    out = extend_adam_state(np.full((5, 2), 1.0), factor="B", new_rank=3)
    assert out.shape == (5, 3) and (out[:, 2] == 0).all()


def test_convergence_monitor():
    m = ConvergenceMonitor(epsilon=0.05, window=5)
    for i in range(10):
        m.update(2.0 - 0.2 * i)  # still improving fast
    assert not m.converged()
    m.reset()
    for _ in range(10):
        m.update(1.0)  # flat
    assert m.converged()
    m2 = ConvergenceMonitor(epsilon=0.05, window=5)
    m2.update(1.0)
    assert not m2.converged()  # not enough history


def test_plan_growth_step_respects_budget_and_cap():
    ranks = {"l0": 4, "l1": 4}
    new = plan_growth_step(
        {"l0": 10.0, "l1": 1.0}, ranks, {"l0": 100, "l1": 100},
        extra_budget_params=800, r_max=8, step=4,
    )
    assert new == {"l0": 8, "l1": 8}  # 400 + 400 = 800 spent exactly
    new = plan_growth_step(
        {"l0": 10.0, "l1": 1.0}, ranks, {"l0": 100, "l1": 100},
        extra_budget_params=400, r_max=64, step=4,
    )
    assert sum(new.values()) - sum(ranks.values()) == 4 * 1  # only one 400-param step fits... on l0
    assert new["l0"] > new["l1"]


def test_frontier_and_report():
    pts = [
        FrontierPoint("uniform-8", 1_000_000, 0.60, 100),
        FrontierPoint("uniform-16", 2_000_000, 0.70, 180),
        FrontierPoint("allocated", 1_000_000, 0.68, 120),   # dominates uniform-8
        FrontierPoint("bad", 3_000_000, 0.65),              # dominated by uniform-16
    ]
    labels = [p.label for p in frontier(pts)]
    assert labels == ["allocated", "uniform-16"]
    report = format_frontier(pts)
    assert "allocated" in report and "*" in report
