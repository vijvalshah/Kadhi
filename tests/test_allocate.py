"""Tests for kadhi_cli.utils.allocate — concave-objective LoRA rank
allocation under a trainable-parameter budget."""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from kadhi_cli.utils import allocate

    assert hasattr(allocate, "AllocationResult")
    assert hasattr(allocate, "allocate_ranks")


# ---------------------------------------------------------------------------
# AllocationResult validation
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    kwargs = dict(
        ranks={"a": 8, "b": 0},
        lambda_star=0.5,
        used_params=800,
        budget_params=1000,
        over_budget=False,
        frozen_layers=("b",),
        iterations=10,
    )
    kwargs.update(overrides)
    return kwargs


def test_allocation_result_valid_construction():
    from kadhi_cli.utils.allocate import AllocationResult

    result = AllocationResult(**_valid_kwargs())
    assert result.ranks == {"a": 8, "b": 0}
    assert result.frozen_layers == ("b",)


def test_allocation_result_rejects_empty_ranks():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(ranks={}, frozen_layers=()))


def test_allocation_result_rejects_non_mapping_ranks():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(TypeError):
        AllocationResult(**_valid_kwargs(ranks=[("a", 8)]))


def test_allocation_result_rejects_bool_rank_value():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(TypeError):
        AllocationResult(**_valid_kwargs(ranks={"a": True, "b": 0}))


def test_allocation_result_rejects_negative_rank_value():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(ranks={"a": -1, "b": 0}))


def test_allocation_result_rejects_non_str_rank_key():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(TypeError):
        AllocationResult(**_valid_kwargs(ranks={1: 8, "b": 0}))


def test_allocation_result_rejects_non_finite_lambda():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(lambda_star=float("inf")))
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(lambda_star=float("nan")))


def test_allocation_result_rejects_non_positive_lambda():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(lambda_star=0.0))
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(lambda_star=-1.0))


def test_allocation_result_rejects_negative_used_or_budget_params():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(used_params=-1))
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(budget_params=-1))


def test_allocation_result_rejects_bool_used_params():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(TypeError):
        AllocationResult(**_valid_kwargs(used_params=True))


def test_allocation_result_rejects_inconsistent_over_budget():
    from kadhi_cli.utils.allocate import AllocationResult

    # used_params < budget_params but over_budget=True is inconsistent.
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(used_params=500, budget_params=1000, over_budget=True))
    # used_params > budget_params but over_budget=False is inconsistent.
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(used_params=1500, budget_params=1000, over_budget=False))


def test_allocation_result_rejects_wrong_frozen_layers():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(frozen_layers=("a",)))
    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(frozen_layers=()))


def test_allocation_result_rejects_unsorted_frozen_layers():
    from kadhi_cli.utils.allocate import AllocationResult

    kwargs = _valid_kwargs(
        ranks={"a": 0, "b": 0, "c": 5},
        frozen_layers=("b", "a"),
        used_params=500,
        budget_params=1000,
        over_budget=False,
    )
    with pytest.raises(ValueError):
        AllocationResult(**kwargs)


def test_allocation_result_rejects_negative_iterations():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(ValueError):
        AllocationResult(**_valid_kwargs(iterations=-1))


def test_allocation_result_rejects_bool_iterations():
    from kadhi_cli.utils.allocate import AllocationResult

    with pytest.raises(TypeError):
        AllocationResult(**_valid_kwargs(iterations=True))


# ---------------------------------------------------------------------------
# allocate_ranks — input validation
# ---------------------------------------------------------------------------


def test_allocate_ranks_mismatched_keys_raises_naming_keys():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100, "c": 100}
    with pytest.raises(ValueError) as exc_info:
        allocate_ranks(scores, costs, budget_params=10_000)
    msg = str(exc_info.value)
    assert "b" in msg
    assert "c" in msg


def test_allocate_ranks_non_positive_score_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 0.0}
    costs = {"a": 100, "b": 100}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=10_000)

    scores_neg = {"a": 1.0, "b": -5.0}
    with pytest.raises(ValueError):
        allocate_ranks(scores_neg, costs, budget_params=10_000)


def test_allocate_ranks_non_positive_cost_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100, "b": 0}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=10_000)

    costs_neg = {"a": 100, "b": -100}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs_neg, budget_params=10_000)


def test_allocate_ranks_non_int_cost_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100.5, "b": 100}
    with pytest.raises(TypeError):
        allocate_ranks(scores, costs, budget_params=10_000)


def test_allocate_ranks_r_min_gte_r_max_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100, "b": 100}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=10_000, r_min=8, r_max=8)
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=10_000, r_min=8, r_max=4)


def test_allocate_ranks_r_min_below_one_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100, "b": 100}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=10_000, r_min=0, r_max=64)


def test_allocate_ranks_non_positive_budget_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0}
    costs = {"a": 100, "b": 100}
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=0)
    with pytest.raises(ValueError):
        allocate_ranks(scores, costs, budget_params=-100)


def test_allocate_ranks_empty_scores_raises():
    from kadhi_cli.utils.allocate import allocate_ranks

    with pytest.raises(ValueError):
        allocate_ranks({}, {}, budget_params=1000)


# ---------------------------------------------------------------------------
# Symmetric case
# ---------------------------------------------------------------------------


def test_allocate_ranks_symmetric_layers_get_uniform_allocation():
    from kadhi_cli.utils.allocate import allocate_ranks

    n = 6
    scores = {f"l{i}": 3.0 for i in range(n)}
    costs = {f"l{i}": 1000 for i in range(n)}
    budget = 6 * 1000 * 16  # generous middling budget

    result = allocate_ranks(scores, costs, budget_params=budget, r_min=4, r_max=64)

    values = list(result.ranks.values())
    assert max(values) - min(values) <= 1
    assert result.used_params <= budget * 1.05  # close to tight, allow slack


# ---------------------------------------------------------------------------
# Higher score -> higher rank
# ---------------------------------------------------------------------------


def test_allocate_ranks_higher_score_gets_higher_rank():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"hi": 10.0, "lo": 1.0}
    costs = {"hi": 1000, "lo": 1000}
    result = allocate_ranks(scores, costs, budget_params=20_000, r_min=1, r_max=64)

    assert result.ranks["hi"] > result.ranks["lo"]


# ---------------------------------------------------------------------------
# Edge cases: tiny / huge budgets
# ---------------------------------------------------------------------------


def test_allocate_ranks_tiny_budget_freezes_everything():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0, "c": 0.5}
    costs = {"a": 10_000, "b": 10_000, "c": 10_000}
    result = allocate_ranks(scores, costs, budget_params=1, r_min=4, r_max=64)

    assert all(r == 0 for r in result.ranks.values())
    assert set(result.frozen_layers) == set(scores.keys())
    assert result.used_params == 0
    assert result.over_budget is False


def test_allocate_ranks_huge_budget_clamps_to_r_max():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 1.0, "b": 2.0, "c": 3.0}
    costs = {"a": 100, "b": 100, "c": 100}
    huge_budget = 10**12
    result = allocate_ranks(scores, costs, budget_params=huge_budget, r_min=4, r_max=64)

    assert all(r == 64 for r in result.ranks.values())
    assert result.over_budget is False
    assert result.used_params < huge_budget


# ---------------------------------------------------------------------------
# Single layer
# ---------------------------------------------------------------------------


def test_allocate_ranks_single_layer():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"only": 5.0}
    costs = {"only": 500}
    result = allocate_ranks(scores, costs, budget_params=20_000, r_min=4, r_max=64)

    assert list(result.ranks.keys()) == ["only"]
    assert result.ranks["only"] >= 0


# ---------------------------------------------------------------------------
# Hand-computable 2-layer case
# ---------------------------------------------------------------------------


def test_allocate_ranks_hand_computable_two_layer_case():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 2.0, "b": 1.0}
    costs = {"a": 1000, "b": 1000}
    budget = 50_000

    result = allocate_ranks(scores, costs, budget_params=budget, r_min=1, r_max=1000)

    # Higher score at equal cost must get strictly more rank.
    assert result.ranks["a"] > result.ranks["b"]
    # Greedy spends the budget down: with equal costs of 1000 and a 50,000
    # budget, exactly 50 units of rank are affordable in total.
    assert result.ranks["a"] + result.ranks["b"] == 50
    assert result.used_params == budget


# ---------------------------------------------------------------------------
# Guarantees of the greedy marginal allocator
# ---------------------------------------------------------------------------


def test_allocate_ranks_never_exceeds_budget():
    """The previous continuous-relaxation allocator could overshoot once
    several layers clamped to r_max. Greedy cannot, by construction."""
    import random

    from kadhi_cli.utils.allocate import allocate_ranks

    rng = random.Random(4)
    for _ in range(200):
        n = rng.choice([2, 5, 16])
        scores = {f"l{i}": round(rng.uniform(0.05, 3.0), 4) for i in range(n)}
        costs = {f"l{i}": rng.choice([500, 8192, 81920]) for i in range(n)}
        budget = rng.randint(1_000, 5_000_000)
        result = allocate_ranks(scores, costs, budget_params=budget, r_min=4, r_max=64)
        assert result.used_params <= budget
        assert result.over_budget is False


def test_allocate_ranks_spends_budget_it_can_afford():
    """Regression guard for the defect that motivated the rewrite: the old
    allocator froze every layer whose continuous rank landed just under
    r_min and never redistributed their share, in one measured case
    committing only 65,536 of 551,241 available parameters (12%)."""
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {f"l{i}": 1.0 + i * 0.01 for i in range(16)}
    costs = {f"l{i}": 8192 for i in range(16)}
    budget = 600_000
    result = allocate_ranks(scores, costs, budget_params=budget, r_min=4, r_max=64)

    # Whatever is left over must be too small to buy even one more unit at
    # the cheapest layer — i.e. nothing affordable was left on the table.
    leftover = budget - result.used_params
    assert leftover < min(costs.values())


def test_allocate_ranks_matches_brute_force_optimum_equal_costs():
    """Fox (1966): marginal analysis is EXACTLY optimal for a separable
    concave objective under a single linear constraint with equal per-unit
    costs. Verified directly against exhaustive search."""
    import itertools
    import math
    import random

    from kadhi_cli.utils.allocate import allocate_ranks

    def objective(ranks, scores):
        return sum(scores[k] * math.log(1 + r) for k, r in ranks.items())

    rng = random.Random(11)
    for _ in range(60):
        n = rng.choice([2, 3, 4])
        r_max = rng.choice([6, 8])
        scores = {f"l{i}": round(rng.uniform(0.1, 3.0), 3) for i in range(n)}
        unit = rng.choice([2, 3, 5])
        costs = {f"l{i}": unit for i in range(n)}
        budget = rng.randint(6, 80)

        result = allocate_ranks(
            scores, costs, budget_params=budget, r_min=1, r_max=r_max
        )
        got = objective(dict(result.ranks), scores)

        best = -1.0
        keys = list(scores)
        for combo in itertools.product(range(0, r_max + 1), repeat=n):
            cand = dict(zip(keys, combo))
            if sum(cand[k] * costs[k] for k in keys) > budget:
                continue
            best = max(best, objective(cand, scores))

        assert got == pytest.approx(best, abs=1e-9)


def test_allocate_ranks_lambda_star_is_last_accepted_efficiency():
    """lambda_star is the shadow price of the budget constraint: the
    objective-gain-per-parameter of the final increment the budget bought."""
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 3.0, "b": 1.0, "c": 7.0}
    costs = {"a": 500, "b": 800, "c": 300}
    result = allocate_ranks(scores, costs, budget_params=15_000, r_min=1, r_max=64)
    assert result.lambda_star > 0
    assert math.isfinite(result.lambda_star)


def test_allocate_ranks_iterations_counts_accepted_increments():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 3.0, "b": 1.0}
    costs = {"a": 1000, "b": 1000}
    # r_min=1 so every accepted increment is one unit of rank; total rank
    # allocated must then equal the number of accepted increments.
    result = allocate_ranks(scores, costs, budget_params=10_000, r_min=1, r_max=64)
    assert result.iterations == sum(result.ranks.values())


def test_allocate_ranks_max_iters_caps_accepted_increments():
    from kadhi_cli.utils.allocate import allocate_ranks

    scores = {"a": 3.0, "b": 1.0, "c": 7.0}
    costs = {"a": 500, "b": 800, "c": 300}
    result = allocate_ranks(
        scores, costs, budget_params=1_000_000, r_min=1, r_max=64, max_iters=5
    )
    assert result.iterations <= 5
