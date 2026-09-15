"""Tests for kadhi_cli.utils.plan — the capacity/sensitivity/allocate bridge.

Unlike sensitivity.py's tests, everything here is REAL: capacity.py and
allocate.py are both pure-Python and already correct/tested, so this file
constructs real ``LoraModuleShape`` and ``AllocationResult`` objects and
exercises the actual glue with zero mocking.
"""

from __future__ import annotations

import pytest


def _shape(name, in_features, out_features):
    from kadhi_cli.utils.capacity import LoraModuleShape

    return LoraModuleShape(name=name, in_features=in_features, out_features=out_features)


def _allocation(ranks, *, budget_params=10_000):
    from kadhi_cli.utils.allocate import AllocationResult

    used = sum(v for v in ranks.values())  # cost is irrelevant to these tests
    return AllocationResult(
        ranks=ranks,
        lambda_star=1.0,
        used_params=used,
        budget_params=budget_params,
        over_budget=used > budget_params,
        frozen_layers=tuple(sorted(k for k, v in ranks.items() if v == 0)),
        iterations=1,
    )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------

def test_module_imports():
    from kadhi_cli.utils import plan

    assert hasattr(plan, "layer_key")
    assert hasattr(plan, "group_shapes_by_layer")
    assert hasattr(plan, "cost_per_unit_rank_from_shapes")
    assert hasattr(plan, "PlanResult")
    assert hasattr(plan, "rank_pattern_from_allocation")


def test_no_heavy_deps_imported():
    import sys

    from kadhi_cli.utils import plan  # noqa: F401

    for heavy in ("torch", "numpy", "transformers", "peft", "safetensors"):
        assert heavy not in sys.modules, f"{heavy} should not be imported by plan.py"


# ---------------------------------------------------------------------------
# layer_key
# ---------------------------------------------------------------------------

def test_layer_key_format():
    from kadhi_cli.utils.plan import layer_key

    assert layer_key(0) == "layer.0"
    assert layer_key(12) == "layer.12"


def test_layer_key_rejects_bool():
    from kadhi_cli.utils.plan import layer_key

    with pytest.raises(TypeError, match="bool"):
        layer_key(True)


def test_layer_key_rejects_negative():
    from kadhi_cli.utils.plan import layer_key

    with pytest.raises(ValueError, match=">= 0"):
        layer_key(-1)


def test_layer_key_rejects_non_int():
    from kadhi_cli.utils.plan import layer_key

    with pytest.raises(TypeError):
        layer_key("0")


# ---------------------------------------------------------------------------
# group_shapes_by_layer
# ---------------------------------------------------------------------------

def test_group_shapes_by_layer_basic():
    from kadhi_cli.utils.plan import group_shapes_by_layer

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 4096, 4096),
        _shape("model.layers.0.mlp.gate_proj", 4096, 11008),
        _shape("model.layers.1.self_attn.q_proj", 4096, 4096),
    ]
    grouped = group_shapes_by_layer(shapes)
    assert set(grouped) == {"layer.0", "layer.1"}
    assert len(grouped["layer.0"]) == 2
    assert len(grouped["layer.1"]) == 1


def test_group_shapes_by_layer_skips_unindexed_modules():
    from kadhi_cli.utils.plan import group_shapes_by_layer

    shapes = [
        _shape("model.embed_tokens", 32000, 4096),  # no layers.N. segment
        _shape("model.layers.3.self_attn.q_proj", 4096, 4096),
    ]
    grouped = group_shapes_by_layer(shapes)
    assert set(grouped) == {"layer.3"}


def test_group_shapes_by_layer_gpt2_h_naming():
    from kadhi_cli.utils.plan import group_shapes_by_layer

    shapes = [_shape("transformer.h.2.attn.c_attn", 768, 2304)]
    grouped = group_shapes_by_layer(shapes)
    assert set(grouped) == {"layer.2"}


def test_group_shapes_by_layer_empty_raises():
    from kadhi_cli.utils.plan import group_shapes_by_layer

    with pytest.raises(ValueError, match="nothing to allocate"):
        group_shapes_by_layer([_shape("model.embed_tokens", 32000, 4096)])


# ---------------------------------------------------------------------------
# cost_per_unit_rank_from_shapes
# ---------------------------------------------------------------------------

def test_cost_per_unit_rank_sums_in_plus_out():
    from kadhi_cli.utils.plan import cost_per_unit_rank_from_shapes

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 4096, 4096),   # 8192
        _shape("model.layers.0.mlp.gate_proj", 4096, 11008),      # 15104
    ]
    cost = cost_per_unit_rank_from_shapes(shapes)
    assert cost == {"layer.0": 4096 + 4096 + 4096 + 11008}


def test_cost_per_unit_rank_multi_layer():
    from kadhi_cli.utils.plan import cost_per_unit_rank_from_shapes

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.1.self_attn.q_proj", 200, 200),
    ]
    cost = cost_per_unit_rank_from_shapes(shapes)
    assert cost == {"layer.0": 200, "layer.1": 400}


# ---------------------------------------------------------------------------
# PlanResult validation
# ---------------------------------------------------------------------------

def test_plan_result_rejects_zero_rank_pattern_entry():
    from kadhi_cli.utils.plan import PlanResult

    with pytest.raises(ValueError):
        PlanResult(rank_pattern={"x": 0}, frozen_module_paths=(), default_r=8)


def test_plan_result_rejects_bool_rank():
    from kadhi_cli.utils.plan import PlanResult

    with pytest.raises(TypeError):
        PlanResult(rank_pattern={"x": True}, frozen_module_paths=(), default_r=8)


def test_plan_result_rejects_non_positive_default_r():
    from kadhi_cli.utils.plan import PlanResult

    with pytest.raises(ValueError):
        PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=0)


def test_plan_result_accepts_valid():
    from kadhi_cli.utils.plan import PlanResult

    pr = PlanResult(rank_pattern={"model.layers.0.q_proj": 16}, frozen_module_paths=("x",), default_r=8)
    assert pr.rank_pattern["model.layers.0.q_proj"] == 16


# ---------------------------------------------------------------------------
# rank_pattern_from_allocation — the real end-to-end bridge
# ---------------------------------------------------------------------------

def test_rank_pattern_omits_default_rank_layers():
    from kadhi_cli.utils.plan import rank_pattern_from_allocation

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.1.self_attn.q_proj", 100, 100),
    ]
    allocation = _allocation({"layer.0": 8, "layer.1": 32})  # 8 == default_r
    result = rank_pattern_from_allocation(shapes, allocation, default_r=8)
    assert "model.layers.0.self_attn.q_proj" not in result.rank_pattern
    assert result.rank_pattern["model.layers.1.self_attn.q_proj"] == 32
    assert result.frozen_module_paths == ()


def test_rank_pattern_expands_every_module_at_a_layer():
    from kadhi_cli.utils.plan import rank_pattern_from_allocation

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.0.self_attn.v_proj", 100, 100),
        _shape("model.layers.0.mlp.gate_proj", 100, 200),
    ]
    allocation = _allocation({"layer.0": 16})
    result = rank_pattern_from_allocation(shapes, allocation, default_r=8)
    assert result.rank_pattern == {
        "model.layers.0.self_attn.q_proj": 16,
        "model.layers.0.self_attn.v_proj": 16,
        "model.layers.0.mlp.gate_proj": 16,
    }


def test_rank_pattern_frozen_layer_goes_to_frozen_paths_not_rank_pattern():
    from kadhi_cli.utils.plan import rank_pattern_from_allocation

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.1.self_attn.q_proj", 100, 100),
    ]
    allocation = _allocation({"layer.0": 0, "layer.1": 16})
    result = rank_pattern_from_allocation(shapes, allocation, default_r=8)
    assert "model.layers.0.self_attn.q_proj" not in result.rank_pattern
    assert result.frozen_module_paths == ("model.layers.0.self_attn.q_proj",)
    assert result.rank_pattern == {"model.layers.1.self_attn.q_proj": 16}


def test_rank_pattern_mismatched_keys_raises_naming_both_directions():
    from kadhi_cli.utils.plan import rank_pattern_from_allocation

    shapes = [_shape("model.layers.0.self_attn.q_proj", 100, 100)]
    # allocation has layer.5, which does not exist in shapes; shapes has
    # layer.0, which is absent from the allocation.
    allocation = _allocation({"layer.5": 16})
    with pytest.raises(ValueError, match="layer.0"):
        rank_pattern_from_allocation(shapes, allocation, default_r=8)


def test_rank_pattern_rejects_bad_default_r():
    from kadhi_cli.utils.plan import rank_pattern_from_allocation

    shapes = [_shape("model.layers.0.self_attn.q_proj", 100, 100)]
    allocation = _allocation({"layer.0": 16})
    with pytest.raises(ValueError):
        rank_pattern_from_allocation(shapes, allocation, default_r=0)
    with pytest.raises(TypeError):
        rank_pattern_from_allocation(shapes, allocation, default_r=True)


def test_full_pipeline_capacity_to_allocate_to_plan():
    """The real end-to-end path: shapes -> cost -> allocate_ranks -> plan.

    This is the strongest test in this file: it runs allocate.allocate_ranks
    for real (not a hand-built AllocationResult) against costs derived from
    real LoraModuleShape objects, then bridges the result into a rank_pattern
    — proving the three modules actually compose, not just that each has
    internally-consistent unit tests.
    """
    from kadhi_cli.utils.allocate import allocate_ranks
    from kadhi_cli.utils.plan import cost_per_unit_rank_from_shapes, rank_pattern_from_allocation

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.0.self_attn.v_proj", 100, 100),
        _shape("model.layers.1.self_attn.q_proj", 100, 100),
        _shape("model.layers.1.self_attn.v_proj", 100, 100),
    ]
    cost = cost_per_unit_rank_from_shapes(shapes)
    # layer.1 is far more "important" than layer.0 under this fake score.
    scores = {"layer.0": 1.0, "layer.1": 10.0}

    allocation = allocate_ranks(
        scores, cost, budget_params=4000, r_min=2, r_max=32,
    )
    result = rank_pattern_from_allocation(shapes, allocation, default_r=8)

    layer0_ranks = {
        v for k, v in result.rank_pattern.items() if "layers.0." in k
    } or {8}  # 8 == default_r, omitted if that's what was allocated
    layer1_ranks = {
        v for k, v in result.rank_pattern.items() if "layers.1." in k
    } or {8}
    # The higher-scoring layer must not end up with a strictly lower rank.
    assert max(layer1_ranks) >= max(layer0_ranks)
