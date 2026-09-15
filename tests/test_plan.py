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
    """Must run in a clean subprocess: other tests in this same file (the
    real-checkpoint end-to-end tests) legitimately import numpy/safetensors
    themselves, which would poison a same-process sys.modules check
    regardless of plan.py's own imports — this proves plan.py's TOP-LEVEL
    import graph specifically, independent of test execution order.
    """
    import subprocess
    import sys

    src_root = __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; "
        "from kadhi_cli.utils import plan; "
        "heavy = [m for m in ('torch', 'numpy', 'transformers', 'peft', 'safetensors') "
        "if m in sys.modules]; "
        "sys.exit(1) if heavy else sys.exit(0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**__import__("os").environ, "PYTHONPATH": str(src_root)},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"plan.py's top-level import pulled in a heavy dependency "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


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


# ---------------------------------------------------------------------------
# aggregate_layer_snr
# ---------------------------------------------------------------------------

def _snr(name, snr, shape=(4, 4)):
    from kadhi_cli.utils.spectrum_scan import LayerSNR

    return LayerSNR(name=name, module_type="attn", group="self_attn.q_proj", snr=snr, shape=shape)


def test_aggregate_layer_snr_averages_within_a_layer():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr("model.layers.0.self_attn.q_proj.weight", 2.0),
        _snr("model.layers.0.mlp.gate_proj.weight", 4.0),
        _snr("model.layers.1.self_attn.q_proj.weight", 10.0),
    ]
    scores = aggregate_layer_snr(records)
    assert scores == {"layer.0": 3.0, "layer.1": 10.0}


def test_aggregate_layer_snr_skips_unindexed_and_raises_if_all_skipped():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    with pytest.raises(ValueError, match="nothing to score"):
        aggregate_layer_snr([_snr("model.embed_tokens.weight", 5.0)])


# ---------------------------------------------------------------------------
# build_static_plan — the full real pipeline against a REAL safetensors
# checkpoint with REAL float32 data (not zero-byte fixtures): capacity.py's
# shape discovery + spectrum_scan's actual SVD-based SNR + allocate.py's
# actual bisection, composed exactly as build_static_plan wires them.
# ---------------------------------------------------------------------------

def _write_real_checkpoint(path, *, low_rank_layer: int, noise_layer: int, dim: int = 32):
    """A tiny 2-layer synthetic checkpoint with genuinely different spectral
    structure per layer, so compute_snr's real SVD produces a real,
    directionally-meaningful difference (not an arbitrary number).

    `low_rank_layer`'s q_proj is a rank-2 matrix (a few outer products) —
    almost all its singular-value mass sits in a couple of dominant
    components, which is exactly the "signal-heavy" shape compute_snr scores
    high. `noise_layer`'s q_proj is iid Gaussian noise — a flat singular-value
    spectrum with most mass below the Marchenko-Pastur threshold, which
    compute_snr scores low.
    """
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    u = rng.standard_normal((dim, 2)).astype(np.float32)
    v = rng.standard_normal((2, dim)).astype(np.float32)
    low_rank = (u @ v).astype(np.float32)
    noise = rng.standard_normal((dim, dim)).astype(np.float32)

    tensors = {
        f"model.layers.{low_rank_layer}.self_attn.q_proj.weight": low_rank,
        f"model.layers.{low_rank_layer}.self_attn.v_proj.weight": low_rank.copy(),
        f"model.layers.{noise_layer}.self_attn.q_proj.weight": noise,
        f"model.layers.{noise_layer}.self_attn.v_proj.weight": noise.copy(),
    }
    save_file(tensors, str(path))


def test_build_static_plan_end_to_end_on_real_checkpoint(tmp_path):
    pytest.importorskip("safetensors")
    np = pytest.importorskip("numpy")

    from kadhi_cli.utils.plan import build_static_plan

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)

    result = build_static_plan(
        str(tmp_path),
        target_modules=["q_proj", "v_proj"],
        budget_params=100_000,
        default_r=8,
        r_min=2,
        r_max=32,
    )

    layer0_ranks = [v for k, v in result.rank_pattern.items() if ".layers.0." in k]
    layer1_ranks = [v for k, v in result.rank_pattern.items() if ".layers.1." in k]
    r0 = layer0_ranks[0] if layer0_ranks else 8  # 8 == default_r, omitted if allocated
    r1 = layer1_ranks[0] if layer1_ranks else 8

    # The low-rank (signal-heavy) layer must score higher on real SNR and
    # therefore receive rank >= the noise layer's — this is the whole point
    # of the allocator, verified against REAL spectral math, not a hand-fed
    # score dict.
    assert r0 >= r1, (
        f"low-rank layer got rank {r0}, noise layer got rank {r1} — the "
        "real SNR signal should favor the structured layer"
    )


def test_build_static_plan_raises_on_no_matching_target_modules(tmp_path):
    from kadhi_cli.utils.plan import build_static_plan

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)

    with pytest.raises(ValueError):
        build_static_plan(
            str(tmp_path),
            target_modules=["definitely_not_a_real_module"],
            budget_params=100_000,
            default_r=8,
        )


# ---------------------------------------------------------------------------
# trainable_params_for_plan
# ---------------------------------------------------------------------------

def test_trainable_params_for_plan_uses_default_r_when_unmentioned():
    from kadhi_cli.utils.plan import PlanResult, trainable_params_for_plan

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)
    assert trainable_params_for_plan(shapes, pr) == 8 * 200


def test_trainable_params_for_plan_uses_explicit_rank_when_present():
    from kadhi_cli.utils.plan import PlanResult, trainable_params_for_plan

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={"model.layers.0.q_proj": 16}, frozen_module_paths=(), default_r=8)
    assert trainable_params_for_plan(shapes, pr) == 16 * 200


def test_trainable_params_for_plan_frozen_layer_contributes_zero():
    from kadhi_cli.utils.plan import PlanResult, trainable_params_for_plan

    shapes = [_shape("model.layers.0.q_proj", 100, 100), _shape("model.layers.1.q_proj", 100, 100)]
    pr = PlanResult(
        rank_pattern={"model.layers.1.q_proj": 16},
        frozen_module_paths=("model.layers.0.q_proj",),
        default_r=8,
    )
    # Without the frozen distinction this would wrongly price layer 0 at
    # default_r=8 instead of 0 — this is the exact pitfall the function's
    # docstring documents (a direct-lookup analogue of what PEFT-style
    # pattern re-matching would get wrong for the same reason).
    assert trainable_params_for_plan(shapes, pr) == 0 + 16 * 200


def test_trainable_params_for_plan_use_dora_adds_out_features():
    from kadhi_cli.utils.plan import PlanResult, trainable_params_for_plan

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={"model.layers.0.q_proj": 16}, frozen_module_paths=(), default_r=8)
    without_dora = trainable_params_for_plan(shapes, pr, use_dora=False)
    with_dora = trainable_params_for_plan(shapes, pr, use_dora=True)
    assert with_dora == without_dora + 100  # + out_features


def test_trainable_params_for_plan_matches_direct_hand_computation_from_a_real_allocation():
    """Ties this function back to the real allocate_ranks pipeline: build a
    real PlanResult via rank_pattern_from_allocation, then confirm
    trainable_params_for_plan's total equals a hand-summed total over the
    ORIGINAL allocation.ranks (the ground truth), proving the PlanResult
    round-trip doesn't lose or double-count anything.
    """
    from kadhi_cli.utils.allocate import allocate_ranks
    from kadhi_cli.utils.plan import (
        cost_per_unit_rank_from_shapes,
        rank_pattern_from_allocation,
        trainable_params_for_plan,
    )

    shapes = [
        _shape("model.layers.0.self_attn.q_proj", 100, 100),
        _shape("model.layers.1.self_attn.q_proj", 100, 100),
    ]
    cost = cost_per_unit_rank_from_shapes(shapes)
    scores = {"layer.0": 1.0, "layer.1": 5.0}
    allocation = allocate_ranks(scores, cost, budget_params=3000, r_min=2, r_max=32)
    plan_result = rank_pattern_from_allocation(shapes, allocation, default_r=8)

    ground_truth = sum(
        allocation.ranks[key] * cost[key] for key in allocation.ranks
    )
    assert trainable_params_for_plan(shapes, plan_result) == ground_truth
