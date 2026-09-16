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
    """Aggregates per layer — but over MEAN-NORMALIZED values, not raw SNR.

    Raw SNR is not comparable across module types (its magnitude depends on
    matrix shape — see aggregate_layer_snr's docstring), so each
    layer_type_signature group is divided by its own mean across layers
    before the per-layer average:

      group "self_attn.q_proj" = {2.0, 10.0}, mean 6.0 -> 1/3, 5/3
      group "mlp.gate_proj"    = {4.0},       mean 4.0 -> 1.0

      layer.0 = (1/3 + 1.0) / 2 = 2/3
      layer.1 = 5/3
    """
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr("model.layers.0.self_attn.q_proj.weight", 2.0),
        _snr("model.layers.0.mlp.gate_proj.weight", 4.0),
        _snr("model.layers.1.self_attn.q_proj.weight", 10.0),
    ]
    scores = aggregate_layer_snr(records)
    assert set(scores) == {"layer.0", "layer.1"}
    assert scores["layer.0"] == pytest.approx(2.0 / 3.0)
    assert scores["layer.1"] == pytest.approx(5.0 / 3.0)


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


# ---------------------------------------------------------------------------
# aggregate_layer_snr — the module-type scale correction (Defect 1)
#
# Measured on this machine with an IDENTICAL generative process (same low-rank
# signal + same noise), varying only the SHAPE:
#     square (512x512)  -> SNR ~0.0060
#     wide  (1376x512)  -> SNR ~0.0411   (~7x)
#     tall  (512x1376)  -> SNR ~0.0402   (~7x)
# so in a Llama-style model the rectangular MLP matrices dominate the
# square-ish attention ones in a plain mean, and a layer's score reflects
# essentially only its MLP SNR. These tests pin the fix.
# ---------------------------------------------------------------------------

def _snr_named(name, snr):
    """A LayerSNR whose ``group`` matches its own name (the real scan does
    this too) — aggregate_layer_snr derives the group from the NAME via
    spectrum_scan.layer_type_signature, not from this field.
    """
    from kadhi_cli.utils.spectrum_scan import LayerSNR, layer_type_signature

    return LayerSNR(
        name=name,
        module_type="mlp" if ".mlp." in name else "attn",
        group=layer_type_signature(name),
        snr=snr,
        shape=(4, 4),
    )


def test_aggregate_layer_snr_normalization_flips_a_plain_mean_ranking():
    """The test that proves the fix does something.

    Two layers x two module types, with the MLP group's raw SNR ~10x the
    attention group's — exactly the shape-driven scale gap measured above.

        layer.0:  q_proj =  10,  gate_proj = 100
        layer.1:  q_proj =   1,  gate_proj = 180

    PLAIN MEAN (the old, broken behaviour):
        layer.0 = (10 + 100)/2 = 55.0
        layer.1 = (1  + 180)/2 = 90.5     ->  layer.1 > layer.0

    MEAN-NORMALIZED PER GROUP (correct):
        q_proj mean    = (10 + 1)/2   =   5.5
        gate_proj mean = (100 + 180)/2 = 140.0
        layer.0 = (10/5.5 + 100/140)/2 = (1.8182 + 0.7143)/2 = 1.2662
        layer.1 = ( 1/5.5 + 180/140)/2 = (0.1818 + 1.2857)/2 = 0.7338
                                          ->  layer.0 > layer.1

    The two orderings are OPPOSITE, so this test genuinely fails against a
    plain mean. layer.0 is the right answer: it is 10x better than layer.1 at
    q_proj while being only ~1.8x worse at gate_proj — a fact the raw mean
    cannot see because gate_proj's absolute magnitude swamps q_proj's.
    """
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr_named("model.layers.0.self_attn.q_proj.weight", 10.0),
        _snr_named("model.layers.0.mlp.gate_proj.weight", 100.0),
        _snr_named("model.layers.1.self_attn.q_proj.weight", 1.0),
        _snr_named("model.layers.1.mlp.gate_proj.weight", 180.0),
    ]

    # Sanity: confirm the plain mean really does rank them the other way, so
    # this test cannot silently pass against a regression to the old code.
    plain = {
        "layer.0": (10.0 + 100.0) / 2,
        "layer.1": (1.0 + 180.0) / 2,
    }
    assert plain["layer.1"] > plain["layer.0"]

    scores = aggregate_layer_snr(records)
    assert scores["layer.0"] == pytest.approx((10 / 5.5 + 100 / 140) / 2)
    assert scores["layer.1"] == pytest.approx((1 / 5.5 + 180 / 140) / 2)
    assert scores["layer.0"] > scores["layer.1"], (
        "mean-normalization must rank the layer that is better RELATIVE TO "
        "ITS PEERS within each module type first, not the one whose MLP "
        "matrices happen to carry the larger absolute SNR"
    )


def test_aggregate_layer_snr_scores_are_all_finite_and_strictly_positive():
    import math

    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr_named("model.layers.0.self_attn.q_proj.weight", 1e-9),
        _snr_named("model.layers.0.mlp.down_proj.weight", 12345.0),
        _snr_named("model.layers.1.self_attn.q_proj.weight", 500.0),
        _snr_named("model.layers.1.mlp.down_proj.weight", 1e-9),
        _snr_named("model.layers.2.self_attn.q_proj.weight", 3.0),
        _snr_named("model.layers.2.mlp.down_proj.weight", 4.0),
    ]
    scores = aggregate_layer_snr(records)
    assert len(scores) == 3
    for key, value in scores.items():
        assert math.isfinite(value), f"{key} -> {value!r} is not finite"
        assert value > 0.0, f"{key} -> {value!r} is not strictly positive"


def test_aggregate_layer_snr_skips_an_all_zero_module_type_group():
    """A group whose mean SNR is 0 must be skipped whole — never divided by,
    which would emit NaN/inf, and never emitted as 0.0, which
    allocate_ranks would later reject with a confusing far-away error.
    """
    import math

    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        # This whole group is dead — every member is 0.0.
        _snr_named("model.layers.0.self_attn.q_proj.weight", 0.0),
        _snr_named("model.layers.1.self_attn.q_proj.weight", 0.0),
        # This group is healthy and carries both layers on its own.
        _snr_named("model.layers.0.mlp.gate_proj.weight", 1.0),
        _snr_named("model.layers.1.mlp.gate_proj.weight", 2.0),
    ]
    scores = aggregate_layer_snr(records)
    assert set(scores) == {"layer.0", "layer.1"}
    # gate_proj mean = 1.5 -> 1/1.5, 2/1.5 (the dead group contributes
    # nothing at all, rather than dragging both layers to 0.5x).
    assert scores["layer.0"] == pytest.approx(1.0 / 1.5)
    assert scores["layer.1"] == pytest.approx(2.0 / 1.5)
    for value in scores.values():
        assert math.isfinite(value) and value > 0.0


def test_aggregate_layer_snr_omits_a_layer_left_with_no_contributing_matrices():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        # layer.0's ONLY matrix belongs to the dead all-zero q_proj group.
        _snr_named("model.layers.0.self_attn.q_proj.weight", 0.0),
        _snr_named("model.layers.1.self_attn.q_proj.weight", 0.0),
        # layer.1 also has a healthy matrix, so it survives.
        _snr_named("model.layers.1.mlp.gate_proj.weight", 5.0),
    ]
    scores = aggregate_layer_snr(records)
    assert set(scores) == {"layer.1"}, (
        "layer.0 lost every contributing matrix and must be OMITTED, not "
        "emitted with a 0.0 score"
    )
    assert scores["layer.1"] == pytest.approx(1.0)  # 5.0 / mean(5.0)


def test_aggregate_layer_snr_all_non_finite_raises():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr_named("model.layers.0.self_attn.q_proj.weight", float("nan")),
        _snr_named("model.layers.1.self_attn.q_proj.weight", float("inf")),
    ]
    with pytest.raises(ValueError, match="nothing to score"):
        aggregate_layer_snr(records)


def test_aggregate_layer_snr_skips_individual_non_finite_records():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr_named("model.layers.0.self_attn.q_proj.weight", float("nan")),
        _snr_named("model.layers.1.self_attn.q_proj.weight", 4.0),
    ]
    scores = aggregate_layer_snr(records)
    # The NaN record is dropped before the group mean is taken, so the group
    # mean is 4.0 (not NaN) and layer.1 normalizes to exactly 1.0.
    assert set(scores) == {"layer.1"}
    assert scores["layer.1"] == pytest.approx(1.0)


def test_aggregate_layer_snr_all_groups_dead_raises():
    from kadhi_cli.utils.plan import aggregate_layer_snr

    records = [
        _snr_named("model.layers.0.self_attn.q_proj.weight", 0.0),
        _snr_named("model.layers.1.mlp.gate_proj.weight", 0.0),
    ]
    with pytest.raises(ValueError, match="nothing to score"):
        aggregate_layer_snr(records)


# ---------------------------------------------------------------------------
# LayerSignals validation
# ---------------------------------------------------------------------------

def _signals_kwargs():
    return {
        "shapes": (_shape("model.layers.0.self_attn.q_proj", 100, 100),),
        "scores": {"layer.0": 1.0},
        "cost_per_unit_rank": {"layer.0": 200},
    }


def test_layer_signals_accepts_valid():
    from kadhi_cli.utils.plan import LayerSignals

    signals = LayerSignals(**_signals_kwargs())
    assert signals.scores == {"layer.0": 1.0}
    assert signals.cost_per_unit_rank == {"layer.0": 200}
    assert len(signals.shapes) == 1


def test_layer_signals_rejects_mismatched_key_sets():
    from kadhi_cli.utils.plan import LayerSignals

    kwargs = _signals_kwargs()
    kwargs["cost_per_unit_rank"] = {"layer.1": 200}
    with pytest.raises(ValueError, match="identical key sets"):
        LayerSignals(**kwargs)


def test_layer_signals_rejects_non_positive_score():
    from kadhi_cli.utils.plan import LayerSignals

    kwargs = _signals_kwargs()
    kwargs["scores"] = {"layer.0": 0.0}
    with pytest.raises(ValueError, match="finite and > 0"):
        LayerSignals(**kwargs)

    kwargs["scores"] = {"layer.0": float("nan")}
    with pytest.raises(ValueError, match="finite and > 0"):
        LayerSignals(**kwargs)


def test_layer_signals_rejects_empty_shapes():
    from kadhi_cli.utils.plan import LayerSignals

    kwargs = _signals_kwargs()
    kwargs["shapes"] = ()
    with pytest.raises(ValueError, match="shapes must not be empty"):
        LayerSignals(**kwargs)


def test_layer_signals_rejects_bool_and_non_positive_cost():
    from kadhi_cli.utils.plan import LayerSignals

    kwargs = _signals_kwargs()
    kwargs["cost_per_unit_rank"] = {"layer.0": True}
    with pytest.raises(TypeError, match="not bool"):
        LayerSignals(**kwargs)

    kwargs["cost_per_unit_rank"] = {"layer.0": 0}
    with pytest.raises(ValueError, match="must be > 0"):
        LayerSignals(**kwargs)


def test_layer_signals_rejects_empty_mappings():
    from kadhi_cli.utils.plan import LayerSignals

    kwargs = _signals_kwargs()
    kwargs["scores"] = {}
    kwargs["cost_per_unit_rank"] = {}
    with pytest.raises(ValueError, match="must not be empty"):
        LayerSignals(**kwargs)


# ---------------------------------------------------------------------------
# compute_layer_signals + plan_from_signals (Defect 2) — against a REAL
# synthetic safetensors checkpoint.
# ---------------------------------------------------------------------------

def test_two_step_api_matches_build_static_plan_exactly(tmp_path):
    pytest.importorskip("safetensors")
    pytest.importorskip("numpy")

    from kadhi_cli.utils.plan import (
        build_static_plan,
        compute_layer_signals,
        plan_from_signals,
    )

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)

    kwargs = dict(budget_params=100_000, default_r=8, r_min=2, r_max=32)

    one_shot = build_static_plan(
        str(tmp_path), target_modules=["q_proj", "v_proj"], **kwargs
    )
    signals = compute_layer_signals(str(tmp_path), ["q_proj", "v_proj"])
    two_step = plan_from_signals(signals, **kwargs)

    assert dict(two_step.rank_pattern) == dict(one_shot.rank_pattern)
    assert two_step.frozen_module_paths == one_shot.frozen_module_paths
    assert two_step.default_r == one_shot.default_r
    # The signals themselves are internally consistent by construction.
    assert set(signals.scores) == set(signals.cost_per_unit_rank)
    assert signals.shapes


def test_plan_from_signals_rejects_a_non_layer_signals_argument():
    from kadhi_cli.utils.plan import plan_from_signals

    with pytest.raises(TypeError, match="LayerSignals"):
        plan_from_signals({"scores": {}}, budget_params=1000, default_r=8)


def test_plan_from_signals_does_no_model_io(tmp_path, monkeypatch):
    """plan_from_signals must never touch the weights — proven by making the
    scan explode if it is called at all."""
    pytest.importorskip("safetensors")
    pytest.importorskip("numpy")

    from kadhi_cli.utils import spectrum_scan
    from kadhi_cli.utils.plan import compute_layer_signals, plan_from_signals

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)
    signals = compute_layer_signals(str(tmp_path), ["q_proj", "v_proj"])

    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("plan_from_signals must not scan weights")

    monkeypatch.setattr(spectrum_scan, "scan_weights_dir", boom)
    for budget in (100_000, 50_000, 10_000):
        plan_from_signals(signals, budget_params=budget, default_r=8, r_min=2, r_max=32)


# ---------------------------------------------------------------------------
# The performance fix itself: the SVD scan must run ONCE across a whole
# feasibility shrink loop, not once per iteration.
# ---------------------------------------------------------------------------

def _hw_base(params_b=8.0):
    from kadhi_cli.utils.hardware_fit import HardwareFitInput

    return HardwareFitInput(
        params_b=params_b, seq_len=2048, batch_size=1, optimizer="adamw_torch",
        quant="4bit", peft="lora", gradient_checkpointing=False,
    )


def _counting_scan(monkeypatch):
    """Wrap spectrum_scan.scan_weights_dir in a call counter. plan.py imports
    it lazily inside the function body, so patching the module attribute is
    what the real call site actually resolves.
    """
    from kadhi_cli.utils import spectrum_scan

    real = spectrum_scan.scan_weights_dir
    calls = []

    def counting(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(spectrum_scan, "scan_weights_dir", counting)
    return calls


def test_feasibility_loop_scans_the_model_exactly_once_with_the_two_step_api(
    tmp_path, monkeypatch,
):
    pytest.importorskip("safetensors")
    pytest.importorskip("numpy")

    from kadhi_cli.utils.capacity import discover_lora_module_shapes
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import compute_layer_signals, plan_from_signals

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)
    shapes = discover_lora_module_shapes(str(tmp_path), ["q_proj", "v_proj"])

    calls = _counting_scan(monkeypatch)

    # The expensive half, hoisted OUT of the loop.
    signals = compute_layer_signals(str(tmp_path), ["q_proj", "v_proj"])
    assert len(calls) == 1

    def build(budget_params):
        return plan_from_signals(
            signals, budget_params=budget_params, default_r=8, r_min=2, r_max=32,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=100_000, use_dora=False,
        # An impossible ceiling, so the loop never succeeds and burns every
        # one of its max_iters shrink iterations.
        hardware_fit_base=_hw_base(params_b=8.0), vram_gb=0.0001,
        max_iters=6,
    )
    assert result.check.feasible is False
    assert result.iterations == 6
    assert len(calls) == 1, (
        f"the SVD scan ran {len(calls)} times across {result.iterations} "
        "feasibility iterations — compute_layer_signals must be called once "
        "and reused"
    )


def test_old_pattern_rescans_once_per_iteration_pinning_the_regression(
    tmp_path, monkeypatch,
):
    """Pins the measured defect: a closure calling build_static_plan re-runs
    the ENTIRE SVD scan on every shrink iteration (6 scans for 6 iterations,
    when 1 is correct). If someone reverts commands/allocate.py to that
    pattern, the test above fails and this one explains why.
    """
    pytest.importorskip("safetensors")
    pytest.importorskip("numpy")

    from kadhi_cli.utils.capacity import discover_lora_module_shapes
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import build_static_plan

    shard = tmp_path / "model.safetensors"
    _write_real_checkpoint(shard, low_rank_layer=0, noise_layer=1)
    shapes = discover_lora_module_shapes(str(tmp_path), ["q_proj", "v_proj"])

    calls = _counting_scan(monkeypatch)

    def build(budget_params):
        return build_static_plan(
            str(tmp_path), ["q_proj", "v_proj"], budget_params=budget_params,
            default_r=8, r_min=2, r_max=32,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=100_000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=8.0), vram_gb=0.0001,
        max_iters=6,
    )
    assert result.iterations == 6
    assert len(calls) == result.iterations == 6
