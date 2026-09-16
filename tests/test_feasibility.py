"""Tests for kadhi_cli.utils.feasibility — the allocate <-> hardware_fit loop.

Everything here uses the REAL capacity/allocate/hardware_fit/plan modules
(no mocking) — all four are pure Python and already independently correct,
so this file's job is to prove the composition is correct, especially the
accounting subtlety plan.trainable_params_for_plan exists to avoid (see its
docstring and test_plan.py).
"""

from __future__ import annotations

import pytest


def _shape(name, in_features, out_features):
    from kadhi_cli.utils.capacity import LoraModuleShape

    return LoraModuleShape(name=name, in_features=in_features, out_features=out_features)


def _hw_base(params_b=8.0):
    from kadhi_cli.utils.hardware_fit import HardwareFitInput

    return HardwareFitInput(
        params_b=params_b, seq_len=2048, batch_size=1, optimizer="adamw_torch",
        quant="4bit", peft="lora", gradient_checkpointing=False,
    )


# ---------------------------------------------------------------------------
# check_plan_feasibility
# ---------------------------------------------------------------------------

def test_check_plan_feasibility_ok_case():
    from kadhi_cli.utils.feasibility import check_plan_feasibility
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)
    check = check_plan_feasibility(
        pr, shapes, use_dora=False,
        hardware_fit_base=_hw_base(params_b=0.001),  # tiny model, easily fits
        vram_gb=1000.0,
    )
    assert check.feasible is True
    assert check.hardware_report.ok is True


def test_check_plan_feasibility_infeasible_case():
    from kadhi_cli.utils.feasibility import check_plan_feasibility
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)
    check = check_plan_feasibility(
        pr, shapes, use_dora=False,
        hardware_fit_base=_hw_base(params_b=8.0),
        vram_gb=0.0001,  # impossibly small ceiling
    )
    assert check.feasible is False
    assert check.hardware_report.ok is False


def test_check_plan_feasibility_matches_trainable_params_for_plan_directly():
    from kadhi_cli.utils.feasibility import check_plan_feasibility
    from kadhi_cli.utils.plan import PlanResult, trainable_params_for_plan

    shapes = [_shape("model.layers.0.q_proj", 100, 100), _shape("model.layers.1.q_proj", 100, 100)]
    pr = PlanResult(rank_pattern={"model.layers.1.q_proj": 16}, frozen_module_paths=(), default_r=8)
    check = check_plan_feasibility(
        pr, shapes, use_dora=False, hardware_fit_base=_hw_base(), vram_gb=1000.0,
    )
    expected = trainable_params_for_plan(shapes, pr, use_dora=False)
    assert check.trainable_params == expected
    # Sanity: layer.0 falls back to default_r=8 (200 in+out), layer.1 is
    # explicitly allocated 16 (200 in+out) -> 8*200 + 16*200 = 4800.
    assert expected == 8 * 200 + 16 * 200


def test_check_plan_feasibility_frozen_layer_contributes_zero_not_default_r():
    """The exact bug plan.trainable_params_for_plan exists to avoid: a frozen
    layer must price at 0, not default_r, even though it's absent from
    rank_pattern (the same reason it would be absent if it were simply at
    the default) — frozen_module_paths is what disambiguates the two.
    """
    from kadhi_cli.utils.feasibility import check_plan_feasibility
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100), _shape("model.layers.1.q_proj", 100, 100)]
    pr = PlanResult(
        rank_pattern={"model.layers.1.q_proj": 16},
        frozen_module_paths=("model.layers.0.q_proj",),
        default_r=8,
    )
    check = check_plan_feasibility(
        pr, shapes, use_dora=False, hardware_fit_base=_hw_base(), vram_gb=1000.0,
    )
    # layer.0 frozen -> 0; layer.1 allocated 16 -> 200*16=3200. NOT 8*200+16*200.
    assert check.trainable_params == 16 * 200


def test_feasibility_check_validates_consistency():
    from kadhi_cli.utils.feasibility import FeasibilityCheck
    from kadhi_cli.utils.hardware_fit import HardwareFitReport, VRAMBreakdown

    breakdown = VRAMBreakdown(
        weights_gb=1, optimizer_gb=1, gradients_gb=1, activations_gb=1, overhead_gb=1,
    )
    report = HardwareFitReport(
        ok=True, peak_vram_gb=5, required_with_margin_gb=5.5,
        available_vram_gb=10, breakdown=breakdown, reason="fits",
    )
    with pytest.raises(ValueError, match="must match"):
        FeasibilityCheck(feasible=False, trainable_params=100, hardware_report=report)


# ---------------------------------------------------------------------------
# fit_plan_to_budget
# ---------------------------------------------------------------------------

def test_fit_plan_to_budget_succeeds_immediately_when_first_budget_fits():
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]

    def build(budget_params):
        return PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=1000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=0.001), vram_gb=1000.0,
    )
    assert result.iterations == 1
    assert result.check.feasible is True
    assert result.budget_params_used == 1000


def test_fit_plan_to_budget_finds_the_largest_feasible_budget():
    """Bisection must land on (or just under) the true feasibility frontier,
    not merely on the first budget that happens to fit. The previous
    geometric-shrink implementation could only ever land on 0.75**k of the
    starting budget, leaving up to 25% of feasible capacity unreachable."""
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 1000, 1000)]
    # Rank scales directly with budget, so cost is monotone in budget and a
    # true frontier exists.
    def build(budget_params):
        rank = max(0, min(64, budget_params // 2000))
        return PlanResult(
            rank_pattern={"model.layers.0.q_proj": rank} if rank not in (0, 8) else {},
            frozen_module_paths=() if rank else ("model.layers.0.q_proj",),
            default_r=8,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=10_000_000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=8.0), vram_gb=4.2, max_iters=40,
    )
    if result.check.feasible:
        # Anything materially larger than what we settled on must NOT fit —
        # that is what "largest feasible" means.
        from kadhi_cli.utils.feasibility import check_plan_feasibility

        bigger = int(result.budget_params_used * 1.5) + 4000
        worse = check_plan_feasibility(
            build(bigger), shapes, use_dora=False,
            hardware_fit_base=_hw_base(params_b=8.0), vram_gb=4.2,
        )
        assert not worse.feasible, (
            f"settled on {result.budget_params_used} but {bigger} also fits — "
            "bisection undershot the frontier"
        )


def test_fit_plan_to_budget_uses_one_evaluation_when_full_budget_fits():
    """The common case must not pay for a search it does not need."""
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]

    def build(budget_params):
        return PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=1000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=0.001), vram_gb=1000.0,
    )
    assert result.iterations == 1
    assert result.check.feasible is True
    assert result.budget_params_used == 1000


def test_fit_plan_to_budget_reports_last_attempt_when_nothing_is_feasible():
    """When even a fully-frozen allocation cannot fit, no rank allocation can
    help. Report the attempt (so the caller can show the real gap) instead of
    raising."""
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 1000, 1000)]

    def build(budget_params):
        return PlanResult(
            rank_pattern={"model.layers.0.q_proj": 64}, frozen_module_paths=(), default_r=8,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=1_000_000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=8.0), vram_gb=0.0001, max_iters=8,
    )
    assert result.check.feasible is False
    assert result.iterations >= 1
    assert result.budget_params_used >= 1


def test_fit_plan_to_budget_bounded_by_max_iters():
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 1000, 1000)]

    def build(budget_params):
        rank = max(0, min(64, budget_params // 2000))
        return PlanResult(
            rank_pattern={"model.layers.0.q_proj": rank} if rank not in (0, 8) else {},
            frozen_module_paths=() if rank else ("model.layers.0.q_proj",),
            default_r=8,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=10_000_000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=8.0), vram_gb=4.2, max_iters=5,
    )
    assert result.iterations <= 5


def test_fit_plan_to_budget_rejects_bad_inputs():
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import PlanResult

    shapes = [_shape("model.layers.0.q_proj", 100, 100)]

    def build(budget_params):
        return PlanResult(rank_pattern={}, frozen_module_paths=(), default_r=8)

    with pytest.raises(ValueError):
        fit_plan_to_budget(
            build, shapes, initial_budget_params=0, use_dora=False,
            hardware_fit_base=_hw_base(), vram_gb=10.0,
        )
    with pytest.raises(ValueError):
        fit_plan_to_budget(
            build, shapes, initial_budget_params=100, use_dora=False,
            hardware_fit_base=_hw_base(), vram_gb=10.0, max_iters=0,
        )
    with pytest.raises(TypeError):
        fit_plan_to_budget(
            build, shapes, initial_budget_params=True, use_dora=False,
            hardware_fit_base=_hw_base(), vram_gb=10.0,
        )


# ---------------------------------------------------------------------------
# real end-to-end: plan.build_static_plan feeding straight into the loop
# ---------------------------------------------------------------------------

def test_fit_plan_to_budget_with_real_build_static_plan(tmp_path):
    """Uses the actual safetensors-backed build_static_plan, proving the
    feasibility loop composes with the real static pipeline, not just a
    hand-built PlanResult closure."""
    import numpy as np
    from safetensors.numpy import save_file

    from kadhi_cli.utils.capacity import discover_lora_module_shapes
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import build_static_plan

    rng = np.random.default_rng(1)
    dim = 32
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((dim, dim)).astype("float32"),
        "model.layers.1.self_attn.q_proj.weight": rng.standard_normal((dim, dim)).astype("float32"),
    }
    shard = tmp_path / "model.safetensors"
    save_file(tensors, str(shard))

    shapes = discover_lora_module_shapes(str(tmp_path), ["q_proj"])

    def build(budget_params):
        return build_static_plan(
            str(tmp_path), ["q_proj"], budget_params=budget_params,
            default_r=8, r_min=2, r_max=32,
        )

    result = fit_plan_to_budget(
        build, shapes, initial_budget_params=100_000, use_dora=False,
        hardware_fit_base=_hw_base(params_b=0.001), vram_gb=1000.0,
        max_iters=3,
    )
    assert result.check.feasible is True
    assert result.iterations == 1
