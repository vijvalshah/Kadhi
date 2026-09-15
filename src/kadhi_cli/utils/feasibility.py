"""kadhi_cli.utils.feasibility — closes the loop between the capacity
allocator and the resource model.

See docs/adaptation-controller.md §3.3 ("Resource model") and
adaptation-controller-plan.md §1.5's remaining gap: ``plan.build_static_plan``
can emit a ``rank_pattern`` the device cannot actually fit — nothing
previously checked an allocation against ``hardware_fit`` before handing it
back, so the VRAM ceiling was a config field with no enforcement behind it.

Accounting for a :class:`plan.PlanResult`'s exact trainable-parameter cost is
delegated to ``plan.trainable_params_for_plan`` — see that function's
docstring for a real correctness pitfall it exists to avoid (routing a
``PlanResult``'s own emitted ``rank_pattern`` back through PEFT-style suffix
matching against the SAME shape names it was built from can never
self-match, and silently falls back to ``default_r`` for every module the
allocator actually changed; direct dict lookup sidesteps this because the
mapping here is already known to be exact, not a fuzzy pattern).

Public surface:
- ``FeasibilityCheck`` frozen dataclass.
- ``check_plan_feasibility(plan_result, shapes, *, use_dora,
  hardware_fit_base, vram_gb)`` -> FeasibilityCheck.
- ``FitResult`` frozen dataclass.
- ``fit_plan_to_budget(build_plan_fn, *, initial_budget_params, ...)`` ->
  FitResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:
    from kadhi_cli.utils.capacity import LoraModuleShape
    from kadhi_cli.utils.hardware_fit import HardwareFitInput, HardwareFitReport
    from kadhi_cli.utils.plan import PlanResult


@dataclass(frozen=True)
class FeasibilityCheck:
    """The outcome of checking one plan against the resource model."""

    feasible: bool
    trainable_params: int
    hardware_report: "HardwareFitReport"

    def __post_init__(self) -> None:
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be bool")
        if isinstance(self.trainable_params, bool):
            raise TypeError("trainable_params must be int, not bool")
        if not isinstance(self.trainable_params, int):
            raise TypeError(
                f"trainable_params must be int, got {type(self.trainable_params).__name__}"
            )
        if self.trainable_params < 0:
            raise ValueError(f"trainable_params must be >= 0, got {self.trainable_params}")
        # hardware_report's own type/validity is enforced by HardwareFitReport
        # itself (imported lazily below in the function that builds one) —
        # this dataclass only re-checks the field it derives from it.
        if self.feasible != bool(self.hardware_report.ok):
            raise ValueError(
                f"feasible ({self.feasible}) must match hardware_report.ok "
                f"({self.hardware_report.ok})"
            )


def check_plan_feasibility(
    plan_result: "PlanResult",
    shapes: "Sequence[LoraModuleShape]",
    *,
    use_dora: bool,
    hardware_fit_base: "HardwareFitInput",
    vram_gb: float,
) -> FeasibilityCheck:
    """Check one plan's EXACT trainable-parameter cost against the resource
    model.

    ``hardware_fit_base`` supplies every field ``HardwareFitInput`` needs
    EXCEPT ``trainable_params`` (that field is computed here, exactly, from
    ``plan_result`` — any value already set on ``hardware_fit_base`` for it
    is overwritten, not read). ``vram_gb`` is the feasibility ceiling —
    passed separately rather than read off ``hardware_fit_base`` because a
    caller shrinking the SEARCH budget across iterations (see
    ``fit_plan_to_budget``) keeps this ceiling fixed throughout.
    """
    from kadhi_cli.utils.hardware_fit import HardwareFitInput, decide_hardware_fit
    from kadhi_cli.utils.plan import trainable_params_for_plan

    trainable_params = trainable_params_for_plan(shapes, plan_result, use_dora=use_dora)
    inp = HardwareFitInput(
        params_b=hardware_fit_base.params_b,
        seq_len=hardware_fit_base.seq_len,
        batch_size=hardware_fit_base.batch_size,
        optimizer=hardware_fit_base.optimizer,
        quant=hardware_fit_base.quant,
        peft=hardware_fit_base.peft,
        gradient_checkpointing=hardware_fit_base.gradient_checkpointing,
        trainable_params=trainable_params,
    )
    report = decide_hardware_fit(inp, available_vram_gb=vram_gb)
    return FeasibilityCheck(
        feasible=report.ok, trainable_params=trainable_params, hardware_report=report,
    )


@dataclass(frozen=True)
class FitResult:
    """The outcome of :func:`fit_plan_to_budget`."""

    plan: "PlanResult"
    check: FeasibilityCheck
    budget_params_used: int
    iterations: int

    def __post_init__(self) -> None:
        if isinstance(self.budget_params_used, bool):
            raise TypeError("budget_params_used must be int, not bool")
        if not isinstance(self.budget_params_used, int) or self.budget_params_used <= 0:
            raise ValueError(
                f"budget_params_used must be a positive int, got {self.budget_params_used!r}"
            )
        if isinstance(self.iterations, bool):
            raise TypeError("iterations must be int, not bool")
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError(f"iterations must be a positive int, got {self.iterations!r}")


def fit_plan_to_budget(
    build_plan_fn: "Callable[[int], PlanResult]",
    shapes: "Sequence[LoraModuleShape]",
    *,
    initial_budget_params: int,
    use_dora: bool,
    hardware_fit_base: "HardwareFitInput",
    vram_gb: float,
    shrink_factor: float = 0.75,
    max_iters: int = 6,
) -> FitResult:
    """Allocate, check feasibility, shrink the budget and retry on rejection.

    ``build_plan_fn(budget_params) -> PlanResult`` is supplied by the caller
    rather than fixed to ``plan.build_static_plan`` so this loop works
    identically whether the underlying scores came from the static SNR path
    or a live gradient probe — this module only owns the feasibility
    check-and-shrink, not how a plan gets produced. The expensive part of
    producing a plan (an SNR scan, or a gradient probe) should be done ONCE
    by the caller's closure and reused across iterations; only the
    allocation itself needs to re-run per shrink, which is exactly what a
    closure over already-computed scores/costs lets ``build_plan_fn`` do
    cheaply.

    Stops as soon as a feasible plan is found. If ``max_iters`` is exhausted
    without one, returns the LAST (still-infeasible) attempt rather than
    raising — the caller can inspect ``FitResult.check.feasible`` and decide
    what to do; silently raising would hide exactly the information (how
    close did it get, what was the final gap) that makes an infeasible
    result actionable rather than merely a failure.
    """
    if isinstance(initial_budget_params, bool):
        raise TypeError("initial_budget_params must be int, not bool")
    if not isinstance(initial_budget_params, int) or initial_budget_params <= 0:
        raise ValueError(
            f"initial_budget_params must be a positive int, got {initial_budget_params!r}"
        )
    if not (0.0 < shrink_factor < 1.0):
        raise ValueError(f"shrink_factor must be in (0, 1), got {shrink_factor!r}")
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError(f"max_iters must be a positive int, got {max_iters!r}")

    next_budget = initial_budget_params
    plan_result: Optional["PlanResult"] = None
    check: Optional[FeasibilityCheck] = None
    # `used_budget` always names the budget that actually produced
    # `plan_result`/`check` in the current loop body — kept distinct from
    # `next_budget` (the shrunk value queued for the FOLLOWING iteration) so
    # the final return, however the loop exits, reports the budget that was
    # really used rather than one shrink-step ahead of it.
    used_budget = next_budget
    for iteration in range(1, max_iters + 1):
        used_budget = next_budget
        plan_result = build_plan_fn(used_budget)
        check = check_plan_feasibility(
            plan_result, shapes, use_dora=use_dora,
            hardware_fit_base=hardware_fit_base, vram_gb=vram_gb,
        )
        if check.feasible:
            return FitResult(
                plan=plan_result, check=check, budget_params_used=used_budget, iterations=iteration,
            )
        next_budget = max(1, int(used_budget * shrink_factor))
        if next_budget == used_budget:
            # Shrinking stalled (budget already at the floor) — stop rather
            # than loop uselessly to max_iters on an unchanging budget.
            return FitResult(
                plan=plan_result, check=check, budget_params_used=used_budget, iterations=iteration,
            )

    # max_iters exhausted, never feasible — plan_result/check/used_budget are
    # all set from the last iteration actually run (the loop always runs at
    # least once, since max_iters >= 1 is validated above).
    assert plan_result is not None and check is not None  # noqa: S101 — loop invariant
    return FitResult(
        plan=plan_result, check=check, budget_params_used=used_budget, iterations=max_iters,
    )
