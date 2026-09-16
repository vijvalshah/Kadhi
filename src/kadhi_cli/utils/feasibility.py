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
    max_iters: int = 24,
) -> FitResult:
    """Find the LARGEST feasible trainable-parameter budget, by bisection.

    ``build_plan_fn(budget_params) -> PlanResult`` is supplied by the caller
    rather than fixed to ``plan.build_static_plan``, so this loop works
    identically whether the scores came from the static SNR path or a live
    gradient probe — this module owns the feasibility search, not how a plan
    gets produced. **The caller must do the expensive measurement (an SNR
    scan, or a gradient probe) ONCE, outside its closure**, and have
    ``build_plan_fn`` only re-run the cheap allocation; see
    ``plan.compute_layer_signals`` / ``plan.plan_from_signals``. A closure
    that calls ``plan.build_static_plan`` per invocation re-scans the whole
    model on every iteration.

    *Why bisection rather than shrinking by a fixed factor.* The previous
    implementation multiplied the budget by 0.75 on each rejection. That was
    wrong in two ways. It **undershoots**: the first budget that happens to
    fit after geometric decay is not the largest that would have fit, so
    capacity is silently thrown away (0.75 decay can only ever land on
    0.75^k of the original, so up to 25% of the feasible budget is
    unreachable by construction). And it **fails to find feasible plans that
    exist**: from a 50,000,000-parameter starting budget, six 0.75 steps only
    reach 11,865,234 — measured on this codebase, that run reported "not
    feasible" while feasible allocations existed far below.

    Bisection is valid here because feasibility is monotone in the budget: a
    larger ``budget_params`` lets the allocator spend at least as much, which
    can only increase trainable parameters, which can only increase predicted
    peak VRAM. So if budget B fits, every smaller budget fits too, and the
    feasible set is an interval ``[0, B*]``. ``max_iters`` bisection steps
    locate ``B*`` to within ``initial_budget_params / 2**max_iters`` — at the
    default of 24, that is a relative precision of ~6e-8, far finer than the
    granularity of a single unit of rank.

    Returns the best FEASIBLE plan found. If none is feasible — including the
    degenerate case where even a fully-frozen allocation does not fit, which
    means no rank allocation can ever help — returns the last infeasible
    attempt rather than raising, so the caller can inspect
    ``FitResult.check`` and report the actual gap. Raising would discard
    exactly the information that makes an infeasible result actionable.
    """
    if isinstance(initial_budget_params, bool):
        raise TypeError("initial_budget_params must be int, not bool")
    if not isinstance(initial_budget_params, int) or initial_budget_params <= 0:
        raise ValueError(
            f"initial_budget_params must be a positive int, got {initial_budget_params!r}"
        )
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError(f"max_iters must be a positive int, got {max_iters!r}")

    iterations = 0

    def attempt(budget: int) -> "tuple[PlanResult, FeasibilityCheck]":
        nonlocal iterations
        iterations += 1
        plan_result = build_plan_fn(budget)
        return plan_result, check_plan_feasibility(
            plan_result, shapes, use_dora=use_dora,
            hardware_fit_base=hardware_fit_base, vram_gb=vram_gb,
        )

    # The full budget is the best possible answer; try it before bisecting so
    # the common case (it fits) costs exactly one evaluation.
    plan_result, check = attempt(initial_budget_params)
    if check.feasible:
        return FitResult(
            plan=plan_result, check=check,
            budget_params_used=initial_budget_params, iterations=iterations,
        )

    best: "Optional[tuple[PlanResult, FeasibilityCheck, int]]" = None
    last = (plan_result, check, initial_budget_params)
    lo, hi = 1, initial_budget_params  # hi is known infeasible from above
    while lo < hi and iterations < max_iters:
        mid = (lo + hi) // 2
        if mid == hi:  # integer bisection converged
            break
        plan_result, check = attempt(mid)
        last = (plan_result, check, mid)
        if check.feasible:
            best = (plan_result, check, mid)
            lo = mid + 1
        else:
            hi = mid

    plan_result, check, budget_used = best if best is not None else last
    return FitResult(
        plan=plan_result, check=check,
        budget_params_used=budget_used, iterations=iterations,
    )
