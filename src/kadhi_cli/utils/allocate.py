"""kadhi_cli.utils.allocate — concave-objective LoRA rank allocation under a
trainable-parameter budget.

See docs/adaptation-controller.md §3.2 for the design rationale (why the
objective must be concave, not linear) and adaptation-controller-plan.md §1.5
for why this module exists (lora.rank_pattern has validated consumers
throughout the codebase and no producer).

Public surface:
- ``AllocationResult`` frozen dataclass.
- ``allocate_ranks(scores, cost_per_unit_rank, *, budget_params, r_min=4,
  r_max=64, tolerance=1e-6, max_iters=100)`` -> ``AllocationResult``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AllocationResult:
    """The outcome of one :func:`allocate_ranks` call.

    ``ranks`` maps each caller-supplied layer key to its allocated integer
    rank (``0`` means frozen — not adapted at all). ``lambda_star`` is the
    converged Lagrange multiplier on the budget constraint. ``used_params``
    is the EXACT trainable-parameter cost of the final integer ``ranks``
    (``Σ cost(ℓ)·r(ℓ)``), which is not guaranteed to equal ``budget_params``
    because of rounding and box clamping — see ``over_budget``.
    """

    ranks: "Mapping[str, int]"
    lambda_star: float
    used_params: int
    budget_params: int
    over_budget: bool
    frozen_layers: "tuple[str, ...]"
    iterations: int

    def __post_init__(self) -> None:
        if not isinstance(self.ranks, Mapping):
            raise TypeError(f"ranks must be a Mapping, got {type(self.ranks).__name__}")
        if not self.ranks:
            raise ValueError("ranks must not be empty")
        for key, value in self.ranks.items():
            if not isinstance(key, str):
                raise TypeError(f"ranks keys must be str, got {type(key).__name__}")
            if isinstance(value, bool):
                raise TypeError(f"ranks[{key!r}] must be int, not bool")
            if not isinstance(value, int):
                raise TypeError(
                    f"ranks[{key!r}] must be int, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"ranks[{key!r}] must be >= 0, got {value}")

        if isinstance(self.lambda_star, bool):
            raise TypeError("lambda_star must be float, not bool")
        if not isinstance(self.lambda_star, (int, float)):
            raise TypeError(
                f"lambda_star must be a real number, got {type(self.lambda_star).__name__}"
            )
        if not math.isfinite(self.lambda_star):
            raise ValueError(f"lambda_star must be finite, got {self.lambda_star}")
        if self.lambda_star <= 0:
            raise ValueError(f"lambda_star must be > 0, got {self.lambda_star}")

        for field_name in ("used_params", "budget_params"):
            val = getattr(self, field_name)
            if isinstance(val, bool):
                raise TypeError(f"{field_name} must be int, not bool")
            if not isinstance(val, int):
                raise TypeError(f"{field_name} must be int, got {type(val).__name__}")
            if val < 0:
                raise ValueError(f"{field_name} must be >= 0, got {val}")

        if not isinstance(self.over_budget, bool):
            raise TypeError(
                f"over_budget must be bool, got {type(self.over_budget).__name__}"
            )
        expected_over_budget = self.used_params > self.budget_params
        if self.over_budget != expected_over_budget:
            raise ValueError(
                "over_budget is inconsistent with used_params/budget_params: "
                f"used_params={self.used_params}, budget_params={self.budget_params}, "
                f"over_budget={self.over_budget} (expected {expected_over_budget})"
            )

        if not isinstance(self.frozen_layers, tuple):
            raise TypeError(
                f"frozen_layers must be a tuple, got {type(self.frozen_layers).__name__}"
            )
        for key in self.frozen_layers:
            if not isinstance(key, str):
                raise TypeError(
                    f"frozen_layers entries must be str, got {type(key).__name__}"
                )
        expected_frozen = tuple(sorted(k for k, v in self.ranks.items() if v == 0))
        if self.frozen_layers != expected_frozen:
            raise ValueError(
                f"frozen_layers must be exactly the sorted keys with rank 0; "
                f"got {self.frozen_layers!r}, expected {expected_frozen!r}"
            )

        if isinstance(self.iterations, bool):
            raise TypeError("iterations must be int, not bool")
        if not isinstance(self.iterations, int):
            raise TypeError(
                f"iterations must be int, got {type(self.iterations).__name__}"
            )
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")


def _continuous_rank(score: float, lam: float, cost: int) -> float:
    """Unconstrained closed-form ``r(ℓ; λ) = max(0, s(ℓ)/(λ·cost(ℓ)) − 1)``."""
    return max(0.0, score / (lam * cost) - 1.0)


def _budget_used_at(
    lam: float, scores: Mapping[str, float], costs: Mapping[str, int]
) -> float:
    total = 0.0
    for key, score in scores.items():
        total += costs[key] * _continuous_rank(score, lam, costs[key])
    return total


def allocate_ranks(
    scores: "Mapping[str, float]",
    cost_per_unit_rank: "Mapping[str, int]",
    *,
    budget_params: int,
    r_min: int = 4,
    r_max: int = 64,
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> AllocationResult:
    """Allocate integer LoRA rank per layer under a trainable-parameter budget.

    Maximises the concave objective ``Σ_ℓ s(ℓ)·log(1+r(ℓ))`` subject to
    ``Σ_ℓ cost(ℓ)·r(ℓ) <= budget_params`` and ``r(ℓ) ∈ {0} ∪ [r_min, r_max]``
    (integer). A linear objective under this linear constraint is degenerate
    — it dumps the whole budget into the single highest-scoring layer — so
    the concave ``log(1+r)`` term (diminishing returns per layer) is used
    instead, per docs/adaptation-controller.md §3.2.

    Derivation (see module/plan docs for the full write-up): relaxing
    integrality and the box constraint, Lagrangian stationarity against a
    shared multiplier ``λ`` on the budget constraint gives the closed form
    ``r(ℓ; λ) = max(0, s(ℓ)/(λ·cost(ℓ)) − 1)``. The implied budget usage
    ``Σ cost(ℓ)·r(ℓ; λ)`` is monotonically non-increasing in ``λ``, so the
    unique ``λ*`` solving the budget constraint at equality is found by
    bisection. The final integer allocation is then obtained per layer by:
    rounding ``r(ℓ; λ*)`` to the nearest integer, freezing (``r(ℓ) = 0``) if
    that rounds below ``r_min``, clamping to ``r_max`` if it exceeds it, and
    forcing ``r(ℓ) = 0`` for any non-positive score (defensive; such scores
    should not be passed in at all — see below).

    `scores` and `cost_per_unit_rank` MUST have identical key sets — raises
    ``ValueError`` naming the mismatched keys otherwise (mismatched dicts
    indicate a caller bug that should be surfaced, not silently unioned or
    intersected). Every score must be > 0 — a layer with a non-positive
    score should simply be OMITTED by the caller, not passed in at 0; a
    non-positive score raises ``ValueError`` rather than being silently
    treated as "freeze" (that ambiguity would hide a caller bug). Every cost
    must be a positive int — a zero or negative cost breaks the monotonicity
    the bisection relies on and raises ``ValueError``.

    ``r_min`` must be >= 1 and ``r_max`` must be > ``r_min`` (raises
    ``ValueError`` otherwise). ``budget_params`` must be a positive int.

    Because rounding and clamping happen AFTER ``λ*`` converges, the final
    allocation generally does not hit ``budget_params`` exactly:
    ``used_params`` (computed from the FINAL integer ranks) may land
    slightly under or over the budget. If many layers clamp to ``r_max``,
    ``used_params`` can exceed ``budget_params`` — this is reported honestly
    via ``over_budget`` rather than corrected by a secondary "shave layers
    down" pass, which is out of scope for this pure allocation function.

    Edge cases (all legitimate, none raise):
    - A budget so small every layer rounds below ``r_min``: every rank is 0
      (fully frozen). Deciding whether that is acceptable is the caller's
      job (a future ``kadhi plan``), not this function's.
    - A budget so large every layer wants to exceed ``r_max``: every rank
      clamps to ``r_max``; ``used_params`` is typically well under
      ``budget_params`` and ``over_budget`` is False.
    """
    if not isinstance(budget_params, int) or isinstance(budget_params, bool):
        raise TypeError(
            f"budget_params must be int, got {type(budget_params).__name__}"
        )
    if budget_params <= 0:
        raise ValueError(f"budget_params must be > 0, got {budget_params}")

    if not isinstance(r_min, int) or isinstance(r_min, bool):
        raise TypeError(f"r_min must be int, got {type(r_min).__name__}")
    if not isinstance(r_max, int) or isinstance(r_max, bool):
        raise TypeError(f"r_max must be int, got {type(r_max).__name__}")
    if r_min < 1:
        raise ValueError(f"r_min must be >= 1, got {r_min}")
    if r_max <= r_min:
        raise ValueError(f"r_max must be > r_min, got r_max={r_max}, r_min={r_min}")

    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError(f"tolerance must be float, got {type(tolerance).__name__}")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"tolerance must be finite and > 0, got {tolerance}")

    if not isinstance(max_iters, int) or isinstance(max_iters, bool):
        raise TypeError(f"max_iters must be int, got {type(max_iters).__name__}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be > 0, got {max_iters}")

    if not isinstance(scores, Mapping):
        raise TypeError(f"scores must be a Mapping, got {type(scores).__name__}")
    if not isinstance(cost_per_unit_rank, Mapping):
        raise TypeError(
            f"cost_per_unit_rank must be a Mapping, got {type(cost_per_unit_rank).__name__}"
        )
    if not scores:
        raise ValueError("scores must not be empty")

    score_keys = set(scores.keys())
    cost_keys = set(cost_per_unit_rank.keys())
    if score_keys != cost_keys:
        missing_from_cost = sorted(score_keys - cost_keys)
        missing_from_scores = sorted(cost_keys - score_keys)
        details = []
        if missing_from_cost:
            details.append(
                f"in scores but missing from cost_per_unit_rank: {missing_from_cost}"
            )
        if missing_from_scores:
            details.append(
                f"in cost_per_unit_rank but missing from scores: {missing_from_scores}"
            )
        raise ValueError(
            "scores and cost_per_unit_rank must have identical key sets; "
            + "; ".join(details)
        )

    for key, score in scores.items():
        if isinstance(score, bool):
            raise TypeError(f"scores[{key!r}] must be a real number, not bool")
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"scores[{key!r}] must be a real number, got {type(score).__name__}"
            )
        if not math.isfinite(score):
            raise ValueError(f"scores[{key!r}] must be finite, got {score}")
        if score <= 0:
            raise ValueError(
                f"scores[{key!r}] = {score} is non-positive; omit non-positive-score "
                "layers from the input instead of passing them at <= 0"
            )

    for key, cost in cost_per_unit_rank.items():
        if isinstance(cost, bool):
            raise TypeError(f"cost_per_unit_rank[{key!r}] must be int, not bool")
        if not isinstance(cost, int):
            raise TypeError(
                f"cost_per_unit_rank[{key!r}] must be int, got {type(cost).__name__}"
            )
        if cost <= 0:
            raise ValueError(
                f"cost_per_unit_rank[{key!r}] must be > 0, got {cost}"
            )

    scores_f = {k: float(v) for k, v in scores.items()}
    costs = dict(cost_per_unit_rank)

    # Bracket lambda so the bisection always has a valid root inside it,
    # regardless of r_min/r_max: for lambda small enough, the UNCLAMPED
    # continuous usage sum_i [score_i/lambda - cost_i] (every r_i(lambda) > 0
    # since lambda is tiny) exceeds budget_params; solving that inequality
    # for lambda gives a safe, budget-aware lambda_lo. For lambda large
    # enough (>= max score_i/cost_i), every r_i(lambda) <= 0, so usage is
    # exactly 0 <= budget_params. This brackets the true root purely from
    # budget_params and the inputs, independent of r_min/r_max.
    sum_scores = sum(scores_f.values())
    sum_costs = sum(costs.values())
    lambda_lo = sum_scores / (budget_params + sum_costs + 1)
    lambda_lo = max(lambda_lo, 1e-300)
    lambda_hi = max(score / costs[key] for key, score in scores_f.items())
    if lambda_hi <= lambda_lo:
        lambda_hi = lambda_lo * 2.0 + 1e-300

    def used_at(lam: float) -> float:
        return _budget_used_at(lam, scores_f, costs)

    lo, hi = lambda_lo, lambda_hi
    lam_star = hi
    iterations = 0
    for iterations in range(1, max_iters + 1):
        mid = (lo + hi) / 2.0
        usage = used_at(mid)
        diff = usage - budget_params
        lam_star = mid
        if abs(diff) <= tolerance * max(1.0, budget_params):
            break
        if usage > budget_params:
            # too much spending at this lambda -> need larger lambda
            lo = mid
        else:
            hi = mid

    ranks: dict = {}
    for key, score in scores_f.items():
        if score <= 0:
            ranks[key] = 0
            continue
        continuous = _continuous_rank(score, lam_star, costs[key])
        rounded = round(continuous)
        if rounded < r_min:
            ranks[key] = 0
        elif rounded > r_max:
            ranks[key] = r_max
        else:
            ranks[key] = int(rounded)

    used_params = sum(costs[k] * r for k, r in ranks.items())
    over_budget = used_params > budget_params
    frozen_layers = tuple(sorted(k for k, v in ranks.items() if v == 0))

    return AllocationResult(
        ranks=ranks,
        lambda_star=lam_star,
        used_params=used_params,
        budget_params=budget_params,
        over_budget=over_budget,
        frozen_layers=frozen_layers,
        iterations=iterations,
    )


__all__ = [
    "AllocationResult",
    "allocate_ranks",
]
