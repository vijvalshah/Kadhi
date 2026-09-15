"""kadhi_cli.utils.allocate — concave-objective LoRA rank allocation under a
trainable-parameter budget.

See docs/adaptation-controller.md §3.2 for the design rationale (why the
objective must be concave, not linear) and adaptation-controller-plan.md §1.5
for why this module exists (lora.rank_pattern has validated consumers
throughout the codebase and no producer).

The problem solved here is::

    maximise   Σ_ℓ  s(ℓ) · log(1 + r(ℓ))
    subject to Σ_ℓ  cost(ℓ) · r(ℓ)  ≤  budget_params
               r(ℓ) ∈ {0} ∪ [r_min, r_max],  integer

**Algorithm: greedy marginal allocation** (Fox 1966, *Discrete optimization
via marginal analysis*; see also Federgruen & Groenevelt 1986). Starting from
every layer frozen, repeatedly spend the next slice of budget wherever it buys
the most objective per parameter, until nothing affordable remains.

*Optimality, stated precisely.* For a separable concave objective under a
single linear constraint with **equal per-unit costs**, marginal analysis is
exactly optimal. Verified here against brute force on 200 randomised small
instances: **200/200 exact** at ``r_min=1``. Every figure in this docstring is
reproduced by ``benchmarks/harness/allocator_optimality.py`` — run it rather
than trusting these numbers. Two caveats, both measured rather
than assumed:

* ``r_min > 1`` makes the feasible set ``{0} ∪ [r_min, r_max]`` non-convex
  (a semi-continuous variable), so greedy becomes a heuristic: 193/200 exact,
  worst observed shortfall 9.5%. That shortfall is an artefact of TOY sizes
  (2-4 layers, ``r_max`` 6-8), where one admission decision dominates the
  whole objective. Measured against a much stronger reference (greedy plus
  one restart per layer that force-admits that layer at ``r_min`` first),
  the gap collapses at realistic layer counts::

      layers:     4        8       16       32       64
      mean gap:   0.0078%  0.0000%  0.0002%  0.0005%  0.0000%
      max  gap:   0.3115%  0.0000%  0.0073%  0.0187%  0.0000%

  So the non-convexity is not worth engineering around: a repair pass would
  add real complexity to recover ~0.0005% on the sizes anyone actually
  trains. Documented rather than built.
* Heterogeneous per-layer costs turn the problem into a knapsack, where
  greedy-by-efficiency is likewise a heuristic (163/200 exact, worst
  shortfall 24.8%). In practice this rarely
  bites: every decoder layer of a standard dense transformer has identical
  target-module dimensions, so ``cost(ℓ)`` is the SAME for every layer (for
  Llama-3.1-8B over q/k/v/o/gate/up/down it is 81,920 parameters per unit of
  rank, at every layer). Heterogeneous costs arise only for irregular
  architectures such as MoE with varying expert counts.

*Why this replaced the previous approach.* The first implementation solved
the continuous relaxation by bisection on the Lagrange multiplier
(``r(ℓ;λ) = max(0, s(ℓ)/(λ·cost(ℓ)) − 1)``), then rounded and applied the
``r_min`` floor. That closed form is correct for the relaxation, but flooring
AFTER λ converged silently discarded budget: layers whose continuous rank
landed just under ``r_min`` were frozen and their share was never
redistributed. Measured over 12 randomised realistic instances, the greedy
allocator scores **+26% mean / +82% worst-case** higher on the objective, and
the old approach left as much as 88% of the budget unspent (one instance
committed 65,536 of 551,241 available parameters). The relaxation could also
*exceed* the budget once several layers clamped to ``r_max`` — it reported
that honestly via ``over_budget``, but reporting a violated constraint is
worse than not violating it. Greedy cannot exceed the budget by construction,
so ``over_budget`` is now always ``False`` and is retained only so the public
dataclass shape is stable.

Public surface:
- ``AllocationResult`` frozen dataclass.
- ``allocate_ranks(scores, cost_per_unit_rank, *, budget_params, r_min=4,
  r_max=64, max_iters=1_000_000)`` -> ``AllocationResult``.
"""

from __future__ import annotations

import heapq
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


def allocate_ranks(
    scores: "Mapping[str, float]",
    cost_per_unit_rank: "Mapping[str, int]",
    *,
    budget_params: int,
    r_min: int = 4,
    r_max: int = 64,
    max_iters: int = 1_000_000,
) -> AllocationResult:
    """Allocate integer LoRA rank per layer under a trainable-parameter budget.

    Maximises ``Σ_ℓ s(ℓ)·log(1+r(ℓ))`` subject to
    ``Σ_ℓ cost(ℓ)·r(ℓ) <= budget_params`` and
    ``r(ℓ) ∈ {0} ∪ [r_min, r_max]`` (integer), by greedy marginal analysis:
    repeatedly take the increment with the highest objective-gain-per-
    parameter that still fits. A layer's first increment is lumpy (0 ->
    ``r_min``, since intermediate ranks are inadmissible) and is priced on
    its average gain per parameter over that whole lump; later increments
    are single units. See the module docstring for the optimality analysis,
    the measured comparison against the previous approach, and why the
    objective is concave rather than linear.

    Guarantees:

    * ``used_params <= budget_params`` ALWAYS — no increment is accepted
      that would breach the budget, so ``over_budget`` is always ``False``
      (the field is retained only to keep the public dataclass stable).
    * The budget is spent down as far as it can be: the loop only stops when
      every remaining candidate is either unaffordable or already at
      ``r_max``. It does not terminate at the first unaffordable candidate,
      because with unequal ``cost(ℓ)`` a cheaper increment further down the
      heap may still fit.
    * Deterministic: equal-efficiency candidates are broken by layer key.

    ``lambda_star`` is the efficiency (objective gain per parameter) of the
    LAST ACCEPTED increment — the shadow price of the budget constraint at
    this allocation. When nothing was affordable at all, it falls back to
    the best REJECTED efficiency, i.e. the price the budget would have had
    to support, which keeps it strictly positive and interpretable.
    ``iterations`` is the number of accepted increments.

    `scores` and `cost_per_unit_rank` MUST have identical key sets — raises
    ``ValueError`` naming the mismatched keys otherwise (mismatched dicts
    indicate a caller bug that should be surfaced, not silently unioned or
    intersected). Every score must be > 0 — a layer with a non-positive
    score should simply be OMITTED by the caller, not passed in at 0; a
    non-positive score raises ``ValueError`` rather than being silently
    treated as "freeze" (that ambiguity would hide a caller bug). Every cost
    must be a positive int, since a zero or negative cost would make
    gain-per-parameter meaningless (and infinite).

    ``r_min`` must be >= 1 and ``r_max`` must be > ``r_min`` (raises
    ``ValueError`` otherwise). ``budget_params`` must be a positive int.
    ``max_iters`` caps accepted increments as a defensive bound; the loop is
    already structurally bounded by ``len(scores) * r_max``, so the default
    is set far above any realistic allocation and exists only to guarantee
    termination if that invariant is ever broken by a future change.

    Edge cases (all legitimate, none raise):
    - A budget too small to admit even one layer at ``r_min``: every rank is
      0 (fully frozen). Deciding whether that is acceptable is the caller's
      job, not this function's.
    - A budget large enough for every layer to reach ``r_max``: every rank
      clamps there and ``used_params`` lands well under ``budget_params``,
      because there is nothing left worth buying.
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

    # ------------------------------------------------------------------
    # Greedy marginal allocation (Fox 1966, "Discrete optimization via
    # marginal analysis"). Repeatedly spend the next unit of budget wherever
    # it buys the most objective per parameter, until nothing affordable is
    # left. See this module's docstring for the optimality analysis and the
    # measured comparison against the continuous-relaxation approach this
    # replaced.
    #
    # Heap entries are (-efficiency, key, step, step_cost) where efficiency
    # is objective-gain-per-parameter for that candidate increment. `key` is
    # a str and acts as a deterministic tiebreaker, so equal-efficiency
    # candidates resolve in a stable order rather than raising on an
    # uncomparable 3rd element.
    #
    # Invariant: at most ONE live heap entry per layer at any time (one is
    # seeded per layer, and accepting one pushes at most its single
    # successor). A dropped entry is never replaced, which is correct in
    # both drop cases:
    #   * r_max reached      -> that layer can never take another increment.
    #   * unaffordable       -> `spent` only ever grows and the step cost is
    #                           fixed, so an unaffordable step stays
    #                           unaffordable forever.
    # Together these bound the loop at L + L*r_max pops, so it terminates.
    heap: "list[tuple[float, str, int, int]]" = []
    for key, score in scores_f.items():
        # A layer's FIRST increment is lumpy: it must jump straight to r_min
        # (ranks below r_min are not admissible), so it is priced as the
        # average gain per parameter over that whole lump. Pricing it as a
        # single unit step would systematically over-rate admitting new
        # layers relative to deepening existing ones.
        lump_gain = score * math.log1p(r_min)
        lump_cost = r_min * costs[key]
        heapq.heappush(heap, (-lump_gain / lump_cost, key, r_min, lump_cost))

    ranks: "dict[str, int]" = {key: 0 for key in scores_f}
    spent = 0
    accepted = 0
    # The efficiency of the last ACCEPTED increment is the shadow price of
    # the budget constraint at this allocation. If nothing is affordable at
    # all, fall back to the best REJECTED efficiency — "the price the budget
    # would have had to support" — which keeps lambda_star strictly positive
    # and meaningful rather than an arbitrary sentinel.
    last_accepted_efficiency = 0.0
    best_rejected_efficiency = 0.0

    while heap and accepted < max_iters:
        neg_efficiency, key, step, step_cost = heapq.heappop(heap)
        efficiency = -neg_efficiency
        if ranks[key] + step > r_max:
            continue
        if spent + step_cost > budget_params:
            best_rejected_efficiency = max(best_rejected_efficiency, efficiency)
            continue
        ranks[key] += step
        spent += step_cost
        accepted += 1
        last_accepted_efficiency = efficiency
        if ranks[key] < r_max:
            current = ranks[key]
            unit_gain = scores_f[key] * (math.log1p(current + 1) - math.log1p(current))
            heapq.heappush(
                heap, (-unit_gain / costs[key], key, 1, costs[key])
            )

    used_params = sum(costs[k] * r for k, r in ranks.items())
    # Structurally guaranteed by the affordability check above: no increment
    # is ever accepted that would push `spent` past the budget. Asserted
    # rather than merely documented, because the previous algorithm COULD
    # exceed the budget and this is the property that replaced that defect.
    if used_params > budget_params:  # pragma: no cover - defensive invariant
        raise AssertionError(
            f"greedy allocation exceeded its budget ({used_params} > "
            f"{budget_params}); this is a bug in allocate_ranks"
        )

    lambda_star = last_accepted_efficiency or best_rejected_efficiency
    if not (lambda_star > 0 and math.isfinite(lambda_star)):
        # Only reachable if every candidate was rejected for r_max reasons
        # with no affordability rejection recorded; keep the documented
        # "strictly positive, finite" contract of the dataclass.
        lambda_star = min(
            score / costs[key] for key, score in scores_f.items()
        )

    frozen_layers = tuple(sorted(k for k, v in ranks.items() if v == 0))

    return AllocationResult(
        ranks=ranks,
        lambda_star=float(lambda_star),
        used_params=used_params,
        budget_params=budget_params,
        over_budget=False,  # structurally impossible; see the guard above
        frozen_layers=frozen_layers,
        iterations=accepted,
    )


__all__ = [
    "AllocationResult",
    "allocate_ranks",
]
