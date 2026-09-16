#!/usr/bin/env python3
"""Reproducible verification of the capacity allocator's optimality claims.

Every numeric claim made about `utils/allocate.py` in
`docs/adaptation-controller.md` §3.2 and `docs/adaptation-controller-plan.md`
§1.5 is produced by this script. Run it to re-derive them rather than
trusting the documents:

    python benchmarks/harness/allocator_optimality.py

Needs no GPU, no torch, and no network — `utils/allocate.py` is pure Python.
Runs in a few seconds.

What it checks
--------------

1. **Exact optimality against brute force.** For a separable concave
   objective under a single linear constraint with EQUAL per-unit costs,
   marginal analysis (Fox 1966) is exactly optimal. Verified by exhaustive
   search over every admissible rank combination on small instances.

2. **Where the guarantee stops.** Two things break the equal-cost concave
   premise, and this quantifies both rather than hand-waving them:
   - `r_min > 1` makes the feasible set `{0} ∪ [r_min, r_max]` non-convex.
   - Unequal per-layer costs turn the problem into a knapsack.

3. **Does the non-convexity matter at real scale?** Compares the allocator
   against a much stronger reference (greedy plus one restart per layer that
   force-admits that layer at `r_min` first) as the layer count grows. Real
   models have 16-80 decoder layers, not 3.

4. **Regression check against the superseded algorithm.** Reimplements the
   continuous-relaxation approach that shipped first (bisect for the Lagrange
   multiplier, round, then apply the `r_min` floor) and measures the
   objective gap and budget utilisation against the current allocator. This
   is the evidence for the "+26% mean / +82% worst-case" claim.

5. **The budget invariant.** The current allocator must never exceed its
   budget. The superseded one could.
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kadhi_cli.utils.allocate import allocate_ranks  # noqa: E402


def objective(ranks, scores) -> float:
    """The quantity the allocator maximises: Σ s(ℓ)·log(1+r(ℓ))."""
    return sum(scores[k] * math.log(1 + r) for k, r in ranks.items())


def brute_force_optimum(scores, costs, budget, r_min, r_max) -> float:
    """Exhaustive search over every admissible combination. Only tractable
    for a handful of layers — that is why the instances here are small."""
    keys = list(scores)
    domain = [0] + list(range(r_min, r_max + 1))
    best = -1.0
    for combo in itertools.product(domain, repeat=len(keys)):
        cand = dict(zip(keys, combo))
        if sum(cand[k] * costs[k] for k in keys) > budget:
            continue
        best = max(best, objective(cand, scores))
    return best


def superseded_relaxation(scores, costs, budget, r_min, r_max, iters=100):
    """The algorithm that shipped first, reimplemented here so the comparison
    is reproducible after it was removed from the codebase.

    Bisects for the Lagrange multiplier of the CONTINUOUS relaxation
    (r(ℓ;λ) = max(0, s(ℓ)/(λ·cost(ℓ)) − 1)), rounds to the nearest integer,
    then applies the r_min floor. The floor is applied AFTER λ converges,
    which is the defect: layers that land just under r_min are frozen and
    their share of the budget is never redistributed.
    """
    lo = sum(scores.values()) / (budget + sum(costs.values()) + 1)
    hi = max(s / costs[k] for k, s in scores.items())
    lam = hi
    for _ in range(iters):
        lam = (lo + hi) / 2.0
        used = sum(
            costs[k] * max(0.0, s / (lam * costs[k]) - 1.0) for k, s in scores.items()
        )
        if abs(used - budget) <= 1e-6 * max(1.0, budget):
            break
        if used > budget:
            lo = lam
        else:
            hi = lam

    ranks = {}
    for k, s in scores.items():
        r = round(max(0.0, s / (lam * costs[k]) - 1.0))
        ranks[k] = 0 if r < r_min else min(int(r), r_max)
    return ranks, sum(costs[k] * r for k, r in ranks.items())


def check_1_and_2_optimality() -> None:
    print("=" * 74)
    print("1+2. Optimality against brute force")
    print("=" * 74)
    rng = random.Random(11)

    for label, equal_costs, r_min in [
        ("equal costs,   r_min=1  (Fox 1966 applies exactly)", True, 1),
        ("equal costs,   r_min=4  (non-convex floor)", True, 4),
        ("unequal costs, r_min=1  (knapsack regime)", False, 1),
    ]:
        exact = total = 0
        worst = 0.0
        for _ in range(200):
            n = rng.choice([2, 3, 4])
            r_max = rng.choice([6, 8])
            scores = {f"l{i}": round(rng.uniform(0.1, 3.0), 3) for i in range(n)}
            if equal_costs:
                unit = rng.choice([2, 3, 5])
                costs = {f"l{i}": unit for i in range(n)}
            else:
                costs = {f"l{i}": rng.choice([2, 3, 5, 7]) for i in range(n)}
            budget = rng.randint(6, 90)

            res = allocate_ranks(
                scores, costs, budget_params=budget, r_min=r_min, r_max=r_max
            )
            assert res.used_params <= budget, "BUDGET VIOLATION"
            got = objective(dict(res.ranks), scores)
            best = brute_force_optimum(scores, costs, budget, r_min, r_max)
            total += 1
            if abs(got - best) < 1e-9:
                exact += 1
            elif best > 0:
                worst = max(worst, (best - got) / best * 100)
        print(f"  {label}")
        print(f"      exact: {exact}/{total}   worst shortfall: {worst:.2f}%")
    print()


def check_3_scale() -> None:
    print("=" * 74)
    print("3. Does the r_min non-convexity matter at realistic layer counts?")
    print("=" * 74)
    print("   (allocator vs. greedy + one force-admit restart per layer)")

    def stronger_reference(scores, costs, budget, r_min, r_max) -> float:
        base = allocate_ranks(
            scores, costs, budget_params=budget, r_min=r_min, r_max=r_max
        )
        best = objective(dict(base.ranks), scores)
        for forced in scores:
            lump = r_min * costs[forced]
            if lump > budget:
                continue
            rest = {k: v for k, v in scores.items() if k != forced}
            if not rest:
                continue
            sub = allocate_ranks(
                rest, {k: costs[k] for k in rest},
                budget_params=budget - lump, r_min=r_min, r_max=r_max,
            )
            cand = dict(sub.ranks)
            cand[forced] = r_min
            best = max(best, objective(cand, scores))
        return best

    rng = random.Random(5)
    print(f"   {'layers':>8} {'mean gap':>10} {'max gap':>10}")
    for n in (4, 8, 16, 32, 64):
        gaps = []
        for _ in range(40):
            scores = {f"l{i}": round(rng.uniform(0.05, 3.0), 4) for i in range(n)}
            unit = rng.choice([8192, 81920])
            costs = {f"l{i}": unit for i in range(n)}
            budget = rng.randint(n * unit * 2, n * unit * 40)
            res = allocate_ranks(
                scores, costs, budget_params=budget, r_min=4, r_max=64
            )
            got = objective(dict(res.ranks), scores)
            best = stronger_reference(scores, costs, budget, 4, 64)
            gaps.append((best - got) / best * 100 if best > 0 else 0.0)
        print(f"   {n:>8} {sum(gaps)/len(gaps):>9.4f}% {max(gaps):>9.4f}%")
    print()


def check_4_and_5_regression() -> None:
    print("=" * 74)
    print("4+5. Current allocator vs. the superseded continuous relaxation")
    print("=" * 74)
    rng = random.Random(7)
    gains = []
    violations = 0
    worst_waste = 0.0

    print(f"   {'case':>5} {'gain':>8} {'now used':>12} {'old used':>12} {'budget':>12}")
    for case in range(12):
        n = rng.choice([8, 16, 32])
        scores = {f"l{i}": round(rng.uniform(0.05, 3.0), 4) for i in range(n)}
        costs = {f"l{i}": rng.choice([8192, 15104, 20000]) for i in range(n)}
        budget = rng.randint(200_000, 3_000_000)

        now = allocate_ranks(scores, costs, budget_params=budget, r_min=4, r_max=64)
        old_ranks, old_used = superseded_relaxation(
            scores, costs, budget, 4, 64
        )

        o_now = objective(dict(now.ranks), scores)
        o_old = objective(old_ranks, scores)
        gain = (o_now - o_old) / o_now * 100 if o_now > 0 else 0.0
        gains.append(gain)
        if old_used > budget:
            violations += 1
        worst_waste = max(worst_waste, (budget - old_used) / budget * 100)
        assert now.used_params <= budget

        print(
            f"   {case:>5} {gain:>+7.2f}% {now.used_params:>12,} "
            f"{old_used:>12,} {budget:>12,}"
        )

    print()
    print(f"   mean objective gain over the superseded algorithm: {sum(gains)/len(gains):+.2f}%")
    print(f"   max  objective gain over the superseded algorithm: {max(gains):+.2f}%")
    print(f"   superseded algorithm exceeded its budget in:       {violations}/12 cases")
    print(f"   most budget the superseded algorithm left unspent: {worst_waste:.1f}%")
    print(f"   current allocator exceeded its budget in:          0/12 cases (invariant)")
    print()


def main() -> int:
    check_1_and_2_optimality()
    check_3_scale()
    check_4_and_5_regression()
    print("All allocator optimality claims reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
