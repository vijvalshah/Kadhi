"""kadhi_cli.utils.growth — in-place LoRA rank growth, convergence-driven
stopping, and frontier reporting (Adaptation Controller, Phase 4).

See docs/adaptation-controller.md and adaptation-controller-plan.md §Phase 4.

Pure numpy and torch-free: tensors are ``numpy.ndarray``; a torch caller
passes ``tensor.detach().cpu().numpy()`` in and copies the result back.

Public surface:
- ``expand_lora_pair(A, B, new_rank, rng)`` -> (A', B'): grow one adapter's
  rank *without changing the function it computes*.
- ``extend_adam_state(exp_avg, exp_avg_sq, ..., new_rank)``: extend the
  optimizer moments for the grown factors with zeros.
- ``ConvergenceMonitor``: decides, against the phase-2 noise floor ``epsilon``,
  whether the loss has stopped improving and capacity should grow.
- ``plan_growth_step``: spend a stage's extra budget on the highest-scoring layers.
- ``FrontierPoint`` / ``frontier`` / ``format_frontier``: non-dominated
  (trainable params, quality) reporting over accumulated plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------- expansion

def expand_lora_pair(
    A: np.ndarray, B: np.ndarray, new_rank: int, rng: Optional[np.random.Generator] = None,
    init_scale: float = 0.01,
) -> "tuple[np.ndarray, np.ndarray]":
    """Grow a LoRA pair from rank ``r`` to ``new_rank`` preserving ``B @ A`` exactly.

    ``A`` is ``(r, in)``, ``B`` is ``(out, r)``; the adapter delta is ``B @ A``
    (any ``alpha / r`` scaling is applied by the caller and must be held
    constant across the expansion, i.e. rescale alpha with the rank). The new
    rows of ``A`` are small random values and the matching new columns of
    ``B`` are zero, so the added rank contributes exactly 0 to the output at
    the expansion step. ``B``'s new columns still receive nonzero gradient
    (their gradient is ``dL/dY @ (A_new x)``), so the new capacity trains.
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
        raise ValueError(f"incompatible LoRA pair: A{A.shape}, B{B.shape}")
    r = A.shape[0]
    if isinstance(new_rank, bool) or not isinstance(new_rank, int) or new_rank < r:
        raise ValueError(f"new_rank must be an int >= current rank {r}, got {new_rank!r}")
    if new_rank == r:
        return A.copy(), B.copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    extra = new_rank - r
    new_A = np.concatenate(
        [A, (init_scale * rng.standard_normal((extra, A.shape[1]))).astype(A.dtype)], axis=0,
    )
    new_B = np.concatenate([B, np.zeros((B.shape[0], extra), dtype=B.dtype)], axis=1)
    return new_A, new_B


def extend_adam_state(
    state: np.ndarray, *, factor: str, new_rank: int,
) -> np.ndarray:
    """Extend one Adam moment buffer (``exp_avg`` or ``exp_avg_sq``) for a grown factor.

    ``factor`` is ``"A"`` (grow along rows) or ``"B"`` (grow along columns).
    New slots start at zero: they have seen no gradient, which is exactly the
    state a freshly created parameter has, and keeps existing slots' history
    untouched so training continues without a loss spike.
    """
    if factor not in ("A", "B"):
        raise ValueError(f"factor must be 'A' or 'B', got {factor!r}")
    axis = 0 if factor == "A" else 1
    cur = state.shape[axis]
    if new_rank < cur:
        raise ValueError(f"new_rank {new_rank} < current {cur}")
    pad = [(0, 0), (0, 0)]
    pad[axis] = (0, new_rank - cur)
    return np.pad(state, pad)


# -------------------------------------------------------------- convergence

@dataclass
class ConvergenceMonitor:
    """Declares convergence when loss improvement over ``window`` steps is
    within the measured seed-noise floor ``epsilon`` — an improvement the run
    cannot distinguish from noise is not evidence the current rank is still
    learning, so growth (or stopping) is warranted.
    """

    epsilon: float
    window: int = 50
    losses: "list[float]" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.epsilon > 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon!r}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window!r}")
        self.losses = []

    def update(self, loss: float) -> None:
        if not np.isfinite(loss):
            raise ValueError("loss must be finite")
        self.losses.append(float(loss))

    def converged(self) -> bool:
        """True once ``2*window`` losses exist and the mean of the latest
        window improved on the previous window by no more than ``epsilon``."""
        if len(self.losses) < 2 * self.window:
            return False
        recent = np.mean(self.losses[-self.window:])
        prior = np.mean(self.losses[-2 * self.window:-self.window])
        return bool(prior - recent <= self.epsilon)

    def reset(self) -> None:
        """Call after a growth step: the new capacity gets a fresh window."""
        self.losses = []


def plan_growth_step(
    scores: Mapping[str, float],
    ranks: Mapping[str, int],
    cost_per_unit_rank: Mapping[str, int],
    *,
    extra_budget_params: int,
    r_max: int = 64,
    step: int = 4,
) -> "dict[str, int]":
    """Grow ranks by ``step`` on the best score-per-parameter layers until
    ``extra_budget_params`` is spent. Returns the new per-layer ranks (a
    layer already at ``r_max`` or too costly for the remainder is skipped)."""
    if step < 1:
        raise ValueError("step must be >= 1")
    new = dict(ranks)
    remaining = extra_budget_params
    order = sorted(
        new, key=lambda k: scores.get(k, 0.0) / cost_per_unit_rank[k], reverse=True,
    )
    progressed = True
    while progressed:
        progressed = False
        for key in order:
            inc = min(step, r_max - new[key])
            cost = inc * cost_per_unit_rank[key]
            if inc > 0 and cost <= remaining:
                new[key] += inc
                remaining -= cost
                progressed = True
    return new


# ----------------------------------------------------------------- frontier

@dataclass(frozen=True)
class FrontierPoint:
    label: str
    trainable_params: int
    quality: float  # higher is better
    wall_clock_s: Optional[float] = None


def frontier(points: Sequence[FrontierPoint]) -> "tuple[FrontierPoint, ...]":
    """Non-dominated points: maximise ``quality``, minimise ``trainable_params``
    (same dominance rule as ``tunability.pareto_frontier``), sorted by params."""
    keep = []
    for i, p in enumerate(points):
        dominated = any(
            q.quality >= p.quality and q.trainable_params <= p.trainable_params
            and (q.quality > p.quality or q.trainable_params < p.trainable_params)
            for j, q in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(p)
    return tuple(sorted(keep, key=lambda p: p.trainable_params))


def format_frontier(points: Sequence[FrontierPoint]) -> str:
    """Plain-text report of every point, marking which are on the frontier."""
    on = set(map(id, frontier(points)))
    lines = [f"{'plan':<24}{'params':>14}{'quality':>10}{'time_s':>9}  frontier"]
    for p in sorted(points, key=lambda p: p.trainable_params):
        t = f"{p.wall_clock_s:.0f}" if p.wall_clock_s is not None else "-"
        lines.append(
            f"{p.label:<24}{p.trainable_params:>14,}{p.quality:>10.4f}{t:>9}  "
            f"{'*' if id(p) in on else ''}"
        )
    return "\n".join(lines)


__all__ = [
    "expand_lora_pair", "extend_adam_state", "ConvergenceMonitor", "plan_growth_step",
    "FrontierPoint", "frontier", "format_frontier",
]
