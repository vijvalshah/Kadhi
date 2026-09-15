"""kadhi_cli.utils.plan — bridges the sensitivity probe and the capacity
allocator into a concrete ``lora.rank_pattern``.

Three modules exist and are independently tested, but nothing connected them
end to end (see docs/adaptation-controller-plan.md §1.5 and the note at the
end of that document's Phase 3 section):

- ``utils/capacity.py`` reads real per-module shapes off an already-local
  checkpoint.
- ``utils/sensitivity.py`` scores each decoder layer's task-conditional
  importance.
- ``utils/allocate.py`` turns (scores, per-layer cost) into an integer rank
  per layer under a parameter budget — but its ``ranks`` mapping is keyed by
  whatever opaque string the caller supplies; it deliberately knows nothing
  about PEFT's ``rank_pattern`` matching rules.

This module is that missing connective layer, and only that: it groups
``capacity.LoraModuleShape`` entries by decoder-layer index (reusing
``lisa.py``'s existing layer-index regex, the same single source of truth
every other layer-indexed feature in this codebase already uses), derives
the per-layer cost ``allocate_ranks`` needs, and — once an allocation comes
back — expands the per-layer rank into the per-MODULE ``rank_pattern`` PEFT
actually consumes.

**The one thing this module does NOT solve, on purpose**: PEFT's
``rank_pattern`` cannot express "rank 0" as "exclude this layer from LoRA
entirely" — a real ``peft.LoraConfig`` needs a positive rank wherever it
attaches an adapter at all. A layer the allocator freezes (rank 0) has to be
excluded some other way — restricting ``target_modules``, or PEFT's
``layers_to_transform`` — which is a decision the CALLER makes, not this
module. ``rank_pattern_from_allocation`` therefore returns the frozen
layers' module paths SEPARATELY (``frozen_module_paths``) rather than
guessing how to encode them, matching the open design question already
recorded in adaptation-controller-plan.md §1.5.

A separate concern this module also covers: producing a real ``rank_pattern``
does not actually require the gradient-based sensitivity probe. Kadhi already
has a static, task-independent layer signal — ``spectrum_scan``'s spectral
SNR — computed straight from the same on-disk checkpoint capacity.py reads,
with no live model load and no torch dependency at all (it falls back to a
pure-numpy safetensors reader when torch is unavailable). ``aggregate_layer_snr``
and ``build_static_plan`` compose capacity + spectrum_scan + allocate into a
fully working, fully static Adaptation Controller pass — the gradient probe
(``sensitivity.py``) is a strictly better, task-conditional signal to use in
its place once a live training dataset and model are available, not a
prerequisite for having a controller at all.

Public surface:
- ``layer_key(layer_index)`` -> str — the canonical string key used
  consistently across scores / costs / ranks throughout this bridge.
- ``group_shapes_by_layer(shapes)`` -> dict[str, tuple[LoraModuleShape, ...]].
- ``cost_per_unit_rank_from_shapes(shapes)`` -> dict[str, int].
- ``aggregate_layer_snr(layer_snrs)`` -> dict[str, float].
- ``PlanResult`` frozen dataclass.
- ``rank_pattern_from_allocation(shapes, allocation, *, default_r)`` -> PlanResult.
- ``LayerSignals`` frozen dataclass — the expensive measurement, done once.
- ``compute_layer_signals(weights_dir, target_modules, *, modules="all")``
  -> LayerSignals (EXPENSIVE: the only step that touches model weights).
- ``plan_from_signals(signals, *, budget_params, default_r, r_min=4,
  r_max=64)`` -> PlanResult (cheap; safe to call in a loop).
- ``build_static_plan(weights_dir, target_modules, *, budget_params,
  default_r, r_min=4, r_max=64, modules="all")`` -> PlanResult — a thin
  wrapper over the two above, for one-shot callers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

# Deliberate cross-module reuse of lisa.py's single source of truth for
# layer-index parsing ("layers.N." / "h.N."), the same regex sensitivity.py
# already reuses for the same reason — not a second, possibly-drifting copy.
from kadhi_cli.utils.lisa import _LAYER_RE

if TYPE_CHECKING:
    from kadhi_cli.utils.allocate import AllocationResult
    from kadhi_cli.utils.capacity import LoraModuleShape
    from kadhi_cli.utils.spectrum_scan import LayerSNR


def layer_key(layer_index: int) -> str:
    """Canonical string key for a decoder-layer index.

    Every function in this bridge (and ``allocate_ranks``, which is agnostic
    to what its keys mean) uses this exact format, so a score dict, a cost
    dict, and an allocation's ``ranks`` dict are always key-compatible.
    """
    if isinstance(layer_index, bool):
        raise TypeError("layer_index must be int, not bool")
    if not isinstance(layer_index, int):
        raise TypeError(f"layer_index must be int, got {type(layer_index).__name__}")
    if layer_index < 0:
        raise ValueError(f"layer_index must be >= 0, got {layer_index}")
    return f"layer.{layer_index}"


def group_shapes_by_layer(
    shapes: "Sequence[LoraModuleShape]",
) -> "dict[str, tuple[LoraModuleShape, ...]]":
    """Group LoRA-eligible module shapes by their decoder-layer index.

    A shape whose ``name`` does not contain a ``layers.N.`` / ``h.N.``
    segment (e.g. an embedding or lm_head matched by an unusual
    ``target_modules`` list) is skipped — this bridge only allocates rank
    across INDEXED decoder layers, matching what ``sensitivity.py`` and
    ``lisa.py`` both already assume about the architectures Kadhi's layer-
    indexed features support. Raises ValueError if NO shape matches any
    layer (nothing for the allocator to work with).
    """
    grouped: dict[str, list["LoraModuleShape"]] = {}
    for shape in shapes:
        match = _LAYER_RE.search(shape.name)
        if match is None:
            continue
        key = layer_key(int(match.group(1)))
        grouped.setdefault(key, []).append(shape)
    if not grouped:
        raise ValueError(
            "group_shapes_by_layer: no shape name contains a 'layers.N.' / "
            "'h.N.' segment — nothing to allocate rank across for this "
            "architecture."
        )
    return {key: tuple(group) for key, group in grouped.items()}


def cost_per_unit_rank_from_shapes(shapes: "Sequence[LoraModuleShape]") -> "dict[str, int]":
    """Per-layer trainable-parameter cost of one unit of LoRA rank.

    ``params(r) = r · Σ_{shapes at this layer} (in_features + out_features)``
    is linear in ``r``, so the per-layer coefficient — this function's
    output — is exactly the ``cost_per_unit_rank`` argument
    ``allocate.allocate_ranks`` needs.
    """
    grouped = group_shapes_by_layer(shapes)
    return {
        key: sum(shape.in_features + shape.out_features for shape in group)
        for key, group in grouped.items()
    }


@dataclass(frozen=True)
class PlanResult:
    """The output of turning an :class:`allocate.AllocationResult` into a
    real ``lora.rank_pattern``.

    ``rank_pattern`` holds one entry per (layer, module) pair whose
    allocated rank is both non-zero and different from ``default_r`` —
    entries equal to ``default_r`` are omitted on purpose, so an operator
    reading the emitted config sees only the layers the allocator actually
    changed, not a restatement of the default at every layer.

    ``frozen_module_paths`` lists every module path belonging to a layer the
    allocator zeroed out. See this module's docstring: encoding these as an
    actual exclusion (``target_modules`` restriction, ``layers_to_transform``,
    or similar) is left to the caller.
    """

    rank_pattern: "Mapping[str, int]"
    frozen_module_paths: "tuple[str, ...]"
    default_r: int

    def __post_init__(self) -> None:
        if not isinstance(self.rank_pattern, Mapping):
            raise TypeError(
                f"rank_pattern must be a Mapping, got {type(self.rank_pattern).__name__}"
            )
        for key, value in self.rank_pattern.items():
            if not isinstance(key, str):
                raise TypeError(f"rank_pattern keys must be str, got {type(key).__name__}")
            if isinstance(value, bool):
                raise TypeError(f"rank_pattern[{key!r}] must be int, not bool")
            if not isinstance(value, int):
                raise TypeError(
                    f"rank_pattern[{key!r}] must be int, got {type(value).__name__}"
                )
            if value <= 0:
                raise ValueError(
                    f"rank_pattern[{key!r}] must be > 0 (0 means frozen and "
                    f"belongs in frozen_module_paths instead), got {value}"
                )
        if not isinstance(self.frozen_module_paths, tuple):
            raise TypeError(
                "frozen_module_paths must be a tuple, got "
                f"{type(self.frozen_module_paths).__name__}"
            )
        for path in self.frozen_module_paths:
            if not isinstance(path, str):
                raise TypeError(
                    f"frozen_module_paths entries must be str, got {type(path).__name__}"
                )
        if isinstance(self.default_r, bool):
            raise TypeError("default_r must be int, not bool")
        if not isinstance(self.default_r, int):
            raise TypeError(f"default_r must be int, got {type(self.default_r).__name__}")
        if self.default_r <= 0:
            raise ValueError(f"default_r must be > 0, got {self.default_r}")


def rank_pattern_from_allocation(
    shapes: "Sequence[LoraModuleShape]",
    allocation: "AllocationResult",
    *,
    default_r: int,
) -> PlanResult:
    """Expand a per-layer :class:`allocate.AllocationResult` into a
    per-module ``rank_pattern``.

    ``allocation.ranks`` must be keyed exactly as :func:`layer_key` produces
    (i.e. it must have come from an allocation run against
    :func:`cost_per_unit_rank_from_shapes`'s output, or an equivalently-keyed
    dict) — raises ValueError naming any allocation key absent from
    ``shapes``'s own grouping, and any layer present in ``shapes`` but absent
    from the allocation (both directions are a caller bug worth surfacing,
    not silently ignoring).
    """
    if isinstance(default_r, bool):
        raise TypeError("default_r must be int, not bool")
    if not isinstance(default_r, int) or default_r <= 0:
        raise ValueError(f"default_r must be a positive int, got {default_r!r}")

    grouped = group_shapes_by_layer(shapes)

    shape_keys = set(grouped)
    alloc_keys = set(allocation.ranks)
    if shape_keys != alloc_keys:
        missing_in_alloc = sorted(shape_keys - alloc_keys)
        missing_in_shapes = sorted(alloc_keys - shape_keys)
        raise ValueError(
            "rank_pattern_from_allocation: shapes and allocation.ranks "
            "disagree on which layers exist — "
            f"layers in shapes but not allocated: {missing_in_alloc}; "
            f"allocated but not present in shapes: {missing_in_shapes}"
        )

    rank_pattern: dict[str, int] = {}
    frozen_module_paths: list[str] = []
    for key, group in grouped.items():
        rank = allocation.ranks[key]
        if rank == 0:
            frozen_module_paths.extend(shape.name for shape in group)
            continue
        if rank == default_r:
            continue
        for shape in group:
            rank_pattern[shape.name] = rank

    return PlanResult(
        rank_pattern=rank_pattern,
        frozen_module_paths=tuple(sorted(frozen_module_paths)),
        default_r=default_r,
    )


def trainable_params_for_plan(
    shapes: "Sequence[LoraModuleShape]", plan_result: PlanResult, *, use_dora: bool = False,
) -> int:
    """Exact trainable-parameter count implied by a :class:`PlanResult`.

    Looks ``plan_result.rank_pattern`` up by DIRECT dict key against each
    shape's exact ``name`` — never through
    ``capacity.resolve_pattern_rank``'s PEFT-style suffix matching. That
    distinction is load-bearing, not stylistic: PEFT's real matching
    (``.*\\.{pattern}$``) requires a literal ``.`` immediately before the
    matched pattern within the full parameter path — true in production,
    where ``base_model.model.`` (or similar) always prefixes ``shape.name``
    — but ``plan_result.rank_pattern``'s keys ARE ``shape.name`` verbatim, so
    re-matching them against ``shape.name`` itself (no prefix) can never
    succeed: there is no character before the start of a string. Routing
    this function's accounting through that matcher would therefore silently
    fall back to ``default_r`` for every module the allocator actually
    changed. Since the mapping here is already known to be EXACT rather than
    a fuzzy pattern, a direct lookup sidesteps the issue rather than working
    around it.
    """
    frozen = set(plan_result.frozen_module_paths)
    total = 0
    for shape in shapes:
        if shape.name in frozen:
            continue
        rank = plan_result.rank_pattern.get(shape.name, plan_result.default_r)
        if rank <= 0:
            continue
        total += rank * (shape.in_features + shape.out_features)
        if use_dora:
            total += shape.out_features
    return total


def aggregate_layer_snr(layer_snrs: "Sequence[LayerSNR]") -> "dict[str, float]":
    """Scale-corrected per-layer score from per-matrix spectral SNR.

    ``spectrum_scan.scan_weights_dir`` returns one :class:`LayerSNR` per
    weight MATRIX; the allocator needs one score per LAYER. Averaging the raw
    ``snr`` values of a layer's matrices — what this function used to do — is
    statistically invalid, because spectral SNR magnitude depends on the
    matrix SHAPE, not only on how informative the layer is. Measured on an
    IDENTICAL generative process (same low-rank signal + same noise), varying
    only the shape:

    ===============  ============  ==================
    matrix           shape         SNR
    ===============  ============  ==================
    square           512x512       ~0.0060
    wide             1376x512      ~0.0411  (~7x)
    tall             512x1376      ~0.0402  (~7x)
    ===============  ============  ==================

    So in a Llama-style model the rectangular MLP matrices (gate/up/down)
    systematically dominate the square-ish attention matrices (q/o) by ~7x in
    a plain mean, and a layer's score ends up reflecting almost purely its
    MLP SNR — regardless of actual importance.

    The fix, following the precedent already established by
    ``spectrum_scan.select_unfrozen_parameters`` (which groups by
    :func:`spectrum_scan.layer_type_signature` and selects WITHIN each
    module-type group precisely so the choice "keeps the unfreeze balanced
    across module types"): normalize each matrix's SNR **within its
    ``layer_type_signature`` group, across layers**, before averaging the —
    now comparable — values per layer::

        normalized(m)  =  snr(m) / mean{ snr(m') : group(m') == group(m) }
        score(layer)   =  mean{ normalized(m) : m in layer }

    Mean-normalization specifically (not a z-score, not rank-normalization):

    - It is **scale-correcting**: every module-type group ends up with mean
      1.0, so each module type contributes equally to a layer's score.
    - It **preserves within-group RATIOS**, which is where the actual signal
      lives — a layer 2x better than its peers at ``q_proj`` stays 2x better.
      Rank-normalization would throw that magnitude information away.
    - It keeps every value **strictly positive**, which
      ``allocate.allocate_ranks`` REQUIRES (it raises ``ValueError`` on a
      non-positive score). A z-score would produce negative values for every
      below-average layer and break that contract outright.

    Skipping rules (all deliberate, none silent-but-wrong):

    - A matrix whose name carries no ``layers.N.`` / ``h.N.`` segment is
      skipped (mirrors :func:`group_shapes_by_layer`).
    - An individual non-finite ``snr`` is skipped.
    - A whole group whose mean SNR is 0 or non-finite is skipped — dividing
      by it would emit zeros/NaNs that later trip ``allocate_ranks``'
      positivity check with a confusing, far-away error.
    - A layer left with NO contributing matrices after that skipping is
      OMITTED from the result entirely, rather than emitted as a 0.0 score
      that ``allocate_ranks`` would reject.

    Raises ``ValueError`` if nothing aggregates at all. Every returned score
    is guaranteed finite and > 0.
    """
    import math

    from kadhi_cli.utils.spectrum_scan import layer_type_signature

    # (layer_key, group, snr) for every usable record.
    usable: list[tuple[str, str, float]] = []
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for record in layer_snrs:
        match = _LAYER_RE.search(record.name)
        if match is None:
            continue
        snr = float(record.snr)
        if not math.isfinite(snr):
            continue
        key = layer_key(int(match.group(1)))
        group = layer_type_signature(record.name)
        usable.append((key, group, snr))
        group_sums[group] = group_sums.get(group, 0.0) + snr
        group_counts[group] = group_counts.get(group, 0) + 1

    if not usable:
        raise ValueError(
            "aggregate_layer_snr: no LayerSNR name contains a 'layers.N.' / "
            "'h.N.' segment — nothing to score."
        )

    group_means = {
        group: group_sums[group] / group_counts[group] for group in group_sums
    }

    normalized: dict[str, list[float]] = {}
    for key, group, snr in usable:
        mean = group_means[group]
        if not math.isfinite(mean) or mean <= 0.0:
            continue  # whole group skipped — see docstring
        normalized.setdefault(key, []).append(snr / mean)

    if not normalized:
        raise ValueError(
            "aggregate_layer_snr: every module-type group had a zero or "
            "non-finite mean SNR — nothing to score."
        )

    scores = {key: sum(values) / len(values) for key, values in normalized.items()}

    # Guard the contract allocate_ranks depends on, here rather than three
    # calls away where the error would be unintelligible.
    for key, value in scores.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"aggregate_layer_snr: computed a non-positive/non-finite "
                f"score for {key!r} ({value!r}) — allocate_ranks requires "
                "every score to be finite and > 0."
            )
    return scores


@dataclass(frozen=True)
class LayerSignals:
    """Everything expensive about planning, measured ONCE and reusable.

    Producing a plan has two halves with wildly different costs: MEASURING
    the model (a full SVD of every weight matrix — minutes on a real 8B) and
    ALLOCATING under a budget (microseconds of arithmetic). This dataclass is
    the boundary between them: it holds the measured half, so a caller that
    plans repeatedly — most importantly ``feasibility.fit_plan_to_budget``,
    which re-plans once per budget-shrink iteration — measures once and
    allocates many times.

    - ``shapes``: the LoRA-eligible module shapes, already restricted to the
      layers both the SNR scan and the shape discovery agree exist.
    - ``scores``: ``layer_key`` -> normalized per-layer score (see
      :func:`aggregate_layer_snr`), all finite and > 0 as
      ``allocate.allocate_ranks`` requires.
    - ``cost_per_unit_rank``: ``layer_key`` -> trainable params per unit of
      rank, all positive ints.

    ``scores`` and ``cost_per_unit_rank`` are required to have IDENTICAL key
    sets, so an instance is internally consistent by construction and
    :func:`plan_from_signals` needs no reconciliation of its own.
    """

    shapes: "tuple[LoraModuleShape, ...]"
    scores: "Mapping[str, float]"
    cost_per_unit_rank: "Mapping[str, int]"

    def __post_init__(self) -> None:
        import math

        if not isinstance(self.shapes, tuple):
            raise TypeError(f"shapes must be a tuple, got {type(self.shapes).__name__}")
        if not self.shapes:
            raise ValueError("shapes must not be empty")
        for field_name in ("scores", "cost_per_unit_rank"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{field_name} must be a Mapping, got {type(value).__name__}"
                )
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            for key in value:
                if not isinstance(key, str):
                    raise TypeError(
                        f"{field_name} keys must be str, got {type(key).__name__}"
                    )
        score_keys = set(self.scores)
        cost_keys = set(self.cost_per_unit_rank)
        if score_keys != cost_keys:
            raise ValueError(
                "LayerSignals: scores and cost_per_unit_rank must have "
                "identical key sets — "
                f"scored but uncosted: {sorted(score_keys - cost_keys)}; "
                f"costed but unscored: {sorted(cost_keys - score_keys)}"
            )
        for key, score in self.scores.items():
            if isinstance(score, bool):
                raise TypeError(f"scores[{key!r}] must be a float, not bool")
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"scores[{key!r}] must be a number, got {type(score).__name__}"
                )
            if not math.isfinite(float(score)) or float(score) <= 0.0:
                raise ValueError(
                    f"scores[{key!r}] must be finite and > 0, got {score!r} "
                    "(allocate_ranks rejects non-positive scores; omit the "
                    "layer instead of scoring it 0)"
                )
        for key, cost in self.cost_per_unit_rank.items():
            if isinstance(cost, bool):
                raise TypeError(f"cost_per_unit_rank[{key!r}] must be int, not bool")
            if not isinstance(cost, int):
                raise TypeError(
                    f"cost_per_unit_rank[{key!r}] must be int, got "
                    f"{type(cost).__name__}"
                )
            if cost <= 0:
                raise ValueError(
                    f"cost_per_unit_rank[{key!r}] must be > 0, got {cost}"
                )


def compute_layer_signals(
    weights_dir: str,
    target_modules: "str | Sequence[str]",
    *,
    modules: str = "all",
) -> LayerSignals:
    """Run the expensive measurement once: discover shapes, scan spectral
    SNR, normalize + aggregate to per-layer scores, derive per-layer cost.

    This is the ONLY part of planning that touches the model's weights (a
    full SVD per matrix). Call it once and reuse the result across as many
    allocations as you like via :func:`plan_from_signals`.

    Steps:

    1. ``capacity.discover_lora_module_shapes`` — real per-module shapes.
    2. ``spectrum_scan.scan_weights_dir`` — real spectral SNR per matrix
       (the STATIC, task-independent signal; see this module's docstring for
       why the gradient probe is a better-but-optional upgrade, not a
       prerequisite).
    3. :func:`aggregate_layer_snr` — scale-corrected per-layer score.
    4. :func:`cost_per_unit_rank_from_shapes` — per-layer cost.
    5. Reconciliation: the SNR scan and the shape discovery can disagree on
       which layers exist (different module filters), so everything is
       restricted to their intersection here — which is what makes the
       returned :class:`LayerSignals` internally consistent by construction.

    Raises whatever the underlying step raises (a checkpoint with no matching
    target modules, an unindexed architecture) — no extra error handling is
    layered on top, since a caller needs to see exactly which step failed.
    """
    from kadhi_cli.utils.capacity import discover_lora_module_shapes
    from kadhi_cli.utils.spectrum_scan import scan_weights_dir

    shapes = discover_lora_module_shapes(weights_dir, target_modules)
    if not shapes:
        raise ValueError(
            f"compute_layer_signals: no matching LoRA-eligible weights found "
            f"under {weights_dir!r} for target_modules={target_modules!r}"
        )

    # NOTE on spectrum_scan's on-disk scan cache (read_cached_scan /
    # write_cached_scan / scan_model): deliberately NOT wired in here.
    # Those entry points key the cache on a MODEL IDENTITY
    # (``model_slug(model)``, truncated to 128 chars) with no content
    # fingerprint — no shard size, mtime or digest — whereas this function is
    # handed a weights DIRECTORY. Kadhi materializes a given model into one
    # fixed cache directory, so re-fetching or updating a checkpoint leaves
    # the path unchanged while the weights change, and a cache hit would then
    # return SNRs measured from different weights, silently; two long
    # directory paths sharing a 128-char prefix additionally slug-collide.
    # The `modules` filter IS part of the cache key (read_cached_scan rejects
    # a mismatch), so that half is fine — what is missing is a weights
    # fingerprint. Wiring the cache safely needs a content-addressed key
    # (e.g. the ``_weight_file_manifest`` tuple folded into the slug); until
    # then a slow scan beats a wrong one. The severe cost this function's
    # existence fixes — re-scanning once per feasibility iteration — is
    # already eliminated by hoisting the scan out of the loop entirely.
    layer_snrs = scan_weights_dir(weights_dir, modules=modules)
    scores = aggregate_layer_snr(layer_snrs)
    cost = cost_per_unit_rank_from_shapes(shapes)

    common = set(scores) & set(cost)
    if not common:
        raise ValueError(
            "compute_layer_signals: the SNR scan and the LoRA module "
            "discovery share no layer in common — check `modules` vs "
            "`target_modules`."
        )
    scores = {k: v for k, v in scores.items() if k in common}
    cost = {k: v for k, v in cost.items() if k in common}
    # Restrict shapes to layers present in `common` so rank_pattern_from_
    # allocation's shapes/allocation key-set check passes even when a filter
    # mismatch dropped some layers above.
    shapes = tuple(
        s for s in shapes
        if (m := _LAYER_RE.search(s.name)) is not None and layer_key(int(m.group(1))) in common
    )

    return LayerSignals(shapes=shapes, scores=scores, cost_per_unit_rank=cost)


def plan_from_signals(
    signals: LayerSignals,
    *,
    budget_params: int,
    default_r: int,
    r_min: int = 4,
    r_max: int = 64,
) -> PlanResult:
    """Cheap: allocate under a budget from already-computed signals.

    Does NO model I/O — no SVD, no safetensors read — so it is safe to call
    many times, which is exactly what ``feasibility.fit_plan_to_budget``'s
    shrink loop needs. Composes ``allocate.allocate_ranks`` with
    :func:`rank_pattern_from_allocation`.

    ``signals`` is already internally consistent (see :class:`LayerSignals`),
    so no reconciliation happens here.
    """
    from kadhi_cli.utils.allocate import allocate_ranks

    if not isinstance(signals, LayerSignals):
        raise TypeError(
            f"signals must be a LayerSignals, got {type(signals).__name__}"
        )
    allocation = allocate_ranks(
        signals.scores,
        signals.cost_per_unit_rank,
        budget_params=budget_params,
        r_min=r_min,
        r_max=r_max,
    )
    return rank_pattern_from_allocation(
        signals.shapes, allocation, default_r=default_r
    )


def build_static_plan(
    weights_dir: str,
    target_modules: "str | Sequence[str]",
    *,
    budget_params: int,
    default_r: int,
    r_min: int = 4,
    r_max: int = 64,
    modules: str = "all",
) -> PlanResult:
    """Produce a real ``rank_pattern`` from an on-disk checkpoint alone.

    A thin convenience wrapper: :func:`compute_layer_signals` followed by
    :func:`plan_from_signals`. Composes, in order:

    1. ``capacity.discover_lora_module_shapes`` — real per-module shapes.
    2. ``spectrum_scan.scan_weights_dir`` — real spectral SNR per matrix
       (this is the STATIC, task-independent signal; see this module's
       docstring for why the gradient probe is a better-but-optional
       upgrade, not a prerequisite).
    3. ``aggregate_layer_snr`` — per-layer score.
    4. ``cost_per_unit_rank_from_shapes`` — per-layer cost.
    5. ``allocate.allocate_ranks`` — the actual allocation.
    6. ``rank_pattern_from_allocation`` — the final ``PlanResult``.

    Every step is real: no live model load, no gradient computation, no
    dataset — everything is derived from the checkpoint's own weights on
    disk. Raises whatever the underlying step raises (a checkpoint with no
    matching target modules, an unindexed architecture, an unreachable
    budget) — this function does not add its own error handling on top,
    since a caller needs to see exactly which step failed and why.

    **Do not call this in a loop.** Steps 1-2 are the expensive ones (a full
    SVD of every weight matrix — minutes on a real 8B), and this function
    re-runs them on EVERY call, so a caller that plans repeatedly under
    different budgets — ``feasibility.fit_plan_to_budget``'s shrink loop
    being the one that matters — re-does the entire scan per iteration. Use
    the two-step API instead: call :func:`compute_layer_signals` once, then
    :func:`plan_from_signals` per budget.
    """
    signals = compute_layer_signals(weights_dir, target_modules, modules=modules)
    return plan_from_signals(
        signals,
        budget_params=budget_params,
        default_r=default_r,
        r_min=r_min,
        r_max=r_max,
    )
