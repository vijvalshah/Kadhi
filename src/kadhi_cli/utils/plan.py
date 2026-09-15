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

Public surface:
- ``layer_key(layer_index)`` -> str — the canonical string key used
  consistently across scores / costs / ranks throughout this bridge.
- ``group_shapes_by_layer(shapes)`` -> dict[str, tuple[LoraModuleShape, ...]].
- ``cost_per_unit_rank_from_shapes(shapes)`` -> dict[str, int].
- ``PlanResult`` frozen dataclass.
- ``rank_pattern_from_allocation(shapes, allocation, *, default_r)`` -> PlanResult.
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
