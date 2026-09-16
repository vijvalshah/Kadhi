"""kadhi_cli.utils.capacity — exact LoRA/DoRA trainable-parameter accounting.

Phase 1 of the Adaptation Controller project (see
``docs/adaptation-controller.md`` / ``docs/adaptation-controller-plan.md``,
sections 1.1/1.2). ``lora.rank_pattern`` (``src/kadhi_cli/config/schema.py``)
lets an operator override the LoRA rank for a module-name pattern, and it is
passed straight through to ``peft.LoraConfig(rank_pattern=...)`` by
``peft_builder.py`` — but nothing in the codebase computes an exact
trainable-parameter count from it. This module is that accounting primitive:
given real on-disk tensor shapes and a rank_pattern, it counts EXACT LoRA /
DoRA trainable parameters. It intentionally does no architecture guessing —
if there is no local checkpoint to inspect, callers get ``None`` and are
expected to fall back elsewhere.

Self-contained and dependency-light by design: no ``torch`` / ``transformers``
/ ``peft`` / ``safetensors`` import anywhere, even lazily, so this module
stays importable in the lightest install tier. Safetensors shard headers are
parsed by hand (little-endian u64 header length + JSON header), mirroring
``kadhi_cli.utils.gpu._params_from_local_safetensors``'s existing precedent.

Public surface:
- ``LoraModuleShape`` frozen dataclass (name / in_features / out_features).
- ``resolve_pattern_rank(rank_pattern, full_key, default_r)`` -> int.
- ``discover_lora_module_shapes(weights_dir, target_modules)`` -> tuple[LoraModuleShape, ...].
- ``count_lora_trainable_params(shapes, *, default_r, rank_pattern=None, use_dora=False)`` -> int.
- ``estimate_lora_trainable_params_from_checkpoint(weights_dir, target_modules, *, default_r, rank_pattern=None, use_dora=False)`` -> Optional[int].
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

# A safetensors file starts with a u64 little-endian header length, then a
# JSON header carrying each tensor's dtype + shape. Reading it costs a few
# KB — the weights themselves are never touched. Mirrors
# kadhi_cli.utils.gpu._params_from_local_safetensors exactly.
_ST_HEADER_LEN_BYTES = 8
_MAX_ST_HEADER_BYTES = 100 * 1024 * 1024  # sanity bound on a crafted file

# Defensive cap on total tensors inspected across all shards in a directory —
# mirrors spectrum_scan.py's streaming-cap posture. Never silently truncate:
# raise instead once this is exceeded.
_MAX_TENSORS = 200_000

# Default LoRA target-module segments for "auto"/"all", covering:
#   - Llama/Mistral/Qwen/Gemma-style attention + MLP projections
#   - GPT-2 Conv1D naming
#   - common fused variants (qkv_proj, query_key_value, gate_up_proj)
_DEFAULT_TARGET_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "c_attn", "c_fc", "c_proj",
    "qkv_proj", "query_key_value", "gate_up_proj",
})

_AUTO_ALIASES = frozenset({"auto", "all"})


@dataclass(frozen=True)
class LoraModuleShape:
    """One 2-D LoRA-eligible weight's module path and shape.

    ``name`` is the module path WITHOUT the trailing ``.weight`` suffix, e.g.
    ``"model.layers.5.self_attn.q_proj"``. ``in_features``/``out_features``
    follow the on-disk tensor's ``[out, in]`` shape convention, i.e.
    ``in_features == shape[1]`` and ``out_features == shape[0]``.
    """

    name: str
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be str, got {type(self.name).__name__}")
        if not self.name:
            raise ValueError("name must not be empty")
        for fld in ("in_features", "out_features"):
            val = getattr(self, fld)
            if isinstance(val, bool):
                raise TypeError(f"{fld} must be int, not bool")
            if not isinstance(val, int):
                raise TypeError(f"{fld} must be int, got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"{fld} must be positive, got {val}")


def resolve_pattern_rank(
    rank_pattern: Optional[Mapping[str, int]],
    full_key: str,
    default_r: int,
) -> int:
    """Resolve the LoRA rank that applies to ``full_key``.

    Mirrors PEFT's ``get_pattern_key`` exactly (``peft/utils/other.py``)::

        def get_pattern_key(pattern_keys, key_to_match):
            return next(
                filter(lambda key: re.match(rf".*\\.{key}$", key_to_match), pattern_keys),
                key_to_match,
            )

    ``rank_pattern`` keys are tried in INSERTION order; the first key ``p``
    such that ``re.match(rf".*\\.{p}$", full_key)`` matches wins — ``p`` is
    used as a raw regex fragment (not escaped), so ``.`` and ``*`` behave as
    regex metacharacters on purpose (patterns like ``"experts.*.w1"`` rely on
    this). Returns ``default_r`` if ``rank_pattern`` is falsy or nothing
    matches.
    """
    if not rank_pattern:
        return default_r
    for pattern, rank in rank_pattern.items():
        if re.match(rf".*\.{pattern}$", full_key):
            return rank
    return default_r


def _module_segment(name: str) -> str:
    """Final dot-separated segment of a module path (e.g. 'q_proj')."""
    return name.rsplit(".", 1)[-1]


def _normalize_target_modules(
    target_modules: Union[str, Sequence[str]],
) -> frozenset:
    if isinstance(target_modules, str):
        if target_modules.strip().lower() in _AUTO_ALIASES:
            return _DEFAULT_TARGET_MODULES
        return frozenset({target_modules})
    normalized = {str(m) for m in target_modules}
    if not normalized:
        return frozenset()
    return frozenset(normalized)


def _read_safetensors_header(path: str) -> Optional[dict]:
    """Parse a shard's header (name -> {shape, ...}), or None if unreadable.

    Manual little-endian u64 length prefix + JSON header, no ``safetensors``
    package import — mirrors ``gpu.py``'s ``_params_from_local_safetensors``.
    Defensive per-shard: any (OSError, ValueError, TypeError,
    json.JSONDecodeError) is swallowed by the caller so one corrupt/crafted
    shard never aborts discovery across the rest of the directory.
    """
    with open(path, "rb") as handle:
        raw = handle.read(_ST_HEADER_LEN_BYTES)
        if len(raw) < _ST_HEADER_LEN_BYTES:
            return None
        header_len = int.from_bytes(raw, "little")
        if header_len <= 0 or header_len > _MAX_ST_HEADER_BYTES:
            return None
        header = json.loads(handle.read(header_len))
    if not isinstance(header, dict):
        return None
    return header


def discover_lora_module_shapes(
    weights_dir: str,
    target_modules: Union[str, Sequence[str]],
) -> "tuple[LoraModuleShape, ...]":
    """Stream ``.safetensors`` shard headers in ``weights_dir`` for shapes.

    Shape-only introspection: only tensor headers are ever read, never
    tensor data. Returns one ``LoraModuleShape`` per 2-D ``.weight`` tensor
    whose final dot-separated segment (before ``.weight``) matches
    ``target_modules`` — an EXACT segment match, never a substring match.

    ``target_modules == "auto"`` or ``"all"`` matches against a built-in
    default set covering Llama/Mistral/Qwen/Gemma-style naming, GPT-2 Conv1D
    naming, and common fused variants. An explicit list/tuple of strings
    matches those exact segments instead.

    Only 2-D tensors are considered (1-D norms/biases are skipped). Total
    tensors inspected across all shards is capped at ``_MAX_TENSORS``
    (200_000); exceeding it raises ``ValueError`` rather than silently
    truncating.

    Returns ``()`` (never ``None``) when ``weights_dir`` has no matching
    shards/tensors. A missing/non-directory ``weights_dir``, or a shard with
    an unreadable/corrupt header, is never fatal here — such shards are
    skipped and the rest of the directory is still scanned, mirroring
    ``gpu.py``'s ``_params_from_local_safetensors`` defensive posture.
    """
    wanted = _normalize_target_modules(target_modules)
    if not wanted:
        return ()
    if not isinstance(weights_dir, str) or not os.path.isdir(weights_dir):
        return ()

    try:
        shards = sorted(
            entry.path
            for entry in os.scandir(weights_dir)
            if entry.is_file() and entry.name.endswith(".safetensors")
        )
    except OSError:
        return ()

    results: list[LoraModuleShape] = []
    inspected = 0
    for shard in shards:
        try:
            header = _read_safetensors_header(shard)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            inspected += 1
            if inspected > _MAX_TENSORS:
                raise ValueError(
                    f"more than {_MAX_TENSORS} tensors in {weights_dir!r}; "
                    "refusing to continue rather than silently truncate"
                )
            if not name.endswith(".weight"):
                continue
            shape = meta.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                continue
            out_dim, in_dim = shape[0], shape[1]
            if (
                isinstance(out_dim, bool)
                or isinstance(in_dim, bool)
                or not isinstance(out_dim, int)
                or not isinstance(in_dim, int)
                or out_dim <= 0
                or in_dim <= 0
            ):
                continue
            module_path = name[: -len(".weight")]
            if _module_segment(module_path) not in wanted:
                continue
            try:
                results.append(
                    LoraModuleShape(
                        name=module_path,
                        in_features=in_dim,
                        out_features=out_dim,
                    )
                )
            except (TypeError, ValueError):
                continue
    return tuple(results)


def count_lora_trainable_params(
    shapes: Sequence[LoraModuleShape],
    *,
    default_r: int,
    rank_pattern: Optional[Mapping[str, int]] = None,
    use_dora: bool = False,
) -> int:
    """Exact LoRA/DoRA trainable-parameter count over ``shapes``.

    For each shape, resolves its rank via :func:`resolve_pattern_rank` and
    adds ``rank * (in_features + out_features)`` (the standard LoRA A/B
    low-rank factor parameter count). A shape whose resolved rank is ``<= 0``
    is skipped entirely (a frozen/excluded layer contributes zero trainable
    parameters). If ``use_dora`` is True, each COUNTED shape additionally
    contributes ``out_features`` parameters — the DoRA magnitude vector ``m``
    is one learned scalar per output feature (arXiv:2402.09353, DoRA:
    Weight-Decomposed Low-Rank Adaptation).
    """
    if isinstance(default_r, bool):
        raise TypeError("default_r must be int, not bool")
    if not isinstance(default_r, int):
        raise TypeError(f"default_r must be int, got {type(default_r).__name__}")
    if default_r < 0:
        raise ValueError(f"default_r must be >= 0, got {default_r}")
    if not isinstance(use_dora, bool):
        raise TypeError("use_dora must be bool")

    total = 0
    for shape in shapes:
        if not isinstance(shape, LoraModuleShape):
            raise TypeError(
                f"shapes must contain LoraModuleShape, got {type(shape).__name__}"
            )
        rank = resolve_pattern_rank(rank_pattern, shape.name, default_r)
        if rank <= 0:
            continue
        total += rank * (shape.in_features + shape.out_features)
        if use_dora:
            total += shape.out_features
    return total


def estimate_lora_trainable_params_from_checkpoint(
    weights_dir: str,
    target_modules: Union[str, Sequence[str]],
    *,
    default_r: int,
    rank_pattern: Optional[Mapping[str, int]] = None,
    use_dora: bool = False,
) -> Optional[int]:
    """Orchestrate discovery + counting for a local checkpoint directory.

    Returns ``None`` (never 0, never a raised exception) when ``weights_dir``
    yields zero matching shapes — this lets callers distinguish "no local
    checkpoint / nothing matched" from "a real zero-parameter plan" and fall
    back to a different estimate elsewhere, matching the None-fallback
    convention used throughout ``gpu.py`` and ``spectrum_scan.py``. A
    missing/invalid ``weights_dir`` never raises here either — it simply
    yields no shapes and therefore ``None``.
    """
    try:
        shapes = discover_lora_module_shapes(weights_dir, target_modules)
    except ValueError:
        return None
    if not shapes:
        return None
    return count_lora_trainable_params(
        shapes,
        default_r=default_r,
        rank_pattern=rank_pattern,
        use_dora=use_dora,
    )


__all__ = [
    "LoraModuleShape",
    "resolve_pattern_rank",
    "discover_lora_module_shapes",
    "count_lora_trainable_params",
    "estimate_lora_trainable_params_from_checkpoint",
]
