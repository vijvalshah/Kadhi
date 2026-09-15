"""kadhi_cli.utils.sensitivity — task-conditional layer sensitivity probe.

See docs/adaptation-controller.md §3.1 for the design rationale and
adaptation-controller-plan.md §1.4 for why this exists (and what it does
NOT claim — the sensitivity metric is AdaLoRA's, applied here at layer
granularity as a cheap one-shot score rather than continuously during
training).

Public surface:
- ``LayerSensitivity`` frozen dataclass (layer_index / score / param_count).
- ``SensitivityReport`` frozen dataclass (scores / probe_steps / model_slug).
- ``compute_layer_sensitivity(model, batches, *, max_steps=50)`` -> SensitivityReport.
- ``spearman_correlation(a, b)`` -> float.
- ``correlate_with_static_signals(report, *, spectrum_scores=None, shrink_scores=None)`` -> dict.
- ``cache_path_for(model, dataset_fingerprint, cache_dir=None)`` -> str.
- ``save_sensitivity_report(report, model, dataset_fingerprint, *, cache_dir=None)`` -> str.
- ``load_sensitivity_report(model, dataset_fingerprint, *, cache_dir=None)`` -> Optional[SensitivityReport].

No top-level torch/numpy/scipy import anywhere — ``compute_layer_sensitivity``
is the only function that touches a real model and it imports torch lazily
inside itself, matching the lazy-import discipline documented in
``live_eval.py``'s module docstring and followed by ``lisa.py`` /
``spectrum_scan.py``. Spearman correlation is hand-rolled pure Python (rank +
tie-averaging + Pearson-of-ranks) so this module adds no numpy/scipy
dependency, matching ``capacity.py`` / ``allocate.py``'s zero-new-deps posture.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class LayerSensitivity:
    """One decoder layer's task-conditional sensitivity score."""

    layer_index: int
    score: float
    param_count: int

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or not isinstance(self.layer_index, int):
            raise TypeError(
                f"layer_index must be int, not {type(self.layer_index).__name__}"
            )
        if self.layer_index < 0:
            raise ValueError(f"layer_index must be >= 0, got {self.layer_index}")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError(f"score must be a number, got {type(self.score).__name__}")
        score = float(self.score)
        import math

        if not math.isfinite(score):
            raise ValueError("score must be finite")
        if score < 0.0:
            raise ValueError(f"score must be >= 0.0, got {score}")
        if isinstance(self.param_count, bool) or not isinstance(self.param_count, int):
            raise TypeError(
                f"param_count must be int, not {type(self.param_count).__name__}"
            )
        if self.param_count <= 0:
            raise ValueError(f"param_count must be > 0, got {self.param_count}")


@dataclass(frozen=True)
class SensitivityReport:
    """A full probe run: per-layer scores plus the run's provenance."""

    scores: "tuple[LayerSensitivity, ...]"
    probe_steps: int
    model_slug_: str

    def __post_init__(self) -> None:
        if not isinstance(self.scores, tuple):
            raise TypeError(f"scores must be a tuple, got {type(self.scores).__name__}")
        if not self.scores:
            raise ValueError("scores must not be empty")
        if not all(isinstance(item, LayerSensitivity) for item in self.scores):
            raise TypeError("scores must contain only LayerSensitivity instances")
        indices = [item.layer_index for item in self.scores]
        if indices != sorted(indices):
            raise ValueError("scores must be sorted ascending by layer_index")
        if len(set(indices)) != len(indices):
            raise ValueError("scores must not contain duplicate layer_index values")
        if isinstance(self.probe_steps, bool) or not isinstance(self.probe_steps, int):
            raise TypeError(
                f"probe_steps must be int, not {type(self.probe_steps).__name__}"
            )
        if self.probe_steps <= 0:
            raise ValueError(f"probe_steps must be > 0, got {self.probe_steps}")
        if not isinstance(self.model_slug_, str) or not self.model_slug_:
            raise TypeError("model_slug_ must be a non-empty str")


def compute_layer_sensitivity(
    model: Any, batches: Iterable[Any], *, max_steps: int = 50,
) -> SensitivityReport:
    """Run up to max_steps forward+backward passes accumulating per-layer
    sensitivity. `model` is an already-loaded torch model in TRAIN mode with
    a LOSS-PRODUCING forward (i.e. call signature `model(**batch)` returns
    an object with a `.loss` attribute — matches every HF CausalLM forward
    Kadhi already trains against). `batches` yields dicts of already-tokenized
    tensors ready to unpack as `model(**batch)` kwargs (input_ids, attention_mask,
    labels, etc — exactly what a DataLoader over a Kadhi training dataset yields).

    For each batch (up to max_steps, stopping early and reporting the REAL
    count if the iterable is exhausted first):
      1. zero any existing .grad on every parameter
      2. forward: out = model(**batch); loss = out.loss
      3. loss.backward()
      4. for every named parameter with a non-None .grad, compute
         (param.grad * param).abs().sum().item(), determine its layer via
         locate_decoder_layer_indices-style regex match on the param NAME
         (reuse the actual regex/matching mechanism `lisa.py` uses — do not
         duplicate the regex constant, import it or the matching helper if
         it's exposed at module level in lisa.py; if lisa.py only exposes
         the compiled regex as a private `_LAYER_RE` module constant, import
         it explicitly as `from kadhi_cli.utils.lisa import _LAYER_RE` with a
         one-line comment explaining that's a deliberate cross-module reuse
         of the single source of truth for layer-index parsing, not a new
         one), accumulate the per-layer running sum and a running
         parameter count.
      5. zero grads again before the next batch (never call optimizer.step()
         anywhere in this function — a probe must not train the model)

    After all steps: score(ℓ) = accumulated |grad⊙weight| sum for layer ℓ,
    divided by that layer's total accumulated parameter count (a parameter
    counted once per step it had a grad, OR counted once total — pick
    ONCE-TOTAL: param_count per layer should reflect the layer's actual
    parameter count, not step-multiplied; accumulate sum-of-|grad⊙weight|
    across steps, but param_count from a single pass over named_parameters()
    since it doesn't change between steps).

    Raise ValueError if `locate_decoder_layer_indices`-style matching finds
    zero layers (mirror lisa.py's own error message style and RuntimeError
    choice for "could not detect numbered decoder layers"). Raise ValueError
    if `batches` yields zero batches before any step runs.

    Import torch INSIDE this function, not at module scope.
    """
    import torch  # noqa: PLC0415 — lazy per this module's zero-top-level-torch policy

    # Deliberate cross-module reuse of lisa.py's single source of truth for
    # layer-index parsing (the module only exposes it as a private constant,
    # not a public re-export) — not a new regex.
    from kadhi_cli.utils.lisa import _LAYER_RE

    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError(f"max_steps must be int, not {type(max_steps).__name__}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    # Single pass over named_parameters(): layer index per param name (once),
    # and each layer's total parameter count (once — never step-multiplied).
    layer_of_param: dict[int, int] = {}
    param_count: dict[int, int] = {}
    named_params = list(model.named_parameters())
    for name, param in named_params:
        match = _LAYER_RE.search(name)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        layer_of_param[id(param)] = layer_idx
        param_count[layer_idx] = param_count.get(layer_idx, 0) + param.numel()

    if not layer_of_param:
        raise ValueError(
            "sensitivity probe: could not detect numbered decoder layers "
            "('layers.N.' / 'h.N.') in the model — the sensitivity probe "
            "cannot be applied to this architecture."
        )

    running_sum: dict[int, float] = {idx: 0.0 for idx in param_count}

    def _zero_grads() -> None:
        for _name, param in named_params:
            param.grad = None

    _zero_grads()

    steps_run = 0
    for batch in batches:
        if steps_run >= max_steps:
            break
        out = model(**batch)
        loss = out.loss
        loss.backward()

        with torch.no_grad():
            for _name, param in named_params:
                grad = param.grad
                if grad is None:
                    continue
                layer_idx = layer_of_param.get(id(param))
                if layer_idx is None:
                    continue
                contribution = (grad * param).abs().sum().item()
                running_sum[layer_idx] = running_sum[layer_idx] + float(contribution)

        _zero_grads()
        steps_run += 1

    if steps_run == 0:
        raise ValueError("sensitivity probe: `batches` yielded zero batches")

    scores = tuple(
        LayerSensitivity(
            layer_index=layer_idx,
            score=running_sum[layer_idx] / param_count[layer_idx],
            param_count=param_count[layer_idx],
        )
        for layer_idx in sorted(param_count)
    )

    from kadhi_cli.utils.spectrum_scan import model_slug

    slug = model_slug(getattr(model, "name_or_path", None) or type(model).__name__)

    return SensitivityReport(scores=scores, probe_steps=steps_run, model_slug_=slug)


def _rank_with_tie_averaging(values: Sequence[float]) -> list[float]:
    """Fractional (tie-averaged) ranks, 1-indexed, ascending by value."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Ranks i+1 .. j+1 (1-indexed) tie -> average rank.
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation between two equal-length numeric sequences,
    implemented by hand (rank both sequences, average ranks for ties, then
    Pearson correlation of the rank sequences) — no scipy/numpy dependency.
    Raise ValueError if len(a) != len(b) or len(a) < 2. Returns a float in
    [-1.0, 1.0]; returns 0.0 (not NaN, not a raise) for the degenerate case
    where one sequence is constant (zero variance) — document this choice."""
    if len(a) != len(b):
        raise ValueError(f"sequences must have equal length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError(f"sequences must have length >= 2, got {len(a)}")

    ranks_a = _rank_with_tie_averaging(list(a))
    ranks_b = _rank_with_tie_averaging(list(b))

    n = len(ranks_a)
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n

    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b))
    var_a = sum((x - mean_a) ** 2 for x in ranks_a)
    var_b = sum((y - mean_b) ** 2 for y in ranks_b)

    # Zero variance in either rank sequence means one input was constant
    # (every value tied) — Pearson correlation is undefined (0/0); return
    # 0.0 rather than NaN or raising, since "no discernible correlation" is
    # the honest reading of a constant signal.
    if var_a == 0.0 or var_b == 0.0:
        return 0.0

    correlation = cov / ((var_a ** 0.5) * (var_b ** 0.5))
    # Guard against float noise pushing a hair outside [-1, 1].
    return max(-1.0, min(1.0, correlation))


def correlate_with_static_signals(
    report: SensitivityReport, *,
    spectrum_scores: Optional[Mapping[int, float]] = None,
    shrink_scores: Optional[Mapping[int, float]] = None,
) -> dict:
    """Returns {"spectrum": float | None, "shrink": float | None} — Spearman
    correlation between `report`'s per-layer scores and each supplied static
    signal, restricted to the INTERSECTION of layer indices present in both
    (a static signal may not cover every layer the probe touched, e.g. if it
    was computed with a different `--modules` filter) — if the intersection
    has fewer than 2 layers, that entry is None rather than raising. A signal
    argument of None (not measured / not supplied) maps to None in the output,
    not 0.0 — these are different states and must not be conflated (None =
    "we don't know", 0.0 would falsely claim "we measured zero correlation")."""
    probe_scores = {item.layer_index: item.score for item in report.scores}

    def _correlate(static: Optional[Mapping[int, float]]) -> Optional[float]:
        if static is None:
            return None
        common = sorted(set(probe_scores) & set(static))
        if len(common) < 2:
            return None
        probe_vals = [probe_scores[idx] for idx in common]
        static_vals = [static[idx] for idx in common]
        return spearman_correlation(probe_vals, static_vals)

    return {
        "spectrum": _correlate(spectrum_scores),
        "shrink": _correlate(shrink_scores),
    }


def cache_path_for(model: str, dataset_fingerprint: str, cache_dir: Optional[str] = None) -> str:
    """os.path.join(resolve_cache_dir(cache_dir), "sensitivity",
    model_slug(model) + "-" + dataset_fingerprint + ".json") — import
    resolve_cache_dir and model_slug from kadhi_cli.utils.spectrum_scan."""
    from kadhi_cli.utils.spectrum_scan import model_slug, resolve_cache_dir

    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint.strip():
        raise ValueError("dataset_fingerprint must be a non-empty string")

    return os.path.join(
        resolve_cache_dir(cache_dir),
        "sensitivity",
        model_slug(model) + "-" + dataset_fingerprint + ".json",
    )


def _atomic_write_json(payload: dict, path: str) -> str:
    """Atomic JSON write, mirroring spectrum_scan.py's ``_atomic_write_json``
    precedent (mkstemp in the parent dir, write, ``os.replace``) — reimplemented
    locally rather than imported since spectrum_scan's helper is a private
    module-level function, not part of its public surface.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".kadhi.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return os.path.realpath(path)


def save_sensitivity_report(
    report: SensitivityReport, model: str, dataset_fingerprint: str, *,
    cache_dir: Optional[str] = None,
) -> str:
    """Serialize `report` to JSON at cache_path_for(...), atomic write (mirror
    spectrum_scan.py's _atomic_write_json pattern: mkstemp in the parent dir,
    write, os.replace — import and reuse spectrum_scan's helper if it's
    accessible, otherwise reimplement the same atomic-write shape locally with
    a comment noting the precedent). Returns the path written."""
    path = cache_path_for(model, dataset_fingerprint, cache_dir=cache_dir)
    payload = {
        "schema": _CACHE_SCHEMA,
        "model_slug_": report.model_slug_,
        "probe_steps": report.probe_steps,
        "scores": [
            {
                "layer_index": item.layer_index,
                "score": item.score,
                "param_count": item.param_count,
            }
            for item in report.scores
        ],
    }
    return _atomic_write_json(payload, path)


def load_sensitivity_report(
    model: str, dataset_fingerprint: str, *, cache_dir: Optional[str] = None,
) -> Optional[SensitivityReport]:
    """Read back a report saved by save_sensitivity_report, or None if the
    cache file doesn't exist / is corrupt (catch OSError, ValueError,
    TypeError, json.JSONDecodeError — never raise on a missing/bad cache,
    matching this repo's None-fallback convention throughout gpu.py /
    spectrum_scan.py / capacity.py)."""
    path = cache_path_for(model, dataset_fingerprint, cache_dir=cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("schema") != _CACHE_SCHEMA:
            return None
        scores = tuple(
            LayerSensitivity(
                layer_index=int(item["layer_index"]),
                score=float(item["score"]),
                param_count=int(item["param_count"]),
            )
            for item in payload["scores"]
        )
        return SensitivityReport(
            scores=scores,
            probe_steps=int(payload["probe_steps"]),
            model_slug_=str(payload["model_slug_"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
