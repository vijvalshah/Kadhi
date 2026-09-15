"""Tests for kadhi_cli.utils.sensitivity — task-conditional layer sensitivity probe."""

from __future__ import annotations

import json
import sys

import pytest


# ---------------------------------------------------------------------------
# Module imports (Tier 1 — no torch required)
# ---------------------------------------------------------------------------


def test_module_imports_without_heavy_deps():
    from kadhi_cli.utils import sensitivity

    assert hasattr(sensitivity, "LayerSensitivity")
    assert hasattr(sensitivity, "SensitivityReport")
    assert hasattr(sensitivity, "compute_layer_sensitivity")
    assert hasattr(sensitivity, "spearman_correlation")
    assert hasattr(sensitivity, "correlate_with_static_signals")
    assert hasattr(sensitivity, "cache_path_for")
    assert hasattr(sensitivity, "save_sensitivity_report")
    assert hasattr(sensitivity, "load_sensitivity_report")
    assert "torch" not in sys.modules
    assert "numpy" not in sys.modules


# ---------------------------------------------------------------------------
# LayerSensitivity validation
# ---------------------------------------------------------------------------


def test_layer_sensitivity_valid():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    ls = LayerSensitivity(layer_index=0, score=1.5, param_count=100)
    assert ls.layer_index == 0
    assert ls.score == 1.5
    assert ls.param_count == 100


def test_layer_sensitivity_rejects_bool_layer_index():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(TypeError):
        LayerSensitivity(layer_index=True, score=1.0, param_count=10)


def test_layer_sensitivity_rejects_negative_layer_index():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=-1, score=1.0, param_count=10)


def test_layer_sensitivity_rejects_non_int_layer_index():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(TypeError):
        LayerSensitivity(layer_index=1.5, score=1.0, param_count=10)


def test_layer_sensitivity_rejects_negative_score():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=0, score=-0.1, param_count=10)


def test_layer_sensitivity_rejects_non_finite_score():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=0, score=float("inf"), param_count=10)
    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=0, score=float("nan"), param_count=10)


def test_layer_sensitivity_rejects_bool_param_count():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(TypeError):
        LayerSensitivity(layer_index=0, score=1.0, param_count=True)


def test_layer_sensitivity_rejects_non_positive_param_count():
    from kadhi_cli.utils.sensitivity import LayerSensitivity

    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=0, score=1.0, param_count=0)
    with pytest.raises(ValueError):
        LayerSensitivity(layer_index=0, score=1.0, param_count=-5)


# ---------------------------------------------------------------------------
# SensitivityReport validation
# ---------------------------------------------------------------------------


def test_sensitivity_report_valid():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    report = SensitivityReport(
        scores=(
            LayerSensitivity(layer_index=0, score=1.0, param_count=10),
            LayerSensitivity(layer_index=1, score=2.0, param_count=20),
        ),
        probe_steps=5,
        model_slug_="my-model",
    )
    assert report.probe_steps == 5
    assert report.model_slug_ == "my-model"


def test_sensitivity_report_rejects_empty_scores():
    from kadhi_cli.utils.sensitivity import SensitivityReport

    with pytest.raises(ValueError):
        SensitivityReport(scores=(), probe_steps=1, model_slug_="m")


def test_sensitivity_report_rejects_unsorted_scores():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    with pytest.raises(ValueError):
        SensitivityReport(
            scores=(
                LayerSensitivity(layer_index=1, score=1.0, param_count=10),
                LayerSensitivity(layer_index=0, score=2.0, param_count=20),
            ),
            probe_steps=1,
            model_slug_="m",
        )


def test_sensitivity_report_rejects_duplicate_layer_index():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    with pytest.raises(ValueError):
        SensitivityReport(
            scores=(
                LayerSensitivity(layer_index=0, score=1.0, param_count=10),
                LayerSensitivity(layer_index=0, score=2.0, param_count=20),
            ),
            probe_steps=1,
            model_slug_="m",
        )


def test_sensitivity_report_rejects_non_positive_probe_steps():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    with pytest.raises(ValueError):
        SensitivityReport(
            scores=(LayerSensitivity(layer_index=0, score=1.0, param_count=10),),
            probe_steps=0,
            model_slug_="m",
        )


def test_sensitivity_report_rejects_bool_probe_steps():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    with pytest.raises(TypeError):
        SensitivityReport(
            scores=(LayerSensitivity(layer_index=0, score=1.0, param_count=10),),
            probe_steps=True,
            model_slug_="m",
        )


def test_sensitivity_report_rejects_empty_model_slug():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    with pytest.raises(TypeError):
        SensitivityReport(
            scores=(LayerSensitivity(layer_index=0, score=1.0, param_count=10),),
            probe_steps=1,
            model_slug_="",
        )


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------


def test_spearman_correlation_perfect_positive():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_correlation_perfect_negative():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_correlation_tie_averaging():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    # a has a tie at positions 1,2 (values 2,2) -> averaged ranks [1, 2.5, 2.5, 4]
    # naive integer ranking (no tie-average) would give ranks [1, 2, 3, 4] and a
    # DIFFERENT (higher-magnitude) correlation with b's ranks [1, 2, 3, 4].
    a = [1, 2, 2, 3]
    b = [1, 2, 3, 4]
    # Hand-computed: ranks_a = [1, 2.5, 2.5, 4], ranks_b = [1, 2, 3, 4]
    # mean_a = 2.5, mean_b = 2.5
    # deviations_a = [-1.5, 0, 0, 1.5]; deviations_b = [-1.5, -0.5, 0.5, 1.5]
    # cov = (-1.5*-1.5) + (0*-0.5) + (0*0.5) + (1.5*1.5) = 2.25 + 0 + 0 + 2.25 = 4.5
    # var_a = 1.5^2*2 = 4.5; var_b = 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 = 2.25+0.25+0.25+2.25=5.0
    # corr = 4.5 / sqrt(4.5 * 5.0) = 4.5 / sqrt(22.5) = 4.5 / 4.743... = 0.94868...
    expected = 4.5 / ((4.5 * 5.0) ** 0.5)
    assert spearman_correlation(a, b) == pytest.approx(expected)
    # And it must NOT equal the naive (no-tie-averaging) integer-rank result.
    naive_a_ranks = [1, 2, 3, 4]
    naive_mean = 2.5
    naive_dev = [x - naive_mean for x in naive_a_ranks]
    b_dev = [x - naive_mean for x in [1, 2, 3, 4]]
    naive_cov = sum(x * y for x, y in zip(naive_dev, b_dev))
    naive_var_a = sum(x * x for x in naive_dev)
    naive_var_b = sum(y * y for y in b_dev)
    naive_corr = naive_cov / ((naive_var_a * naive_var_b) ** 0.5)
    assert spearman_correlation(a, b) != pytest.approx(naive_corr)


def test_spearman_correlation_constant_sequence_returns_zero():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    result = spearman_correlation([5, 5, 5, 5], [1, 2, 3, 4])
    assert result == 0.0


def test_spearman_correlation_mismatched_lengths_raises():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    with pytest.raises(ValueError):
        spearman_correlation([1, 2, 3], [1, 2])


def test_spearman_correlation_too_short_raises():
    from kadhi_cli.utils.sensitivity import spearman_correlation

    with pytest.raises(ValueError):
        spearman_correlation([1], [1])


# ---------------------------------------------------------------------------
# correlate_with_static_signals
# ---------------------------------------------------------------------------


def _make_report():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport

    return SensitivityReport(
        scores=(
            LayerSensitivity(layer_index=0, score=1.0, param_count=10),
            LayerSensitivity(layer_index=1, score=2.0, param_count=10),
            LayerSensitivity(layer_index=2, score=3.0, param_count=10),
        ),
        probe_steps=3,
        model_slug_="m",
    )


def test_correlate_with_static_signals_both_none():
    from kadhi_cli.utils.sensitivity import correlate_with_static_signals

    report = _make_report()
    result = correlate_with_static_signals(report)
    assert result == {"spectrum": None, "shrink": None}


def test_correlate_with_static_signals_disjoint_layers_is_none():
    from kadhi_cli.utils.sensitivity import correlate_with_static_signals

    report = _make_report()
    result = correlate_with_static_signals(report, spectrum_scores={10: 1.0, 11: 2.0})
    assert result["spectrum"] is None
    assert result["shrink"] is None


def test_correlate_with_static_signals_two_overlapping_layers_returns_float():
    from kadhi_cli.utils.sensitivity import correlate_with_static_signals, spearman_correlation

    report = _make_report()
    spectrum = {0: 5.0, 1: 10.0}  # overlaps layers 0,1 only
    result = correlate_with_static_signals(report, spectrum_scores=spectrum)
    assert isinstance(result["spectrum"], float)
    expected = spearman_correlation([1.0, 2.0], [5.0, 10.0])
    assert result["spectrum"] == pytest.approx(expected)
    assert result["shrink"] is None


def test_correlate_with_static_signals_hand_computed_example():
    from kadhi_cli.utils.sensitivity import LayerSensitivity, SensitivityReport, correlate_with_static_signals

    report = SensitivityReport(
        scores=(
            LayerSensitivity(layer_index=0, score=1.0, param_count=10),
            LayerSensitivity(layer_index=1, score=2.0, param_count=10),
            LayerSensitivity(layer_index=2, score=2.0, param_count=10),
            LayerSensitivity(layer_index=3, score=3.0, param_count=10),
        ),
        probe_steps=3,
        model_slug_="m",
    )
    shrink = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    result = correlate_with_static_signals(report, shrink_scores=shrink)
    # probe scores [1,2,2,3] vs shrink [1,2,3,4] -- same as the tie-averaging test above.
    expected = 4.5 / ((4.5 * 5.0) ** 0.5)
    assert result["shrink"] == pytest.approx(expected)
    assert result["spectrum"] is None


# ---------------------------------------------------------------------------
# cache_path_for / save / load
# ---------------------------------------------------------------------------


def test_cache_path_for_contains_slug_and_fingerprint(tmp_path):
    from kadhi_cli.utils.sensitivity import cache_path_for

    path = cache_path_for("meta-llama/Llama-3.1-8B", "abc123", cache_dir=str(tmp_path))
    assert "sensitivity" in path
    assert "abc123" in path
    assert "meta-llama" in path or "Llama-3.1-8B" in path or "meta-llama__Llama-3.1-8B" in path


def test_save_and_load_sensitivity_report_round_trip(tmp_path):
    from kadhi_cli.utils.sensitivity import (
        LayerSensitivity,
        SensitivityReport,
        load_sensitivity_report,
        save_sensitivity_report,
    )

    report = SensitivityReport(
        scores=(
            LayerSensitivity(layer_index=0, score=1.25, param_count=100),
            LayerSensitivity(layer_index=1, score=2.5, param_count=200),
        ),
        probe_steps=10,
        model_slug_="my-model-slug",
    )
    path = save_sensitivity_report(report, "my-model", "fp-1", cache_dir=str(tmp_path))
    assert path

    loaded = load_sensitivity_report("my-model", "fp-1", cache_dir=str(tmp_path))
    assert loaded is not None
    assert loaded.probe_steps == report.probe_steps
    assert loaded.model_slug_ == report.model_slug_
    assert loaded.scores == report.scores


def test_load_sensitivity_report_missing_file_returns_none(tmp_path):
    from kadhi_cli.utils.sensitivity import load_sensitivity_report

    assert load_sensitivity_report("no-such-model", "fp", cache_dir=str(tmp_path)) is None


def test_load_sensitivity_report_corrupt_json_returns_none(tmp_path):
    from kadhi_cli.utils.sensitivity import cache_path_for, load_sensitivity_report
    import os

    path = cache_path_for("bad-model", "fp", cache_dir=str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not valid json::")

    assert load_sensitivity_report("bad-model", "fp", cache_dir=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Tier 2 — requires torch
# ---------------------------------------------------------------------------


def _build_fake_model(torch, num_layers=3, dim=4):
    """A tiny fake decoder-ish model: layers.0..N-1, each an nn.Linear."""

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.proj(x)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(FakeLayer() for _ in range(num_layers))

        def forward(self, **batch):
            x = batch["input_ids"]
            for layer in self.layers:
                x = layer(x)
            target = batch["labels"]
            loss = ((x - target) ** 2).mean()

            class Out:
                pass

            out = Out()
            out.loss = loss
            return out

    model = FakeModel()
    model.train()
    return model


def _make_batches(torch, n_batches, dim=4):
    batches = []
    for _ in range(n_batches):
        batches.append(
            {
                "input_ids": torch.randn(2, dim),
                "labels": torch.randn(2, dim),
            }
        )
    return batches


def test_compute_layer_sensitivity_basic():
    torch = pytest.importorskip("torch")
    from kadhi_cli.utils.sensitivity import SensitivityReport, compute_layer_sensitivity

    model = _build_fake_model(torch, num_layers=3)
    snapshot = [p.data.clone() for p in model.parameters()]

    batches = _make_batches(torch, 2)
    report = compute_layer_sensitivity(model, batches, max_steps=2)

    assert isinstance(report, SensitivityReport)
    assert len(report.scores) == 3
    assert report.probe_steps == 2
    for item in report.scores:
        assert item.score >= 0.0
        import math

        assert math.isfinite(item.score)

    for before, after in zip(snapshot, model.parameters()):
        assert torch.equal(before, after.data)


def test_compute_layer_sensitivity_reports_real_step_count():
    torch = pytest.importorskip("torch")
    from kadhi_cli.utils.sensitivity import compute_layer_sensitivity

    model = _build_fake_model(torch, num_layers=2)
    batches = _make_batches(torch, 1)
    report = compute_layer_sensitivity(model, batches, max_steps=50)
    assert report.probe_steps == 1


def test_compute_layer_sensitivity_zero_batches_raises():
    torch = pytest.importorskip("torch")
    from kadhi_cli.utils.sensitivity import compute_layer_sensitivity

    model = _build_fake_model(torch, num_layers=2)
    with pytest.raises(ValueError):
        compute_layer_sensitivity(model, [], max_steps=5)


def test_compute_layer_sensitivity_no_layers_raises():
    torch = pytest.importorskip("torch")
    from kadhi_cli.utils.sensitivity import compute_layer_sensitivity

    class NoLayerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, **batch):
            x = self.proj(batch["input_ids"])
            target = batch["labels"]

            class Out:
                pass

            out = Out()
            out.loss = ((x - target) ** 2).mean()
            return out

    model = NoLayerModel()
    model.train()
    batches = _make_batches(torch, 2)
    with pytest.raises(ValueError):
        compute_layer_sensitivity(model, batches, max_steps=2)
