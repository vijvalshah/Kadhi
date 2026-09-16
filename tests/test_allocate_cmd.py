"""Tests for `kadhi allocate` (commands/allocate.py) — end to end via the
real Typer CLI app against a real synthetic checkpoint. No mocking: this
exercises config loading, the local-checkpoint check, capacity.py's shape
discovery, spectrum_scan's real SVD-based SNR, allocate.py's real greedy allocator,
and hardware_fit's real VRAM predictor, all through the actual `kadhi
allocate` command a user would run.
"""

from __future__ import annotations

import textwrap

import pytest
from typer.testing import CliRunner

from kadhi_cli.cli import app

runner = CliRunner()


def _write_checkpoint(tmp_path, dim=32):
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(2)
    tensors = {
        f"model.layers.{i}.self_attn.q_proj.weight": rng.standard_normal((dim, dim)).astype("float32")
        for i in range(2)
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))


def _write_config(tmp_path, *, vram_gb=None):
    # Built as a flat list of already-correctly-indented lines rather than a
    # single textwrap.dedent(f\"\"\"...\"\"\") block: an interpolated snippet
    # indented independently of the surrounding Python source line breaks
    # dedent's "strip the common leading whitespace" logic — it computes the
    # common prefix across ALL lines including the interpolated one, silently
    # under-dedenting everything else into invalid YAML. Caught by an actual
    # yaml.parser.ParserError the first time this test ran.
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    lines = [
        f"base: {tmp_path}",
        "task: sft",
        "data:",
        f"  train: {data_path}",
        "  format: chatml",
        "training:",
        "  lora:",
        "    r: 8",
        "    target_modules: [q_proj]",
        "controller:",
        "  enabled: true",
        "  budget:",
        "    trainable_params: 100000",
    ]
    if vram_gb is not None:
        lines.append(f"    vram_gb: {vram_gb}")
    cfg_path = tmp_path / "kadhi.yaml"
    cfg_path.write_text("\n".join(lines) + "\n")
    return cfg_path


def test_allocate_cmd_feasible_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    cfg_path = _write_config(tmp_path, vram_gb=1_000_000.0)

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "feasible" in result.output.lower()
    assert "True" in result.output


def test_allocate_cmd_explain_shows_rank_pattern_and_vram_breakdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    cfg_path = _write_config(tmp_path, vram_gb=1_000_000.0)

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path), "--explain"])
    assert result.exit_code == 0, result.output
    assert "rank_pattern" in result.output
    # Rich wraps this table's title across lines at narrow test-terminal
    # widths ("predicted peak VRAM" / "breakdown") — check the column
    # headers, which survive wrapping, rather than the exact title string.
    assert "bucket" in result.output and "activations" in result.output


def test_allocate_cmd_infeasible_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    cfg_path = _write_config(tmp_path, vram_gb=0.0000001)

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 3, result.output
    assert "not feasible" in result.output.lower()


def test_allocate_cmd_reports_infeasible_at_any_rank_distinctly(tmp_path, monkeypatch):
    """When even a fully-frozen allocation cannot fit, the overflow is base
    weights/activations/overhead — not the adapter. Saying only "not feasible"
    would send the operator off to tune a budget that was never the problem,
    so that case gets its own actionable message."""
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    cfg_path = _write_config(tmp_path, vram_gb=0.001)

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 3, result.output
    assert "Not feasible at ANY rank" in result.output
    assert "0 trainable" in result.output
    # It must name a lever that actually helps, and rule out the one that doesn't.
    assert "gradient_checkpointing" in result.output
    assert "Raising the parameter budget cannot help" in result.output


def test_allocate_cmd_requires_controller_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    cfg_path = tmp_path / "kadhi.yaml"
    cfg_path.write_text(textwrap.dedent(f"""\
        base: {tmp_path}
        task: sft
        data:
          train: {data_path}
          format: chatml
        """))

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 1, result.output
    assert "controller.enabled" in result.output


def test_allocate_cmd_requires_local_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    cfg_path = tmp_path / "kadhi.yaml"
    cfg_path.write_text(textwrap.dedent(f"""\
        base: definitely/not-a-local-path-{{__name__}}
        task: sft
        data:
          train: {data_path}
          format: chatml
        controller:
          enabled: true
          budget:
            trainable_params: 100000
        """))

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 1, result.output
    assert "not a local directory" in result.output


def test_allocate_cmd_requires_trainable_params_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_checkpoint(tmp_path)
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    cfg_path = tmp_path / "kadhi.yaml"
    cfg_path.write_text(textwrap.dedent(f"""\
        base: {tmp_path}
        task: sft
        data:
          train: {data_path}
          format: chatml
        controller:
          enabled: true
          budget:
            vram_gb: 4.0
        """))

    result = runner.invoke(app, ["allocate", "--config", str(cfg_path)])
    assert result.exit_code == 1, result.output
    assert "trainable_params is required" in result.output


def test_allocate_cmd_registered_in_help():
    """The command must actually be reachable from the top-level CLI, not
    just importable as a bare function — proves the cli.py registration."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "allocate" in result.output
