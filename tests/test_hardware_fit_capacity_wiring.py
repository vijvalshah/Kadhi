"""Tests for the Phase 1 resource-model correction (adaptation-controller-plan.md
§1.1): ``HardwareFitInput.trainable_params`` (an exact, opt-in override) and its
wiring through ``kadhi_cli.commands.train._build_hardware_fit_input``.

Covers:
1. ``trainable_params`` accepted (int) and accepted as the omitted/None default,
   with the None path byte-identical to a HardwareFitInput built without the
   field at all.
2. Validation: negative int rejected (ValueError), bool rejected (TypeError),
   float rejected (TypeError).
3. ``estimate_peak_vram_gb`` actually uses the override when supplied, and it
   materially undercuts the flat 1%-heuristic fallback for the same config.
4. ``_build_hardware_fit_input`` fails open (trainable_params=None) when
   ``cfg.base`` doesn't resolve to a local directory — the existing,
   unregressed behavior.
5. Integration: when pointed at a real local checkpoint directory, the wiring
   populates a non-None trainable_params for a ``peft: lora`` config.
"""

from __future__ import annotations

import pytest

from kadhi_cli.config.loader import load_config_from_string
from kadhi_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

_BASE_KW = dict(
    params_b=8.0,
    seq_len=2048,
    batch_size=4,
    optimizer="adamw_torch",
    quant="none",
    peft="lora",
    gradient_checkpointing=False,
)


# ─────────────────────────── field acceptance ───────────────────────────


def test_trainable_params_accepted_and_defaults_to_none():
    inp = HardwareFitInput(trainable_params=12345, **_BASE_KW)
    assert inp.trainable_params == 12345

    inp_default = HardwareFitInput(**_BASE_KW)
    assert inp_default.trainable_params is None

    inp_explicit_none = HardwareFitInput(trainable_params=None, **_BASE_KW)
    assert inp_explicit_none.trainable_params is None

    # None (default or explicit) must behave byte-for-byte identically to a
    # HardwareFitInput built without the field at all.
    b1 = estimate_peak_vram_gb(inp_default)
    b2 = estimate_peak_vram_gb(inp_explicit_none)
    assert b1 == b2


# ─────────────────────────── validation ───────────────────────────


def test_trainable_params_rejects_negative():
    with pytest.raises(ValueError):
        HardwareFitInput(trainable_params=-1, **_BASE_KW)


def test_trainable_params_rejects_bool():
    with pytest.raises(TypeError):
        HardwareFitInput(trainable_params=True, **_BASE_KW)


def test_trainable_params_rejects_float():
    with pytest.raises(TypeError):
        HardwareFitInput(trainable_params=1.5, **_BASE_KW)


# ─────────────────────────── override actually takes effect ───────────────


def test_override_produces_materially_smaller_optimizer_and_gradients():
    # 8B model, LoRA -> fallback heuristic is 1% = 80,000,000 params.
    fallback = estimate_peak_vram_gb(HardwareFitInput(trainable_params=None, **_BASE_KW))
    exact = estimate_peak_vram_gb(HardwareFitInput(trainable_params=8_000_000, **_BASE_KW))

    fallback_total = fallback.optimizer_gb + fallback.gradients_gb
    exact_total = exact.optimizer_gb + exact.gradients_gb

    assert exact_total < fallback_total
    # 8,000,000 / 80,000,000 == 0.1 -> optimizer+gradients should track that
    # ratio exactly, since both scale linearly with trainable_params.
    ratio = exact_total / fallback_total
    assert ratio == pytest.approx(0.1, rel=1e-6)


# ─────────────────────────── _build_hardware_fit_input wiring ─────────────

_NONLOCAL_BASE_YAML = """
base: meta-llama/Llama-2-7b-hf
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 8
  quantization: none
  lora:
    r: 16
"""


def test_build_hardware_fit_input_fails_open_for_nonlocal_checkpoint():
    from kadhi_cli.commands.train import _build_hardware_fit_input

    cfg = load_config_from_string(_NONLOCAL_BASE_YAML)
    inp = _build_hardware_fit_input(cfg)
    assert inp is not None
    assert inp.trainable_params is None
    assert inp.peft == "lora"


def test_build_hardware_fit_input_integration_local_checkpoint(tmp_path):
    """When cfg.base is a local directory with a real checkpoint, the exact
    accounting from utils/capacity.py should populate trainable_params."""
    try:
        import kadhi_cli.utils.capacity as capacity  # noqa: F401
    except ImportError:
        pytest.skip("capacity.py not yet available")

    import json
    import struct

    from kadhi_cli.commands.train import _build_hardware_fit_input

    # Minimal synthetic safetensors shard: one 2-D q_proj weight tensor,
    # header-only (shape info), same manual little-endian-u64-length + JSON
    # header format capacity.py's discovery reads. No tensor data needed
    # since only shapes are inspected.
    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [4096, 4096],
            "data_offsets": [0, 0],
        },
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    weights_dir = tmp_path / "tiny-checkpoint"
    weights_dir.mkdir()
    shard_path = weights_dir / "model.safetensors"
    with open(shard_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)

    yaml_str = f"""
base: {weights_dir}
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 8
  quantization: none
  lora:
    r: 16
    target_modules: auto
"""
    cfg = load_config_from_string(yaml_str)
    inp = _build_hardware_fit_input(cfg)
    assert inp is not None
    assert inp.peft == "lora"
    assert inp.trainable_params is not None
    assert inp.trainable_params > 0
    # 16 * (4096 + 4096) for the single q_proj shape.
    assert inp.trainable_params == 16 * (4096 + 4096)
