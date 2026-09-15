"""Tests for kadhi_cli.utils.capacity — exact LoRA/DoRA trainable-param accounting."""

from __future__ import annotations

import json
import struct

import pytest


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from kadhi_cli.utils import capacity

    assert hasattr(capacity, "LoraModuleShape")
    assert hasattr(capacity, "resolve_pattern_rank")
    assert hasattr(capacity, "discover_lora_module_shapes")
    assert hasattr(capacity, "count_lora_trainable_params")
    assert hasattr(capacity, "estimate_lora_trainable_params_from_checkpoint")


# ---------------------------------------------------------------------------
# resolve_pattern_rank
# ---------------------------------------------------------------------------


def test_resolve_pattern_rank_no_pattern_returns_default():
    from kadhi_cli.utils.capacity import resolve_pattern_rank

    assert resolve_pattern_rank(None, "model.layers.5.self_attn.q_proj", 8) == 8
    assert resolve_pattern_rank({}, "model.layers.5.self_attn.q_proj", 8) == 8


def test_resolve_pattern_rank_plain_segment_matches_any_layer():
    from kadhi_cli.utils.capacity import resolve_pattern_rank

    pattern = {"q_proj": 32}
    assert resolve_pattern_rank(pattern, "model.layers.0.self_attn.q_proj", 8) == 32
    assert resolve_pattern_rank(pattern, "model.layers.31.self_attn.q_proj", 8) == 32


def test_resolve_pattern_rank_layer_specific_matches_only_that_layer():
    from kadhi_cli.utils.capacity import resolve_pattern_rank

    pattern = {"layers.5.self_attn.q_proj": 64}
    assert (
        resolve_pattern_rank(pattern, "model.layers.5.self_attn.q_proj", 8) == 64
    )
    # layer 6 must NOT match layer 5's specific pattern -> falls to default
    assert resolve_pattern_rank(pattern, "model.layers.6.self_attn.q_proj", 8) == 8


def test_resolve_pattern_rank_first_match_wins_by_insertion_order():
    from kadhi_cli.utils.capacity import resolve_pattern_rank

    # Both keys COULD match "model.layers.5.self_attn.q_proj" — the plain
    # "q_proj" key is inserted first, so it must win even though the more
    # specific "layers.5.self_attn.q_proj" key also matches. This proves
    # order, not specificity, governs — matching real PEFT behavior exactly.
    pattern = {"q_proj": 16, "layers.5.self_attn.q_proj": 64}
    assert resolve_pattern_rank(pattern, "model.layers.5.self_attn.q_proj", 8) == 16


def test_resolve_pattern_rank_regex_fragment_pattern():
    from kadhi_cli.utils.capacity import resolve_pattern_rank

    pattern = {"experts.*.w1": 4}
    assert (
        resolve_pattern_rank(pattern, "model.layers.3.experts.7.w1", 8) == 4
    )
    # No "experts" segment at all -> regex cannot match -> falls to default.
    assert resolve_pattern_rank(pattern, "model.layers.3.moe.w2", 8) == 8


# ---------------------------------------------------------------------------
# LoraModuleShape validation
# ---------------------------------------------------------------------------


def test_lora_module_shape_happy_path():
    from kadhi_cli.utils.capacity import LoraModuleShape

    shape = LoraModuleShape(name="model.layers.0.self_attn.q_proj", in_features=4096, out_features=4096)
    assert shape.in_features == 4096
    assert shape.out_features == 4096


@pytest.mark.parametrize("bad_in,bad_out", [(0, 4096), (-1, 4096), (4096, 0), (4096, -5)])
def test_lora_module_shape_rejects_non_positive_dims(bad_in, bad_out):
    from kadhi_cli.utils.capacity import LoraModuleShape

    with pytest.raises(ValueError):
        LoraModuleShape(name="x", in_features=bad_in, out_features=bad_out)


def test_lora_module_shape_rejects_bool_dims():
    from kadhi_cli.utils.capacity import LoraModuleShape

    with pytest.raises(TypeError):
        LoraModuleShape(name="x", in_features=True, out_features=4096)
    with pytest.raises(TypeError):
        LoraModuleShape(name="x", in_features=4096, out_features=False)


def test_lora_module_shape_rejects_empty_name():
    from kadhi_cli.utils.capacity import LoraModuleShape

    with pytest.raises(ValueError):
        LoraModuleShape(name="", in_features=4096, out_features=4096)


# ---------------------------------------------------------------------------
# count_lora_trainable_params
# ---------------------------------------------------------------------------


def test_count_lora_trainable_params_known_sum():
    from kadhi_cli.utils.capacity import LoraModuleShape, count_lora_trainable_params

    shapes = [
        LoraModuleShape(name="model.layers.0.self_attn.q_proj", in_features=4096, out_features=4096),
        LoraModuleShape(name="model.layers.0.mlp.gate_proj", in_features=4096, out_features=11008),
    ]
    default_r = 8
    rank_pattern = {"gate_proj": 16}
    expected = 8 * (4096 + 4096) + 16 * (4096 + 11008)
    assert (
        count_lora_trainable_params(shapes, default_r=default_r, rank_pattern=rank_pattern)
        == expected
    )


def test_count_lora_trainable_params_zero_rank_excludes_shape():
    from kadhi_cli.utils.capacity import LoraModuleShape, count_lora_trainable_params

    shapes = [
        LoraModuleShape(name="model.layers.0.self_attn.q_proj", in_features=4096, out_features=4096),
        LoraModuleShape(name="model.layers.0.self_attn.k_proj", in_features=4096, out_features=4096),
    ]
    rank_pattern = {"k_proj": 0}
    expected = 8 * (4096 + 4096)  # only q_proj counted
    assert (
        count_lora_trainable_params(shapes, default_r=8, rank_pattern=rank_pattern)
        == expected
    )


def test_count_lora_trainable_params_use_dora_adds_out_features():
    from kadhi_cli.utils.capacity import LoraModuleShape, count_lora_trainable_params

    shapes = [
        LoraModuleShape(name="q_proj", in_features=100, out_features=200),
    ]
    without_dora = count_lora_trainable_params(shapes, default_r=4)
    with_dora = count_lora_trainable_params(shapes, default_r=4, use_dora=True)
    assert with_dora - without_dora == 200


def test_count_lora_trainable_params_rejects_bad_default_r():
    from kadhi_cli.utils.capacity import LoraModuleShape, count_lora_trainable_params

    shapes = [LoraModuleShape(name="q_proj", in_features=10, out_features=10)]
    with pytest.raises(TypeError):
        count_lora_trainable_params(shapes, default_r=True)
    with pytest.raises(ValueError):
        count_lora_trainable_params(shapes, default_r=-1)


# ---------------------------------------------------------------------------
# Synthetic safetensors fixture helpers
# ---------------------------------------------------------------------------


def _write_safetensors(path, tensors: dict) -> None:
    """Write a minimal valid safetensors file by hand (header only matters).

    ``tensors`` maps name -> (dtype_str, shape_list). Data bytes are dummy
    zero bytes sized to match the declared shape/dtype (only the header is
    ever read by the module under test, but real byte counts keep the file
    internally consistent for anyone who does open it with real tooling).
    """
    itemsize = {"F32": 4, "F16": 2, "BF16": 2}
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * itemsize[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(b"\x00" * offset)


# ---------------------------------------------------------------------------
# discover_lora_module_shapes + estimate_lora_trainable_params_from_checkpoint
# ---------------------------------------------------------------------------


def test_discover_lora_module_shapes_from_synthetic_checkpoint(tmp_path):
    from kadhi_cli.utils.capacity import discover_lora_module_shapes

    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors(
        shard,
        {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [4096, 4096]),
            "model.layers.0.mlp.gate_proj.weight": ("F32", [11008, 4096]),
            "model.layers.0.input_layernorm.weight": ("F32", [4096]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", [64]),
        },
    )

    shapes = discover_lora_module_shapes(str(tmp_path), "auto")
    by_name = {s.name: (s.in_features, s.out_features) for s in shapes}

    assert by_name == {
        "model.layers.0.self_attn.q_proj": (4096, 4096),
        "model.layers.0.mlp.gate_proj": (4096, 11008),
    }
    # 1-D tensors and non-target modules must be excluded.
    assert "model.layers.0.input_layernorm" not in by_name


def test_discover_lora_module_shapes_explicit_target_list(tmp_path):
    from kadhi_cli.utils.capacity import discover_lora_module_shapes

    shard = tmp_path / "model.safetensors"
    _write_safetensors(
        shard,
        {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [4096, 4096]),
            "model.layers.0.mlp.gate_proj.weight": ("F32", [11008, 4096]),
        },
    )

    shapes = discover_lora_module_shapes(str(tmp_path), ["gate_proj"])
    assert len(shapes) == 1
    assert shapes[0].name == "model.layers.0.mlp.gate_proj"


def test_estimate_lora_trainable_params_from_checkpoint_matches_manual_count(tmp_path):
    from kadhi_cli.utils.capacity import estimate_lora_trainable_params_from_checkpoint

    shard = tmp_path / "model.safetensors"
    _write_safetensors(
        shard,
        {
            "model.layers.0.self_attn.q_proj.weight": ("F32", [4096, 4096]),
            "model.layers.0.mlp.gate_proj.weight": ("F32", [11008, 4096]),
        },
    )

    result = estimate_lora_trainable_params_from_checkpoint(
        str(tmp_path), "auto", default_r=8
    )
    expected = 8 * (4096 + 4096) + 8 * (4096 + 11008)
    assert result == expected


# ---------------------------------------------------------------------------
# None-fallback contract
# ---------------------------------------------------------------------------


def test_estimate_returns_none_for_nonexistent_directory(tmp_path):
    from kadhi_cli.utils.capacity import estimate_lora_trainable_params_from_checkpoint

    missing = tmp_path / "does-not-exist"
    assert (
        estimate_lora_trainable_params_from_checkpoint(str(missing), "auto", default_r=8)
        is None
    )


def test_estimate_returns_none_for_empty_directory(tmp_path):
    from kadhi_cli.utils.capacity import estimate_lora_trainable_params_from_checkpoint

    assert (
        estimate_lora_trainable_params_from_checkpoint(str(tmp_path), "auto", default_r=8)
        is None
    )


def test_discover_returns_empty_tuple_not_none_for_empty_directory(tmp_path):
    from kadhi_cli.utils.capacity import discover_lora_module_shapes

    result = discover_lora_module_shapes(str(tmp_path), "auto")
    assert result == ()


# ---------------------------------------------------------------------------
# Defensive: corrupt/truncated shard must not raise
# ---------------------------------------------------------------------------


def test_corrupt_safetensors_file_does_not_raise(tmp_path):
    from kadhi_cli.utils.capacity import (
        discover_lora_module_shapes,
        estimate_lora_trainable_params_from_checkpoint,
    )

    garbage = tmp_path / "corrupt.safetensors"
    garbage.write_bytes(b"\xff" * 64)

    # Neither call should raise; both fall back to "nothing found".
    assert discover_lora_module_shapes(str(tmp_path), "auto") == ()
    assert (
        estimate_lora_trainable_params_from_checkpoint(str(tmp_path), "auto", default_r=8)
        is None
    )
