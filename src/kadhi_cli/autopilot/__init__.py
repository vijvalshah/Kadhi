"""Autopilot — zero-config fine-tuning decision engine (Part H of v0.25.0)."""

from kadhi_cli.autopilot.analyzer import (
    DatasetProfile,
    HardwareProfile,
    ModelProfile,
    analyze_dataset,
    analyze_hardware,
    analyze_model,
)
from kadhi_cli.autopilot.decisions import (
    decide_batch_size,
    decide_epochs,
    decide_lr,
    decide_max_length,
    decide_peft,
    decide_performance_flags,
    decide_quantization,
    decide_task,
    detect_prequantized_format,
    detect_prequantized_format_from_path,
    parse_gpu_budget,
)
from kadhi_cli.autopilot.generate_config import build_kadhi_config, write_yaml

__all__ = [
    "DatasetProfile",
    "HardwareProfile",
    "ModelProfile",
    "analyze_dataset",
    "analyze_hardware",
    "analyze_model",
    "build_kadhi_config",
    "decide_batch_size",
    "decide_epochs",
    "decide_lr",
    "decide_max_length",
    "decide_peft",
    "decide_performance_flags",
    "decide_quantization",
    "decide_task",
    "detect_prequantized_format",
    "detect_prequantized_format_from_path",
    "parse_gpu_budget",
    "write_yaml",
]
