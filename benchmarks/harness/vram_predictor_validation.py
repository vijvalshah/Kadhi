#!/usr/bin/env python3
"""Resource-model validation: predicted vs measured peak training VRAM.

``docs/adaptation-controller-plan.md`` phase 1 ("Resource model validation",
§1.2) asks for exactly this: the analytical predictor in
``kadhi_cli.utils.hardware_fit.estimate_peak_vram_gb`` has never been checked
against measurement. This script compares its prediction against the real
peak CUDA memory of an actual training step, across a sweep of
(quantization, peft, rank) points at one fixed small model size.

REQUIREMENTS TO RUN MEANINGFULLY: a CUDA GPU, and a small model available
locally (either already cached by ``transformers``/the HF hub, or a synthetic
checkpoint from ``benchmarks/harness/synth_checkpoint.py``). This sandbox has
no GPU, so this script has NOT been run here and NO measured numbers exist
yet in this repo from this change — do not treat any number in this docstring
or in code as a measured result. Only a real run on real hardware produces
data for ``benchmarks/results/``.

TODO (follow-up PR): ``run_one``'s measurement half is a deliberate stub.
Wiring a real forward+backward training step needs, at minimum: resolving a
tokenizer + dataset batch of the right ``seq_len``/``batch_size`` shape,
building the actual quantization config (bnb 4-bit/8-bit) and PEFT wrapper
(LoRA/DoRA) that ``kadhi_cli.trainer.sft.SFTTrainerWrapper`` would build for
the given config, and running exactly one optimizer step before reading
``torch.cuda.max_memory_allocated()``. The SFT trainer's setup path is the
right thing to reuse rather than re-implementing PEFT wiring here (see
``src/kadhi_cli/trainer/sft.py``), but that setup is not a small standalone
call — it currently lives inside ``SFTTrainerWrapper``'s larger training
loop, and lifting the minimal slice out is bigger than fits this harness
script. A correct, honest partial script beats a fragile speculative one:
``run_one`` therefore ALWAYS returns ``measured_gb=None`` for now, and a real
measurement should replace the ``# TODO: real training step`` block below.

Typical invocation::

    python benchmarks/harness/vram_predictor_validation.py \
        --model-dir /path/to/local/small/model --seq-len 512 --batch-size 1

A machine without CUDA still runs the (cheap, real) prediction half and
prints a clear notice that no measured numbers are available in this
environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def _kadhi_src_on_path() -> None:
    """Make ``kadhi_cli`` importable regardless of invocation cwd/install."""
    here = Path(__file__).resolve()
    src = here.parents[2] / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_kadhi_src_on_path()


def run_one(
    model_dir: str,
    *,
    seq_len: int,
    batch_size: int,
    quant: str,
    peft: str,
    optimizer: str,
    gradient_checkpointing: bool,
    lora_r: int = 16,
) -> dict:
    """Predict (and, where wired, measure) peak VRAM for one config point.

    Mirrors ``kadhi_cli.commands.train._build_hardware_fit_input``'s exact-
    accounting wiring exactly: ``trainable_params`` is computed via
    ``utils.capacity.estimate_lora_trainable_params_from_checkpoint`` when
    ``peft != "full"`` and ``model_dir`` is a local directory with a real
    checkpoint; otherwise it stays ``None`` and ``estimate_peak_vram_gb``
    falls back to its flat 1%-of-params heuristic. Do not diverge from this
    logic — the whole point of this harness is validating the SAME code path
    training pre-flight uses, not a parallel approximation of it.

    Returns a dict with ``predicted_gb`` (always populated) and
    ``measured_gb`` (``None`` unless a CUDA training step has been wired in,
    see the module TODO above).
    """
    import os

    from kadhi_cli.utils.gpu import model_size_from_name
    from kadhi_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

    params_b = model_size_from_name(model_dir)
    if not isinstance(params_b, (int, float)) or params_b <= 0:
        raise ValueError(
            f"could not resolve a parameter count for model_dir={model_dir!r}; "
            "pass a directory/name utils.gpu.model_size_from_name recognizes"
        )

    trainable_params: Optional[int] = None
    if peft != "full" and os.path.isdir(model_dir):
        try:
            from kadhi_cli.utils.capacity import (
                estimate_lora_trainable_params_from_checkpoint,
            )

            trainable_params = estimate_lora_trainable_params_from_checkpoint(
                weights_dir=model_dir,
                target_modules="auto",
                default_r=lora_r,
                rank_pattern=None,
                use_dora=(peft == "dora"),
            )
        except (AttributeError, TypeError, ValueError, OSError):
            trainable_params = None  # fail open, matches train.py's wiring

    inp = HardwareFitInput(
        params_b=float(params_b),
        seq_len=seq_len,
        batch_size=batch_size,
        optimizer=optimizer,
        quant=quant,
        peft=peft,
        gradient_checkpointing=gradient_checkpointing,
        trainable_params=trainable_params,
    )
    breakdown = estimate_peak_vram_gb(inp)
    predicted_gb = breakdown.total_gb

    measured_gb: Optional[float] = None
    try:
        import torch

        if torch.cuda.is_available():
            # TODO: real training step. See module docstring — this is the
            # deliberate stub. A follow-up PR should: load the model at
            # `model_dir` with the requested `quant`/`peft` config through
            # the SFT trainer's own setup path, build one batch of shape
            # (batch_size, seq_len), run one forward + backward + optimizer
            # step, and read torch.cuda.max_memory_allocated() after a
            # torch.cuda.reset_peak_memory_stats() taken before the step.
            # Left unimplemented here rather than faked.
            pass
    except ImportError:
        pass  # torch not installed in this environment — prediction still works

    return {
        "model_dir": model_dir,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "quant": quant,
        "peft": peft,
        "optimizer": optimizer,
        "gradient_checkpointing": gradient_checkpointing,
        "lora_r": lora_r,
        "predicted_gb": predicted_gb,
        "measured_gb": measured_gb,
        "breakdown": {
            "weights_gb": breakdown.weights_gb,
            "optimizer_gb": breakdown.optimizer_gb,
            "gradients_gb": breakdown.gradients_gb,
            "activations_gb": breakdown.activations_gb,
            "overhead_gb": breakdown.overhead_gb,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model-dir",
        default="meta-llama/Llama-3.2-1B",
        help="local checkpoint directory (preferred, enables exact LoRA "
        "accounting) or a model name utils.gpu.model_size_from_name "
        "recognizes for the size-only fallback",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optimizer", default="adamw_torch")
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=False
    )
    return parser.parse_args()


# One fixed small model size; sweep quant × peft × rank.
_SWEEP_POINTS = [
    # (quant, peft, lora_r)
    ("none", "full", 0),
    ("none", "lora", 4),
    ("none", "lora", 16),
    ("none", "lora", 64),
    ("4bit", "qlora", 16),
    ("none", "dora", 16),
]


def main() -> int:
    args = parse_args()

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    rows = []
    for quant, peft, lora_r in _SWEEP_POINTS:
        row = run_one(
            args.model_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            quant=quant,
            peft=peft,
            optimizer=args.optimizer,
            gradient_checkpointing=args.gradient_checkpointing,
            lora_r=lora_r,
        )
        rows.append(row)

    header = f"{'quant':<8} {'peft':<6} {'r':>4} {'predicted_gb':>14} {'measured_gb':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        measured = "n/a" if row["measured_gb"] is None else f"{row['measured_gb']:.2f}"
        print(
            f"{row['quant']:<8} {row['peft']:<6} {row['lora_r']:>4} "
            f"{row['predicted_gb']:>14.3f} {measured:>12}"
        )

    if not cuda_available:
        print()
        print(
            "NOTICE: no CUDA GPU available in this environment — only "
            "predicted_gb was computed. Re-run this script on real hardware "
            "with a locally available model to populate measured_gb and "
            "commit the sweep to benchmarks/results/. Do not fabricate "
            "measured numbers."
        )
    else:
        print()
        print(
            "NOTICE: CUDA is available, but the measurement half of this "
            "script is still a stub (see the module TODO) — measured_gb is "
            "None for every row above. Wire the real training step before "
            "trusting any comparison here."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
