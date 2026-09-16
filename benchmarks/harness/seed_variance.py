#!/usr/bin/env python3
"""Seed-variance calibration for the Adaptation Controller (Phase 2).

Validates the plan's Phase 2 acceptance criteria in
``docs/adaptation-controller-plan.md``: before any rank-allocation claim can
be trusted, the noise floor of the evaluation metric across random seeds has
to be measured, because a stopping rule of the form ``ΔQ/ΔP < ε`` is
meaningless if ``ε`` is smaller than ordinary seed-to-seed spread. Published
evaluations of fine-tuned models report double-digit swings in held-out
metrics across seeds at small sample counts; this harness measures Kadhi's
own spread on its own trainer rather than importing that number from
elsewhere.

Three things come out of one run of this script, for a FIXED dataset and
model:

1. Seed spread of final training loss and held-out eval loss, for one fixed
   LoRA configuration, across ``N`` seeds. This is ``ε``'s source: a
   candidate capacity change has to move the metric by more than this
   spread before the controller may treat it as a real improvement.
2. The same spread for the CURRENT automatic decision
   (``autopilot.decisions.decide_peft`` — a three-branch lookup on dataset
   size alone) — the baseline the allocator (Phase 3) has to beat.
3. The same spread for a uniform ``r=32`` LoRA pattern — the second
   baseline named in the plan.

Drives ``kadhi_cli.trainer.sft.SFTTrainerWrapper`` directly, in-process,
mirroring the pattern in ``mlx_sft_smoke.py`` (in-process fixture, dispatch
the real trainer wrapper, extract the real result dict) rather than
shelling out to the ``kadhi`` CLI, so this can run inside CI/dev containers
with no filesystem side effects beyond a temp directory.

Requires ``pip install -e ".[train]"`` and a network connection on first run
(downloads ``HuggingFaceTB/SmolLM2-135M``, ~270 MB, then reads from the HF
cache). CPU-only is fine — SmolLM2-135M trains a handful of steps on CPU in
well under a minute; this script does not require a GPU, unlike
``vram_predictor_validation.py`` (that one measures VRAM, which does not
exist without a GPU — this one measures loss/eval spread, which is
GPU-independent).

Honesty note: this script is written so it CAN be run to produce real
numbers, but running it is out of scope for the sandbox that authored it
(no torch/transformers/peft install here, and even if there were, the
network fetch is not guaranteed). Do not treat any number printed by a run
of this file as calibrated until you have actually run it and inspected the
output; do not hand-copy illustrative numbers from this docstring or from
docs/adaptation-controller-plan.md into a config as if they were measured.

Usage:
    python seed_variance.py [--rows N] [--epochs N] [--seeds S1,S2,S3] [--model REPO_ID]

    python seed_variance.py --rows 64 --epochs 1 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Deliberately trivial and repetitive, matching mlx_sft_smoke.py's fixture —
# this harness measures TRAINER variance, not data quality, so the content
# of the fixture is irrelevant as long as it is fixed across every seed and
# every configuration compared below.
_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Name a primary color.", "Red is a primary color."),
    ("What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("How many days are in a week?", "There are 7 days in a week."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet."),
    ("What language is spoken in Japan?", "Japanese is spoken in Japan."),
]


def build_rows(n: int) -> list[dict]:
    out = []
    for i in range(n):
        q, a = _PAIRS[i % len(_PAIRS)]
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


def _config_yaml(
    *, model: str, data_path: Path, out_dir: Path, seed: int,
    lora_r: int, lora_alpha: int, epochs: int,
) -> str:
    # Every field that moves step count or the optimizer trajectory is
    # pinned explicitly, not left at a schema default — the same discipline
    # mlx_sft_smoke.py documents (#716): a spread measurement is worthless
    # if two "identical" runs secretly differ in batch size or schedule.
    return f"""
base: {model}
task: sft
data:
  train: {data_path}
  format: chatml
  max_length: 256
  train_on_responses_only: true
training:
  seed: {seed}
  epochs: {epochs}
  lr: 1e-4
  batch_size: 2
  gradient_accumulation_steps: 1
  scheduler: constant
  warmup_ratio: 0.0
  weight_decay: 0.0
  logging_steps: 1
  lora:
    r: {lora_r}
    alpha: {lora_alpha}
output: {out_dir}
"""


def run_one(
    *, model: str, data_path: Path, seed: int, lora_r: int, lora_alpha: int,
    rows_n: int, epochs: int, workdir: Path,
) -> dict[str, Any]:
    """Train one seed of one fixed config; return final train loss + eval loss.

    Builds a fresh ``SFTTrainerWrapper`` per call — trainer state must not
    leak between seeds, which a shared instance would risk.
    """
    from kadhi_cli.config.loader import load_config_from_string

    out_dir = workdir / f"seed{seed}_r{lora_r}"
    cfg = load_config_from_string(_config_yaml(
        model=model, data_path=data_path, out_dir=out_dir, seed=seed,
        lora_r=lora_r, lora_alpha=lora_alpha, epochs=epochs,
    ))

    from kadhi_cli.trainer.sft import SFTTrainerWrapper
    from kadhi_cli.utils.live_eval import resolve_device

    rows = build_rows(rows_n)
    split = max(1, rows_n // 5)
    train_rows, val_rows = rows[split:], rows[:split]

    # SFTTrainerWrapper defaults device="cuda" — wrong on the CPU-only boxes
    # this script's docstring explicitly claims to support. resolve_device
    # picks CUDA when available, else CPU, same as every other live-eval path.
    wrapper = SFTTrainerWrapper(cfg, device=resolve_device())
    wrapper.setup({"train": train_rows, "val": val_rows})
    result = wrapper.train()

    eval_loss: Optional[float] = None
    try:
        eval_metrics = wrapper.trainer.evaluate()
        eval_loss = float(eval_metrics.get("eval_loss")) if eval_metrics else None
    except Exception as exc:  # noqa: BLE001 — a calibration run must not die on eval
        print(f"  [seed {seed}] eval() raised {type(exc).__name__}: {exc}", file=sys.stderr)

    return {
        "seed": seed,
        "lora_r": lora_r,
        "initial_loss": result.get("initial_loss"),
        "final_loss": result.get("final_loss"),
        "eval_loss": eval_loss,
        "total_steps": result.get("total_steps"),
    }


def _spread(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if v is not None]
    if len(finite) < 2:
        return {"n": len(finite), "mean": finite[0] if finite else float("nan"),
                "stdev": 0.0, "range": 0.0}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "stdev": statistics.stdev(finite),
        "range": max(finite) - min(finite),
    }


def run_sweep(
    *, model: str, rows_n: int, epochs: int, seeds: list[int], workdir: Path,
) -> dict[str, Any]:
    """Run every seed against three configs: the fixed probe rank, the
    current autopilot lookup, and a uniform r=32 baseline. Returns the raw
    per-run results plus the derived spread for each config."""
    from kadhi_cli.autopilot.decisions import decide_peft

    data_path = workdir / "train.jsonl"
    rows = build_rows(rows_n)
    data_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    # decide_peft needs (data_size, model_size_b, vram_gb); SmolLM2-135M is
    # ~0.135B params, and vram_gb is set high enough that use_dora's
    # threshold never flips the comparison — this sweep is about rank
    # variance, not DoRA.
    autopilot_peft = decide_peft(data_size=rows_n, model_size_b=0.135, vram_gb=64.0)

    configs = {
        "probe_r8": {"lora_r": 8, "lora_alpha": 16},
        "autopilot_lookup": {
            "lora_r": autopilot_peft["r"], "lora_alpha": autopilot_peft["alpha"],
        },
        "uniform_r32": {"lora_r": 32, "lora_alpha": 64},
    }

    raw: dict[str, list[dict]] = {name: [] for name in configs}
    for name, cfg in configs.items():
        print(f"\n=== {name} (r={cfg['lora_r']}) ===")
        for seed in seeds:
            print(f"  seed {seed} ...", end=" ", flush=True)
            row = run_one(
                model=model, data_path=data_path, seed=seed,
                lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                rows_n=rows_n, epochs=epochs, workdir=workdir,
            )
            raw[name].append(row)
            print(f"final_loss={row['final_loss']} eval_loss={row['eval_loss']}")

    spread = {
        name: {
            "final_loss": _spread([r["final_loss"] for r in rows_]),
            "eval_loss": _spread([r["eval_loss"] for r in rows_]),
        }
        for name, rows_ in raw.items()
    }
    return {"raw": raw, "spread": spread, "configs": configs}


def _print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("SEED-VARIANCE REPORT")
    print("=" * 72)
    for name, spread in result["spread"].items():
        r = result["configs"][name]["lora_r"]
        print(f"\n{name} (r={r}):")
        for metric in ("final_loss", "eval_loss"):
            s = spread[metric]
            print(
                f"  {metric:12s}  n={s['n']}  mean={s['mean']:.4f}  "
                f"stdev={s['stdev']:.4f}  range={s['range']:.4f}"
            )
    max_eval_range = max(
        spread["eval_loss"]["range"] for spread in result["spread"].values()
    )
    print(
        f"\nSuggested epsilon (max observed eval_loss range across configs, "
        f"{max_eval_range:.4f}): a capacity decision may only be trusted if it "
        "moves held-out eval_loss by more than this. This is a starting point "
        "for Phase 3, not a value to hardcode — recompute it for the real "
        "task/dataset/model the allocator will run against; SmolLM2-135M on 8 "
        "synthetic rows is a mechanism check, not a production calibration."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated list of int seeds, e.g. 0,1,2,3,4",
    )
    parser.add_argument("--out", default=None, help="path to write JSON results")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "torch is not installed in this environment "
            "(pip install -e '.[train]'). This script cannot run here; "
            "it is written to run on a machine with the training extras "
            "installed — CPU is sufficient, no GPU required.",
            file=sys.stderr,
        )
        return 1

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 3:
        print(
            "warning: fewer than 3 seeds gives an unreliable spread estimate "
            "(stdev of 2 points is not meaningful); pass at least 3-5.",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(prefix="kadhi_seed_variance_") as tmp:
        workdir = Path(tmp)
        result = run_sweep(
            model=args.model, rows_n=args.rows, epochs=args.epochs,
            seeds=seeds, workdir=workdir,
        )

    _print_report(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.out}")
    else:
        print(
            "\nNo --out given: results were not saved. Pass --out to write "
            "them for benchmarks/results/, and only commit a results file "
            "that came from a real run on real hardware."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
