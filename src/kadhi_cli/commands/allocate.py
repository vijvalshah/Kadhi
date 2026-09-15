"""`kadhi allocate` — produce a real `lora.rank_pattern` from a checkpoint's
own weights (Adaptation Controller, Phase 3).

See docs/adaptation-controller.md and docs/adaptation-controller-plan.md.
Named ``allocate`` rather than ``plan`` deliberately: `kadhi plan` already
exists (the unrelated Terraform-shape cost/ETA/VRAM pre-flight command in
``commands/plan.py``) — this command produces a different kind of plan
entirely (a rank allocation, not a cost estimate) and needed a name that
does not collide with a shipped one.

Static-signal mode only: this command scores layers with
``spectrum_scan``'s spectral SNR (real, torch-free, computed straight off
the on-disk checkpoint), not the gradient-based sensitivity probe in
``utils/sensitivity.py`` — that probe needs a loaded model and a real
dataset batch iterator, which this command does not yet assemble. See
``utils/plan.py``'s module docstring for why a working controller does not
require the gradient probe to exist first.
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

# Same quant-name subset commands/train.py's _build_hardware_fit_input
# restricts to — hardware_fit.HardwareFitInput only models these three; an
# unmapped quant (gptq/awq/...) means the predictor's bytes-per-param table
# doesn't apply cleanly, so this command declines to guess rather than
# silently mis-predicting VRAM.
_HW_FIT_QUANT = {"none": "none", "4bit": "4bit", "8bit": "8bit"}


def allocate_cmd(
    config: str = typer.Option(
        "kadhi.yaml", "--config", "-c", help="Path to kadhi.yaml.",
    ),
    explain: bool = typer.Option(
        False, "--explain",
        help="Print the per-layer SNR ranking and the full VRAM breakdown, not just the summary.",
    ),
) -> None:
    """Allocate LoRA rank per layer under `controller.budget`, from the
    checkpoint's own weights, checked against the VRAM ceiling."""
    from kadhi_cli.config.loader import load_config

    cfg = load_config(config)

    if cfg.controller is None or not cfg.controller.enabled:
        console.print(
            "[red]controller.enabled is not set.[/] Add a `controller:` "
            "block to your config with `enabled: true` and a budget "
            "(`trainable_params` and/or `vram_gb`) — see "
            "docs/adaptation-controller.md §4."
        )
        raise typer.Exit(1)

    if not os.path.isdir(cfg.base):
        console.print(
            f"[red]{escape(cfg.base)} is not a local directory.[/] "
            "`kadhi allocate` reads real module shapes straight off an "
            "already-downloaded checkpoint (no network fetch, by design — "
            "see docs/adaptation-controller-plan.md §1.1). Fetch the model "
            "first, or point `base:` at a local path."
        )
        raise typer.Exit(1)

    budget = cfg.controller.budget
    if budget.trainable_params is None:
        console.print(
            "[red]controller.budget.trainable_params is required for "
            "`kadhi allocate` today.[/] Inferring a parameter budget from "
            "`vram_gb` alone is not yet implemented — set "
            "`controller.budget.trainable_params` explicitly."
        )
        raise typer.Exit(1)

    lora_cfg = cfg.training.lora
    quant = _HW_FIT_QUANT.get(str(cfg.training.quantization or "none"))
    if quant is None:
        console.print(
            f"[red]quantization {escape(str(cfg.training.quantization))!r} is not "
            "supported by the VRAM feasibility check yet[/] "
            f"(supported: {', '.join(sorted(_HW_FIT_QUANT))})."
        )
        raise typer.Exit(1)

    from kadhi_cli.utils.gpu import model_size_from_name

    params_b = model_size_from_name(cfg.base)
    if not isinstance(params_b, (int, float)) or params_b <= 0:
        console.print(f"[red]Could not determine model size for {escape(cfg.base)}.[/]")
        raise typer.Exit(1)

    seq_len = cfg.data.max_length if isinstance(cfg.data.max_length, int) else 2048
    batch_size = cfg.training.batch_size if isinstance(cfg.training.batch_size, int) else 1

    from kadhi_cli.utils.hardware_fit import HardwareFitInput
    from kadhi_cli.utils.capacity import discover_lora_module_shapes
    from kadhi_cli.utils.feasibility import fit_plan_to_budget
    from kadhi_cli.utils.plan import compute_layer_signals, plan_from_signals

    hardware_fit_base = HardwareFitInput(
        params_b=float(params_b), seq_len=seq_len, batch_size=batch_size,
        optimizer="adamw_torch", quant=quant, peft="lora",
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
    )

    shapes = discover_lora_module_shapes(cfg.base, lora_cfg.target_modules)
    if not shapes:
        console.print(
            f"[red]No LoRA-eligible weights matched target_modules="
            f"{lora_cfg.target_modules!r} under {escape(cfg.base)}.[/]"
        )
        raise typer.Exit(1)

    try:
        # Measure the model ONCE. fit_plan_to_budget calls build_plan_fn once
        # per budget-shrink iteration; a closure over build_static_plan would
        # re-run the full SVD scan of every weight matrix on every one of them
        # (6 scans for 6 iterations — minutes each on a real 8B). The two-step
        # API keeps the expensive half out of the loop.
        signals = compute_layer_signals(cfg.base, lora_cfg.target_modules)

        def build(budget_params: int):
            # plan_from_signals doesn't take use_dora — DoRA's extra magnitude
            # vector doesn't change the rank ALLOCATION (the concave objective
            # is independent of it), only the trainable-parameter ACCOUNTING,
            # which is where use_dora is applied instead (see
            # fit_plan_to_budget below).
            return plan_from_signals(
                signals, budget_params=budget_params, default_r=lora_cfg.r,
            )

        vram_gb = budget.vram_gb if budget.vram_gb is not None else 1_000_000.0
        result = fit_plan_to_budget(
            build, shapes, initial_budget_params=budget.trainable_params,
            use_dora=lora_cfg.use_dora, hardware_fit_base=hardware_fit_base,
            vram_gb=vram_gb,
        )
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    plan_result = result.plan
    check = result.check
    style = "green" if check.feasible else "red"

    summary = Table(title="Adaptation Controller — allocation")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("base", escape(cfg.base))
    summary.add_row("feasible", str(check.feasible))
    summary.add_row("budget_params (requested)", str(budget.trainable_params))
    summary.add_row("budget_params (used)", str(result.budget_params_used))
    summary.add_row("trainable_params (exact)", str(check.trainable_params))
    summary.add_row("iterations", str(result.iterations))
    summary.add_row("peak_vram_gb (predicted)", f"{check.hardware_report.peak_vram_gb:.3f}")
    if budget.vram_gb is not None:
        summary.add_row("vram_gb (ceiling)", f"{budget.vram_gb:.3f}")
    summary.add_row("rank_pattern entries", str(len(plan_result.rank_pattern)))
    summary.add_row("frozen modules", str(len(plan_result.frozen_module_paths)))
    console.print(summary)

    if explain:
        detail = Table(title="rank_pattern")
        detail.add_column("module")
        detail.add_column("rank")
        for path, rank in sorted(plan_result.rank_pattern.items()):
            detail.add_row(escape(path), str(rank))
        console.print(detail)

        if plan_result.frozen_module_paths:
            frozen = Table(title="frozen modules (rank 0 — not adapted)")
            frozen.add_column("module")
            for path in plan_result.frozen_module_paths:
                frozen.add_row(escape(path))
            console.print(frozen)

        breakdown = check.hardware_report.breakdown
        vram_table = Table(title="predicted peak VRAM breakdown")
        vram_table.add_column("bucket")
        vram_table.add_column("GB")
        vram_table.add_row("weights", f"{breakdown.weights_gb:.3f}")
        vram_table.add_row("optimizer", f"{breakdown.optimizer_gb:.3f}")
        vram_table.add_row("gradients", f"{breakdown.gradients_gb:.3f}")
        vram_table.add_row("activations", f"{breakdown.activations_gb:.3f}")
        vram_table.add_row("overhead", f"{breakdown.overhead_gb:.3f}")
        vram_table.add_row("total", f"{breakdown.total_gb:.3f}")
        console.print(vram_table)

    if check.feasible:
        message = (
            "[green]Feasible.[/] Paste `lora.rank_pattern` from --explain "
            "into your kadhi.yaml, or wire it in programmatically."
        )
    elif check.trainable_params == 0:
        # The budget search bisects down to a fully-frozen allocation and it
        # STILL does not fit, so the overflow is not the adapter's — it is
        # base weights + activations + overhead alone. No rank allocation can
        # help, and saying "not feasible" without saying that would send the
        # operator off to tune a budget that was never the problem.
        message = (
            "[red]Not feasible at ANY rank.[/] Even with every layer frozen "
            "(0 trainable parameters) the predicted peak is "
            f"{check.hardware_report.peak_vram_gb:.2f} GB against a "
            f"{check.hardware_report.available_vram_gb:.2f} GB ceiling — "
            "the overflow is base weights, "
            "activations and overhead, not the adapter. Lower "
            "`data.max_length` or `training.batch_size`, enable "
            "`training.gradient_checkpointing`, quantize further, or use "
            "`training.stream_layers`. Raising the parameter budget cannot help."
        )
    else:
        message = (
            f"[red]Not feasible.[/] Searched down to "
            f"{result.budget_params_used:,} trainable parameters over "
            f"{result.iterations} evaluation(s). "
            f"{escape(check.hardware_report.reason)}"
        )

    console.print(Panel(message, title="allocate", border_style=style))
    if not check.feasible:
        raise typer.Exit(3)


__all__ = ["allocate_cmd"]
