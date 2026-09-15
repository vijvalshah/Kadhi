"""kadhi ci — fine-tuning CI helpers (v0.71.35).

``kadhi ci init`` writes a GitHub Actions workflow that gates every PR on
``kadhi data validate`` -> ``kadhi expect`` -> ``kadhi ship --evidence``.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from kadhi_cli.utils.ci_workflow import write_kadhi_gate_workflow

console = Console()

app = typer.Typer(no_args_is_help=True, help="Fine-tuning CI helpers.")


@app.command("init")
def init(
    data: str = typer.Option(
        "./data/train.jsonl", "--data", help="Training data path (repo-relative)"
    ),
    suite: str = typer.Option(
        "expectations.yaml", "--suite", help="Expectations suite YAML (repo-relative)"
    ),
    evidence: str = typer.Option(
        "ship_evidence.json", "--evidence", help="Ship evidence JSON (repo-relative)"
    ),
    python_version: str = typer.Option(
        "3.11", "--python", help="Python version for the CI runner (e.g. 3.11)"
    ),
    branch: str = typer.Option("main", "--branch", help="Branch the workflow triggers on"),
    config: Optional[str] = typer.Option(
        None, "--config",
        help="Bind the ship gate to a committed kadhi.yaml (refuses stale evidence)",
    ),
    output: str = typer.Option(
        ".github/workflows/kadhi-gate.yml", "-o", "--output", help="Workflow output path"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing workflow"),
) -> None:
    """Write a GitHub Actions fine-tuning gate workflow."""
    try:
        written = write_kadhi_gate_workflow(
            data_path=data,
            suite_path=suite,
            evidence_path=evidence,
            python_version=python_version,
            branch=branch,
            output_path=output,
            overwrite=force,
            config_path=config,
        )
    except (ValueError, TypeError, OSError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"Wrote [bold]{escape(written)}[/]\n\n"
            "The gate runs: [cyan]kadhi data validate[/] -> [cyan]kadhi expect[/] -> "
            "[cyan]kadhi ship --evidence[/].\n"
            "Edit the paths in the workflow to match your repo.",
            title="kadhi ci init",
            border_style="green",
        )
    )
