"""kadhi lock — shared run lockfile (v0.67.0 Part E).

Subcommands:

- ``kadhi lock write``: render a ``kadhi.lock`` from operator-supplied
  base-model / dataset / env hashes.
- ``kadhi lock check``: compare a tracked ``kadhi.lock`` against a
  freshly-computed closure; exit 3 on drift.
- ``kadhi lock show``: print a tracked lock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

app = typer.Typer(no_args_is_help=True, help="Shared run lockfile")


@app.command(name="write")
def write_lock_cmd(
    base_model: str = typer.Option(..., "--base-model", help="HF model id / path"),
    base_sha: str = typer.Option(..., "--base-sha", help="64-hex base-model SHA"),
    dataset_sha: str = typer.Option(..., "--dataset-sha", help="64-hex dataset SHA"),
    env_hash: Optional[str] = typer.Option(
        None,
        "--env-hash",
        help="64-hex env hash. If omitted, auto-derived from --env-lock "
        "(kadhi-env.lock) via `kadhi env lock`.",
    ),
    env_lock: str = typer.Option(
        "kadhi-env.lock",
        "--env-lock",
        help="Path to a kadhi-env.lock to auto-derive --env-hash from "
        "(used only when --env-hash is omitted).",
    ),
    output: str = typer.Option("kadhi.lock", "--output", "-o", help="Output path"),
):
    """Render a ``kadhi.lock`` from the base/dataset/env hashes.

    When ``--env-hash`` is omitted, it is auto-derived from ``--env-lock``
    (default ``kadhi-env.lock``) so an operator who ran ``kadhi env lock`` does
    not have to copy the hash by hand (v0.71.1 #224).
    """
    from kadhi_cli import __version__
    from kadhi_cli.utils.kadhi_lock import KadhiLock, compute_lock_closure, write_lock

    # v0.71.1 #224 — auto-glue: derive the env hash from kadhi-env.lock when
    # the operator did not pass --env-hash explicitly. Treat an empty string
    # the same as omitted so `--env-hash ""` auto-derives rather than tripping
    # the generic 64-hex closure error.
    if not env_hash:
        from kadhi_cli.utils.env_lock import compute_env_hash
        from kadhi_cli.utils.env_lock import read_lock as read_env_lock

        try:
            env_lock_obj = read_env_lock(env_lock)
        except FileNotFoundError as exc:
            console.print(
                f"[red]--env-hash not provided and {escape(env_lock)!s} not found.[/]\n"
                "Either pass --env-hash <64-hex> explicitly, or run "
                "`kadhi env lock` first to create kadhi-env.lock."
            )
            raise typer.Exit(2) from exc
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc
        env_hash = compute_env_hash(env_lock_obj)

    try:
        closure = compute_lock_closure(
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
        )
        lock = KadhiLock(
            kadhi_version=__version__,
            base_model=base_model,
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
            closure_sha=closure,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        write_lock(lock, output)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Lock:        [bold]{escape(output)}[/]\n"
            f"Base:        [bold]{escape(base_model)}[/]\n"
            f"Closure SHA: [bold]{closure[:12]}…[/]",
            title="kadhi.lock written",
        )
    )


@app.command(name="show")
def show_lock_cmd(
    path: str = typer.Argument("kadhi.lock", help="Path to kadhi.lock"),
):
    """Print a tracked lock file."""
    from kadhi_cli.utils.kadhi_lock import read_lock

    try:
        lock = read_lock(path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            f"Kadhi version:    [bold]{escape(lock.kadhi_version)}[/]\n"
            f"Base model:      [bold]{escape(lock.base_model)}[/]\n"
            f"Base SHA:        [bold]{lock.base_model_sha[:16]}…[/]\n"
            f"Dataset SHA:     [bold]{lock.dataset_sha[:16]}…[/]\n"
            f"Env hash:        [bold]{lock.env_hash[:16]}…[/]\n"
            f"Closure SHA:     [bold]{lock.closure_sha[:16]}…[/]\n"
            f"Created at:      [bold]{escape(lock.created_at)}[/]",
            title=f"kadhi.lock — {escape(path)}",
        )
    )


@app.command(name="check")
def check_lock_cmd(
    path: str = typer.Argument("kadhi.lock", help="Path to tracked kadhi.lock"),
    base_sha: str = typer.Option(..., "--base-sha", help="64-hex current base-model SHA"),
    dataset_sha: str = typer.Option(..., "--dataset-sha", help="64-hex current dataset SHA"),
    env_hash: str = typer.Option(..., "--env-hash", help="64-hex current env hash"),
    base_model: str = typer.Option(..., "--base-model", help="Current base model id"),
):
    """Refuse with exit 3 if the lock has drifted from current state."""
    from kadhi_cli import __version__
    from kadhi_cli.utils.kadhi_lock import (
        KadhiLock,
        check_lock_drift,
        compute_lock_closure,
        read_lock,
    )

    try:
        expected = read_lock(path)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    try:
        closure = compute_lock_closure(
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
        )
        # Use the existing kadhi_version + created_at from `expected` so the
        # comparison stays content-only (drift only counts the 5 content
        # fields per `check_lock_drift`).
        actual = KadhiLock(
            kadhi_version=expected.kadhi_version,
            base_model=base_model,
            base_model_sha=base_sha,
            dataset_sha=dataset_sha,
            env_hash=env_hash,
            closure_sha=closure,
            created_at=expected.created_at,
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(2) from exc

    drift = check_lock_drift(expected, actual)
    if drift.ok:
        console.print(
            Panel(
                f"Lock:        [bold]{escape(path)}[/]\n"
                f"Status:      [green]OK[/] — closure matches",
                title="kadhi lock check",
            )
        )
        return

    console.print(
        Panel(
            f"Lock:    [bold]{escape(path)}[/]\n"
            f"Status:  [red]DRIFT[/]",
            title="kadhi lock check",
        )
    )
    for change in drift.changes:
        console.print(f"  [red]- {escape(change)}[/]")
    if __version__ != expected.kadhi_version:
        console.print(
            f"[yellow]Note: kadhi version changed "
            f"({escape(expected.kadhi_version)} -> {escape(__version__)})[/]"
        )
    raise typer.Exit(3)
