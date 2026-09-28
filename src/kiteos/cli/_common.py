# Copyright 2026 Kite ML.
"""Shared CLI plumbing: console, station loading, prompts and subprocess helpers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn, Optional

import click
import typer
from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

from kiteos import __version__, brand
from kiteos.config import StationConfig, StationError, repo_root

if TYPE_CHECKING:
    from kiteos.cli.runs import RunState

Level = Literal["PASS", "WARN", "FAIL", "INFO"]

THEME = Theme(
    {
        "kite": f"bold {brand.PRIMARY}",
        "accent": brand.ACCENT,
        "ok": f"bold {brand.SUCCESS}",
        "warn": f"bold {brand.WARNING}",
        "fail": f"bold {brand.ERROR}",
        "muted": brand.MUTED,
    }
)
LEVEL_STYLE: dict[str, str] = {"PASS": "ok", "WARN": "warn", "FAIL": "fail", "INFO": "muted"}

# file=None makes rich resolve sys.stdout at write time (so CliRunner captures output).
console = Console(theme=THEME, highlight=False)
# The banner goes to stderr, so machine-readable output (`kiteos list --json`) stays clean.
banner_console = Console(theme=THEME, highlight=False, stderr=True)

StationOption = Annotated[
    Optional[Path],
    typer.Option(
        "--station",
        "-s",
        help="Station file (default: $KITEOS_STATION or <repo>/config/station.yaml).",
        show_default=False,
    ),
]


@dataclass
class _State:
    station: Path | None = None
    banner: bool = True


STATE = _State()


def print_banner(compact: bool = True) -> None:
    if compact:
        banner_console.print(f"[kite]{brand.BRAND_NAME}[/kite] [muted]{__version__} · {brand.TAGLINE}[/muted]")
    else:
        banner_console.print(f"[kite]{brand.banner(__version__)}[/kite]")


def die(message: str, code: int = 1) -> NoReturn:
    """Print an error in KiteOS style and exit with ``code``."""
    console.print(f"[fail]error:[/fail] {message}")
    raise typer.Exit(code)


def warn(message: str) -> None:
    console.print(f"[warn]warning:[/warn] {message}")


def note(message: str) -> None:
    console.print(f"[muted]{message}[/muted]")


def status_line(level: Level, name: str, detail: str = "", fix: str | None = None) -> None:
    """One doctor-style result line, with remediation text underneath when given."""
    style = LEVEL_STYLE[level]
    console.print(f"[{style}]{level:<4}[/{style}]  {name}" + (f"  [muted]{detail}[/muted]" if detail else ""))
    if fix:
        for line in fix.splitlines():
            console.print(f"      [accent]->[/accent] {line}")


def load_station(path: Path | None = None) -> StationConfig:
    """Load the station from ``--station``, the global option, or the default lookup."""
    chosen = path or STATE.station
    try:
        return StationConfig.load(chosen)
    except StationError as exc:
        die(str(exc), code=2)


def prompt_text(message: str) -> str | None:
    """Prompt for one line; None on EOF/Ctrl-C (callers treat that as a refusal)."""
    try:
        return click.prompt(message, default="", show_default=False, prompt_suffix=": ")
    except (click.Abort, EOFError):
        console.print()
        return None


def confirm_typed(expected: str, message: str) -> bool:
    """Require the operator to type ``expected`` exactly (fails closed on EOF)."""
    answer = prompt_text(message)
    return answer is not None and answer.strip() == expected


def kiteos_pythonpath(environ: Mapping[str, str] | None = None, repo: Path | None = None) -> str:
    """PYTHONPATH with ``<repo>/src`` first, so the runtime workers and the isolated env import KiteOS."""
    environ = os.environ if environ is None else environ
    src = str((repo or repo_root()) / "src")
    existing = [p for p in environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != src]
    return os.pathsep.join([src, *existing])


def kiteos_env(environ: Mapping[str, str] | None = None, repo: Path | None = None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    env["PYTHONPATH"] = kiteos_pythonpath(env, repo)
    return env


@dataclass(frozen=True)
class CmdResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()


def run_capture(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> CmdResult:
    """Run a command and capture its output; a missing binary or timeout is a failed result."""
    try:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CmdResult(127, "", f"{argv[0]}: not found ({exc})")
    except subprocess.TimeoutExpired:
        return CmdResult(124, "", f"{' '.join(argv)}: timed out after {timeout:.0f} s")
    except OSError as exc:
        return CmdResult(126, "", f"{argv[0]}: {exc}")
    return CmdResult(proc.returncode, proc.stdout or "", proc.stderr or "")


RUNTIME_COMMAND = "kiteos-runtime"  # console script of this package, installed in the runtime virtualenv


def runtime_argv(*args: str) -> list[str]:
    """``uv run --no-sync kiteos-runtime ...`` (never a plain ``uv run``: it would uninstall KiteOS)."""
    return ["uv", "run", "--no-sync", RUNTIME_COMMAND, *args]


def run_runtime(station: StationConfig, *args: str, timeout: float = 120.0) -> CmdResult:
    """Run a runtime CLI command in the station's runtime checkout and capture its output."""
    root = station.runtime_root_path()
    if not root.is_dir():
        return CmdResult(2, "", f"runtime checkout not found at {root}")
    return run_capture(runtime_argv(*args), cwd=root, env=kiteos_env(), timeout=timeout)


def attach_to_run(purpose: str) -> RunState | None:
    """Join the bus of the running KiteOS stack (from its run-state file) before importing the runtime.

    Mock runs use a private bus, so a client must use the same ``LCM_DEFAULT_URL`` /
    ``ZENOH_SCOUT_ADDR`` as the stack. Refuses when several KiteOS runs are alive (a call could
    reach the wrong one). With no KiteOS run, this shell's settings (normally the runtime's default bus)
    are used unchanged.
    """
    from kiteos.cli.runs import alive_run_states, apply_bus, bus_of, runtime_already_imported

    runs = alive_run_states()
    if len(runs) > 1:
        listing = "\n".join(f"  - {escape(run.describe())}" for run in runs)
        die(
            f"more than one KiteOS stack is running; {purpose} could reach the wrong one:\n{listing}\n"
            "Stop the extra stack first (`kiteos stop --pid <PID>`).",
            code=2,
        )
    if not runs:
        note(f"{purpose}: no KiteOS run found on this host; using this shell's runtime bus settings")
        return None
    run = runs[0]
    if runtime_already_imported():
        if bus_of(os.environ) != {k: v for k, v in run.bus.items()}:
            die(f"the runtime is already loaded on another bus; cannot attach to {run.describe()}", code=2)
        return run
    apply_bus(run.bus)
    note(f"{purpose}: attached to {escape(run.describe())}")
    return run
