# Copyright 2026 Kite ML.
"""``kiteos stop``: graceful shutdown of the running robot stack.

Only SIGTERM is ever sent by KiteOS. The runtime then stops every module (the collection recorder gets
``--recorder.shutdown-timeout``, 100 s under KiteOS, to finalize its file before the runtime kills it),
and finally sweeps leftover worker processes (terminate, 2 s, kill). KiteOS waits up to 300 s for
the stack to exit and never kills it itself.

Which stack: the KiteOS run-state files and the runtime's run registry. A run is stopped without the
typed confirmation only when it is PROVEN to be mock arms (its run-state file, or ``expect-mock
true`` in its registered command line); anything else (real, or unknown) needs ``stop`` typed
after someone holds the arms. With several runs alive, ``--pid`` picks one. A robot stack that is
in neither list (for example one started outside KiteOS) is reported, never guessed at.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Optional

import typer
from rich.markup import escape
from rich.panel import Panel

from kiteos.cli import runs
from kiteos.cli._common import StationOption, confirm_typed, console, die, load_station, note
from kiteos.config import StationConfig

STOP_TIMEOUT_S = 300.0
STOP_WORD = "stop"


@dataclass(frozen=True)
class RunningStack:
    pid: int
    run_id: str | None = None
    blueprint: str | None = None
    mock: bool | None = None  # True proven mock, False proven real, None unknown
    source: str = "runtime registry"
    related: tuple[int, ...] = ()  # other pids of the same run (uv wrapper, the runtime process)
    hardware: str | None = None  # what a hardware-free `kiteos run` drives (its run-state file)

    @property
    def mode(self) -> str:
        if self.mock is True and self.hardware:
            return self.hardware
        return {True: "MOCK arms", False: "REAL ARMS"}.get(self.mock, "arms UNKNOWN (treated as real)")  # type: ignore[arg-type]

    def label(self) -> str:
        return f"PID {self.pid} {self.blueprint or 'robot stack'} · {self.mode} · {self.source}"


def parse_runtime_status(text: str) -> RunningStack | None:
    """Parse ``kiteos-runtime status`` output (``PID:``, ``Run ID:``, ``Blueprint:`` lines)."""
    pid = re.search(r"PID:\s*(\d+)", text)
    if pid is None:
        return None
    run_id = re.search(r"Run ID:\s*(\S+)", text)
    blueprint = re.search(r"Blueprint:\s*(.+)", text)
    return RunningStack(
        pid=int(pid.group(1)),
        run_id=run_id.group(1) if run_id else None,
        blueprint=blueprint.group(1).strip() if blueprint else None,
    )


def _process_table() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, command line)."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    table = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]), parts[2] if len(parts) == 3 else "")
    return table


def find_stacks(
    *,
    kiteos_runs: Callable[[], list[runs.RunState]] = runs.alive_run_states,
    registry: Callable[[], list[dict]] = runs.runtime_registry_runs,
    processes: Callable[[], dict[int, tuple[int, str]]] = _process_table,
) -> list[RunningStack]:
    """Running stacks from KiteOS run-state files and the runtime's registry (one entry per run)."""
    table = processes()
    entries = registry()
    found: list[RunningStack] = []
    claimed: set[int] = set()
    for state in kiteos_runs():
        # The run-state pid is `uv run ... kiteos-runtime run`; the runtime process itself is its child (the
        # registered one, else the child running `kiteos-runtime run`): SIGTERM goes to the runtime directly.
        child = next((e for e in entries if table.get(int(e["pid"]), (None,))[0] == state.pid), None)
        if child is not None:
            pid = int(child["pid"])
        else:
            pid = next(
                (
                    p
                    for p, (parent, cmd) in table.items()
                    if parent == state.pid and runs.STACK_PATTERNS[0].search(cmd)
                ),
                state.pid,
            )
        claimed |= {state.pid, pid}
        found.append(
            RunningStack(
                pid=pid,
                run_id=child.get("run_id") if child else None,
                blueprint=state.blueprint,
                mock=not state.real,
                source="KiteOS run",
                related=(state.pid,),
                hardware=None if state.real else state.hardware,
            )
        )
    for entry in entries:
        pid = int(entry["pid"])
        if pid in claimed:
            continue
        found.append(
            RunningStack(
                pid=pid,
                run_id=entry.get("run_id"),
                blueprint=entry.get("blueprint"),
                mock=runs.registry_run_is_mock(entry),
                source="runtime registry",
            )
        )
    return found


def _alive(pid: int, kill: Callable[[int, int], None]) -> bool:
    try:
        kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else
    return True


def graceful_stop(
    pid: int,
    *,
    timeout_s: float = STOP_TIMEOUT_S,
    poll_s: float = 1.0,
    kill: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[], None] | None = None,
) -> bool:
    """SIGTERM ``pid`` and wait until it exits. Returns True once gone. Never sends SIGKILL."""
    try:
        kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    waited = 0.0
    while waited < timeout_s:
        if not _alive(pid, kill):
            return True
        if progress is not None:
            progress()
        sleep(poll_s)
        waited += poll_s
    return not _alive(pid, kill)


def power_down_warning() -> None:
    console.print(
        Panel.fit(
            "Stopping the stack disables the motors: the arms go limp and may DROP.\n"
            "Park the arms away from the glass and have someone hold both arms first.",
            border_style="fail",
            title="KiteOS stop",
        )
    )


def choose_stack(pid: int | None = None) -> RunningStack | None:
    """The one stack to stop, or None when nothing is running. Exits when it cannot decide."""
    stacks = find_stacks()
    if pid is not None:
        for stack in stacks:
            if pid == stack.pid or pid in stack.related:
                return stack
        die(f"PID {pid} is not a running KiteOS stack", code=2)
    if len(stacks) > 1:
        listing = "\n".join(f"  - {escape(s.label())}" for s in stacks)
        die(f"more than one robot stack is running:\n{listing}\nChoose one with --pid.", code=2)
    if stacks:
        return stacks[0]
    unregistered = runs.stack_processes()
    if unregistered:
        listing = "\n".join(f"  - PID {p}: {escape(c[:140])}" for p, c in unregistered[:8])
        die(
            "a robot stack is running but was not started by KiteOS and is not in the runtime's run "
            f"registry:\n{listing}\nKiteOS will not guess. Support the arms, then stop it in its own "
            "terminal (Ctrl-C).",
            code=1,
        )
    return None


def stop_stack(target: RunningStack, *, timeout_s: float = STOP_TIMEOUT_S) -> bool:
    """SIGTERM ``target`` and wait; True if it exited."""
    console.print(
        f"Stopping {target.blueprint or 'robot stack'} (PID {target.pid}, {target.mode}) with SIGTERM; "
        "finalizing can take minutes",
        end="",
    )
    done = graceful_stop(target.pid, timeout_s=timeout_s, progress=lambda: console.print(".", end=""))
    console.print()
    if done:
        console.print("[ok]Stack stopped[/ok]; recordings are finalized.")
    else:
        console.print(
            f"[warn]The stack is still shutting down after {timeout_s:.0f} s.[/warn] It was NOT killed; "
            "check its terminal, then run `kiteos stop` again."
        )
    return done


def stop_running(station: StationConfig, *, timeout_s: float = STOP_TIMEOUT_S, pid: int | None = None) -> bool:
    """Stop the (single) running stack after the caller confirmed; True if nothing is left running."""
    target = choose_stack(pid)
    if target is None:
        note("no running robot stack found; nothing to stop")
        return True
    return stop_stack(target, timeout_s=timeout_s)


def stop(
    pid: Annotated[Optional[int], typer.Option("--pid", help="Which stack, when several run.")] = None,
    real: Annotated[
        bool,
        typer.Option("--real", hidden=True, help="Deprecated: confirmation is required unless the run is proven mock."),
    ] = False,
    timeout: Annotated[float, typer.Option("--timeout", min=1.0, help="Seconds to wait for a clean exit.")] = (
        STOP_TIMEOUT_S
    ),
    station: StationOption = None,
) -> None:
    """Stop the running KiteOS stack gracefully (SIGTERM only; typed confirmation unless mock)."""
    load_station(station)
    target = choose_stack(pid)
    if target is None:
        note("no running robot stack found; nothing to stop")
        return
    console.print(f"Running: {escape(target.label())}")
    if target.mock is not True:
        power_down_warning()
        if not confirm_typed(STOP_WORD, f"Type '{STOP_WORD}' once someone is holding both arms"):
            die("not confirmed; the robot is still running", code=1)
    if not stop_stack(target, timeout_s=timeout):
        raise typer.Exit(1)
