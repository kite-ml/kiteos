# Copyright 2026 Kite ML.
"""``kiteos collect start|controls``: record teleoperated demonstrations.

``start`` runs the collection blueprint in the foreground, recording into the next free
``<recordings>/<prefix>-NNN`` (the runtime never reuses a recording folder). Without ``--real`` the
arms are mock and the session is recorded as ``mock-<prefix>-NNN``, which ``kiteos upload`` never
pushes. ``controls`` attaches the KiteOS episode controls to the running stack from a second
terminal; U (delete last saved) and P (pause and upload) act on exactly the session that stack is
recording (from its KiteOS run-state file), never on a guess.

On macOS run both in the foreground from a terminal that has camera permission:
background (daemon) runs crash there (pyarrow's libcurl after fork).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from kiteos import brand
from kiteos.cli._common import StationOption, attach_to_run, confirm_typed, console, die, load_station, note, warn
from kiteos.cli.launch import collection_flags, launch
from kiteos.cli.upload import DELETED_FILE, MOCK_PREFIX, find_sessions, read_indices, run_upload
from kiteos.config import StationConfig, platform_key

app = typer.Typer(help="Record teleoperated demonstrations.", no_args_is_help=True)

PAUSE_STOP_WORD = "stop"


def next_free_session(recordings_dir: Path, prefix: str) -> Path:
    """``<recordings_dir>/<prefix>-NNN`` for the lowest NNN >= 1 not taken yet."""
    n = 1
    while (recordings_dir / f"{prefix}-{n:03d}").exists():
        n += 1
    return recordings_dir / f"{prefix}-{n:03d}"


def latest_session(recordings_dir: Path, prefix: str) -> Path | None:
    """Highest-numbered session directory (the one a running collection writes to)."""
    sessions = find_sessions(recordings_dir, prefix)
    return sessions[-1] if sessions else None


def delete_last_saved(session_dir: Path, episodes_saved: int) -> tuple[int | None, int]:
    """Mark the newest not-yet-deleted saved episode for removal before upload.

    Returns ``(deleted_index, remaining_count)``; ``deleted_index`` is None when nothing is left.
    """
    marks = session_dir / DELETED_FILE
    deleted = read_indices(marks)
    remaining = [i for i in range(episodes_saved) if i not in deleted]
    if not remaining:
        return None, 0
    with marks.open("a") as fh:
        fh.write(f"{remaining[-1]}\n")
    return remaining[-1], len(remaining) - 1


@app.command("start")
def start(
    real: Annotated[bool, typer.Option("--real", help="Drive the real arms (default: simulated).")] = False,
    session: Annotated[
        Optional[str], typer.Option("--session", help="Session prefix (default: station session_prefix).")
    ] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task instruction (default: station task).")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the runtime command and exit.")] = False,
    allow_concurrent: Annotated[
        bool,
        typer.Option("--allow-concurrent", help="MOCK only: start even if another robot stack runs on this host."),
    ] = False,
    station_ports: Annotated[
        bool, typer.Option("--station-ports", help="MOCK only: use the station's WebXR/viser ports (Quest practice).")
    ] = False,
    station: StationOption = None,
) -> None:
    """Start a collection session in the foreground, recording into the next free folder."""
    config = load_station(station)
    prefix = session or config.prefix
    if not real:
        prefix = prefix if prefix.startswith(MOCK_PREFIX) else f"{MOCK_PREFIX}{prefix}"
    recording = next_free_session(config.recordings_path(), prefix)
    task_text = task or config.task
    if platform_key() == "darwin":
        note("macOS: run this from a terminal with camera permission; it stays in the foreground")
    console.print(f"Recording to [kite]{recording}[/kite] · task [kite]{task_text}[/kite]")
    if not real:
        warn("MOCK ARMS: this session records simulated joint states and is never uploaded")
    note("attach the episode controls from another terminal: kiteos collect controls")
    launch(
        config,
        "collection",
        real=real,
        extra_flags=collection_flags(recording, task_text),
        dry_run=dry_run,
        allow_concurrent=allow_concurrent,
        station_ports_for_mock=station_ports,
        run_info={"recording": recording, "session_prefix": prefix},
    )


def power_down(station: StationConfig, pid: int | None = None) -> bool:
    """After P: confirm someone holds the arms, then stop the runtime gracefully (SIGTERM only)."""
    from kiteos.cli.stop import power_down_warning, stop_running

    console.print("\nPausing: the take in progress was saved.")
    power_down_warning()
    if not confirm_typed(PAUSE_STOP_WORD, f"Type '{PAUSE_STOP_WORD}' once someone is holding both arms"):
        console.print("Not stopping. The robot is still running; run `kiteos collect controls` to continue.")
        return False
    time.sleep(1.0)  # let the save marker reach the recorder before shutdown
    return stop_running(station, pid=pid)


def running_session(run_recording: str | None, session: str | None) -> tuple[Path | None, str | None]:
    """(session dir, prefix) the running collection records into, or (None, None) if unknown."""
    if not run_recording:
        return None, None
    recording = Path(run_recording)
    match = re.fullmatch(r"(.+)-(\d+)", recording.name)
    prefix = match.group(1) if match else None
    if session is not None and prefix is not None and session != prefix:
        die(
            f"the running collection records {recording.name}, not a {session}-NNN session; "
            "drop --session (the controls follow the running stack)",
            code=2,
        )
    return recording, prefix


@app.command("controls")
def controls(
    session: Annotated[
        Optional[str],
        typer.Option("--session", help="Only a check: the running collection must record this prefix."),
    ] = None,
    station: StationOption = None,
) -> None:
    """Attach episode controls to the running collection (Space/R/D/U/P/Q)."""
    config = load_station(station)
    run = attach_to_run("collect controls")
    session_dir, prefix = running_session(run.recording if run else None, session)
    if session_dir is None:
        warn(
            "cannot tell which session the running stack records (it was not started with "
            "`kiteos collect start`): U (delete last saved) is disabled and P will not upload"
        )
    try:
        from dimos.imitation.tui import CollectionSession
        from dimos.porcelain.dimos import Dimos

        from kiteos.cli.collect_tui import PAUSE_RESULT, KiteCollectionApp
    except ImportError as exc:
        die(f"the KiteOS runtime is not importable here ({exc}); run kiteos from the runtime virtualenv (see `kiteos doctor`)", code=2)

    driver = None
    try:
        driver = Dimos.connect()
        app_ = KiteCollectionApp(
            CollectionSession(driver),
            title=brand.title(config.task),
            session_dir=lambda: session_dir,
        )
        result = app_.run()
    except Exception as exc:
        die(f"collection controls failed: {exc}\nIs `kiteos collect start` running?", code=1)
    finally:
        if driver is not None:
            driver.stop()

    if result != PAUSE_RESULT:
        return
    if not power_down(config, pid=run.pid if run else None):
        raise typer.Exit(1)
    if prefix is None:
        note("upload skipped (unknown session); run `kiteos upload` for the station's sessions")
        raise typer.Exit(0)
    if prefix.startswith(MOCK_PREFIX):
        note(f"{prefix} was recorded with MOCK arms; nothing is uploaded")
        raise typer.Exit(0)
    raise typer.Exit(run_upload(config, prefix=prefix))
