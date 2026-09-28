# Copyright 2026 Kite ML.
"""``kiteos upload``: convert finished sessions and push them as one dataset.

1. Every ``<recordings>/<prefix>-NNN`` with a finished ``recording.mcap`` and no converted
   dataset yet is converted with ``_isolated/fast_prepare.py`` (fill mode, no slow pre-check),
   dropping the episodes marked with U in ``kiteos collect controls``. A session still being
   written is skipped with a note.
2. Episodes marked after a session was converted are removed from its dataset with
   ``_isolated/drop_episodes.py`` (or the session is re-converted when the history is unknown).
3. All converted sessions are merged into ``<datasets>/_hub`` and pushed to the station's Hub
   repository, private unless the station (or ``--public``) says otherwise.

The helper scripts run in dimOS's isolated LeRobot environment.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import typer

from kiteos.cli._common import StationOption, console, die, kiteos_env, load_station, note, warn
from kiteos.config import StationConfig

ISOLATED_DIR = Path(__file__).resolve().parent / "_isolated"
DELETED_FILE = "deleted_episodes.txt"  # in a recording: 0-based saved-episode indices (U key)
APPLIED_FILE = "kiteos_deleted_episodes.txt"  # in a dataset: the indices already removed
MERGED_DIR = "_hub"
STILL_RECORDING_MARK = "opcode 255"
# Sessions recorded with mock arms (no --real) use this prefix and are never uploaded by default.
MOCK_PREFIX = "mock-"
RECORDING_FILE = "recording.mcap"


def read_indices(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    return {int(tok) for tok in path.read_text().split() if tok.strip().lstrip("-").isdigit()}


def session_number(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}-(\d+)", path.name)
    return int(match.group(1)) if match else None


def find_sessions(recordings_dir: Path, prefix: str) -> list[Path]:
    """All ``<prefix>-NNN`` session directories, in numeric order."""
    if not recordings_dir.is_dir():
        return []
    found = [(session_number(p, prefix), p) for p in recordings_dir.iterdir() if p.is_dir()]
    return [p for n, p in sorted((n, p) for n, p in found if n is not None)]


def find_recordings(recordings_dir: Path, prefix: str) -> list[Path]:
    """Sessions that contain a recording file."""
    return [p for p in find_sessions(recordings_dir, prefix) if (p / "recording.mcap").is_file()]


def dataset_ready(dataset: Path) -> bool:
    return (dataset / "meta" / "info.json").is_file()


def map_to_dataset_indices(new: Sequence[int], applied: set[int]) -> list[int]:
    """Saved-episode indices of a recording -> episode indices of its converted dataset.

    The dataset holds the saved episodes in order, minus those already removed (``applied``).
    """
    return sorted(i - sum(1 for a in applied if a < i) for i in new if i not in applied)


@dataclass
class DropPlan:
    """What to do with deletions marked after a session was converted."""

    reconvert: bool = False
    dataset_indices: list[int] | None = None
    new_marks: list[int] | None = None


def plan_drops(recording: Path, dataset: Path) -> DropPlan:
    marks = read_indices(recording / DELETED_FILE)
    applied_file = dataset / APPLIED_FILE
    if applied_file.is_file():
        applied = read_indices(applied_file)
    else:
        # Converted by an older tool: the marks were all applied only if none came later.
        marks_file = recording / DELETED_FILE
        info = dataset / "meta" / "info.json"
        if not marks or (marks_file.stat().st_mtime <= info.stat().st_mtime):
            return DropPlan()
        return DropPlan(reconvert=True)
    new = sorted(marks - applied)
    if not new:
        return DropPlan()
    return DropPlan(dataset_indices=map_to_dataset_indices(new, applied), new_marks=new)


class IsolatedEnv:
    """Runs KiteOS helper scripts in dimOS's isolated LeRobot project."""

    def __init__(self) -> None:
        try:
            from dimos.experimental.isolated_python.module import (
                isolated_python_environment,
                isolated_python_run_command,
            )
            from dimos.imitation.dataprep.lerobot import lerobot_project
        except ImportError as exc:
            die(f"dimOS is not importable here ({exc}); run kiteos from the dimOS virtualenv", code=2)
        self.project = lerobot_project()
        self._command = isolated_python_run_command
        self.env = kiteos_env(isolated_python_environment(self.project))

    def command(self, script: str, *args: str) -> list[str]:
        return self._command(self.project, "python", str(ISOLATED_DIR / script), *args)

    def run(self, script: str, *args: str, capture: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(script, *args),
            cwd=self.project,
            env=self.env,
            text=True,
            capture_output=capture,
            check=False,
        )


def _tail(text: str, n: int) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def _remove_dataset_dir(path: Path) -> None:
    """Delete a converted dataset folder; never one that holds a raw recording."""
    if path.exists():
        if any(path.rglob(RECORDING_FILE)):
            raise RuntimeError(f"refusing to delete {path}: it contains a raw {RECORDING_FILE}")
        shutil.rmtree(path)


def prepare_session(env: IsolatedEnv, recording: Path, datasets: Path) -> bool:
    output = datasets / recording.name
    if output.resolve() == recording.resolve():
        console.print(f"[fail]FAILED[/fail] {recording.name}: the dataset would overwrite the recording")
        return False
    if dataset_ready(output):
        return apply_late_drops(env, recording, output)
    try:
        _remove_dataset_dir(output)  # left over from a failed attempt
    except RuntimeError as exc:
        console.print(f"[fail]FAILED[/fail] {exc}")
        return False
    console.print(f"Converting [kite]{recording.name}[/kite] ...")
    result = env.run("fast_prepare.py", str(recording), str(output))
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        if STILL_RECORDING_MARK in combined:
            warn(f"skipped {recording.name}: it is still being recorded (stop the stack first: `kiteos stop`)")
        else:
            console.print(f"[fail]FAILED[/fail] {recording.name}:\n{_tail(combined, 8)}")
        try:
            _remove_dataset_dir(output)
        except (RuntimeError, OSError):
            pass
        return False
    lines = [line for line in combined.splitlines() if "[dataprep]" in line or "episodes" in line]
    console.print(f"  [ok]done[/ok]\n{_tail(chr(10).join(lines), 4)}")
    return True


def apply_late_drops(env: IsolatedEnv, recording: Path, dataset: Path) -> bool:
    plan = plan_drops(recording, dataset)
    if plan.reconvert:
        warn(f"{recording.name}: episodes were deleted after conversion; converting it again")
        try:
            _remove_dataset_dir(dataset)
        except RuntimeError as exc:
            console.print(f"[fail]FAILED[/fail] {exc}")
            return False
        return prepare_session(env, recording, dataset.parent)
    if not plan.dataset_indices:
        return True
    console.print(f"Removing deleted episodes {plan.dataset_indices} from [kite]{dataset.name}[/kite] ...")
    result = env.run("drop_episodes.py", str(dataset), *map(str, plan.dataset_indices))
    if result.returncode != 0:
        console.print(f"[fail]FAILED[/fail] {dataset.name}:\n{_tail((result.stdout or '') + (result.stderr or ''), 8)}")
        return False
    applied = read_indices(dataset / APPLIED_FILE) | set(plan.new_marks or [])
    (dataset / APPLIED_FILE).write_text("".join(f"{i}\n" for i in sorted(applied)))
    console.print(f"  [ok]done[/ok] {_tail(result.stdout or '', 1)}")
    return True


def run_upload(
    station: StationConfig,
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    prefix: str | None = None,
    dry_run: bool = False,
    include_mock: bool = False,
) -> int:
    """Convert, merge and push. Returns a process exit code.

    The Hub push replaces the dataset in ``repo`` with the merged sessions of ONE prefix, so a
    prefix other than the station's needs an explicit ``repo_id`` (otherwise it would replace the
    station dataset with only those sessions). Mock-arm sessions (``mock-*``) are refused unless
    ``include_mock``.
    """
    repo = repo_id or station.hub.repo_id
    is_private = station.hub.private if private is None else private
    prefix = prefix or station.prefix
    if prefix.startswith(MOCK_PREFIX) and not include_mock:
        console.print(
            f"[fail]{prefix}[/fail] sessions were recorded with MOCK arms (simulated joint states "
            "next to real images); they are never uploaded. Pass --include-mock to push them "
            "anyway (to a separate --repo-id)."
        )
        return 2
    if prefix != station.prefix and repo_id is None:
        console.print(
            f"Session prefix [kite]{prefix}[/kite] is not the station's ({station.prefix}). "
            f"Pushing it to {repo} would REPLACE that dataset with only {prefix}-NNN sessions. "
            "Pass --repo-id for a separate dataset."
        )
        return 2
    recordings_dir, datasets_dir = station.recordings_path(), station.datasets_path()
    recordings = find_recordings(recordings_dir, prefix)
    if not recordings:
        console.print(f"No recordings found under {recordings_dir}/{prefix}-NNN.")
        return 1
    visibility = "PRIVATE" if is_private else "PUBLIC"
    console.print(
        f"{len(recordings)} session(s) of [kite]{prefix}[/kite] -> "
        f"https://huggingface.co/datasets/{repo} ([{'ok' if is_private else 'warn'}]{visibility}[/])"
    )
    if dry_run:
        for recording in recordings:
            dataset = datasets_dir / recording.name
            if not dataset_ready(dataset):
                marks = sorted(read_indices(recording / DELETED_FILE))
                action, detail = "convert", recording.name + (f" (drop saved episodes {marks})" if marks else "")
            else:
                plan = plan_drops(recording, dataset)
                if plan.reconvert:
                    action, detail = "re-convert", f"{recording.name} (episodes deleted after conversion)"
                elif plan.dataset_indices:
                    action, detail = "drop", f"episodes {plan.dataset_indices} from {dataset.name}"
                else:
                    action, detail = "ready", recording.name
            console.print(f"  {action:<11}{detail}")
        note("dry run: nothing converted or pushed")
        return 0

    env = IsolatedEnv()
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ready = [datasets_dir / r.name for r in recordings if prepare_session(env, r, datasets_dir)]
    if not ready:
        console.print("Nothing converted yet; nothing to upload.")
        return 1
    console.print(f"Merging {len(ready)} session(s) and pushing to https://huggingface.co/datasets/{repo}")
    result = env.run(
        "hub_push.py", repo, str(datasets_dir / MERGED_DIR), str(is_private), *map(str, ready), capture=False
    )
    return result.returncode


def upload(
    repo_id: Annotated[
        Optional[str], typer.Option("--repo-id", help="Hub dataset repo (default: station hub.repo_id).")
    ] = None,
    private: Annotated[
        Optional[bool],
        typer.Option("--private/--public", help="Hub visibility (default: station hub.private, normally private)."),
    ] = None,
    session: Annotated[
        Optional[str], typer.Option("--session", help="Session prefix to upload (default: the station's).")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be converted and pushed.")] = False,
    include_mock: Annotated[
        bool, typer.Option("--include-mock", help="Allow uploading mock-arm (mock-*) sessions.")
    ] = False,
    station: StationOption = None,
) -> None:
    """Convert finished recordings to LeRobot and push them as one dataset to the Hub."""
    config = load_station(station)
    raise typer.Exit(
        run_upload(
            config, repo_id=repo_id, private=private, prefix=session, dry_run=dry_run, include_mock=include_mock
        )
    )
