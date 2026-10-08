# Copyright 2026 Kite ML.
"""The kiteos CLI: runtime command building, safety gates, fence/collect/upload/stop helpers, doctor.

Nothing here touches hardware or launches the runtime: ``launch.execute`` and every subprocess or
live-stack access is replaced by a fake.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import kiteos.safety.fence as safety
from kiteos import __version__
from kiteos.cli import app, doctor, launch, stop, upload
from kiteos.cli import fence as fence_cli
from kiteos.cli._common import STATE, CmdResult, kiteos_pythonpath
from kiteos.cli.cameras import enumerate_darwin, enumerate_linux, missing_station_cameras
from kiteos.cli.can import parse_gs_usb_list, parse_ip_brief
from kiteos.cli.collect import delete_last_saved, latest_session, next_free_session
from kiteos.cli.evaluate import parse_episodes
from kiteos.cli.quest import proxy_argv
from kiteos.config import StationConfig, StationError, repo_root

REPO = Path(__file__).resolve().parents[1]
LEFT_SERIAL = "006C003E4845500F20303750"
RIGHT_SERIAL = "003900374845500F20303750"
STATION_NAME = "kite-test"
HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITEOS_STATION", raising=False)
    monkeypatch.delenv("KITEOS_ROOT", raising=False)
    monkeypatch.setenv("KITEOS_NO_BANNER", "1")
    STATE.station = None

    def no_host_commands(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"test tried to run a host command: {args[0] if args else kwargs}")

    # Nothing in these tests may reach the host (the runtime, adb, CAN or camera listing).
    monkeypatch.setattr(subprocess, "run", no_host_commands)
    monkeypatch.setattr(os, "execvpe", no_host_commands)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A station in tmp: config/station.yaml + config/fence.yaml, runtime root, checkpoint, data dirs."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (tmp_path / "kiteos-runtime").mkdir()
    (tmp_path / "ckpt").mkdir()
    station = {
        "station": STATION_NAME,
        "task": "close the box",
        "session_prefix": "close-box",
        "can": {
            "left": {"darwin": LEFT_SERIAL, "linux": "can_left"},
            "right": {"darwin": RIGHT_SERIAL, "linux": "can_right"},
        },
        "cameras": {
            "left_wrist": {"darwin": 0, "linux": "/dev/v4l/by-path/left"},
            "right_wrist": {"darwin": 2, "linux": "/dev/v4l/by-path/right"},
            "top": {"darwin": 1, "linux": "/dev/v4l/by-path/REPLACE-top"},
        },
        "fence": "config/fence.yaml",
        "recordings_dir": str(tmp_path / "recordings"),
        "datasets_dir": str(tmp_path / "datasets"),
        "hub": {"repo_id": "kiteml/test-data", "private": True},
        "policy": {"checkpoint": str(tmp_path / "ckpt"), "device": "auto"},
        "runtime_root": str(tmp_path / "kiteos-runtime"),
    }
    (config_dir / "station.yaml").write_text(yaml.safe_dump(station))
    # Like the shipped lab fence before teaching: no zones, explicitly allowed for teaching.
    write_fence(config_dir / "fence.yaml", zones=[], allow_no_zones=True)
    return config_dir / "station.yaml"


def write_fence(path: Path, zones: list[dict[str, Any]], **extra: Any) -> Path:
    data = {"version": 1, "station": STATION_NAME, "zones": zones, **extra}
    path.write_text(yaml.safe_dump(data))
    return path


GLASS = {"kind": "half_space", "name": "glass", "point": [0.6, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}
# A plane at x = 0.2 m cuts through the home pose (fingertip x ~ 0.39 m).
BAD_GLASS = {"kind": "half_space", "name": "too_close", "point": [0.2, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}


def invoke(site: Path, *args: str, input: str | None = None) -> Any:
    return runner.invoke(app, ["--station", str(site), *args], input=input)


def load(site: Path) -> StationConfig:
    return StationConfig.load(site)


# =========================================================================== command building


def test_mock_command_never_passes_can_ports(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    station = load(site)
    fence = station.fence_path()
    plan = launch.build_runtime_command(station, "collection", real=False, fence_path=fence, environ={})
    assert plan.argv[:6] == ["uv", "run", "--no-sync", "kiteos-runtime", "run", "kiteos.dual-openyam-collection"]
    assert not any("can-port" in a for a in plan.argv)
    i = plan.argv.index("--controlcoordinator.fence-path")
    assert plan.argv[i + 1] == str(fence) and Path(plan.argv[i + 1]).is_absolute()
    assert plan.argv[plan.argv.index("--left-wrist.hardware.camera-index") + 1] == "0"
    assert plan.argv[plan.argv.index("--right-wrist.hardware.camera-index") + 1] == "2"
    assert plan.argv[plan.argv.index("--top.hardware.camera-index") + 1] == "1"
    assert plan.cwd == station.runtime_root_path()
    assert plan.env["PYTHONPATH"] == str(repo_root() / "src")
    assert any("SIMULATED" in n for n in plan.notes)
    assert not plan.real


def test_real_command_on_darwin_uses_usb_serials(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    station = load(site)
    plan = launch.build_runtime_command(station, "teleop", real=True, fence_path=station.fence_path(), environ={})
    argv = plan.argv
    assert argv[argv.index("--controlcoordinator.left-can-port") + 1] == LEFT_SERIAL
    assert argv[argv.index("--controlcoordinator.right-can-port") + 1] == RIGHT_SERIAL
    assert not any("camera-index" in a for a in argv), "the teleop blueprint has no cameras"
    assert argv.index("--controlcoordinator.fence-path") < argv.index("--controlcoordinator.left-can-port")


def test_real_command_on_linux_uses_socketcan_names(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    station = load(site)
    plan = launch.build_runtime_command(station, "teleop", real=True, fence_path=station.fence_path(), environ={})
    assert plan.argv[plan.argv.index("--controlcoordinator.left-can-port") + 1] == "can_left"
    assert plan.argv[plan.argv.index("--controlcoordinator.right-can-port") + 1] == "can_right"


def test_real_command_requires_fence(site: Path) -> None:
    with pytest.raises(launch.LaunchError, match="fence"):
        launch.build_runtime_command(load(site), "teleop", real=True, fence_path=None)


def test_placeholder_camera_rejected(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    station = load(site)
    with pytest.raises(launch.LaunchError, match="camera top"):
        launch.build_runtime_command(station, "collection", real=False, fence_path=None)
    plan = launch.build_runtime_command(station, "collection", real=False, fence_path=None, cameras=False)
    assert not any("camera-index" in a for a in plan.argv)


def test_station_carry_settings_become_coordinator_flags(site: Path) -> None:
    data = yaml.safe_load(site.read_text())
    data["carry"] = {"max_speed_m_s": 0.15, "yaw": "lock"}
    site.write_text(yaml.safe_dump(data))
    argv = launch.build_runtime_command(load(site), "teleop", real=False, fence_path=None).argv
    flag = lambda name: argv[argv.index(f"--controlcoordinator.{name}") + 1]  # noqa: E731
    assert (flag("carry-max-speed-m-s"), flag("carry-yaw")) == ("0.15", "lock")
    assert (flag("carry-max-accel-m-s2"), flag("carry-max-yaw-rate-deg-s"), flag("carry-max-tilt-error-deg")) == (
        "1",
        "60",
        "3",
    )
    # The runtime's own setups (kiteos run) have no KiteOS coordinator: no carry flags.
    runtime = launch.BlueprintSpec("unitree-go2", False, "a runtime setup", package=None)
    assert not any("carry" in a for a in launch.build_runtime_command(load(site), runtime, real=False, fence_path=None).argv)
    for bad in ({"yaw": "spin"}, {"max_speed_m_s": 0}, {"max_speed_m_s": 2.0}, {"tilt": 3}):
        data["carry"] = bad
        site.write_text(yaml.safe_dump(data))
        with pytest.raises(StationError):
            load(site)


def test_viser_host_flag_only_when_exposed(site: Path) -> None:
    station = load(site)
    plan = launch.build_runtime_command(station, "teleop", real=False, fence_path=None)
    assert launch.VISER_FLAG not in plan.argv
    exposed = station.model_copy(update={"viser_host": "0.0.0.0"})
    exposed._source = station.source
    plan = launch.build_runtime_command(exposed, "teleop", real=False, fence_path=None)
    assert plan.argv[plan.argv.index(launch.VISER_FLAG) + 1] == "0.0.0.0"


def test_pythonpath_puts_kiteos_first_without_duplicates() -> None:
    src = str(repo_root() / "src")
    value = kiteos_pythonpath({"PYTHONPATH": os.pathsep.join(["/other", src])})
    assert value.split(os.pathsep) == [src, "/other"]
    assert kiteos_pythonpath({}) == src


def test_extra_flags_and_helpers(site: Path, tmp_path: Path) -> None:
    rec = tmp_path / "recordings" / "close-box-001"
    flags = launch.collection_flags(rec, "close the box")
    assert flags == [
        "--recorder.recording",
        str(rec),
        "--recorder.format",
        "mcap",
        "--recorder.shutdown-timeout",
        "100",
        "--episodes.task",
        "close the box",
    ]
    rollout = launch.rollout_flags(tmp_path / "ckpt", task="t", device="cuda", speed=0.5, max_start_jump_rad=0.2)
    assert rollout[rollout.index("--policy.fps") + 1] == "15"
    assert rollout[rollout.index("--policy.device") + 1] == "cuda"
    assert rollout[rollout.index("--policy.max-start-jump-rad") + 1] == "0.2"
    plan = launch.build_runtime_command(load(site), "rollout", real=False, fence_path=None, extra_flags=rollout)
    assert plan.argv[5] == "kiteos.dual-openyam-rollout"
    assert plan.argv[-len(rollout) :] == rollout
    assert "PYTHONPATH=" in plan.shell() and "kiteos.dual-openyam-rollout" in plan.shell()


@pytest.mark.parametrize(
    "requested, host, machine, nvidia, expected",
    [
        ("auto", "darwin", "arm64", False, "mps"),
        ("auto", "darwin", "x86_64", False, "cpu"),
        ("auto", "linux", "x86_64", True, "cuda"),
        ("auto", "linux", "aarch64", False, "cpu"),
        ("cpu", "darwin", "arm64", False, "cpu"),
        ("cuda", "linux", "x86_64", False, "cuda"),
    ],
)
def test_resolve_device(requested: str, host: str, machine: str, nvidia: bool, expected: str) -> None:
    got = launch.resolve_device(
        requested,
        host_platform=host,
        machine=machine,
        which=lambda name: "/usr/bin/nvidia-smi" if nvidia else None,
        exists=lambda path: False,
    )
    assert got == expected


def test_resolve_device_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        launch.resolve_device("tpu")


# =========================================================================== launch gates (CLI)


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[launch.LaunchPlan]:
    calls: list[launch.LaunchPlan] = []
    monkeypatch.setattr(launch, "execute", calls.append)
    return calls


def test_teleop_mock_dry_run(site: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "teleop", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "SIMULATED" in result.output
    assert "can-port" not in result.output
    assert "kiteos.dual-openyam-teleop" in result.output
    assert executed == []


def test_teleop_mock_executes(site: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "teleop")
    assert result.exit_code == 0, result.output
    assert len(executed) == 1 and not executed[0].real


def test_real_requires_typed_station_name(site: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "teleop", "--real", input="yes\n")
    assert result.exit_code == 1
    assert "not confirmed" in result.output
    assert executed == []


def test_real_confirmed_launches_with_can_ports(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0, result.output
    assert "NO KEEP-OUT ZONES ARE ENABLED" in result.output
    assert "REAL ARMS" in result.output
    assert len(executed) == 1
    plan = executed[0]
    assert plan.real and LEFT_SERIAL in plan.argv and RIGHT_SERIAL in plan.argv


def test_real_shows_enabled_zones_without_warning(site: Path, executed: list[launch.LaunchPlan]) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS])
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0, result.output
    assert "glass" in result.output
    assert "NO KEEP-OUT ZONES" not in result.output


def test_real_refuses_missing_fence(site: Path, executed: list[launch.LaunchPlan]) -> None:
    (site.parent / "fence.yaml").unlink()
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2
    assert "fence file not found" in result.output
    assert executed == []
    # Mock runs still work without a fence (joint and speed limits only).
    assert invoke(site, "teleop").exit_code == 0


def test_real_refuses_invalid_or_monitor_fence(site: Path, executed: list[launch.LaunchPlan]) -> None:
    write_fence(site.parent / "fence.yaml", zones=[], mode="monitor")
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "monitor" in result.output
    (site.parent / "fence.yaml").write_text("margin_m: 0.001\n")
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "invalid" in result.output
    assert executed == []


def test_collect_start_uses_next_free_session(
    site: Path, tmp_path: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    (tmp_path / "recordings" / "close-box-001").mkdir(parents=True)
    result = invoke(site, "collect", "start", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0, result.output
    argv = executed[0].argv
    assert argv[argv.index("--recorder.recording") + 1] == str(tmp_path / "recordings" / "close-box-002")
    assert argv[argv.index("--episodes.task") + 1] == "close the box"
    assert "--top.hardware.camera-index" in argv


def test_collect_start_session_prefix(site: Path, tmp_path: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "collect", "start", "--session", "pick-cube", "--task", "pick the cube", "--dry-run")
    assert result.exit_code == 0, result.output
    # Mock arms (no --real): recorded under mock-<prefix>, which upload never pushes.
    assert str(tmp_path / "recordings" / "mock-pick-cube-001") in result.output.replace("\n", "")
    assert "MOCK ARMS" in result.output


def test_rollout_start_flags(site: Path, tmp_path: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "rollout", "start", "--speed", "0.5", "--device", "cpu")
    assert result.exit_code == 0, result.output
    argv = executed[0].argv
    assert argv[argv.index("--policy.policy-path") + 1] == str(tmp_path / "ckpt")
    assert argv[argv.index("--policy.fps") + 1] == "15"
    assert argv[argv.index("--policy.device") + 1] == "cpu"


def test_rollout_real_caps_start_jump(site: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "rollout", "start", "--real", "--max-start-jump", "0.5", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "max-start-jump" in result.output
    assert executed == []


def test_rollout_missing_checkpoint(site: Path, tmp_path: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "rollout", "start", "--checkpoint", str(tmp_path / "nope"))
    assert result.exit_code == 2 and "checkpoint not found" in result.output


def test_bad_station_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--station", str(tmp_path / "missing.yaml"), "teleop", "--dry-run"])
    assert result.exit_code == 2
    assert "station file not found" in result.output


# =========================================================================== collect helpers


def test_next_free_session(tmp_path: Path) -> None:
    assert next_free_session(tmp_path, "close-box") == tmp_path / "close-box-001"
    (tmp_path / "close-box-001").mkdir()
    (tmp_path / "close-box-002").mkdir()
    assert next_free_session(tmp_path, "close-box") == tmp_path / "close-box-003"
    assert next_free_session(tmp_path / "missing", "x") == tmp_path / "missing" / "x-001"


def test_latest_session_is_numeric(tmp_path: Path) -> None:
    for n in (9, 10, 2):
        (tmp_path / f"close-box-{n:03d}").mkdir()
    (tmp_path / "close-box-1000").mkdir()
    (tmp_path / "other-999").mkdir()
    assert latest_session(tmp_path, "close-box") == tmp_path / "close-box-1000"
    assert latest_session(tmp_path, "nothing") is None


def test_delete_last_saved_walks_backwards(tmp_path: Path) -> None:
    assert delete_last_saved(tmp_path, 3) == (2, 2)
    assert delete_last_saved(tmp_path, 3) == (1, 1)
    assert delete_last_saved(tmp_path, 3) == (0, 0)
    assert delete_last_saved(tmp_path, 3) == (None, 0)
    assert (tmp_path / upload.DELETED_FILE).read_text().split() == ["2", "1", "0"]


# =========================================================================== upload helpers


def make_recording(root: Path, name: str, marks: list[int] | None = None) -> Path:
    rec = root / name
    rec.mkdir(parents=True)
    (rec / "recording.mcap").write_bytes(b"")
    if marks is not None:
        (rec / upload.DELETED_FILE).write_text("".join(f"{m}\n" for m in marks))
    return rec


def make_dataset(root: Path, name: str, applied: list[int] | None = None) -> Path:
    ds = root / name
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    if applied is not None:
        (ds / upload.APPLIED_FILE).write_text("".join(f"{m}\n" for m in applied))
    return ds


def test_find_recordings(tmp_path: Path) -> None:
    make_recording(tmp_path, "close-box-002")
    make_recording(tmp_path, "close-box-010")
    (tmp_path / "close-box-003").mkdir()  # no recording file yet
    make_recording(tmp_path, "pick-001")
    assert [p.name for p in upload.find_recordings(tmp_path, "close-box")] == ["close-box-002", "close-box-010"]


def test_map_to_dataset_indices() -> None:
    # saved 0..9, episode 3 already removed: dataset holds 0,1,2,4,5,... -> saved 5 is dataset 4.
    assert upload.map_to_dataset_indices([5, 9], {3}) == [4, 8]
    assert upload.map_to_dataset_indices([1], set()) == [1]
    assert upload.map_to_dataset_indices([3], {3}) == []


def test_plan_drops(tmp_path: Path) -> None:
    rec = make_recording(tmp_path / "rec", "close-box-001", marks=[3, 5])
    ds = make_dataset(tmp_path / "ds", "close-box-001", applied=[3])
    plan = upload.plan_drops(rec, ds)
    assert plan.dataset_indices == [4] and plan.new_marks == [5] and not plan.reconvert
    (ds / upload.APPLIED_FILE).write_text("3\n5\n")
    assert upload.plan_drops(rec, ds).dataset_indices is None


def test_plan_drops_legacy_dataset(tmp_path: Path) -> None:
    rec = make_recording(tmp_path / "rec", "close-box-001", marks=[1])
    ds = make_dataset(tmp_path / "ds", "close-box-001")
    info = ds / "meta" / "info.json"
    marks = rec / upload.DELETED_FILE
    os.utime(marks, (1000, 1000))
    os.utime(info, (2000, 2000))
    assert not upload.plan_drops(rec, ds).reconvert  # marks predate the conversion
    os.utime(marks, (3000, 3000))
    assert upload.plan_drops(rec, ds).reconvert  # marked afterwards, history unknown


def test_upload_dry_run(site: Path, tmp_path: Path) -> None:
    make_recording(tmp_path / "recordings", "close-box-001", marks=[0])
    make_recording(tmp_path / "recordings", "close-box-002", marks=[1, 2])
    make_dataset(tmp_path / "datasets", "close-box-002", applied=[1])
    result = invoke(site, "upload", "--dry-run")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "PRIVATE" in out and "kiteml/test-data" in out
    assert "convert    close-box-001 (drop saved episodes [0])" in out
    assert "drop       episodes [1] from close-box-002" in out
    assert "PUBLIC" in invoke(site, "upload", "--dry-run", "--public").output


def test_upload_nothing(site: Path) -> None:
    result = invoke(site, "upload", "--dry-run")
    assert result.exit_code == 1 and "No recordings" in result.output


def test_isolated_scripts_are_standalone() -> None:
    scripts = sorted((upload.ISOLATED_DIR).glob("*.py"))
    assert {p.name for p in scripts} >= {"fast_prepare.py", "hub_push.py", "drop_episodes.py", "eval_policy.py"}
    for script in scripts:
        text = script.read_text()
        assert "import kiteos" not in text and "from kiteos" not in text, script.name
        compile(text, str(script), "exec")


# =========================================================================== stop


def test_parse_runtime_status() -> None:
    text = "  Run ID:    abc-123\n  PID:       4242\n  Blueprint: kiteos.dual-openyam-collection\n"
    running = stop.parse_runtime_status(text)
    assert running == stop.RunningStack(4242, "abc-123", "kiteos.dual-openyam-collection")
    assert stop.parse_runtime_status("No running DimOS instance") is None


class FakeProcess:
    def __init__(self, dies_after: int | None) -> None:
        self.signals: list[int] = []
        self.polls = 0
        self.dies_after = dies_after

    def kill(self, pid: int, sig: int) -> None:
        self.signals.append(sig)
        if sig == 0:
            self.polls += 1
            if self.dies_after is not None and self.polls > self.dies_after:
                raise ProcessLookupError


def test_graceful_stop_only_sigterm() -> None:
    proc = FakeProcess(dies_after=3)
    assert stop.graceful_stop(1, timeout_s=300, kill=proc.kill, sleep=lambda s: None)
    assert proc.signals[0] == signal.SIGTERM
    assert set(proc.signals[1:]) == {0}


def test_graceful_stop_times_out_without_kill() -> None:
    proc = FakeProcess(dies_after=None)
    slept: list[float] = []
    assert not stop.graceful_stop(1, timeout_s=300, poll_s=1.0, kill=proc.kill, sleep=slept.append)
    assert sum(slept) == 300
    assert signal.SIGKILL not in proc.signals
    assert proc.signals.count(signal.SIGTERM) == 1


def fake_stacks(monkeypatch: pytest.MonkeyPatch, stacks: list[stop.RunningStack], unregistered=()) -> list[int]:
    monkeypatch.setattr(stop, "find_stacks", lambda: list(stacks))
    monkeypatch.setattr(stop.runs, "stack_processes", lambda: list(unregistered))
    signalled: list[int] = []
    monkeypatch.setattr(stop, "graceful_stop", lambda pid, **kw: signalled.append(pid) or True)
    return signalled


REAL_RUN = stop.RunningStack(4242, "r1", "kiteos.dual-openyam-collection", mock=False, source="KiteOS run")
UNKNOWN_RUN = stop.RunningStack(4343, "r2", "kiteos.dual-openyam-teleop", mock=None)
MOCK_RUN = stop.RunningStack(4444, "r3", "kiteos.dual-openyam-teleop", mock=True, source="KiteOS run")


def test_plain_stop_asks_for_confirmation_unless_the_run_is_proven_mock(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: `kiteos stop` without --real used to SIGTERM a REAL stack with no confirmation."""
    for run in (REAL_RUN, UNKNOWN_RUN):
        signalled = fake_stacks(monkeypatch, [run])
        result = invoke(site, "stop", input="")
        assert result.exit_code == 1 and signalled == [], result.output
        assert "may DROP" in result.output and run.mode in result.output.replace("\n", " ")
        result = invoke(site, "stop", input="stop\n")
        assert result.exit_code == 0 and signalled == [run.pid], result.output
    signalled = fake_stacks(monkeypatch, [MOCK_RUN])
    result = invoke(site, "stop", input="")
    assert result.exit_code == 0 and signalled == [MOCK_RUN.pid], result.output
    # --real is accepted (deprecated) and changes nothing.
    signalled = fake_stacks(monkeypatch, [REAL_RUN])
    assert invoke(site, "stop", "--real", input="no\n").exit_code == 1 and signalled == []


def test_stop_with_several_runs_needs_a_pid(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signalled = fake_stacks(monkeypatch, [REAL_RUN, MOCK_RUN])
    result = invoke(site, "stop", input="stop\n")
    assert result.exit_code == 2 and "--pid" in result.output and signalled == []
    result = invoke(site, "stop", "--pid", str(MOCK_RUN.pid))
    assert result.exit_code == 0 and signalled == [MOCK_RUN.pid]
    assert invoke(site, "stop", "--pid", "1").exit_code == 2


def test_stop_reports_unregistered_stacks_instead_of_nothing_to_stop(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signalled = fake_stacks(monkeypatch, [], unregistered=[(97794, "python3 -m kite_rollout.run --real")])
    result = invoke(site, "stop", input="stop\n")
    assert result.exit_code == 1 and signalled == []
    assert "kite_rollout" in result.output and "nothing to stop" not in result.output
    fake_stacks(monkeypatch, [])
    result = invoke(site, "stop")
    assert result.exit_code == 0 and "nothing to stop" in result.output


def test_find_stacks_merges_kiteos_runs_and_the_registry() -> None:
    from kiteos.cli.runs import RunState

    state = RunState(pid=100, real=False, blueprint="kiteos.dual-openyam-teleop", station="s")
    registry = [
        {"pid": 101, "run_id": "a", "blueprint": "kiteos.dual-openyam-teleop"},
        {"pid": 200, "run_id": "b", "blueprint": "x", "original_argv": ["dimos", "run", "--controlcoordinator.left-can-port", "L"]},
        {"pid": 300, "run_id": "c", "blueprint": "y", "original_argv": ["dimos", "run", "--controlcoordinator.expect-mock", "true"]},
    ]
    table = {101: (100, "/x/python /x/dimos run kiteos.dual-openyam-teleop")}
    found = stop.find_stacks(kiteos_runs=lambda: [state], registry=lambda: registry, processes=lambda: table)
    assert [(s.pid, s.mock, s.related) for s in found] == [(101, True, (100,)), (200, False, ()), (300, True, ())]
    # Not registered (yet): the `dimos run` child of the uv process is signalled, not uv.
    table = {150: (100, "/x/python /x/dimos run kiteos.dual-openyam-teleop")}
    found = stop.find_stacks(kiteos_runs=lambda: [state], registry=lambda: [], processes=lambda: table)
    assert [(s.pid, s.related) for s in found] == [(150, (100,))]


def test_find_stacks_signals_the_kiteos_runtime_child() -> None:
    from kiteos.cli.runs import RunState

    state = RunState(pid=400, real=True, blueprint="kiteos.dual-openyam-collection", station="s")
    table = {401: (400, "/x/.venv/bin/python /x/.venv/bin/kiteos-runtime run kiteos.dual-openyam-collection")}
    found = stop.find_stacks(kiteos_runs=lambda: [state], registry=lambda: [], processes=lambda: table)
    assert [(s.pid, s.mock, s.related) for s in found] == [(401, False, (400,))]


def test_stop_running_signals_the_chosen_stack(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signalled = fake_stacks(monkeypatch, [REAL_RUN])
    assert stop.stop_running(load(site))
    assert signalled == [REAL_RUN.pid]


# =========================================================================== small parsers


def test_parse_episodes() -> None:
    assert parse_episodes("0,2,5-7") == [0, 2, 5, 6, 7]
    with pytest.raises(ValueError):
        parse_episodes("3-1")
    with pytest.raises(ValueError):
        parse_episodes("")


def test_can_listing_parsers() -> None:
    darwin = f"INDEX  SERIAL\n0      {LEFT_SERIAL}\n1      <missing>\n"
    assert parse_gs_usb_list(darwin) == [LEFT_SERIAL]
    linux = "can_left         UP             \ncan_right@usb    DOWN\n"
    assert parse_ip_brief(linux) == {"can_left": "UP", "can_right": "DOWN"}


def test_camera_enumeration_never_opens_devices() -> None:
    fake_fs = {
        "/dev/v4l/by-id/*": ["/dev/v4l/by-id/usb-cam-video-index0"],
        "/dev/v4l/by-path/*": [],
        "/dev/video*": ["/dev/video0", "/dev/video1"],
    }
    devices = enumerate_linux(lambda pattern: fake_fs[pattern])
    assert devices[0].selector == "/dev/v4l/by-id/usb-cam-video-index0"
    assert {d.selector for d in devices} >= {"/dev/video0", "/dev/video1"}
    listing = json.dumps(
        {
            "SPCameraDataType": [
                {"_name": "MacBook Pro Camera", "spcamera_unique-id": "EAB7A68F-EC2B"},
                {"_name": "USB Camera", "spcamera_unique-id": "0x1234000"},
            ]
        }
    )
    cams = enumerate_darwin(lambda argv, timeout: CmdResult(0, listing, ""))
    # OpenCV orders AVFoundation devices by unique ID, not in system_profiler's order.
    assert [c.selector for c in cams] == ["0", "1"] and "USB Camera" in cams[0].label
    no_ids = json.dumps({"SPCameraDataType": [{"_name": "USB Camera"}, {"_name": "FaceTime"}]})
    assert {c.selector for c in enumerate_darwin(lambda argv, timeout: CmdResult(0, no_ids, ""))} == {"?"}
    assert enumerate_darwin(lambda argv, timeout: CmdResult(1, "", "boom")) == []


def profiler(*names: str) -> Any:
    """A fake ``system_profiler SPCameraDataType -json`` listing these cameras."""
    listing = json.dumps({"SPCameraDataType": [{"_name": n, "spcamera_unique-id": f"id{i}"} for i, n in enumerate(names)]})
    return lambda argv, timeout: CmdResult(0, listing, "")


def never_listed(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("the cameras were listed")


def test_missing_cameras_on_macos_counts_only_usb_cameras(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    station = load(site)
    problem = missing_station_cameras(station, profiler("MacBook Pro Camera", "iPhone (136) Camera"))
    assert problem is not None and problem.startswith("0 of the station's 3 USB cameras are connected")
    assert "not counted: MacBook Pro Camera, iPhone (136) Camera" in problem and "cover test" in problem
    two = missing_station_cameras(station, profiler("USB Camera", "Studio Display Camera", "C920", "OBS Virtual Camera"))
    assert two is not None and two.startswith("2 of the station's 3")
    assert missing_station_cameras(station, profiler("USB Camera", "C920", "Arducam", "MacBook Pro Camera")) is None
    failed = missing_station_cameras(station, lambda argv, timeout: CmdResult(1, "", "boom"))
    assert failed is not None and "system_profiler failed" in failed


def test_missing_cameras_skips_mock_roles(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    data = yaml.safe_load(site.read_text())
    data["cameras"]["top"] = {"darwin": "mock", "linux": "mock"}
    site.write_text(yaml.safe_dump(data))
    assert missing_station_cameras(load(site), profiler("USB Camera", "C920")) is None  # two roles, two cameras
    for role in ("left_wrist", "right_wrist"):
        data["cameras"][role] = {"darwin": "mock", "linux": "mock"}
    site.write_text(yaml.safe_dump(data))
    assert missing_station_cameras(load(site), never_listed) is None


def test_missing_cameras_on_linux_needs_every_device(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    data = yaml.safe_load(site.read_text())
    data["cameras"]["top"]["linux"] = 4  # an index is /dev/video4
    site.write_text(yaml.safe_dump(data))
    present = {"/dev/v4l/by-path/left", "/dev/video4"}
    problem = missing_station_cameras(load(site), never_listed, exists=present.__contains__)
    assert problem is not None and "right_wrist (/dev/v4l/by-path/right)" in problem and "left_wrist" not in problem
    present.add("/dev/v4l/by-path/right")
    assert missing_station_cameras(load(site), never_listed, exists=present.__contains__) is None


def test_collect_and_rollout_refuse_to_start_without_the_cameras(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    asked: list[str] = []

    def missing(station: StationConfig) -> str:
        asked.append(station.station)
        return "0 of the station's 3 USB cameras are connected"

    monkeypatch.setattr(launch, "cameras_missing", missing)
    # Refused before the typed --real confirmation (no input is given).
    result = invoke(site, "collect", "start", "--real")
    assert result.exit_code == 2 and "0 of the station's 3 USB cameras" in result.output
    result = invoke(site, "rollout", "start", "--device", "cpu")
    assert result.exit_code == 2 and "USB cameras" in result.output
    assert executed == []
    dry = invoke(site, "collect", "start", "--dry-run")  # a dry run only warns
    assert dry.exit_code == 0 and "0 of the station's 3 USB cameras" in dry.output
    assert len(asked) == 3
    assert invoke(site, "teleop").exit_code == 0 and len(asked) == 3  # teleop records no camera


def test_quest_proxy_argv(site: Path) -> None:
    station = load(site)
    assert proxy_argv(station) == ["--port", "8080", "--upstream", "https://127.0.0.1:8443"]
    station.quest.adb = "off"
    assert proxy_argv(station)[-1] == "--no-adb"
    station.quest.adb = "/opt/adb"
    assert proxy_argv(station, ["-v"])[-3:] == ["--adb-path", "/opt/adb", "-v"]


# =========================================================================== fence CLI


def test_fence_show(site: Path) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS, {**BAD_GLASS, "enabled": False}])
    result = invoke(site, "fence", "show")
    assert result.exit_code == 0, result.output
    assert "glass" in result.output and "too_close" in result.output and "enforce" in result.output


def test_fence_validate_no_zones_warns(site: Path) -> None:
    result = invoke(site, "fence", "validate")
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output and "none enabled" in result.output


def test_fence_validate_blocks_zone_over_home_pose(site: Path) -> None:
    write_fence(site.parent / "fence.yaml", zones=[BAD_GLASS])
    result = invoke(site, "fence", "validate")
    assert result.exit_code == 1
    assert "FAIL" in result.output and "home pose" in result.output


def test_fence_validate_invalid_file(site: Path) -> None:
    (site.parent / "fence.yaml").write_text("zones:\n  - kind: sphere\n    name: x\n")
    result = invoke(site, "fence", "validate")
    assert result.exit_code == 2


def test_fence_validate_explicit_file() -> None:
    result = runner.invoke(app, ["fence", "validate", "--fence", str(REPO / "config" / "fence.example.yaml")])
    assert result.exit_code == 0, result.output
    assert "glass_window" in result.output


def test_fence_test_sweep(site: Path) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS, {**BAD_GLASS, "enabled": False, "applies_to": "left"}])
    result = invoke(site, "fence", "test", "--samples", "200")
    assert result.exit_code == 0, result.output
    assert "glass" in result.output and "all enabled zones" in result.output


def test_sweep_fractions(site: Path) -> None:
    config = safety.FenceConfig.model_validate({"zones": [GLASS, {**BAD_GLASS, "applies_to": "left"}]})
    rows, combined = fence_cli.sweep(config, samples=300, seed=1)
    by_zone = {r.zone: r for r in rows}
    assert by_zone["too_close"].blocked["right"] is None  # left-only zone
    assert by_zone["too_close"].blocked["left"] > by_zone["glass"].blocked["left"] > 0.0
    assert combined["left"] >= by_zone["too_close"].blocked["left"]
    assert rows == fence_cli.sweep(config, samples=300, seed=1)[0], "seeded sweep is reproducible"


def write_dataset(root: Path, episodes: list[list[list[float]]], names: list[str] | None = None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kiteos.safety.kinematics import DUAL_OPENYAM_JOINTS

    names = names or list(DUAL_OPENYAM_JOINTS)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feature = {"dtype": "float32", "shape": [14], "names": names}
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "features": {"observation.state": feature, "action": feature}})
    )
    rows: dict[str, list[Any]] = {"observation.state": [], "action": [], "episode_index": [], "frame_index": []}
    for ep, frames in enumerate(episodes):
        for i, vec in enumerate(frames):
            rows["observation.state"].append(vec)
            rows["action"].append(vec)
            rows["episode_index"].append(ep)
            rows["frame_index"].append(i)
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def reach(x: float) -> list[float]:
    """Both arms at home but with shoulder/elbow opened by ``x`` rad (reaching forward)."""
    arm = [0.0, 1.047 + x, 1.047 + x, 0.0, 0.0, 0.0]
    return arm + arm + [1.0, 1.0]


def test_fence_test_replays_dataset(site: Path, tmp_path: Path) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS])
    clear = write_dataset(tmp_path / "clear", [[reach(0.0), reach(0.02), reach(0.04)]])
    result = invoke(site, "fence", "test", "--samples", "100", "--dataset", str(clear))
    assert result.exit_code == 0, result.output
    assert "0 action frames held" in result.output

    blocked = write_dataset(tmp_path / "blocked", [[reach(0.0), reach(0.9), reach(0.95)], [reach(0.0)]])
    result = invoke(site, "fence", "test", "--samples", "100", "--dataset", str(blocked))
    assert result.exit_code == 1, result.output
    assert "action frames held" in result.output and "glass" in result.output


def test_replay_reorders_named_features(tmp_path: Path) -> None:
    from kiteos.safety.kinematics import DUAL_OPENYAM_JOINTS

    config = safety.FenceConfig.model_validate({"zones": [{**GLASS, "applies_to": "right"}]})
    swapped = list(DUAL_OPENYAM_JOINTS[6:12]) + list(DUAL_OPENYAM_JOINTS[:6]) + list(DUAL_OPENYAM_JOINTS[12:])
    # Stored order right-first: the far reach is stored in columns 0..5, i.e. it is the RIGHT arm.
    far = [0.0, 1.947, 1.947, 0.0, 0.0, 0.0]
    vec = far + HOME + [1.0, 1.0]
    root = write_dataset(tmp_path / "ds", [[vec]], names=swapped)
    result = fence_cli.replay_dataset(config, root, motion=False)
    assert result.frames == 1 and result.held_action_frames == 1
    assert {h.side for h in result.held} == {"right"}


def test_fence_check_offline(site: Path) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS])
    ok = invoke(site, "fence", "check", "--offline", "--positions", ",".join(map(str, HOME * 2 + [1, 1])))
    assert ok.exit_code == 0, ok.output
    far = [0.0, 1.947, 1.947, 0.0, 0.0, 0.0]
    bad = invoke(site, "fence", "check", "--offline", "--positions", json.dumps(far + HOME))
    assert bad.exit_code == 1 and "VIOLATION" in bad.output
    assert invoke(site, "fence", "check", "--offline").exit_code == 2


class FakeLive:
    """Scripted running stack for fence check/teach."""

    def __init__(self, poses: list[dict[str, float]]) -> None:
        self.poses = list(poses)
        self.reloaded: list[str | None] = []
        self.closed = False

    def joint_positions(self) -> dict[str, float]:
        return self.poses.pop(0) if len(self.poses) > 1 else self.poses[0]

    def fence_status(self) -> dict[str, Any]:
        return {"mode": "enforce", "zones": ["glass"], "holds": 0, "estop": False}

    def fence_check(self, positions: dict[str, float]) -> dict[str, Any]:
        config = safety.FenceConfig.model_validate({"zones": [GLASS]})
        return safety.FenceChecker(config).check(positions).to_dict()

    def fence_reload(self, path: str | None = None) -> dict[str, Any]:
        self.reloaded.append(path)
        return {"ok": True, "zones": 1}

    def close(self) -> None:
        self.closed = True


def named(left: list[float], right: list[float]) -> dict[str, float]:
    from kiteos.safety.kinematics import DUAL_OPENYAM_JOINTS

    return dict(zip(DUAL_OPENYAM_JOINTS, [*left, *right, 1.0, 1.0], strict=True))


def test_fence_check_live(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = FakeLive([named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    write_fence(site.parent / "fence.yaml", zones=[GLASS])
    result = invoke(site, "fence", "check")
    assert result.exit_code == 0, result.output
    assert "Running fence" in result.output and "clear" in result.output
    assert live.closed


def touch_q(side: str, target: list[float]) -> list[float]:
    """Joints 1-3 (Gauss-Newton on the fingertip) that put ``side``'s fingertip on ``target``."""
    from kiteos.safety.kinematics import DualArmKinematics

    kin = DualArmKinematics()
    q = np.array([0.0, 1.6, 1.2, 0.0, 0.0, 0.0])
    for _ in range(50):
        err = np.asarray(target) - kin.fingertip(side, q)
        if np.linalg.norm(err) < 1e-9:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            dq = np.zeros(6)
            dq[k] = 1e-6
            J[:, k] = (kin.fingertip(side, q + dq) - kin.fingertip(side, q)) / 1e-6
        q[:3] += np.linalg.lstsq(J, err, rcond=None)[0]
    assert np.linalg.norm(np.asarray(target) - kin.fingertip(side, q)) < 1e-6
    return [float(v) for v in q]


# Five fingertip touches on a vertical glass at x = 0.50 m, spread about 20 x 20 cm.
TOUCH_POINTS = ([0.50, 0.22, 0.14], [0.50, 0.42, 0.14], [0.50, 0.22, 0.32], [0.50, 0.42, 0.32], [0.50, 0.32, 0.23])


def plane_touch_poses(side: str = "left", points=TOUCH_POINTS) -> list[dict[str, float]]:
    """Poses whose ``side`` fingertips touch the vertical plane x = 0.50 m (world)."""
    offset = 0.0 if side == "left" else -0.62
    poses = []
    for x, y, z in points:
        q = touch_q(side, [x, y + offset, z])
        poses.append(named(q, HOME) if side == "left" else named(HOME, q))
    return poses


def test_fence_teach_writes_zone_with_backup(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    poses = plane_touch_poses()
    live = FakeLive([*poses, named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    fence_path = site.parent / "fence.yaml"
    original = fence_path.read_text()
    result = invoke(
        site,
        "fence",
        "teach",
        "--name",
        "glass_window",
        "--arms",
        "left",
        "--frame",
        "dual_openyam_base",
        "--force",
        input="\n\n\n\n\ndone\ny\ny\n",
    )
    assert result.exit_code == 0, result.output
    config = safety.FenceConfig.load(fence_path)
    zone = next(z for z in config.zones if z.name == "glass_window")
    assert zone.kind == "half_space" and zone.enabled and zone.frame == "dual_openyam_base"
    # The robot (between the bases, above the table) must be on the allowed side.
    ref = np.array([0.0, 0.0, 0.25])
    assert float(np.dot(np.array(zone.normal), ref - np.array(zone.point))) < 0
    backups = list(site.parent.glob("fence.yaml.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == original
    assert live.reloaded == [str(fence_path)]
    assert live.closed


def test_fence_teach_abort_leaves_file(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = FakeLive(plane_touch_poses())
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    fence_path = site.parent / "fence.yaml"
    original = fence_path.read_text()
    result = invoke(site, "fence", "teach", "--arms", "left", "--force", input="\n\n\n\n\ndone\nn\n")
    assert result.exit_code == 1
    assert fence_path.read_text() == original
    assert not list(site.parent.glob("fence.yaml.bak-*"))
    assert live.reloaded == []


def test_upsert_and_backup(tmp_path: Path) -> None:
    config = safety.FenceConfig.model_validate({"zones": [GLASS, BAD_GLASS]})
    moved = safety.FenceConfig.model_validate({"zones": [{**GLASS, "point": [0.7, 0.0, 0.0]}]}).zones[0]
    updated = fence_cli.upsert_zone(config, moved)
    assert [z.name for z in updated.zones] == ["glass", "too_close"]
    assert updated.zones[0].point[0] == pytest.approx(0.7)
    path = tmp_path / "fence.yaml"
    path.write_text("# hand comments\n")
    backup = fence_cli.backup_file(path, datetime(2026, 9, 27, 12, 0, 0))
    assert backup.name == "fence.yaml.bak-20260927-120000"


def test_mount_matrix_matches_kinematics() -> None:
    from kiteos.safety.kinematics import pose_matrix

    np.testing.assert_allclose(
        fence_cli.mount_matrix((0.1, 0.2, 0.3), (0.3, -0.2, 1.1)),
        pose_matrix((0.1, 0.2, 0.3), (0.3, -0.2, 1.1)),
        atol=1e-12,
    )


# =========================================================================== doctor


class FakeProbe(doctor.Probe):
    """A healthy darwin host; tweak attributes to break single checks."""

    def __init__(self, station: StationConfig | None) -> None:
        super().__init__(station)
        self.platform = "darwin"
        self.commit = doctor.read_pin(repo_root() / doctor.PIN_FILE) or "0" * 40
        self.blueprints = list(doctor.EXPECTED_BLUEPRINTS)
        self.serials = [LEFT_SERIAL, RIGHT_SERIAL]
        self.tools = {"uv": "/usr/bin/uv", "adb": "/usr/bin/adb"}
        self.recorder = True
        self.stacks: list[str] = []
        self.calls: list[tuple[str, ...]] = []

    def running_stacks(self) -> list[str]:
        return list(self.stacks)

    def which(self, name: str) -> str | None:
        return self.tools.get(name)

    def is_executable(self, path: Path) -> bool:
        return self.recorder

    def exists(self, path: Path) -> bool:
        # Only the tmp station files are real; never the host's Homebrew adb.
        return str(path) != "/opt/homebrew/bin/adb" and path.exists()

    def glob(self, pattern: str) -> list[str]:
        return []

    def run(self, argv: Any, cwd: Path | None = None, timeout: float = 60.0) -> CmdResult:
        argv = tuple(argv)
        self.calls.append(argv)
        if argv[:1] == ("git",) and "rev-parse" in argv:
            return CmdResult(0, self.commit + "\n", "")
        if argv[:1] == ("git",):
            return CmdResult(0, "", "")
        if argv[0] == "system_profiler":
            cams = [{"_name": "USB cam"}] * 3 + [{"_name": "MacBook Pro Camera"}]
            return CmdResult(0, json.dumps({"SPCameraDataType": cams}), "")
        if argv[-1] == "devices":
            return CmdResult(0, "List of devices attached\n1WMHH0000 device\n", "")
        raise AssertionError(f"unexpected command {argv}")

    def runtime(self, *args: str, timeout: float = 120.0) -> CmdResult:
        self.calls.append(("dimos", *args))
        if args == ("list",):
            return CmdResult(
                0,
                "Built-in blueprints:\n  x\n\nExternal blueprints:\n" + "".join(f"  {b}\n" for b in self.blueprints),
                "",
            )
        if args[:3] == ("hardware", "can", "list"):
            return CmdResult(0, "INDEX  SERIAL\n" + "".join(f"{i}      {s}\n" for i, s in enumerate(self.serials)), "")
        raise AssertionError(f"unexpected dimos command {args}")

    def hf_token(self) -> str | None:
        return "hf_test"

    def imported_runtime_root(self) -> Path | None:
        assert self.station is not None
        return self.station.runtime_root_path()

    def python_version(self) -> tuple[int, int]:
        return (3, 12)


def run_doctor_with(site: Path, **tweaks: Any) -> list[doctor.Check]:
    def factory(station: StationConfig | None) -> FakeProbe:
        probe = FakeProbe(station)
        for key, value in tweaks.items():
            setattr(probe, key, value)
        return probe

    return doctor.run_doctor(site, factory)


def levels(checks: list[doctor.Check]) -> dict[str, str]:
    return {c.name: c.level for c in checks}


def test_doctor_healthy_host(site: Path) -> None:
    checks = run_doctor_with(site)
    by_name = levels(checks)
    assert "FAIL" not in by_name.values(), [c for c in checks if c.level == "FAIL"]
    assert by_name["setups"] == "PASS"
    assert by_name["runtime commit"] == "PASS"
    assert by_name["FK self-test"] == "PASS"
    assert by_name["CAN left"] == by_name["CAN right"] == "PASS"
    assert by_name["fence zones"] == "WARN"  # no zones enabled (allow_no_zones) in the test fence


def test_doctor_blocking_failures(site: Path) -> None:
    checks = run_doctor_with(site, blueprints=[], commit="f" * 40, recorder=False, tools={"adb": "/usr/bin/adb"})
    by_name = levels(checks)
    assert by_name["setups"] == "FAIL"
    assert by_name["runtime commit"] == "FAIL"
    assert by_name["recorder"] == "FAIL"
    assert by_name["uv"] == "FAIL"
    fixes = {c.name: c.fix or "" for c in checks}
    assert "uv pip install --python" in fixes["setups"] and "--no-deps -e" in fixes["setups"]
    assert "checkout" in fixes["runtime commit"]


def test_doctor_hardware_warnings_are_not_blocking(site: Path) -> None:
    checks = run_doctor_with(site, serials=[LEFT_SERIAL], tools={"uv": "/usr/bin/uv"})
    by_name = levels(checks)
    assert by_name["CAN right"] == "WARN"
    assert by_name["adb"] == "WARN"
    assert "FAIL" not in by_name.values()


def test_doctor_linux_can(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    class LinuxProbe(FakeProbe):
        def __init__(self, station: StationConfig | None) -> None:
            super().__init__(station)
            self.platform = "linux"

        def runtime(self, *args: str, timeout: float = 120.0) -> CmdResult:
            if args[:3] == ("hardware", "can", "list"):
                return CmdResult(0, "can_left         UP\ncan_right        DOWN\n", "")
            return super().runtime(*args, timeout=timeout)

        def exists(self, path: Path) -> bool:
            return str(path) != "/dev/v4l/by-path/right" and super().exists(path)

    checks = doctor.run_doctor(site, LinuxProbe)
    by_name = levels(checks)
    assert by_name["CAN left"] == "PASS"
    assert by_name["CAN right"] == "WARN"
    assert by_name["station camera top"] == "FAIL"  # placeholder: collect/rollout refuse to start
    assert by_name["camera right_wrist"] == "WARN"


def test_doctor_never_lists_can_adapters_while_a_stack_runs(site: Path) -> None:
    checks = run_doctor_with(site, stacks=["process: PID 97794: python -m kite_rollout.run --real"])
    can = [c for c in checks if c.name.startswith("CAN")]
    assert len(can) == 1 and can[0].level == "WARN" and "not checked" in can[0].detail


def test_doctor_does_not_count_built_in_or_continuity_cameras(site: Path) -> None:
    class TwoUsbCams(FakeProbe):
        def run(self, argv: Any, cwd: Path | None = None, timeout: float = 60.0) -> CmdResult:
            if tuple(argv)[0] == "system_profiler":
                cams = [{"_name": "USB cam"}] * 2 + [{"_name": "MacBook Pro Camera"}, {"_name": "iPhone (136) Camera"}]
                return CmdResult(0, json.dumps({"SPCameraDataType": cams}), "")
            return super().run(argv, cwd, timeout)

    checks = doctor.run_doctor(site, TwoUsbCams)
    camera = next(c for c in checks if c.name == "cameras")
    assert camera.level == "WARN" and camera.detail.startswith("2 USB camera")


def test_doctor_bad_station_fails_fast(tmp_path: Path) -> None:
    checks = doctor.run_doctor(tmp_path / "missing.yaml", FakeProbe)
    assert levels(checks)["station"] == "FAIL"
    assert "setups" not in levels(checks)


def test_doctor_command_exit_codes(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "Probe", FakeProbe)
    result = invoke(site, "doctor")
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output and "ready" in result.output

    class Broken(FakeProbe):
        def __init__(self, station: StationConfig | None) -> None:
            super().__init__(station)
            self.blueprints = []

    monkeypatch.setattr(doctor, "Probe", Broken)
    result = invoke(site, "doctor")
    assert result.exit_code == 1
    assert "FAIL" in result.output and "setups" in result.output


def test_doctor_reads_pin() -> None:
    assert doctor.read_pin(REPO / "deploy" / "runtime" / "PIN") == "3e7cac5dc34bb766617d30486d7d9f1b2d89d93a"


# =========================================================================== CLI surface


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "doctor",
        "teleop",
        "collect",
        "rollout",
        "eval",
        "quest",
        "cameras",
        "can",
        "upload",
        "stop",
        "fence",
    ):
        assert command in result.output
    for group, subs in {
        "collect": ("start", "controls"),
        "rollout": ("start", "controls"),
        "fence": ("show", "validate", "check", "teach", "test"),
        "cameras": ("list", "check"),
        "can": ("setup", "status"),
    }.items():
        out = runner.invoke(app, [group, "--help"]).output
        for sub in subs:
            assert sub in out, (group, sub)


def test_banner_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITEOS_NO_BANNER")
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and f"KiteOS {__version__}" in result.output
    result = runner.invoke(app, ["fence", "validate", "--fence", str(REPO / "config" / "fence.yaml")])
    assert result.output.startswith("KiteOS")


# =========================================================================== collection TUI


class FakeCollectionSession:
    """Episode monitor stand-in: start/save/discard/toggle over a tiny state machine."""

    def __init__(self) -> None:
        from dimos.msgs.imitation_msgs.EpisodeStatus import EpisodeStatus

        self._status_cls = EpisodeStatus
        self.state, self.saved, self.discarded, self.last = "idle", 0, 0, "init"
        self.commands: list[str] = []
        self.closed = False

    def get_status(self) -> Any:
        return self._status_cls(
            ts=0.0,
            state=self.state,
            episodes_saved=self.saved,
            episodes_discarded=self.discarded,
            last_event=self.last,
            task_label="close the box",
        )

    def command(self, event: str) -> Any:
        self.commands.append(event)
        if event == "toggle":
            event = "save" if self.state == "recording" else "start"
        if event == "start":
            self.state, self.last = "recording", "start"
        elif event == "save" and self.state == "recording":
            self.state, self.saved, self.last = "idle", self.saved + 1, "save"
        elif event == "discard" and self.state == "recording":
            self.state, self.discarded, self.last = "idle", self.discarded + 1, "discard"
        return self.get_status()

    def close(self) -> None:
        self.closed = True


@pytest.mark.runtime
def test_collection_tui_keys(tmp_path: Path) -> None:
    import asyncio

    tui = pytest.importorskip("kiteos.cli.collect_tui")
    session_dir = tmp_path / "close-box-001"
    session_dir.mkdir()
    session = FakeCollectionSession()
    app_ = tui.KiteCollectionApp(session, title="KiteOS · close the box", session_dir=lambda: session_dir)

    async def scenario() -> None:
        async with app_.run_test() as pilot:
            await pilot.press("space")  # start
            await pilot.press("space")  # save -> 1 saved
            await pilot.press("space")  # start
            await pilot.press("space")  # save -> 2 saved
            await pilot.press("u")  # delete saved episode index 1
            await pilot.press("r")  # restart: nothing recording, so just start
            await pilot.press("r")  # restart while recording: discard + start
            await pilot.press("p")  # save the take and exit for power-down

    asyncio.run(scenario())
    assert app_.return_value == tui.PAUSE_RESULT
    assert (session_dir / upload.DELETED_FILE).read_text().split() == ["1"]
    assert session.commands == ["toggle", "toggle", "toggle", "toggle", "start", "discard", "start", "save"]
    assert session.saved == 3 and session.discarded == 1
    assert session.closed


def test_real_refuses_fence_of_another_station(site: Path, executed: list[launch.LaunchPlan]) -> None:
    write_fence(site.parent / "fence.yaml", zones=[GLASS], station="some-other-site")
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "some-other-site" in result.output
    assert executed == []
    assert invoke(site, "teleop", "--dry-run").exit_code == 0  # mock runs only warn


def test_fence_teach_per_arm_frame(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = FakeLive([*plane_touch_poses(), named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    result = invoke(site, "fence", "teach", "--arms", "left", "--name", "glass_left", input="\n\n\n\n\ndone\ny\nn\n")
    assert result.exit_code == 0, result.output
    zone = next(z for z in safety.FenceConfig.load(site.parent / "fence.yaml").zones if z.name == "glass_left")
    assert zone.frame == "left_base" and zone.applies_to == "left"
    assert not zone.enabled  # answered "n" to enabling it now


def test_fence_teach_preview_reports_fit_residual_not_standoff(
    site: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five exact touches on one plane; the 2 cm standoff must not show up as fit error.
    live = FakeLive([*plane_touch_poses(), named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    result = invoke(site, "fence", "teach", "--arms", "left", "--standoff", "0.02", input="\n\n\n\n\ndone\nn\n")
    text = " ".join(result.output.split())  # rich wraps long lines
    assert "rms 0.0 mm, worst 0.0 mm" in text, result.output
    assert "zone plane 20.0 mm in front of the touched surface" in text, result.output
    assert "per-point residuals (mm): +0.0, +0.0, +0.0, +0.0, +0.0" in text, result.output
    assert "tilt uncertainty 0.00 deg" in text, result.output


# =========================================================================== review fixes: launch


def test_mock_plan_uses_a_private_bus_free_ports_and_pins_safety_flags(site: Path) -> None:
    from kiteos.cli import runs

    bus = runs.random_private_bus()
    plan = launch.build_runtime_command(
        load(site), "teleop", real=False, fence_path=None, environ={}, private_bus=bus, ports=(50001, 50002)
    )
    assert plan.env["LCM_DEFAULT_URL"] == bus["LCM_DEFAULT_URL"] and ":7667" not in bus["LCM_DEFAULT_URL"]
    assert plan.env["ZENOH_SCOUT_ADDR"] == bus["ZENOH_SCOUT_ADDR"] and not bus["ZENOH_SCOUT_ADDR"].endswith(":7446")
    argv = plan.argv
    assert argv[argv.index(launch.WEBXR_PORT_FLAG) + 1] == "50001"
    assert argv[argv.index(launch.VISER_PORT_FLAG) + 1] == "50002"
    assert argv[argv.index(launch.VIEWER_FLAG) + 1] == "none"
    assert argv[argv.index("--controlcoordinator.expect-mock") + 1] == "true"
    assert argv[argv.index("--controlcoordinator.require-fence-on-hardware") + 1] == "true"
    assert "LCM_DEFAULT_URL=" in plan.shell()
    with pytest.raises(launch.LaunchError, match="private"):
        launch.build_runtime_command(load(site), "teleop", real=True, fence_path=site, environ={}, private_bus=bus)
    real = launch.build_runtime_command(load(site), "teleop", real=True, fence_path=site, environ={})
    assert real.argv[real.argv.index("--controlcoordinator.expect-mock") + 1] == "false"
    assert not real.private_bus and "LCM_DEFAULT_URL" not in real.env


def test_mock_launch_executes_on_a_private_bus_and_records_the_run(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Any] = []
    monkeypatch.setattr(launch, "record_run", recorded.append)
    result = invoke(site, "teleop")
    assert result.exit_code == 0, result.output
    plan = executed[0]
    assert plan.private_bus and plan.webxr_port not in (8443, None) and plan.viser_port not in (8095, None)
    assert "--config" in plan.argv
    state = recorded[0]
    assert state.real is False and state.private_bus and state.webxr_port == plan.webxr_port
    assert state.bus["ZENOH_SCOUT_ADDR"] == plan.env["ZENOH_SCOUT_ADDR"]
    # --station-ports keeps 8443/8095 (Quest practice), still on a private bus.
    executed.clear()
    assert invoke(site, "teleop", "--station-ports").exit_code == 0
    assert launch.WEBXR_PORT_FLAG not in executed[0].argv and executed[0].private_bus


def test_launch_refuses_while_another_stack_runs(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kiteos.cli import runs

    evidence = [runs.Evidence("process", "PID 97794: python3 -m kite_rollout.run --real ['__main__']")]
    monkeypatch.setattr(launch, "host_stacks", lambda **_: evidence)
    result = invoke(site, "teleop")
    assert result.exit_code == 2 and "kite_rollout" in result.output and executed == []
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and executed == []
    assert invoke(site, "teleop", "--real", "--allow-concurrent").exit_code == 2
    # A MOCK run may share the host explicitly: it is on its own bus and ports.
    result = invoke(site, "teleop", "--allow-concurrent")
    assert result.exit_code == 0 and len(executed) == 1 and executed[0].private_bus


def test_real_launch_refuses_traffic_on_the_target_bus(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kiteos.cli import runs

    probed: list[dict[str, str]] = []

    def busy(env: dict[str, str], **_: Any) -> runs.BusProbe:
        probed.append(env)
        return runs.BusProbe(250, ["dimos/coordinator_joint_state/JointState"])

    monkeypatch.setattr(launch, "bus_probe", busy)
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "coordinator_joint_state" in result.output
    assert executed == [] and probed
    assert invoke(site, "teleop").exit_code == 0  # a mock run never listens on the station bus


def test_launch_refuses_runtime_settings_that_bypass_the_station(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: CONTROLCOORDINATOR__*_CAN_PORT in the environment powered REAL arms in a mock run."""
    monkeypatch.setenv("CONTROLCOORDINATOR__LEFT_CAN_PORT", LEFT_SERIAL)
    result = invoke(site, "teleop")
    assert result.exit_code == 2 and "CONTROLCOORDINATOR__LEFT_CAN_PORT" in result.output and executed == []
    monkeypatch.delenv("CONTROLCOORDINATOR__LEFT_CAN_PORT")
    dotenv = site.parent.parent / "kiteos-runtime" / ".env"
    dotenv.write_text("export CONTROLCOORDINATOR__REQUIRE_FENCE_ON_HARDWARE=false\nOTHER=1\n")
    result = invoke(site, "teleop")
    assert result.exit_code == 2 and "REQUIRE_FENCE_ON_HARDWARE" in result.output
    dotenv.write_text("OTHER=1\n")
    assert invoke(site, "teleop").exit_code == 0


def test_real_launch_refuses_unprotected_arms_and_implausible_speed(
    site: Path, executed: list[launch.LaunchPlan]
) -> None:
    write_fence(site.parent / "fence.yaml", zones=[])  # no allow_no_zones
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "no enabled zone" in result.output
    left_only = {**GLASS, "frame": "left_base", "applies_to": "left"}
    write_fence(site.parent / "fence.yaml", zones=[left_only])
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "right arm" in result.output
    write_fence(site.parent / "fence.yaml", zones=[left_only], allow_no_zones=True)
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0 and "THE RIGHT ARM HAS NO KEEP-OUT ZONE" in result.output
    write_fence(site.parent / "fence.yaml", zones=[GLASS], max_joint_speed_rad_s=10.0)
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 2 and "max_joint_speed_rad_s" in result.output


def test_real_banner_names_the_station_files(site: Path, executed: list[launch.LaunchPlan]) -> None:
    (site.parent / "station.local.yaml").write_text("task: close the box\n")
    result = invoke(site, "teleop", "--real", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0
    assert "station.local.yaml" in result.output.replace("\n", "")


def test_rollout_record_uses_the_recorded_blueprint_and_station_guards(
    site: Path, tmp_path: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[Any] = []
    monkeypatch.setattr(launch, "record_run", recorded.append)
    result = invoke(site, "rollout", "start", "--record", "--device", "cpu")
    assert result.exit_code == 0, result.output
    argv = executed[0].argv
    assert argv[5] == "kiteos.dual-openyam-rollout-recorded"
    recording = argv[argv.index("--recorder.recording") + 1]
    assert Path(recording).name == "mock-rollout-close-box-001"
    assert argv[argv.index("--policy.max-start-jump-rad") + 1] == "0.3"
    assert argv[argv.index("--policy.max-joint-speed-rad-s") + 1] == "3"
    assert recorded[0].recording == recording
    assert invoke(site, "rollout", "start", "--session", "x").exit_code == 2  # --session needs --record


# =========================================================================== review fixes: fence CLI


def test_per_arm_teach_runs_keep_both_zones(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review: `teach --arms left` then `--arms right` with the default name replaced the left
    arm's zone, leaving it with no glass zone and no warning."""
    fence_path = site.parent / "fence.yaml"
    for side in ("left", "right"):
        live = FakeLive([*plane_touch_poses(side), named(HOME, HOME)])
        monkeypatch.setattr(fence_cli, "connect_live", lambda live=live: live)
        result = invoke(site, "fence", "teach", "--arms", side, "--force", input="\n\n\n\n\ndone\ny\ny\n")
        assert result.exit_code == 0, result.output
        if side == "left":
            assert "right arm has NO keep-out zone" in " ".join(result.output.split())
            assert safety.FenceConfig.load(fence_path).allow_no_zones is True
    config = safety.FenceConfig.load(fence_path)
    checker = safety.FenceChecker(config)
    assert checker.zone_names("left") == ["glass_window_left"]
    assert checker.zone_names("right") == ["glass_window_right"]
    assert config.allow_no_zones is False  # both arms covered: the teaching opt-in is cleared


def test_teach_refuses_to_silently_retarget_an_existing_zone(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_fence(site.parent / "fence.yaml", zones=[{**GLASS, "name": "glass_window"}], allow_no_zones=True)
    live = FakeLive([*plane_touch_poses("right"), named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    result = invoke(site, "fence", "teach", "--arms", "right", "--name", "glass_window", input="")
    assert result.exit_code == 2 and "--replace" in result.output


def test_teach_both_arms_in_a_per_arm_frame_needs_applies_to(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = FakeLive([named(HOME, HOME)])
    monkeypatch.setattr(fence_cli, "connect_live", lambda: live)
    result = invoke(site, "fence", "teach", "--arms", "both", "--frame", "left_base", input="")
    assert result.exit_code == 2 and "--applies-to" in result.output


def test_validate_reports_uncovered_arms_open_grippers_and_cameras(site: Path) -> None:
    left_only = {**GLASS, "frame": "left_base", "applies_to": "left"}
    write_fence(site.parent / "fence.yaml", zones=[left_only])
    result = invoke(site, "fence", "validate")
    text = " ".join(result.output.split())
    assert result.exit_code == 1 and "no enabled zone applies to the right arm" in text
    assert "wrist camera" in text
    # Home is clear with the gripper closed but not open (glass at 0.45 m, 3 cm margin).
    close = {**GLASS, "point": [0.45, 0.0, 0.0]}
    write_fence(site.parent / "fence.yaml", zones=[close], margin_m=0.03)
    text = " ".join(invoke(site, "fence", "validate").output.split())
    assert "clear with the gripper closed" in text and "with the gripper open" in text
    write_fence(site.parent / "fence.yaml", zones=[])
    assert invoke(site, "fence", "validate").exit_code == 1  # no zones and no allow_no_zones


# =========================================================================== review fixes: collect/upload


def test_collect_controls_follow_the_running_session() -> None:
    from kiteos.cli.collect import running_session

    session_dir, prefix = running_session("/data/recordings/demo-004", None)
    assert session_dir == Path("/data/recordings/demo-004") and prefix == "demo"
    assert running_session(None, None) == (None, None)
    with pytest.raises(Exception):
        running_session("/data/recordings/demo-004", "close-box")


def test_upload_refuses_mock_sessions_and_foreign_prefixes(site: Path, tmp_path: Path) -> None:
    rec = tmp_path / "recordings"
    for name in ("mock-close-box-001", "demo-001"):
        (rec / name).mkdir(parents=True)
        (rec / name / "recording.mcap").write_bytes(b"x")
    result = invoke(site, "upload", "--session", "mock-close-box", "--dry-run")
    assert result.exit_code == 2 and "MOCK" in result.output
    result = invoke(site, "upload", "--session", "demo", "--dry-run")
    assert result.exit_code == 2 and "--repo-id" in result.output.replace("\n", "")
    assert invoke(site, "upload", "--session", "demo", "--repo-id", "kiteml/demo", "--dry-run").exit_code == 0
    # Station-prefix uploads never pick up mock sessions.
    assert "No recordings found" in invoke(site, "upload", "--dry-run").output


def test_upload_never_deletes_a_raw_recording(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "close-box-001").mkdir(parents=True)
    (data / "close-box-001" / "recording.mcap").write_bytes(b"x" * 10)
    with pytest.raises(RuntimeError, match="raw recording.mcap"):
        upload._remove_dataset_dir(data / "close-box-001")
    assert (data / "close-box-001" / "recording.mcap").exists()


# =========================================================================== review fixes: runs


def test_stack_process_scan_patterns() -> None:
    from kiteos.cli import runs

    ps = "\n".join(
        [
            "  101 uv run --no-sync dimos run kiteos.dual-openyam-teleop --viewer none",
            "  102 /x/.venv/bin/python /x/.venv/bin/dimos run kiteos.dual-openyam-rollout",
            "  103 /x/python3 -m kite_rollout.run --checkpoint c --real",
            "  104 /x/python3 -c from multiprocessing.forkserver import main; main(60, 62, ['__main__'], **{'sys_path': ['/x/dimos_lcm']})",
            "  105 uv run --no-sync dimos imitation rollout",
            "  106 /usr/bin/vim notes-about-dimos",
            "  107 python -m dimos.experimental.isolated_python.bootstrap --declaration kite_rollout.module:X",
        ]
    )
    assert [pid for pid, _ in runs.stack_processes(ps, exclude=[102])] == [101, 103, 104, 107]


def test_stack_process_scan_finds_kiteos_runtime_launches() -> None:
    """KiteOS launches through `kiteos-runtime`; the scan must see those stacks as well as `dimos run`."""
    from kiteos.cli import runs

    ps = "\n".join(
        [
            "  201 uv run --no-sync kiteos-runtime run kiteos.dual-openyam-teleop --viewer none",
            "  202 /x/kiteos-runtime/.venv/bin/python /x/kiteos-runtime/.venv/bin/kiteos-runtime run kiteos.dual-openyam-collection",
            "  203 uv run --no-sync kiteos-runtime imitation rollout",
            "  204 /x/python3 -c from multiprocessing.forkserver import main; main(60, 62, ['__main__'], **{'sys_path': ['/x/kiteos-runtime/.venv/lib']})",
            "  205 /usr/bin/less kiteos-runtime-notes.txt",
        ]
    )
    assert [pid for pid, _ in runs.stack_processes(ps)] == [201, 202, 204]


def test_run_state_stays_alive_while_kiteos_runtime_runs(tmp_path: Path) -> None:
    from kiteos.cli import runs

    state = runs.RunState(pid=21, real=False, blueprint="kiteos.dual-openyam-teleop", station="s", private_bus=True)
    runs.write_run_state(state, tmp_path)
    commands = {21: "uv run --no-sync kiteos-runtime run kiteos.dual-openyam-teleop"}
    found = runs.alive_run_states(tmp_path, alive=lambda pid: pid == 21, command=commands.get)
    assert [s.pid for s in found] == [21]


def test_run_state_files_round_trip_and_stale_ones_are_removed(tmp_path: Path) -> None:
    from kiteos.cli import runs

    alive = runs.RunState(pid=11, real=False, blueprint="kiteos.dual-openyam-teleop", station="s", private_bus=True)
    gone = runs.RunState(pid=12, real=True, blueprint="kiteos.dual-openyam-collection", station="s")
    reused = runs.RunState(pid=13, real=True, blueprint="kiteos.dual-openyam-collection", station="s")
    for state in (alive, gone, reused):
        runs.write_run_state(state, tmp_path)
    commands = {11: "uv run --no-sync dimos run kiteos.dual-openyam-teleop", 13: "/usr/bin/python other.py"}
    found = runs.alive_run_states(tmp_path, alive=lambda pid: pid in (11, 13), command=commands.get)
    assert [s.pid for s in found] == [11] and found[0].describe().startswith("PID 11 MOCK")
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["11.json"]


def test_registry_argv_proves_mock_or_real() -> None:
    from kiteos.cli import runs

    assert runs.registry_run_is_mock({"original_argv": ["run", "--controlcoordinator.left-can-port", "A"]}) is False
    assert runs.registry_run_is_mock({"original_argv": ["run", "--controlcoordinator.expect-mock", "true"]}) is True
    assert runs.registry_run_is_mock({"original_argv": ["run", "kiteos.dual-openyam-teleop"]}) is None


def test_private_bus_is_never_the_default_and_the_probe_skips_it_in_tests() -> None:
    from kiteos.cli import runs

    taken = [runs.random_private_bus()]
    for _ in range(50):
        bus = runs.random_private_bus(taken)
        assert not runs.is_default_bus(bus) and bus != taken[0]
    assert runs.is_default_bus({}) and runs.is_default_bus({"ZENOH_SCOUT_ADDR": "224.0.0.224:7446"})
    probe = runs.probe_bus({"PATH": os.environ.get("PATH", "")})
    assert probe.samples == 0 and "refusing" in (probe.error or "")
    assert runs.free_tcp_port() not in runs.RESERVED_PORTS


def test_other_stacks_collects_every_source() -> None:
    from kiteos.cli import runs

    evidence = runs.other_stacks(
        ports=(8443, 8095),
        registry=lambda: [{"pid": 1, "blueprint": "b", "run_id": "r"}],
        kiteos_runs=lambda: [],
        processes=lambda: [(2, "python -m kite_rollout.run --real")],
        listening=lambda port: port == 8443,
    )
    assert [e.source for e in evidence] == ["runtime registry", "process", "port"]


def test_quest_proxy_follows_a_mock_run_on_its_own_port(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiteos.cli import quest, runs

    state = runs.RunState(pid=5, real=False, blueprint="kiteos.dual-openyam-teleop", station="s", webxr_port=51234)
    monkeypatch.setattr(runs, "alive_run_states", lambda *a, **k: [state])
    assert quest.running_upstream(load(site)) == "https://127.0.0.1:51234"
    monkeypatch.setattr(runs, "alive_run_states", lambda *a, **k: [])
    assert quest.running_upstream(load(site)) == "https://127.0.0.1:8443"


def test_can_setup_refuses_while_a_stack_runs(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiteos.cli import can

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(can, "running_stacks", lambda: ["process: PID 9: dimos run x"])
    result = invoke(site, "can", "setup")
    assert result.exit_code == 2 and "CAN links DOWN" in " ".join(result.output.split())
    assert can.interface_ready("3: can_left: <NOARP,UP,LOWER_UP,ECHO> mtu 16\n can state ERROR-ACTIVE\n bitrate 1000000 sample-point 0.8")
    assert not can.interface_ready("3: can_left: <NOARP,ECHO> mtu 16\n bitrate 1000000")
