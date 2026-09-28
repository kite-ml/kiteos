# Copyright 2026 Kite ML.
"""End-to-end proof of the KiteOS fence through a real ``dimos run`` with MOCK arms.

Launches ``uv run --no-sync dimos run kiteos.dual-openyam-teleop --config <empty>
--controlcoordinator.fence-path tests/fixtures/fence_e2e.yaml
--controlcoordinator.expect-mock true ...`` (the flags the KiteOS CLI pins for a mock run) in the
foreground (never ``--daemon``, never CAN ports, no cameras in this blueprint) on a private bus,
connects with ``Dimos.connect()`` and drives the ``ControlCoordinator`` RPCs and streams:

0. Host plumbing: the WebXR server listens on 127.0.0.1 only; KiteOS's passive bus probe sees the
   stack's traffic on its (private) bus; the process scan finds the stack.
A. ``execute_trajectory`` home -> reach (fingertip beyond the fixture's glass plane) is rejected
   by the coordinator's fence precheck and the arm does not move.
B. The same trajectory through ``task_invoke("joint_trajectory", "execute", ...)`` (a declared
   dimOS task command) and ``task_invoke(..., "on_joint_command", ...)`` is rejected by the same
   precheck; the arm does not move.
B2. A zone-less fence is hot-reloaded, the reach trajectory is accepted, and the glass fence is
   reloaded while it executes: the adapter holds the arm short of the plane and the supervisor
   cancels the trajectory exactly once.
C. ``execute_trajectory`` back home (moving away from the plane) passes the precheck.
D. Teleop-style streaming of small joint steps on the ``joint_command`` stream (no precheck)
   towards the reach pose: the fence holds the arm again, and the stream's re-executions of the
   trajectory task are cancelled at most once per hold episode (not at the stream rate).
E. ``fence_reload`` hot-swaps in a plane 4 cm closer, so the held arm is now inside a zone. A
   trajectory that goes deeper is rejected by the precheck; the straight trajectory home is
   accepted as an escape and executed, with every zone/sphere pair within the margin
   non-decreasing (and every clear pair staying clear) between measured samples, ending clear.
   The fixture fence is then reloaded.
F. ``kiteos fence check`` (the CLI, as a subprocess) reads the running fence over the same RPCs:
   clear at home (exit 0), violation for the reach pose (exit 1).

Throughout, every ``coordinator_joint_state`` message and RPC poll of the measured mock positions
is re-checked by an independent ``FenceChecker``: no sample may come closer to the zone than the
margin (numerical tolerance only), except while backing out of the hot-reloaded closer zone in E,
and ``fence_status`` must report the hold events. Finally the run is stopped with SIGTERM and must
exit cleanly.

Safety interlocks (this machine may also run a real rig on the default bus): the suite conftest
moves this process and the child onto a private bus; this test refuses to run otherwise, moves the
WebXR and viser servers to free ports, never passes CAN flags (and pins expect-mock, so the
coordinator would refuse a real adapter), and verifies over RPC that it is talking to the KiteOS
coordinator with the fixture fence and a ``MockWholeBodyAdapter`` before it sends a single motion
command.

Set ``KITEOS_E2E_ARTIFACTS=<dir>`` to keep the dimOS log and a JSON summary of the evidence.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.mock_e2e, pytest.mark.dimos]
dimos_pkg = pytest.importorskip("dimos", reason="needs the dimOS virtualenv")

from dimos.core.global_config import global_config  # noqa: E402
from dimos.core.transport_factory import make_transport  # noqa: E402
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory  # noqa: E402
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint  # noqa: E402

from kiteos.cli import runs  # noqa: E402
from kiteos.safety.fence import FenceChecker, FenceConfig, escape_path_ok  # noqa: E402
from kiteos.safety.kinematics import ARM_JOINTS, DUAL_OPENYAM_JOINTS  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURE = (REPO / "tests" / "fixtures" / "fence_e2e.yaml").resolve()
# Same glass 4 cm closer: hot-reloaded while the arm is held, which puts the arm inside it.
FIXTURE_CLOSER = (REPO / "tests" / "fixtures" / "fence_e2e_closer.yaml").resolve()
# No zones: lets a trajectory toward the glass pass the precheck (B2).
FIXTURE_OPEN = (REPO / "tests" / "fixtures" / "fence_e2e_open.yaml").resolve()
BLUEPRINT = "kiteos.dual-openyam-teleop"
ZONE = "e2e_glass"
ZONE_CLOSER = "e2e_glass_closer"

HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
REACH = [0.0, 2.0, 2.0, 0.0, 0.0, 0.0]  # fingertip x ~= 0.631 m, beyond the plane at 0.50 m
LEFT = list(ARM_JOINTS["left"])
RIGHT = list(ARM_JOINTS["right"])

STARTUP_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 60.0
# Measured positions may only be closer than the margin by float noise.
NUMERICAL_TOLERANCE_M = 1e-6
KEY_LOG_MARKERS = (
    "KiteOS fence armed",
    "ControlCoordinator started",
    "KiteOS fence hold",
    "cancelled the executing trajectory",
    "Received signal",
    "Module stopped. module=ControlCoordinator",
    "Traceback",
)


# --------------------------------------------------------------------------- helpers


def _dimos_root() -> Path:
    """The dimOS checkout the venv's ``dimos`` package comes from (editable install)."""
    root = Path(dimos_pkg.__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        pytest.skip(f"dimOS is not an editable checkout ({root}); cannot `uv run` from it")
    return root


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _require_private_bus() -> dict[str, str]:
    """The bus settings shared with the child; refuse to run on anything but a private bus."""
    if os.environ.get("KITEOS_TEST_PRIVATE_BUS") != "1":
        pytest.skip("mock e2e only runs on the suite's private bus (KITEOS_TEST_EXTERNAL_BUS set)")
    lcm_url = os.environ.get("LCM_DEFAULT_URL", "")
    scout = os.environ.get("ZENOH_SCOUT_ADDR", "")
    if not lcm_url or ":7667" in lcm_url or not scout or scout.endswith(":7446"):
        pytest.fail(f"refusing to launch dimOS on a default bus: {lcm_url=} {scout=}")
    # dimOS read these at import; they must match what the child process will use.
    if global_config.zenoh_scout_addr != scout:
        pytest.fail(
            f"dimOS in this process scouts {global_config.zenoh_scout_addr!r}, not the private "
            f"{scout!r}; refusing to connect"
        )
    from dimos.protocol.service import lcmservice

    if lcmservice._DEFAULT_LCM_URL != lcm_url:
        pytest.fail(f"dimOS LCM URL is {lcmservice._DEFAULT_LCM_URL!r}, not {lcm_url!r}")
    return {
        "LCM_DEFAULT_URL": lcm_url,
        "ZENOH_SCOUT_ADDR": scout,
        "DIMOS_TRANSPORT": str(global_config.transport),
    }


def _left(named: dict[str, float]) -> np.ndarray:
    return np.array([float(named[n]) for n in LEFT])


def _q14(named: dict[str, float]) -> np.ndarray:
    return np.array([float(named[n]) for n in DUAL_OPENYAM_JOINTS])


def _left_trajectory(q0: np.ndarray | list[float], q1: list[float], duration: float) -> Any:
    """A 2-point left-arm JointTrajectory, the shape dimOS's rollout/planner send."""
    return JointTrajectory(
        joint_names=list(LEFT),
        points=[
            TrajectoryPoint(time_from_start=0.0, positions=[float(v) for v in q0]),
            TrajectoryPoint(time_from_start=duration, positions=[float(v) for v in q1]),
        ],
    )


def _short_of_reach(q_left: np.ndarray) -> bool:
    """Joints 2 and 3 (the ones the reach moves) stopped well before the reach pose."""
    return bool(q_left[1] < REACH[1] - 0.3 and q_left[2] < REACH[2] - 0.3)


def _wait(predicate: Callable[[], bool], timeout: float, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


@dataclass
class Referee:
    """Independent fence check of every measured sample."""

    checker: FenceChecker
    samples: list[tuple[str, float, np.ndarray]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, source: str, named: dict[str, float]) -> None:
        q = _q14(named)
        with self.lock:
            self.samples.append((source, time.monotonic(), q))

    def verdicts(self) -> list[tuple[str, float, float, bool]]:
        with self.lock:
            samples = list(self.samples)
        out = []
        for source, t, q in samples:
            report = self.checker.check(q, gripper_openings={"left": q[12], "right": q[13]})
            worst = min(a.min_clearance_m for a in report.arms.values())
            out.append((source, t, worst, report.ok))
        return out


@dataclass
class MockRun:
    proc: subprocess.Popen[bytes]
    log_path: Path
    argv: list[str]
    env: dict[str, str]  # private bus + PYTHONPATH, for clients such as the kiteos CLI
    webxr_port: int

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")

    def key_lines(self) -> list[str]:
        return [
            line[:240]
            for line in self.log().splitlines()
            if any(marker in line for marker in KEY_LOG_MARKERS)
        ]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MockRun]:
    bus = _require_private_bus()
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        pytest.skip("uv is not installed")
    root = _dimos_root()
    webxr_port = _free_tcp_port()
    empty_config = tmp_path_factory.mktemp("dimos_config") / "config.json"
    empty_config.write_text("{}\n")
    argv = [
        uv,
        "run",
        "--no-sync",
        "dimos",
        "run",
        BLUEPRINT,
        # What `kiteos teleop` (mock) pins: no inherited dimOS config, fence required, mock only.
        "--config",
        str(empty_config),
        "--controlcoordinator.fence-path",
        str(FIXTURE),
        "--controlcoordinator.require-fence-on-hardware",
        "true",
        "--controlcoordinator.expect-mock",
        "true",
        # Never the live rig's WebXR (8443) or viser (8095) ports.
        "--armteleopmodule.server-port",
        str(webxr_port),
        "--manipulationmodule.visualization.port",
        str(_free_tcp_port()),
        "--viewer",
        "none",
    ]
    assert not any("can-port" in a or "can_port" in a for a in argv), "mock e2e must not pass CAN"

    env = dict(os.environ)
    env.update(bus)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    env.pop("DIMOS_RUN_ID", None)
    log_path = tmp_path_factory.mktemp("mock_e2e") / "dimos_run.log"
    with log_path.open("wb") as log_file:
        # Foreground run (macOS --daemon crashes); own session so pytest's Ctrl+C is not
        # delivered to it directly: teardown below sends SIGTERM like `kiteos stop`.
        proc = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    run = MockRun(proc=proc, log_path=log_path, argv=argv, env=env, webxr_port=webxr_port)
    try:
        yield run
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Mock arms only: do not leave a stray stack behind a failed test.
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        _save_artifacts(run, None)


def _save_artifacts(run: MockRun, summary: dict[str, Any] | None) -> None:
    target = os.environ.get("KITEOS_E2E_ARTIFACTS")
    if not target:
        return
    out = Path(target).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run.log_path, out / "dimos_run.log")
    if summary is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def _connect(run: MockRun) -> Any:
    from dimos.porcelain.dimos import Dimos

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if run.proc.poll() is not None:
            pytest.fail(
                f"dimos run exited early with {run.proc.returncode}:\n"
                + "\n".join(run.log().splitlines()[-40:])
            )
        client = None
        try:
            client = Dimos.connect(timeout=3.0)
            coordinator = client.get_module("ControlCoordinator")
            if coordinator.fence_status().get("running"):
                return client
        except Exception as exc:  # not up yet
            last_error = exc
        if client is not None:
            client.stop()
        time.sleep(1.0)
    pytest.fail(
        f"KiteOS mock stack did not come up within {STARTUP_TIMEOUT_S:.0f} s: {last_error!r}"
    )


# --------------------------------------------------------------------------- the test


def test_fence_holds_mock_arms_end_to_end(mock_run: MockRun) -> None:
    config = FenceConfig.load(FIXTURE)
    margin = float(config.margin_m)
    referee = Referee(FenceChecker(config))
    summary: dict[str, Any] = {"argv": mock_run.argv, "fence": str(FIXTURE), "margin_m": margin}

    t_launch = time.monotonic()
    client = _connect(mock_run)
    summary["startup_s"] = round(time.monotonic() - t_launch, 1)
    unsubscribe: Callable[[], None] | None = None
    try:
        cc = client.get_module("ControlCoordinator")

        # ---- interlocks: nothing moves unless this is the mock KiteOS stack with our fence
        status = cc.fence_status()
        hardware = status["hardware"]
        assert status["product"] == "KiteOS"
        assert status["fence_path"] == str(FIXTURE)
        assert list(hardware) == ["dual_openyam"], hardware.keys()
        hw = hardware["dual_openyam"]
        assert hw["real_hardware"] is False and hw["adapter"] == "MockWholeBodyAdapter", hw
        assert hw["mode"] == "enforce" and [z["name"] for z in hw["zones"]] == [ZONE]
        assert hw["holds"] == 0 and status["trajectory_cancels"] == 0
        start = cc.get_joint_positions()
        np.testing.assert_allclose(_left(start), HOME, atol=1e-9)
        summary["fence_status_initial"] = {
            k: status[k] for k in ("product", "running", "fence_path", "trajectory_cancels")
        }

        # ---- 0: host plumbing. The WebXR server listens on loopback only (KiteArmTeleopModule).
        lsof = shutil.which("lsof")
        if lsof:
            listing = subprocess.run(
                [lsof, "-nP", f"-iTCP:{mock_run.webxr_port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            assert f"127.0.0.1:{mock_run.webxr_port}" in listing, listing
            assert f"*:{mock_run.webxr_port}" not in listing, listing
            summary["webxr_listen"] = [ln for ln in listing.splitlines() if "LISTEN" in ln]
        # KiteOS's passive probe (what `--real` runs before starting) sees this stack's traffic
        # on its bus, and the process scan finds it.
        probe = runs.probe_bus(mock_run.env, cwd=_dimos_root())
        assert probe.busy, probe
        assert any("coordinator_joint_state" in t for t in probe.topics), probe.topics
        assert any(pid == mock_run.proc.pid for pid, _ in runs.stack_processes()), "process scan"
        summary["bus_probe"] = {"samples": probe.samples, "topics": probe.topics[:5]}

        # Every published coordinator_joint_state tick, checked independently.
        state_stream = make_transport("/coordinator_joint_state", JointState)
        unsubscribe = state_stream.subscribe(
            lambda msg: referee.add("stream", dict(zip(msg.name, msg.position, strict=True)))
        )
        assert _wait(lambda: len(referee.samples) >= 20, 5.0), "no coordinator_joint_state"

        def poll(seconds: float) -> dict[str, float]:
            deadline = time.monotonic() + seconds
            named = cc.get_joint_positions()
            referee.add("rpc", named)
            while time.monotonic() < deadline:
                time.sleep(0.05)
                named = cc.get_joint_positions()
                referee.add("rpc", named)
            return named

        # The precheck geometry itself, over the fence_check RPC.
        verdict = cc.fence_check(
            {**dict(zip(LEFT, REACH, strict=True)), **dict(zip(RIGHT, HOME, strict=True))}
        )
        assert verdict["ok"] is False and verdict["arms"]["left"]["zone"] == ZONE, verdict
        assert verdict["arms"]["right"]["ok"] is True

        # ---- A: execute_trajectory across the plane is rejected by the precheck
        result = cc.execute_trajectory(_left_trajectory(_left(start), REACH, 3.0))
        assert result.status.name == "INVALID_TRAJECTORY", result
        assert "KiteOS fence rejected" in result.message and ZONE in result.message
        after_a = poll(1.0)
        np.testing.assert_allclose(_left(after_a), HOME, atol=1e-9)
        assert cc.fence_status()["hardware"]["dual_openyam"]["holds"] == 0
        summary["A_precheck"] = {"status": result.status.name, "message": result.message}

        # ---- B: the trajectory task's declared `execute` command gets the same precheck
        named = cc.get_joint_positions()
        result_b = cc.task_invoke(
            "joint_trajectory",
            "execute",
            {"trajectory": _left_trajectory(_left(named), REACH, 3.0), "current_positions": named},
        )
        assert result_b.status.name == "INVALID_TRAJECTORY", result_b
        assert "KiteOS fence rejected" in result_b.message and ZONE in result_b.message
        streamed = cc.task_invoke(
            "joint_trajectory",
            "on_joint_command",
            {"msg": JointState(name=list(LEFT), position=list(REACH)), "t_now": None},
        )
        assert streamed is False
        after_b = poll(1.0)
        np.testing.assert_allclose(_left(after_b), HOME, atol=1e-9)
        assert "joint_trajectory" not in cc.get_active_tasks()
        summary["B_task_invoke_precheck"] = {"status": result_b.status.name, "message": result_b.message}

        # ---- B2: a trajectory accepted under a zone-less fence; the glass returns while it runs
        opened = cc.fence_reload(str(FIXTURE_OPEN))
        assert opened["ok"] and opened["zones"] == [], opened
        cancels_before = cc.fence_status()["trajectory_cancels"]
        named = cc.get_joint_positions()
        result_b2 = cc.execute_trajectory(_left_trajectory(_left(named), REACH, 3.0))
        assert result_b2.status.name == "ACCEPTED", result_b2
        restored = cc.fence_reload(str(FIXTURE))
        assert restored["zones"] == [ZONE] and restored["current"]["dual_openyam"]["ok"], restored

        def held_and_cancelled() -> bool:
            referee.add("rpc", cc.get_joint_positions())
            st = cc.fence_status()
            return (
                st["hardware"]["dual_openyam"]["holds"] >= 1
                and st["trajectory_cancels"] > cancels_before
                and "joint_trajectory" not in cc.get_active_tasks()
            )

        assert _wait(held_and_cancelled, 10.0, poll=0.05), cc.fence_status()
        after_b2 = poll(1.5)
        status_b2 = cc.fence_status()
        hw_b2 = status_b2["hardware"]["dual_openyam"]
        left_b = _left(after_b2)
        assert left_b[1] > HOME[1] + 0.1, "the arm never moved towards the plane"
        assert _short_of_reach(left_b), f"the arm got (nearly) to the reach pose: {left_b}"
        assert hw_b2["last_hold"]["zone"] == ZONE and hw_b2["last_hold"]["side"] == "left"
        assert hw_b2["last_hold"]["enforced"] is True
        assert hw_b2["arms"]["left"]["holding"] is True
        assert status_b2["trajectory_cancels"] == cancels_before + 1, status_b2["trajectory_cancels"]
        summary["B2_hold_and_cancel"] = {
            "held_left_q": left_b.round(4).tolist(),
            "fingertip_m": referee.checker.fingertip("left", left_b).round(4).tolist(),
            "holds": hw_b2["holds"],
            "trajectory_cancels": status_b2["trajectory_cancels"] - cancels_before,
            "last_hold": hw_b2["last_hold"],
        }

        # ---- C: moving away from the plane passes the precheck
        result_c = cc.execute_trajectory(_left_trajectory(left_b, HOME, 2.0))
        assert result_c.status.name == "ACCEPTED", result_c

        def at_home() -> bool:
            named = cc.get_joint_positions()
            referee.add("rpc", named)
            return bool(np.allclose(_left(named), HOME, atol=1e-3))

        assert _wait(at_home, 10.0), cc.get_joint_positions()
        status_c = cc.fence_status()
        holds_before_d = status_c["hardware"]["dual_openyam"]["holds"]
        cancels_before_d = status_c["trajectory_cancels"]
        summary["C_return_home"] = {"status": result_c.status.name}

        # ---- D: teleop-style small steps on the joint_command stream (no precheck)
        commands = make_transport("/joint_command", JointState)
        q = _left(cc.get_joint_positions())
        goal = np.array(REACH)
        sent = 0
        deadline = time.monotonic() + 5.0
        next_poll = 0.0
        while time.monotonic() < deadline:
            q = q + np.clip(goal - q, -0.01, 0.01)  # 0.01 rad per message, 50 Hz
            commands.broadcast(None, JointState(name=list(LEFT), position=[float(v) for v in q]))
            sent += 1
            if time.monotonic() >= next_poll:
                referee.add("rpc", cc.get_joint_positions())
                next_poll = time.monotonic() + 0.1
            time.sleep(0.02)
        assert np.allclose(q, goal), "the streamed command never reached the reach pose"
        after_d = poll(1.0)
        status_d = cc.fence_status()
        hw_d = status_d["hardware"]["dual_openyam"]
        left_d = _left(after_d)
        assert hw_d["holds"] > holds_before_d, "streaming across the plane produced no hold"
        assert left_d[1] > HOME[1] + 0.1 and _short_of_reach(left_d), left_d
        assert hw_d["last_hold"]["zone"] == ZONE and hw_d["last_hold"]["side"] == "left"
        np.testing.assert_allclose(_q14(after_d)[6:12], HOME, atol=1e-9)  # right arm untouched
        # The stream re-executed the trajectory task ~50x/s during the hold; the supervisor must
        # cancel once per hold episode, not at the stream rate (it used to be ~40x/s).
        cancels_d = status_d["trajectory_cancels"] - cancels_before_d
        assert 1 <= cancels_d <= 5, cancels_d
        assert hw_d["holds"] - holds_before_d > 20 * cancels_d
        summary["D_joint_command_stream"] = {
            "messages_sent": sent,
            "held_left_q": left_d.round(4).tolist(),
            "fingertip_m": referee.checker.fingertip("left", left_d).round(4).tolist(),
            "holds_before": holds_before_d,
            "holds_after": hw_d["holds"],
            "trajectory_cancels_during_stream": cancels_d,
            "last_hold": hw_d["last_hold"],
        }
        summary["recent_events_tail"] = status_d["recent_events"][-3:]

        # ---- E: hot reload a closer plane: the held arm is now INSIDE the zone. The precheck
        # accepts only a trajectory that backs straight out, and the adapter lets it through as
        # escapes; no zone/sphere pair within the margin may lose clearance meanwhile.
        reloaded = cc.fence_reload(str(FIXTURE_CLOSER))
        assert reloaded["ok"] and reloaded["zones"] == [ZONE_CLOSER], reloaded
        now = reloaded["current"]["dual_openyam"]
        assert now["ok"] is False and now["arms"]["left"]["zone"] == ZONE_CLOSER, now
        assert now["arms"]["left"]["min_clearance_m"] < 0.0, "the arm should be inside the zone"
        inside_q = _left(cc.get_joint_positions())
        deeper = cc.execute_trajectory(_left_trajectory(inside_q, REACH, 2.0))
        assert deeper.status.name == "INVALID_TRAJECTORY", deeper
        assert "does not move straight out" in deeper.message, deeper.message
        escapes_before = cc.fence_status()["hardware"]["dual_openyam"]["events"]["escape"]
        e_start = len(referee.samples)
        result_e = cc.execute_trajectory(_left_trajectory(inside_q, HOME, 2.0))
        assert result_e.status.name == "ACCEPTED", result_e

        def back_home() -> bool:
            named = cc.get_joint_positions()
            referee.add("rpc", named)
            return bool(np.allclose(_left(named), HOME, atol=1e-3))

        assert _wait(back_home, 15.0, poll=0.02), cc.get_joint_positions()
        after_e = poll(0.5)
        e_end = len(referee.samples)
        status_e = cc.fence_status()["hardware"]["dual_openyam"]
        np.testing.assert_allclose(_left(after_e), HOME, atol=1e-3)
        assert status_e["events"]["escape"] > escapes_before, "no escape was recorded"

        closer = FenceChecker(FenceConfig.load(FIXTURE_CLOSER))
        with referee.lock:
            phase_e = [q14 for source, _t, q14 in referee.samples[e_start:e_end] if source == "stream"]
        distinct = [
            phase_e[0],
            *(b for a, b in zip(phase_e, phase_e[1:], strict=False) if np.any(a != b)),
        ]
        pairs = np.vstack(
            [closer.pair_clearances("left", q14[None, :6], q14[12]) for q14 in distinct]
        )
        escape_path = pairs.min(axis=1)
        assert escape_path[0] < 0.0, escape_path[:3]
        assert escape_path[-1] > closer.margin_m
        # The measured path (sampled per tick) is an escape path: pairs within the margin never
        # lose clearance (beyond float noise), clear pairs stay clear.
        assert escape_path_ok(pairs, closer.margin_m, tol=1e-9), "a pair got closer while inside"
        inside = escape_path < closer.margin_m
        summary["E_hot_reload_escape"] = {
            "reload_current_left": now["arms"]["left"],
            "deeper_trajectory": {"status": deeper.status.name, "message": deeper.message},
            "escape_trajectory": {"status": result_e.status.name},
            "escape_events": status_e["events"]["escape"] - escapes_before,
            "clearance_start_m": float(escape_path[0]),
            "clearance_end_m": float(escape_path[-1]),
            "ticks_inside_margin": int(inside.sum()),
            "samples": len(distinct),
        }
        restored = cc.fence_reload(str(FIXTURE))
        assert restored["zones"] == [ZONE] and restored["current"]["dual_openyam"]["ok"] is True

        # ---- F: the kiteos CLI reads the running fence over the same RPCs
        kiteos = [sys.executable, "-m", "kiteos.cli", "--no-banner", "fence", "check"]
        cli_home = subprocess.run(
            [*kiteos, "--fence", str(FIXTURE)],
            env=mock_run.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert cli_home.returncode == 0, cli_home.stdout + cli_home.stderr
        assert "running stack: clear" in cli_home.stdout
        reach_12 = ",".join(str(v) for v in [*REACH, *HOME])
        cli_reach = subprocess.run(
            [*kiteos, "--fence", str(FIXTURE), "--positions", reach_12],
            env=mock_run.env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert cli_reach.returncode == 1, cli_reach.stdout + cli_reach.stderr
        assert "running stack: VIOLATION" in cli_reach.stdout
        summary["F_cli_fence_check"] = {
            "home": [
                ln for ln in cli_home.stdout.splitlines() if "running stack" in ln or " cm " in ln
            ],
            "reach": [
                ln for ln in cli_reach.stdout.splitlines() if "running stack" in ln or " cm " in ln
            ],
        }
    finally:
        if unsubscribe is not None:
            unsubscribe()
        client.stop()

    # ---- the independent verdict on every measured sample
    verdicts = referee.verdicts()
    stream_n = sum(1 for v in verdicts if v[0] == "stream")
    rpc_n = sum(1 for v in verdicts if v[0] == "rpc")
    worst = min(v[2] for v in verdicts)
    violations = [v for v in verdicts if not v[3] and v[2] < margin - NUMERICAL_TOLERANCE_M]
    summary["measured"] = {
        "stream_samples": stream_n,
        "rpc_samples": rpc_n,
        "min_clearance_m": worst,
        "violations": len(violations),
    }
    assert stream_n >= 300 and rpc_n >= 50, summary["measured"]
    assert not violations, f"measured mock positions violated the fence: {violations[:5]}"
    assert worst >= margin - NUMERICAL_TOLERANCE_M, worst
    # The fence (not something else) stopped the arm: it got right up to the margin.
    assert worst < margin + 0.005, f"arm never approached the fence (min clearance {worst:.4f})"

    # ---- clean shutdown with SIGTERM (what `kiteos stop` sends)
    mock_run.proc.send_signal(signal.SIGTERM)
    returncode = mock_run.proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
    log = mock_run.log()
    summary["returncode"] = returncode
    summary["key_log_lines"] = mock_run.key_lines()
    _save_artifacts(mock_run, summary)
    print(json.dumps(summary, indent=2, default=str))
    assert returncode == 0, "\n".join(log.splitlines()[-40:])
    assert "KiteOS fence armed on dual_openyam (mock_whole_body)" in log
    assert f"zones=['{ZONE}']" in log
    assert "KiteOS fence hold: left arm" in log
    assert "Module stopped. module=ControlCoordinator" in log
    # The only traceback allowed is CPython's benign resource-tracker teardown noise at exit.
    lines = log.splitlines()
    unexpected = [
        "\n".join(lines[max(0, i - 2) : i + 6])
        for i, line in enumerate(lines)
        if "Traceback (most recent call last)" in line
        and not (i > 0 and "ResourceTracker.__del__" in lines[i - 1])
    ]
    assert not unexpected, "unexpected traceback in the dimOS log:\n" + "\n---\n".join(unexpected)
