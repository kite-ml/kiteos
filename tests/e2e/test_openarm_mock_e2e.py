# Copyright 2026 Kite ML.
"""End-to-end proof of the KiteOS fence on the OpenArm through a real ``kiteos-runtime run``.

Launches ``kiteos.openarm-teleop`` with MOCK arms (the flags the KiteOS CLI pins for a mock
run: empty runtime config, the fence for both the coordinator and the viewer, fence required,
expect-mock) in the foreground on the suite's private bus, connects with ``Dimos.connect()``
and checks:

0. It is the KiteOS OpenArm coordinator on ``MockWholeBodyAdapter`` with the test fence; the
   WebXR and viser servers listen on 127.0.0.1 only.
A. ``execute_trajectory`` of both arms to a ready pose (clear of the glass) is accepted and runs.
B. A trajectory that takes the left hand through the glass is rejected by the precheck.
C. Teleop-style streaming of small joint steps toward the glass (``joint_command``, no
   precheck) is held by the adapter at the margin; the right arm does not move.
D. The straight trajectory back to zero (away from the glass) is accepted and runs.

Then the stack is stopped with SIGTERM and must exit. Safety interlocks as in
test_mock_fence_e2e.py: private bus only, never CAN flags, expect-mock pinned.
"""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import signal
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.mock_e2e, pytest.mark.runtime]
runtime_pkg = pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.core.global_config import global_config  # noqa: E402
from dimos.core.transport_factory import make_transport  # noqa: E402
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped  # noqa: E402
from dimos.msgs.geometry_msgs.Quaternion import Quaternion  # noqa: E402
from dimos.msgs.geometry_msgs.Vector3 import Vector3  # noqa: E402
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402
from dimos.msgs.sensor_msgs.Joy import Joy  # noqa: E402
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory  # noqa: E402
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint  # noqa: E402
from dimos.teleop.utils.teleop_transforms import WEBXR_TO_ROBOT_FRAME  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from kiteos.robots import OPENARM  # noqa: E402
from kiteos.safety.fence import FenceChecker, FenceConfig  # noqa: E402
from kiteos.safety.kinematics import DualArmKinematics  # noqa: E402
from kiteos.safety.openarm_coordinator import OPENARM_TELEOP_TASK_NAME  # noqa: E402
from kiteos.teleop.carry import tilt_between  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BLUEPRINT = "kiteos.openarm-teleop"
ZONE = "e2e_glass"
MARGIN_M = 0.05
LEFT, RIGHT = list(OPENARM.arm_joints["left"]), list(OPENARM.arm_joints["right"])
ARMS = LEFT + RIGHT
READY = [-0.5, 0, 0, 1.2, 0, 0, 0] + [0.5, 0, 0, 1.2, 0, 0, 0]
THROUGH_LEFT = [-1.2, 0, 0, 0.3, 0, 0, 0]
REACH_LEFT = [-1.57, 0, 0, 0, 0, 0, 0]
STARTUP_TIMEOUT_S = 150.0
CARRY_SPEED_M_S = 0.3  # a non-default station value: it must reach the teleop task
CARRY_MOVE = np.array([0.10, 0.0, 0.05])
SHUTDOWN_TIMEOUT_S = 60.0


@dataclass
class Run:
    proc: subprocess.Popen[bytes]
    log_path: Path
    webxr_port: int
    viser_port: int
    fence: Path

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _private_bus() -> dict[str, str]:
    if os.environ.get("KITEOS_TEST_PRIVATE_BUS") != "1":
        pytest.skip("mock e2e only runs on the suite's private bus")
    lcm_url = os.environ.get("LCM_DEFAULT_URL", "")
    scout = os.environ.get("ZENOH_SCOUT_ADDR", "")
    if not lcm_url or ":7667" in lcm_url or not scout or scout.endswith(":7446"):
        pytest.fail(f"refusing to launch the runtime on a default bus: {lcm_url=} {scout=}")
    if global_config.zenoh_scout_addr != scout:
        pytest.fail("the runtime in this process is not on the private bus; refusing to connect")
    return {"LCM_DEFAULT_URL": lcm_url, "ZENOH_SCOUT_ADDR": scout, "DIMOS_TRANSPORT": str(global_config.transport)}


def _openarm_cached() -> bool:
    cache = Path.home() / ".cache" / "dimos" / "robot_assets" / "sources"
    return any(cache.glob("*/openarm_description"))


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Run]:
    bus = _private_bus()
    if not _openarm_cached():
        pytest.skip("the OpenArm description is not in the runtime's asset cache (the test never downloads)")
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        pytest.skip("uv is not installed")
    root = Path(runtime_pkg.__file__).resolve().parents[1]
    tmp = tmp_path_factory.mktemp("openarm_e2e")
    fence = tmp / "fence.yaml"
    FenceConfig.model_validate(
        {
            "robot": "openarm",
            "margin_m": MARGIN_M,
            "zones": [{"kind": "half_space", "name": ZONE, "point": [0.60, 0, 0], "normal": [1, 0, 0]}],
        }
    ).dump(fence)
    empty = tmp / "config.json"
    empty.write_text("{}\n")
    webxr, viser = _free_port(), _free_port()
    argv = [
        uv, "run", "--no-sync", "kiteos-runtime", "run", BLUEPRINT,
        "--config", str(empty),
        "--controlcoordinator.fence-path", str(fence),
        "--manipulationmodule.fence-path", str(fence),
        "--controlcoordinator.require-fence-on-hardware", "true",
        "--controlcoordinator.expect-mock", "true",
        "--controlcoordinator.carry-max-speed-m-s", str(CARRY_SPEED_M_S),
        "--armteleopmodule.server-port", str(webxr),
        "--manipulationmodule.visualization.port", str(viser),
        "--viewer", "none",
    ]  # fmt: skip
    assert not any("can-port" in a or "can_port" in a for a in argv), "mock e2e must not pass CAN"
    env = {**os.environ, **bus, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    env.pop("DIMOS_RUN_ID", None)
    log_path = tmp / "runtime_run.log"
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
    handle = Run(proc=proc, log_path=log_path, webxr_port=webxr, viser_port=viser, fence=fence)
    try:
        yield handle
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)  # mock arms only: never leave a stray stack
                proc.wait(timeout=10)
        target = os.environ.get("KITEOS_E2E_ARTIFACTS")
        if target:
            Path(target).expanduser().mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log_path, Path(target).expanduser() / "openarm_runtime_run.log")


def _connect(run: Run) -> Any:
    from dimos.porcelain.dimos import Dimos

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last: BaseException | None = None
    while time.monotonic() < deadline:
        if run.proc.poll() is not None:
            pytest.fail(f"kiteos-runtime run exited early ({run.proc.returncode}):\n" + "\n".join(run.log().splitlines()[-40:]))
        client = None
        try:
            client = Dimos.connect(timeout=3.0)
            if client.get_module("ControlCoordinator").fence_status().get("running"):
                return client
        except Exception as exc:
            last = exc
        if client is not None:
            client.stop()
        time.sleep(1.0)
    pytest.fail(f"the OpenArm mock stack did not come up within {STARTUP_TIMEOUT_S:.0f} s: {last!r}")


def _arms(named: dict[str, float]) -> np.ndarray:
    return np.array([float(named[n]) for n in ARMS])


def _trajectory(q0: np.ndarray, q1: list[float], duration: float) -> JointTrajectory:
    return JointTrajectory(
        joint_names=ARMS,
        points=[
            TrajectoryPoint(time_from_start=0.0, positions=[float(v) for v in q0]),
            TrajectoryPoint(time_from_start=duration, positions=[float(v) for v in q1]),
        ],
    )


def _wait_near(cc: Any, goal: list[float], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if float(np.max(np.abs(_arms(cc.get_joint_positions()) - np.array(goal)))) < 0.02:
            return True
        time.sleep(0.05)
    return False


def _settled(cc: Any, timeout: float = 10.0) -> np.ndarray:
    """The arm positions once they stop changing (the trajectory has finished)."""
    previous = _arms(cc.get_joint_positions())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = _arms(cc.get_joint_positions())
        if float(np.max(np.abs(current - previous))) < 1e-7:
            return current
        previous = current
    return previous


@dataclass
class QuestHand:
    """One controller, in the ROBOT frame (the server maps the WebXR pose back to this)."""

    position: np.ndarray
    rotation: Rotation = field(default_factory=Rotation.identity)
    grip: bool = True
    thumb: bool = False


class FakeQuest:
    """The Quest teleop page's side of the WebXR websocket: LCM PoseStamped + Joy frames at 50 Hz.

    It is what teleop.js sends, so a run goes through the real ArmTeleopModule (pose transform,
    button packing) into the coordinator's teleop task. ``animate`` changes the hands under its lock.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self.hands = {"left": QuestHand(np.array([0.3, 0.2, 1.0])), "right": QuestHand(np.array([0.3, -0.2, 1.0]))}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> FakeQuest:
        self._thread.start()
        if not self._connected.wait(20.0):
            pytest.fail(f"could not connect to the WebXR server on port {self.port}: {self.error!r}")
        return self

    def __exit__(self, *_exc: object) -> None:
        with self._lock:
            for hand in self.hands.values():
                hand.grip = hand.thumb = False
        time.sleep(0.3)  # the released grips reach the task: the session ends cleanly
        self._stop.set()
        self._thread.join(timeout=5.0)

    def animate(self, seconds: float, update: Any) -> None:
        t0 = time.monotonic()
        while (t := time.monotonic() - t0) < seconds:
            if self.error is not None:
                pytest.fail(f"the fake Quest lost its connection: {self.error!r}")
            with self._lock:
                update(t)
            time.sleep(0.01)

    def _frames(self) -> list[bytes]:
        frames = []
        for side, hand in self.hands.items():
            robot = np.eye(4)
            robot[:3, :3] = hand.rotation.as_matrix()
            robot[:3, 3] = hand.position
            webxr = np.linalg.inv(WEBXR_TO_ROBOT_FRAME) @ robot  # invert the server's webxr_to_robot
            turn = Rotation.from_euler("z", 90 if side == "left" else -90, degrees=True).as_matrix()
            webxr[:3, :3] = webxr[:3, :3] @ turn.T
            x, y, z, w = Rotation.from_matrix(webxr[:3, :3]).as_quat()
            pose = PoseStamped(
                ts=time.time(),
                frame_id=side,
                position=Vector3(*map(float, webxr[:3, 3])),
                orientation=Quaternion(float(x), float(y), float(z), float(w)),
            )
            # axes: stick x, stick y, trigger, grip; buttons: trigger, grip, touchpad, thumbstick, X/A, Y/B, menu
            joy = Joy(
                frame_id=side,
                axes=[0.0, 0.0, 0.0, 1.0 if hand.grip else 0.0],
                buttons=[0, int(hand.grip), 0, int(hand.thumb), 0, 0, 0],
            )
            frames += [pose.lcm_encode(), joy.lcm_encode()]
        return frames

    def _run(self) -> None:
        try:
            asyncio.run(self._stream())
        except BaseException as exc:  # reported by animate / __enter__
            self.error = exc

    async def _stream(self) -> None:
        import aiohttp

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # the WebXR server's self-signed certificate
        url = f"wss://127.0.0.1:{self.port}/ws"
        deadline = time.monotonic() + 20.0
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    ws = await session.ws_connect(url, ssl=context)
                    break
                except aiohttp.ClientError:
                    if time.monotonic() > deadline:
                        raise
                    await asyncio.sleep(0.5)
            async with ws:
                self._connected.set()
                while not self._stop.is_set():
                    with self._lock:
                        frames = self._frames()
                    for frame in frames:
                        await ws.send_bytes(frame)
                    await asyncio.sleep(0.02)


def _gripper(cc: Any, kinematics: DualArmKinematics) -> tuple[np.ndarray, Quaternion]:
    named = cc.get_joint_positions()
    pose = kinematics.link_poses("left", [named[n] for n in LEFT])["ee_base_link"]
    return pose[:3, 3].copy(), Quaternion.from_rotation_matrix(pose[:3, :3])


def test_openarm_fence_end_to_end(run: Run) -> None:
    client = _connect(run)
    checker = FenceChecker(FenceConfig.load(run.fence))
    try:
        cc = client.get_module("ControlCoordinator")

        # ---- 0: interlocks and host plumbing
        status = cc.fence_status()
        assert status["product"] == "KiteOS" and status["robot"] == "openarm"
        assert list(status["hardware"]) == ["openarm"], status["hardware"].keys()
        hw = status["hardware"]["openarm"]
        assert hw["real_hardware"] is False and hw["adapter"] == "MockWholeBodyAdapter", hw
        assert [z["name"] for z in hw["zones"]] == [ZONE] and hw["holds"] == 0
        lsof = shutil.which("lsof")
        if lsof:
            for port in (run.webxr_port, run.viser_port):
                listing = subprocess.run(
                    [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=30
                ).stdout
                assert f"127.0.0.1:{port}" in listing and f"*:{port}" not in listing, listing
        assert "KiteOS fence zones in the viewer" in run.log()

        # ---- A: a ready pose clear of the glass is accepted and executed
        start = _arms(cc.get_joint_positions())
        result = cc.execute_trajectory(_trajectory(start, READY, 3.0))
        assert "ACCEPTED" in str(result.status), result
        assert _wait_near(cc, READY, 15.0), cc.get_joint_positions()
        assert checker.check(cc.get_joint_positions()).ok

        # ---- B: a trajectory that takes the left hand through the glass is rejected up front
        now = _settled(cc)
        through = [*THROUGH_LEFT, *now[7:]]
        rejected = cc.execute_trajectory(_trajectory(now, through, 3.0))
        assert "INVALID_TRAJECTORY" in str(rejected.status), rejected
        assert "KiteOS fence rejected the trajectory" in rejected.message and ZONE in rejected.message
        time.sleep(1.0)
        np.testing.assert_allclose(_arms(cc.get_joint_positions()), now, atol=1e-6)

        # ---- C: teleop-style streaming toward the glass is held at the margin
        holds_before = cc.fence_status()["hardware"]["openarm"]["holds"]
        commands = make_transport("/joint_command", JointState)
        q = np.array(now[:7])
        worst = np.inf
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            q = q + np.clip(np.array(REACH_LEFT) - q, -0.01, 0.01)
            commands.broadcast(None, JointState(name=LEFT, position=[float(v) for v in q]))
            report = checker.check_arm("left", _arms(cc.get_joint_positions())[:7], 0.0)
            worst = min(worst, report.min_clearance_m)
            time.sleep(0.02)
        assert np.allclose(q, REACH_LEFT), "the stream never reached the reach pose"
        time.sleep(0.5)
        after = cc.get_joint_positions()
        held = checker.check_arm("left", _arms(after)[:7], 0.0)
        status_c = cc.fence_status()["hardware"]["openarm"]
        assert status_c["holds"] > holds_before, "streaming across the glass produced no hold"
        assert status_c["last_hold"]["zone"] == ZONE and status_c["last_hold"]["side"] == "left"
        assert held.ok and worst > MARGIN_M - 1e-6, (held, worst)  # never inside the margin
        np.testing.assert_allclose(_arms(after)[7:], now[7:], atol=1e-6)  # the right arm did not move

        # ---- D: backing away to zero is accepted and executed
        now = _arms(after)
        home = cc.execute_trajectory(_trajectory(now, [0.0] * 14, 4.0))
        assert "ACCEPTED" in str(home.status), home
        target = checker.clamp_to_limits("left", np.zeros(7)).tolist() + checker.clamp_to_limits("right", np.zeros(7)).tolist()
        assert _wait_near(cc, target, 15.0), cc.get_joint_positions()

        # ---- E: the carry lock through the Quest's own protocol (a fake headset on the WebXR port)
        carry = cc.task_invoke(OPENARM_TELEOP_TASK_NAME, "carry_status", {})
        assert carry["settings"]["max_speed_m_s"] == CARRY_SPEED_M_S, carry  # the station flag reached the task
        kinematics = DualArmKinematics(robot="openarm")
        with FakeQuest(run.webxr_port) as quest:
            left = quest.hands["left"]
            quest.animate(1.0, lambda t: None)  # both grips held: the teleop session engages
            before = _gripper(cc, kinematics)
            start = left.position.copy()

            def lock_then_tilt_and_move(t: float) -> None:
                left.thumb = 0.5 <= t < 0.65  # thumbstick click: lock the left gripper
                s = min(1.0, max(0.0, (t - 1.0) / 2.0))  # then tilt the hand 30 degrees and move it
                left.position = start + s * CARRY_MOVE
                left.rotation = Rotation.from_rotvec([math.radians(30.0) * s, 0.0, 0.0])

            quest.animate(5.0, lock_then_tilt_and_move)
            locked = _gripper(cc, kinematics)
            status = cc.task_invoke(OPENARM_TELEOP_TASK_NAME, "carry_status", {})["hands"]["left"]
            assert status["locked"] and not status["holding"], status
            assert math.degrees(tilt_between(locked[1], before[1])) < 1.0  # the plate stayed level
            np.testing.assert_allclose(locked[0] - before[0], CARRY_MOVE, atol=0.005)  # and followed the hand
            quest.animate(1.0, lambda t: setattr(left, "thumb", 0.2 <= t < 0.35))  # release, hand still tilted
            released = _gripper(cc, kinematics)
            assert not cc.task_invoke(OPENARM_TELEOP_TASK_NAME, "carry_status", {})["hands"]["left"]["locked"]
            assert math.degrees(tilt_between(released[1], locked[1])) < 0.5  # no jump on release
            assert float(np.linalg.norm(released[0] - locked[0])) < 0.002
        log = run.log()
        assert "left arm carry lock ON (thumbstick)" in log and "left arm carry lock OFF (thumbstick)" in log
    finally:
        client.stop()

    # ---- the stack stops cleanly on SIGTERM
    run.proc.send_signal(signal.SIGTERM)
    assert run.proc.wait(timeout=SHUTDOWN_TIMEOUT_S) is not None
    assert "Traceback" not in run.log().split("Received signal")[0], "\n".join(run.log().splitlines()[-40:])
