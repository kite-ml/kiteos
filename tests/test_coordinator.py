# Copyright 2026 Kite ML.
"""KiteDualOpenYamCoordinator: fence wiring into dimOS's coordinator (in-process, mock arms).

Coordinators are constructed the way dimOS's own control tests construct them. "Real" hardware is
a stub registered in place of the Damiao factory; no CAN bus, camera or blueprint is touched.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.dimos
pytest.importorskip("dimos", reason="needs the dimOS virtualenv")

# Keep in-process RPC transports on a private, loopback-only bus (as dimOS's conftest does).
os.environ.setdefault("LCM_DEFAULT_URL", "udpm://239.255.76.67:7917?ttl=0")
os.environ.setdefault("ZENOH_SCOUT_ADDR", "224.0.0.224:17917")

from dimos.control.coordinator import TaskConfig  # noqa: E402
from dimos.control.tasks.trajectory_task.trajectory_task import (  # noqa: E402
    JOINT_TRAJECTORY_TASK_NAME,
    JointTrajectoryTask,
    JointTrajectoryTaskConfig,
    TrajectoryExecutionStatus,
)
from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter  # noqa: E402
from dimos.hardware.whole_body.registry import whole_body_adapter_registry  # noqa: E402
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory  # noqa: E402
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint  # noqa: E402
from dimos.robot.manipulators.dual_openyam.config import (  # noqa: E402
    dual_openyam_hardware,
    dual_openyam_mock_hardware,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_ARM_JOINTS  # noqa: E402

from kiteos.safety.adapter import FencedWholeBodyAdapter  # noqa: E402
from kiteos.safety.coordinator import (  # noqa: E402
    KiteDualOpenYamCoordinator,
    KiteDualOpenYamCoordinatorConfig,
    resolve_fence_path,
)
from kiteos.safety.fence import FenceChecker, FenceConfig  # noqa: E402
from kiteos.safety.zones import HalfSpaceZone  # noqa: E402

HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
REACH = [0.0, 2.0, 2.0, 0.0, 0.0, 0.0]
GLASS = HalfSpaceZone(name="glass", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))


def write_fence(path: Path, **kw: Any) -> str:
    kw.setdefault("margin_m", 0.03)
    kw.setdefault("zones", [GLASS])
    FenceConfig(**kw).dump(path)
    return str(path)


def arm_trajectory(*waypoints: list[float], dt: float = 0.5) -> JointTrajectory:
    """Both arms: left follows ``waypoints``, right stays at home."""
    return JointTrajectory(
        joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
        points=[
            TrajectoryPoint(time_from_start=i * dt, positions=[*wp, *HOME])
            for i, wp in enumerate(waypoints)
        ],
    )


class StubRealAdapter(MockWholeBodyAdapter):
    """Stands in for the Damiao adapter; records that it was built."""

    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        StubRealAdapter.created.append(kwargs)
        super().__init__(dof=kwargs["dof"])


@pytest.fixture
def stub_real_hardware(monkeypatch) -> list[dict[str, Any]]:
    """Route whole-body adapter creation to the stub for non-mock adapter types."""
    StubRealAdapter.created = []
    real_create = whole_body_adapter_registry.create

    def create(name: str, **kwargs: Any) -> Any:
        if name == "mock_whole_body":
            return real_create(name, **kwargs)
        return StubRealAdapter(**kwargs)

    monkeypatch.setattr(whole_body_adapter_registry, "create", create)
    return StubRealAdapter.created


@pytest.fixture
def make_coordinator() -> Iterator[Callable[..., KiteDualOpenYamCoordinator]]:
    """Real coordinators (as in dimOS's test_control.py), all stopped on teardown."""
    made: list[KiteDualOpenYamCoordinator] = []

    def make(**kwargs: Any) -> KiteDualOpenYamCoordinator:
        coordinator = KiteDualOpenYamCoordinator(publish_joint_state=False, **kwargs)
        made.append(coordinator)
        return coordinator

    try:
        yield make
    finally:
        for coordinator in made:
            coordinator.stop()


def wire(coordinator: KiteDualOpenYamCoordinator) -> FencedWholeBodyAdapter:
    """Mock hardware + trajectory task, the way ``start()`` would set them up (no tick loop)."""
    component = dual_openyam_mock_hardware()
    adapter = coordinator._create_whole_body_adapter(component)
    assert isinstance(adapter, FencedWholeBodyAdapter)
    assert adapter.connect() and adapter.activate()
    assert coordinator.add_hardware(adapter, component)
    task = JointTrajectoryTask(JointTrajectoryTaskConfig(joint_names=list(DUAL_OPENYAM_ARM_JOINTS)))
    assert coordinator.add_task(task, task_type="trajectory")
    return adapter


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------- class / config


def test_class_and_config_pickle_for_forkserver_workers():
    assert pickle.loads(pickle.dumps(KiteDualOpenYamCoordinator)) is KiteDualOpenYamCoordinator
    config = KiteDualOpenYamCoordinatorConfig(
        fence_path="config/fence.yaml", left_can_port="can_left", right_can_port="can_right"
    )
    restored = pickle.loads(pickle.dumps(config))
    assert restored.fence_path == "config/fence.yaml"
    assert restored.require_fence_on_hardware is True
    assert restored.left_can_port == "can_left"
    with pytest.raises(ValueError):
        KiteDualOpenYamCoordinatorConfig(fence="nope")  # extra="forbid" like every dimOS config


def test_rpcs_include_the_fence_and_keep_the_overrides():
    rpcs = set(KiteDualOpenYamCoordinator.rpcs)
    assert {"fence_status", "fence_check", "fence_reload"} <= rpcs
    assert {"start", "stop", "set_estop", "execute_trajectory", "cancel_trajectory"} <= rpcs


def test_relative_fence_paths_resolve_from_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_fence_path("fence.yaml") == (tmp_path / "fence.yaml").resolve()
    assert resolve_fence_path(str(tmp_path / "x.yaml")) == (tmp_path / "x.yaml").resolve()


# --------------------------------------------------------------------------- adapter creation


def test_mock_without_a_fence_file_still_enforces_limits(make_coordinator):
    coordinator = make_coordinator()
    adapter = coordinator._create_whole_body_adapter(dual_openyam_mock_hardware())
    assert isinstance(adapter, FencedWholeBodyAdapter)
    assert isinstance(adapter.inner, MockWholeBodyAdapter)
    assert adapter.checker.config.zones == [] and adapter.mode == "enforce"
    assert adapter.connect() and adapter.activate()
    from dimos.hardware.whole_body.spec import MotorCommand

    q = [5.0, *HOME[1:], *HOME, 0.0, 0.0]  # left joint 1 far past its limit
    assert adapter.write_motor_commands([MotorCommand(q=v, dq=0.0) for v in q])
    assert adapter.inner.read_motor_states()[0].q < 5.0
    assert coordinator.fence_status()["fence_path"] is None


def test_mock_with_a_relative_fence_file(make_coordinator, tmp_path, monkeypatch):
    write_fence(tmp_path / "fence.yaml")
    monkeypatch.chdir(tmp_path)
    coordinator = make_coordinator(fence_path="fence.yaml")
    adapter = coordinator._create_whole_body_adapter(dual_openyam_mock_hardware())
    assert adapter.checker.zone_names("left") == ["glass"]
    assert coordinator.fence_status()["fence_path"] == str((tmp_path / "fence.yaml").resolve())


def test_missing_fence_file_is_an_error(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError, match="KiteOS fence file not found"):
        coordinator._create_whole_body_adapter(dual_openyam_mock_hardware())


def test_real_hardware_without_a_fence_is_refused(make_coordinator, stub_real_hardware):
    coordinator = make_coordinator(left_can_port="can_left", right_can_port="can_right")
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    assert component.adapter_type == "dual_openyam_damiao"
    with pytest.raises(RuntimeError, match="without a fence file"):
        coordinator._create_whole_body_adapter(component)
    assert stub_real_hardware == []  # refused before any hardware adapter was built


def test_real_hardware_with_a_fence_is_wrapped(make_coordinator, stub_real_hardware, tmp_path):
    fence = write_fence(tmp_path / "fence.yaml")
    coordinator = make_coordinator(fence_path=fence)
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    adapter = coordinator._create_whole_body_adapter(component)
    assert isinstance(adapter, FencedWholeBodyAdapter)
    assert isinstance(adapter.inner, StubRealAdapter)
    assert stub_real_hardware[0]["dof"] == 14
    assert "runtime_config" in stub_real_hardware[0]  # dimOS's kwargs are forwarded untouched
    assert coordinator.fence_status()["hardware"]["dual_openyam"]["real_hardware"] is True
    with pytest.raises(ValueError, match="monitor"):  # cannot be switched to monitor later
        adapter.reload(FenceConfig(mode="monitor"))


def test_real_hardware_refuses_monitor_mode(make_coordinator, stub_real_hardware, tmp_path):
    fence = write_fence(tmp_path / "fence.yaml", mode="monitor")
    coordinator = make_coordinator(fence_path=fence)
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    with pytest.raises(RuntimeError, match="monitor"):
        coordinator._create_whole_body_adapter(component)
    assert stub_real_hardware == []


def test_monitor_mode_is_allowed_on_mock(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "f.yaml", mode="monitor"))
    adapter = coordinator._create_whole_body_adapter(dual_openyam_mock_hardware())
    assert adapter.mode == "monitor"


def test_fence_requirement_can_be_waived_explicitly(make_coordinator, stub_real_hardware):
    coordinator = make_coordinator(
        left_can_port="can_left", right_can_port="can_right", require_fence_on_hardware=False
    )
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    adapter = coordinator._create_whole_body_adapter(component)
    assert isinstance(adapter.inner, StubRealAdapter)
    assert adapter.checker.config.zones == []


# --------------------------------------------------------------------------- trajectories / estop


def test_execute_trajectory_rejects_a_crossing_trajectory(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    wire(coordinator)
    task = coordinator._trajectory_task
    assert task is not None

    result = coordinator.execute_trajectory(arm_trajectory(HOME, [0.0, 1.5, 1.5, 0, 0, 0], REACH))
    assert result.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY
    assert "glass" in result.message and "left arm" in result.message
    assert not task.is_active()

    safe = arm_trajectory(HOME, [0.2, 1.0, 1.1, 0.0, 0.0, 0.0])
    assert coordinator.execute_trajectory(safe).status is TrajectoryExecutionStatus.ACCEPTED
    assert task.is_active()


def test_estop_cancels_the_trajectory_and_blocks_new_ones(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    adapter = wire(coordinator)
    task = coordinator._trajectory_task
    safe = arm_trajectory(HOME, [0.2, 1.0, 1.1, 0.0, 0.0, 0.0])
    assert coordinator.execute_trajectory(safe).status is TrajectoryExecutionStatus.ACCEPTED

    assert coordinator.set_estop(True) is True
    assert adapter.estopped and not task.is_active()
    result = coordinator.execute_trajectory(safe)
    assert result.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY
    assert "E-STOP" in result.message

    coordinator.set_estop(False)
    assert not adapter.estopped
    assert coordinator.execute_trajectory(safe).status is TrajectoryExecutionStatus.ACCEPTED


def test_fence_rpcs(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    with pytest.raises(RuntimeError, match="not running"):
        coordinator.fence_reload()
    assert coordinator.fence_check({"left_joint1": 0.0})["ok"] is False
    wire(coordinator)

    status = coordinator.fence_status()
    yaml.safe_dump(status)  # the CLI prints it as YAML
    hw = status["hardware"]["dual_openyam"]
    assert hw["mode"] == "enforce" and hw["real_hardware"] is False
    assert [z["name"] for z in hw["zones"]] == ["glass"]

    names = [*DUAL_OPENYAM_ARM_JOINTS, "left_arm/gripper", "right_arm/gripper"]
    reach = dict(zip(names, [*REACH, *HOME, 0.0, 0.0], strict=True))
    home = dict(zip(names, [*HOME, *HOME, 0.0, 0.0], strict=True))
    report = coordinator.fence_check(reach)
    assert report["ok"] is False and report["arms"]["left"]["zone"] == "glass"
    assert coordinator.fence_check(home)["ok"] is True

    wider = write_fence(tmp_path / "wider.yaml", zones=[], margin_m=0.05)
    result = coordinator.fence_reload(wider)
    assert result["ok"] and result["zones"] == [] and result["path"] == str(Path(wider).resolve())
    assert result["current"]["dual_openyam"]["ok"] is True
    assert coordinator.fence_check(reach)["ok"] is True
    with pytest.raises(FileNotFoundError):
        coordinator.fence_reload(str(tmp_path / "missing.yaml"))


def test_fence_reload_rejects_an_invalid_file(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    wire(coordinator)
    bad = tmp_path / "bad.yaml"
    bad.write_text("margin_m: 0.001\n")
    with pytest.raises(ValueError, match="margin_m"):
        coordinator.fence_reload(str(bad))
    assert coordinator.fence_status()["hardware"]["dual_openyam"]["zones"][0]["name"] == "glass"


# --------------------------------------------------------------------------- running, in process


def test_running_coordinator_holds_cancels_and_records_truthfully(make_coordinator, tmp_path):
    """Start the coordinator in-process with mock arms and a trajectory task.

    A slow trajectory toward the glass passes the precheck under an empty fence; the glass is then
    hot-reloaded. The fence must hold the arm, the supervisor must cancel the trajectory, the arm
    must never cross, and every recorded applied command must be what the fence let through.
    """
    empty = write_fence(tmp_path / "empty.yaml", zones=[])
    glass = write_fence(tmp_path / "glass.yaml")
    coordinator = make_coordinator(
        fence_path=empty,
        tasks=[
            TaskConfig(
                name=JOINT_TRAJECTORY_TASK_NAME,
                type="trajectory",
                joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
                priority=10,
            )
        ],
    )
    applied: list[JointState] = []
    lock = threading.Lock()

    def record(msg: JointState) -> None:
        with lock:
            applied.append(msg)

    coordinator.applied_joint_position_command.subscribe(record)
    coordinator.start()
    assert getattr(coordinator._tick_loop._publish_command_callback, "_kiteos_fenced", False)
    adapter = coordinator._fenced["dual_openyam"]
    task = coordinator._trajectory_task
    assert task is not None

    forward = arm_trajectory(HOME, [0.0, 1.35, 1.047, 0.0, 0.0, 0.0], dt=1.5)
    assert coordinator.execute_trajectory(forward).status is TrajectoryExecutionStatus.ACCEPTED
    coordinator.fence_reload(glass)
    with lock:
        reloaded_at = len(applied)

    assert wait_until(lambda: not task.is_active(), timeout=5.0), "trajectory was not cancelled"
    status = coordinator.fence_status()
    assert status["trajectory_cancels"] >= 1
    assert status["hardware"]["dual_openyam"]["events"]["hold"] >= 1

    referee = FenceChecker(FenceConfig.load(glass))
    measured = [s.q for s in adapter.inner.read_motor_states()]
    assert referee.check(measured).ok
    assert measured[1] < 1.35  # held short of the trajectory's goal

    with lock:
        after = applied[reloaded_at + 1 :]
    assert after, "no applied commands were recorded after the reload"
    for msg in after:
        named = dict(zip(msg.name, msg.position, strict=True))
        left = np.array([named[n] for n in DUAL_OPENYAM_ARM_JOINTS[:6]])
        assert referee.check_arm("left", left, 0.0).ok

    coordinator.stop()
    assert coordinator._supervisor is None


# --------------------------------------------------------------------------- review fixes


def left_trajectory(*waypoints: list[float], dt: float = 0.5) -> JointTrajectory:
    return JointTrajectory(
        joint_names=list(DUAL_OPENYAM_ARM_JOINTS[:6]),
        points=[
            TrajectoryPoint(time_from_start=i * dt, positions=list(wp))
            for i, wp in enumerate(waypoints)
        ],
    )


def test_task_invoke_execute_goes_through_the_fence_precheck(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    wire(coordinator)
    task = coordinator._trajectory_task
    crossing = arm_trajectory(HOME, [0.0, 1.5, 1.5, 0, 0, 0], REACH)
    result = coordinator.task_invoke(
        JOINT_TRAJECTORY_TASK_NAME, "execute", {"trajectory": crossing, "current_positions": {}}
    )
    assert result.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY
    assert "KiteOS fence rejected" in result.message and "glass" in result.message
    assert not task.is_active()

    # The undeclared streaming entry point is prechecked the same way.
    msg = JointState(name=list(DUAL_OPENYAM_ARM_JOINTS[:6]), position=REACH)
    assert coordinator.task_invoke(JOINT_TRAJECTORY_TASK_NAME, "on_joint_command", {"msg": msg, "t_now": 0.0}) is False
    assert not task.is_active()

    # Safe motions and read-only commands still reach dimOS's task.
    names = list(DUAL_OPENYAM_ARM_JOINTS)
    current = dict(zip(names, [*HOME, *HOME], strict=True))
    safe = arm_trajectory(HOME, [0.2, 1.0, 1.1, 0.0, 0.0, 0.0])
    result = coordinator.task_invoke(
        JOINT_TRAJECTORY_TASK_NAME, "execute", {"trajectory": safe, "current_positions": current}
    )
    assert result.status is TrajectoryExecutionStatus.ACCEPTED and task.is_active()
    assert coordinator.task_invoke(JOINT_TRAJECTORY_TASK_NAME, "cancel").cancelled

    # E-STOP blocks task_invoke execute like execute_trajectory.
    coordinator.set_estop(True)
    result = coordinator.task_invoke(
        JOINT_TRAJECTORY_TASK_NAME, "execute", {"trajectory": safe, "current_positions": current}
    )
    assert result.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY and "E-STOP" in result.message


def test_precheck_accepts_only_a_straight_escape_when_an_arm_starts_inside(make_coordinator, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml", zones=[]))
    adapter = wire(coordinator)
    task = coordinator._trajectory_task
    from dimos.hardware.whole_body.spec import MotorCommand

    inside = [0.0, 1.35, 1.1, 0.0, 0.0, 0.0]
    # Walk the left arm to `inside` under the empty fence, then reload the glass: now inside.
    q = np.array(HOME)
    for _ in range(200):
        q = q + np.clip(np.array(inside) - q, -0.02, 0.02)
        adapter.read_motor_states()
        time.sleep(0.0)
        adapter.write_motor_commands([MotorCommand(q=float(v), dq=0.0, kp=80.0, kd=5.0) for v in [*q, *HOME, 0.0, 0.0]])
    np.testing.assert_allclose(adapter.last_written("left"), inside, atol=1e-9)
    coordinator.fence_reload(write_fence(tmp_path / "glass.yaml"))
    assert not adapter.checker.check_arm("left", inside, 0.0).ok

    deeper = coordinator.execute_trajectory(left_trajectory(inside, [0.0, 1.5, 1.2, 0, 0, 0]))
    assert deeper.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY
    assert "does not move straight out" in deeper.message
    # Backing out but stopping inside the margin is still rejected.
    partial = coordinator.execute_trajectory(left_trajectory(inside, [0.0, 1.33, 1.1, 0, 0, 0]))
    assert partial.status is TrajectoryExecutionStatus.INVALID_TRAJECTORY
    assert "does not end clear" in partial.message
    out = coordinator.execute_trajectory(left_trajectory(inside, HOME))
    assert out.status is TrajectoryExecutionStatus.ACCEPTED, out.message
    assert task.is_active()


def test_supervisor_cancels_once_per_hold_episode(make_coordinator, tmp_path):
    """A streamed command re-executes the trajectory task every message; a sustained hold must
    cancel it once, not at the stream rate."""
    empty = write_fence(tmp_path / "empty.yaml", zones=[])
    glass = write_fence(tmp_path / "glass.yaml")
    coordinator = make_coordinator(
        fence_path=empty,
        tasks=[
            TaskConfig(
                name=JOINT_TRAJECTORY_TASK_NAME,
                type="trajectory",
                joint_names=list(DUAL_OPENYAM_ARM_JOINTS),
                priority=10,
            )
        ],
    )
    coordinator.start()
    task = coordinator._trajectory_task
    forward = arm_trajectory(HOME, [0.0, 1.35, 1.047, 0.0, 0.0, 0.0], dt=1.5)
    assert coordinator.execute_trajectory(forward).status is TrajectoryExecutionStatus.ACCEPTED
    coordinator.fence_reload(glass)
    assert wait_until(lambda: coordinator.fence_status()["trajectory_cancels"] == 1, 5.0)
    held = coordinator._fenced["dual_openyam"].status()["events"]["hold"]
    # Stream toward the glass for 1 s at 50 Hz (like /joint_command): still one hold episode.
    target = JointState(name=list(DUAL_OPENYAM_ARM_JOINTS[:6]), position=[0.0, 1.6, 1.047, 0, 0, 0])
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        task.on_joint_command(target, time.perf_counter())
        time.sleep(0.02)
    status = coordinator.fence_status()
    assert status["hardware"]["dual_openyam"]["events"]["hold"] > held + 50
    assert status["trajectory_cancels"] == 1, status["trajectory_cancels"]
    # Recent events are coalesced, and the holds are kept separately.
    recent = status["recent_events"]
    by_kind = {(e["kind"], e["zone"]): e["repeats"] for e in recent}
    assert by_kind[("hold", "glass")] > 50 and by_kind[("rate_limit", None)] > 50
    assert status["recent_holds"] and status["recent_holds"][-1]["kind"] == "hold"
    assert not any(h["new_episode"] for h in status["recent_holds"])  # still the first episode
    coordinator.stop()


def test_applied_rewrite_is_installed_before_the_tick_loop_starts(make_coordinator, tmp_path, monkeypatch):
    from dimos.control import tick_loop as tick_loop_module

    seen: list[Any] = []
    original_start = tick_loop_module.TickLoop.start

    def start(self: Any) -> None:
        seen.append(self._publish_command_callback)
        original_start(self)

    monkeypatch.setattr(tick_loop_module.TickLoop, "start", start)
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"))
    coordinator.start()
    assert len(seen) == 1 and getattr(seen[0], "_kiteos_fenced", False)
    coordinator.stop()


def test_expect_mock_refuses_real_can_ports_from_any_source(make_coordinator, stub_real_hardware, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml"), expect_mock=True)
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    with pytest.raises(RuntimeError, match="MOCK arms"):
        coordinator._create_whole_body_adapter(component)
    assert stub_real_hardware == []
    assert isinstance(coordinator._create_whole_body_adapter(dual_openyam_mock_hardware()), FencedWholeBodyAdapter)


def test_real_hardware_refuses_an_unprotected_arm(make_coordinator, stub_real_hardware, tmp_path):
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    left_only = HalfSpaceZone(name="glass_left", frame="left_base", applies_to="left", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    for zones, match in (([], "left and right"), ([left_only], "right arm")):
        coordinator = make_coordinator(fence_path=write_fence(tmp_path / "f.yaml", zones=zones))
        with pytest.raises(RuntimeError, match=match):
            coordinator._create_whole_body_adapter(component)
    assert stub_real_hardware == []
    # Explicit opt-in (teaching the first zone) is allowed, with a warning.
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "t.yaml", zones=[], allow_no_zones=True))
    assert isinstance(coordinator._create_whole_body_adapter(component), FencedWholeBodyAdapter)
    # Implausible rate limits are refused on real hardware.
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "s.yaml", max_joint_speed_rad_s=10.0))
    with pytest.raises(RuntimeError, match="max_joint_speed_rad_s"):
        coordinator._create_whole_body_adapter(component)


def test_real_hardware_reload_refuses_empty_unprotected_or_foreign_fences(make_coordinator, stub_real_hardware, tmp_path):
    coordinator = make_coordinator(fence_path=write_fence(tmp_path / "fence.yaml", station="site-a"))
    component = dual_openyam_hardware(left_can_port="can_left", right_can_port="can_right")
    adapter = coordinator._create_whole_body_adapter(component)
    assert adapter.connect() and adapter.activate()
    assert coordinator.add_hardware(adapter, component)

    truncated = tmp_path / "truncated.yaml"
    truncated.write_text("# zones were here\n")
    with pytest.raises(ValueError, match="empty"):
        coordinator.fence_reload(str(truncated))
    with pytest.raises(RuntimeError, match="no enabled zone"):
        coordinator.fence_reload(write_fence(tmp_path / "none.yaml", station="site-a", zones=[]))
    with pytest.raises(ValueError, match="station"):
        coordinator.fence_reload(write_fence(tmp_path / "other.yaml", station="site-b"))
    assert adapter.checker.zone_names("left") == ["glass"]
    assert coordinator.fence_reload(write_fence(tmp_path / "ok.yaml", station="site-a", margin_m=0.04))["ok"]
