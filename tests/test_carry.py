# Copyright 2026 Kite ML.
"""The Quest carry lock (``kiteos.teleop.carry``): geometry, the follower, the task, its wiring.

The task runs on the runtime's real ``TeleopIKTask`` with a mocked Pink solver, as the runtime's
own teleop tests do. The tests read the targets the task hands the solver, and close the loop by
feeding each target back as the measured gripper pose (perfect tracking) unless a test sets the
measured pose itself.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.runtime
pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.control.coordinator import TaskConfig  # noqa: E402
from dimos.control.task import CoordinatorState, JointStateSnapshot  # noqa: E402
from dimos.control.tasks.pose_target_ik import PinkPoseTargetSolver, PoseTargetIKTaskConfig  # noqa: E402
from dimos.control.tasks.registry import ControlTaskRegistry, control_task_registry  # noqa: E402
from dimos.control.tasks.teleop_ik_task import teleop_ik_task as runtime_teleop  # noqa: E402
from dimos.manipulation.planning.spec.config import RobotModelConfig  # noqa: E402
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped  # noqa: E402
from dimos.msgs.geometry_msgs.Quaternion import Quaternion  # noqa: E402
from dimos.msgs.geometry_msgs.Vector3 import Vector3  # noqa: E402
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402
from dimos.robot.assets.model import RobotModel  # noqa: E402
from dimos.teleop.webxr.controller_types import Buttons  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from kiteos.safety.coordinator import FencedCoordinatorMixin, KiteCarrySettings  # noqa: E402
from kiteos.teleop import carry  # noqa: E402
from kiteos.teleop.carry import (  # noqa: E402
    CarrySettings,
    Follower,
    KiteTeleopIKTask,
    about_vertical,
    tilt_between,
    turn_about_vertical,
    wrap_angle,
)
from kiteos.teleop.tasks import CARRY_COMMANDS, KITEOS_TELEOP_IK, register_kiteos_tasks  # noqa: E402

JOINTS = ("robot/left", "robot/right")
FRAMES = {"left": "left_tool", "right": "right_tool"}
DT = 0.01
LEVEL_HAND = Quaternion(0.0, 0.0, 0.0, 1.0)
# A gripper pointing down (half a turn about x), as an arm reaching over a table holds a plate.
DOWN = Quaternion(1.0, 0.0, 0.0, 0.0)


def about_x(angle: float) -> Quaternion:
    return Quaternion(math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0))


def vec(p: Vector3) -> np.ndarray:
    return np.array([p.x, p.y, p.z])


def robot_model() -> RobotModelConfig:
    return RobotModelConfig(model=RobotModel.from_file(Path("fake.urdf")), joint_names=list(JOINTS))


def teleop_config(*hands: str) -> runtime_teleop.TeleopIKTaskConfig:
    return runtime_teleop.TeleopIKTaskConfig(
        joint_names=JOINTS,
        robot_model=robot_model(),
        bindings=tuple(runtime_teleop.TeleopHandBinding(hand=h, target_frame=FRAMES[h]) for h in hands),
    )


class Rig:
    """One ``KiteTeleopIKTask`` on a mocked solver, ticked as the coordinator does (100 Hz)."""

    def __init__(self, settings: CarrySettings | None = None, hands: tuple[str, ...] = ("left", "right")) -> None:
        self.measured = {
            "left_tool": PoseStamped(position=Vector3(0.3, 0.2, 0.5), orientation=DOWN),
            "right_tool": PoseStamped(position=Vector3(0.3, -0.2, 0.5), orientation=DOWN),
        }
        self.solver = mock.Mock(spec=PinkPoseTargetSolver)
        self.solver.frame_poses.side_effect = lambda _state, frames: {f: self.measured[f] for f in frames}
        self.solver.step.return_value = JointState(name=list(JOINTS), position=[0.0, 0.0])
        self.task = KiteTeleopIKTask("quest", teleop_config(*hands), carry=settings, solver=self.solver)
        self.hands = hands
        self.hand = {h: (np.array([0.0, 0.0, 1.0]), LEVEL_HAND) for h in hands}
        self.thumb = dict.fromkeys(hands, False)
        self.grip = True
        self.tracking = True  # the measured gripper follows its target
        self.t = 1.0

    def tick(self) -> dict[str, PoseStamped] | None:
        buttons = Buttons()
        for h in self.hands:
            setattr(buttons, f"{h}_grip", self.grip)
            setattr(buttons, f"{h}_thumbstick", self.thumb[h])
        self.task.on_teleop_buttons(buttons, self.t)
        for h, (p, q) in self.hand.items():
            getattr(self.task, f"on_{h}_cartesian_command")(PoseStamped(position=Vector3(*p), orientation=q), self.t)
        joints = JointStateSnapshot(joint_positions=dict.fromkeys(JOINTS, 0.0))
        output = self.task.compute(CoordinatorState(joints=joints, t_now=self.t, dt=DT))
        self.t += DT
        if output is None:
            return None
        targets = dict(self.solver.step.call_args.args[0])
        if self.tracking:
            self.measured.update(targets)
        return targets

    def run(self, seconds: float) -> list[dict[str, PoseStamped]]:
        return [t for t in (self.tick() for _ in range(round(seconds / DT))) if t is not None]

    def click(self, hand: str) -> None:
        self.thumb[hand] = True
        self.run(0.05)
        self.thumb[hand] = False
        self.run(0.05)

    def status(self, hand: str = "left") -> dict[str, Any]:
        return self.task.carry_status()["hands"][hand]


# ---------------------------------------------------------------------------- geometry


def test_the_turn_about_the_vertical_is_the_twist_of_a_rotation() -> None:
    for angle in (-3.0, -1.0, 0.0, 0.4, 3.1):
        assert turn_about_vertical(about_vertical(angle)) == pytest.approx(angle, abs=1e-9)
        # a tilt about a horizontal axis on top of it does not change the turn
        assert turn_about_vertical(about_vertical(angle) * about_x(0.7)) == pytest.approx(angle, abs=1e-9)
    assert turn_about_vertical(about_x(math.pi)) is None  # the vertical upside down: undefined


def test_tilt_ignores_turns_about_the_vertical() -> None:
    q = about_x(0.3) * DOWN
    assert tilt_between(about_vertical(1.2) * q, q) == pytest.approx(0.0, abs=1e-7)
    assert tilt_between(about_x(0.25) * q, q) == pytest.approx(0.25, abs=1e-9)
    assert wrap_angle(1.5 * math.pi) == pytest.approx(-0.5 * math.pi)


def test_the_follower_bounds_speed_and_acceleration_and_stops_at_the_goal() -> None:
    follower = Follower([0.0, 0.0, 0.0], max_speed=0.25, max_accel=1.0)
    path = np.array([[0.0, 0.0, 0.0], *(follower.step([0.3, 0.0, 0.0], DT) for _ in range(300))])
    speeds = np.linalg.norm(np.diff(path, axis=0), axis=1) / DT
    arrival = int(np.argmax(np.all(np.isclose(path, [0.3, 0.0, 0.0]), axis=1)))
    assert 0 < arrival < len(path) - 1
    assert speeds.max() <= 0.25 + 1e-9
    assert np.abs(np.diff(speeds[: arrival - 1])).max() / DT <= 1.0 + 1e-6
    assert path[:, 0].max() <= 0.3 + 1e-12 and np.allclose(path[-1], [0.3, 0.0, 0.0])  # no overshoot


# ---------------------------------------------------------------------------- the task


def test_a_thumbstick_press_toggles_that_arm_once() -> None:
    rig = Rig()
    rig.run(0.1)
    rig.thumb["left"] = True
    rig.run(0.3)  # held down: one toggle, not one per button message
    assert rig.status()["locked"] and not rig.status("right")["locked"]
    rig.thumb["left"] = False
    rig.run(0.1)
    assert rig.status()["locked"]
    rig.click("left")
    assert not rig.status()["locked"]


def test_a_locked_gripper_keeps_its_tilt_and_follows_the_hand_gently() -> None:
    rig = Rig(CarrySettings(max_speed_m_s=0.25, max_accel_m_s2=1.0))
    rig.run(0.1)
    rig.click("left")
    start = vec(rig.measured["left_tool"].position)
    held = rig.measured["left_tool"].orientation
    # A jerk of the left hand: 20 cm forward and tilted 30 degrees at once. The right arm is free.
    rig.hand["left"] = (np.array([0.2, 0.0, 1.0]), about_x(math.radians(30)))
    rig.hand["right"] = (np.array([0.0, 0.0, 1.1]), LEVEL_HAND)
    targets = rig.run(3.0)
    left = [t["left_tool"] for t in targets]
    assert max(tilt_between(t.orientation, held) for t in left) < 1e-9
    path = np.array([start, *(vec(t.position) for t in left)])
    assert (np.linalg.norm(np.diff(path, axis=0), axis=1) / DT).max() <= 0.25 + 1e-9
    np.testing.assert_allclose(path[-1], start + [0.2, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(vec(targets[0]["right_tool"].position), [0.3, -0.2, 0.6], atol=1e-9)
    assert rig.status() == {"locked": True, "holding": False, "tilt_error_deg": 0.0, "turn_deg": 0.0}


@pytest.mark.parametrize(("yaw", "expected_deg"), [("follow", 30.0), ("lock", 0.0)])
def test_the_gripper_turns_with_the_hand_about_the_vertical_only(yaw: str, expected_deg: float) -> None:
    rig = Rig(CarrySettings(yaw=yaw, max_yaw_rate_deg_s=60.0))
    rig.run(0.1)
    rig.click("left")
    held = rig.measured["left_tool"].orientation
    rig.hand["left"] = (np.array([0.0, 0.0, 1.0]), about_vertical(math.radians(30)) * about_x(math.radians(20)))
    targets = rig.run(2.0)
    turns = [math.degrees(turn_about_vertical(t["left_tool"].orientation * held.inverse()) or 0.0) for t in targets]
    assert max(tilt_between(t["left_tool"].orientation, held) for t in targets) < 1e-9
    assert turns[-1] == pytest.approx(expected_deg, abs=1e-6)
    assert np.abs(np.diff([0.0, *turns])).max() / DT <= 60.0 + 1e-6


def test_releasing_never_jumps_and_the_hand_moves_the_gripper_from_there() -> None:
    rig = Rig()
    rig.run(0.1)
    rig.click("left")
    rig.hand["left"] = (np.array([0.1, 0.0, 1.0]), about_x(math.radians(30)))
    rig.run(2.0)
    before = rig.measured["left_tool"]
    rig.click("left")  # release while the hand holds still, tilted
    after = rig.measured["left_tool"]
    assert not rig.status()["locked"]
    np.testing.assert_allclose(vec(after.position), vec(before.position), atol=1e-9)
    assert after.orientation.angle_to(before.orientation) < 1e-7
    # Free again: the hand's further motion moves the gripper from where it is, unlimited.
    rig.hand["left"] = (np.array([0.15, 0.0, 1.0]), about_x(math.radians(40)))
    now = rig.run(0.02)[-1]["left_tool"]
    np.testing.assert_allclose(vec(now.position) - vec(after.position), [0.05, 0.0, 0.0], atol=1e-9)
    assert tilt_between(now.orientation, after.orientation) == pytest.approx(math.radians(10), abs=1e-9)


def test_the_lock_outlives_the_deadman_and_takes_the_gripper_as_it_is_on_reengage() -> None:
    rig = Rig()
    rig.run(0.1)
    rig.click("left")
    rig.grip = False
    assert rig.run(0.1) == []  # released grips: no targets
    moved = PoseStamped(position=Vector3(0.35, 0.2, 0.45), orientation=about_x(0.1) * DOWN)
    rig.measured["left_tool"] = moved
    rig.hand["left"] = (np.array([0.5, 0.5, 1.5]), about_x(1.0))  # the hand comes back elsewhere
    rig.grip = True
    first = rig.run(0.01)[0]["left_tool"]
    assert rig.status()["locked"]
    np.testing.assert_allclose(vec(first.position), [0.35, 0.2, 0.45], atol=1e-9)
    rig.hand["left"] = (np.array([0.5, 0.5, 1.5]), about_x(1.5))
    last = rig.run(1.0)[-1]["left_tool"]
    assert tilt_between(last.orientation, moved.orientation) < 1e-9


def test_the_lock_holds_the_gripper_as_it_is_not_where_the_hand_points() -> None:
    rig = Rig()
    rig.run(0.1)
    rig.tracking = False
    lagging = PoseStamped(position=Vector3(0.32, 0.2, 0.5), orientation=about_x(math.radians(4)) * DOWN)
    rig.measured["left_tool"] = lagging
    rig.hand["left"] = (np.array([0.05, 0.0, 1.0]), about_x(math.radians(10)))  # the hand is ahead of the arm
    rig.run(0.05)
    rig.click("left")
    target = rig.run(0.01)[0]["left_tool"]
    np.testing.assert_allclose(vec(target.position), [0.32, 0.2, 0.5], atol=1e-9)
    assert target.orientation.angle_to(lagging.orientation) < 1e-7


def test_the_watchdog_waits_for_the_gripper_to_level_before_following_on() -> None:
    rig = Rig(CarrySettings(max_tilt_error_deg=3.0))
    rig.run(0.1)
    rig.click("left")
    start = vec(rig.measured["left_tool"].position)
    rig.tracking = False
    # The arm cannot keep it level: the measured gripper is 5 degrees off and 1 cm behind.
    stuck = PoseStamped(
        position=Vector3(*(start + [0.01, 0.0, 0.0])),
        orientation=about_x(math.radians(5)) * rig.measured["left_tool"].orientation,
    )
    rig.measured["left_tool"] = stuck
    rig.hand["left"] = (np.array([0.3, 0.0, 1.0]), LEVEL_HAND)
    targets = rig.run(1.0)
    assert rig.status()["holding"] and rig.status()["tilt_error_deg"] == pytest.approx(5.0, abs=0.01)
    np.testing.assert_allclose(vec(targets[-1]["left_tool"].position), vec(stuck.position), atol=1e-6)
    rig.measured["left_tool"] = targets[-1]["left_tool"]  # level again
    rig.tracking = True
    later = rig.run(3.0)
    assert not rig.status()["holding"]
    np.testing.assert_allclose(vec(later[-1]["left_tool"].position), start + [0.3, 0.0, 0.0], atol=1e-6)


def test_scripts_lock_and_read_the_lock_through_the_task_commands() -> None:
    rig = Rig(hands=("right",))
    assert rig.task.set_carry_lock("right", True)["hands"]["right"]["locked"]
    with pytest.raises(ValueError, match="no left hand"):
        rig.task.set_carry_lock("left", True)
    with pytest.raises(ValueError):
        rig.task.set_carry_lock("middle", True)
    status = rig.task.carry_status()
    assert status["settings"] == CarrySettings().model_dump() and set(status["hands"]) == {"right"}


# ---------------------------------------------------------------------------- wiring


class _NoSolver(PinkPoseTargetSolver):
    def __init__(self, config: PoseTargetIKTaskConfig) -> None:  # no robot model needed
        self.received = config


def _task_config(**params: Any) -> TaskConfig:
    return TaskConfig(
        name="quest",
        type=KITEOS_TELEOP_IK,
        joint_names=list(JOINTS),
        priority=20,
        params={
            "robot_model": robot_model(),
            "bindings": [{"hand": "left", "target_frame": "left_tool"}],
            "solver_type": _NoSolver,
            "timeout": 0.4,
            "max_joint_velocity_rad_s": 2.0,
            "joint_command_filter_cutoff_hz": 5.0,
            **params,
        },
    )


def test_the_kiteos_type_registers_beside_the_runtime_teleop_task() -> None:
    registry = ControlTaskRegistry()
    register_kiteos_tasks(registry)
    register_kiteos_tasks(registry)  # idempotent
    assert KITEOS_TELEOP_IK in registry.available()
    ours, runtime = registry.bindings_for(KITEOS_TELEOP_IK), registry.bindings_for("teleop_ik")
    assert ours.consumes == runtime.consumes and set(ours.exposes) == set(CARRY_COMMANDS)


def test_the_factory_builds_the_runtime_task_config_plus_the_carry_settings() -> None:
    cfg = _task_config(carry={"max_speed_m_s": 0.1, "yaw": "lock"})
    task = carry.create_task(cfg, {})
    assert isinstance(task, KiteTeleopIKTask)
    assert task.carry_status()["settings"] == CarrySettings(max_speed_m_s=0.1, yaw="lock").model_dump()
    plain = replace(cfg, type="teleop_ik", params={k: v for k, v in cfg.params.items() if k != "carry"})
    assert task._teleop_config == runtime_teleop.create_task(plain, {})._teleop_config
    with pytest.raises(ValidationError):
        carry.create_task(_task_config(carry={"yaw": "spin"}), {})
    register_kiteos_tasks()
    assert isinstance(control_task_registry.create(KITEOS_TELEOP_IK, _task_config(), hardware={}), KiteTeleopIKTask)


class _RuntimeCoordinator:
    def _create_task_from_config(self, cfg: TaskConfig) -> TaskConfig:
        return cfg


class _Coordinator(FencedCoordinatorMixin, _RuntimeCoordinator):
    def __init__(self, config: Any) -> None:  # the fence state is not needed to build tasks
        self.config = config


def test_kiteos_coordinators_give_the_teleop_task_the_station_carry_settings() -> None:
    from kiteos.safety.coordinator import KiteDualOpenYamCoordinatorConfig
    from kiteos.safety.openarm_coordinator import KiteOpenArmCoordinatorConfig

    coordinator = _Coordinator(KiteCarrySettings(carry_max_speed_m_s=0.1, carry_yaw="lock"))
    built = coordinator._create_task_from_config(_task_config())
    assert built.params["carry"] == {
        "max_speed_m_s": 0.1,
        "max_accel_m_s2": 1.0,
        "max_yaw_rate_deg_s": 60.0,
        "yaw": "lock",
        "max_tilt_error_deg": 3.0,
    }
    other = TaskConfig(name="joint_trajectory", type="trajectory", joint_names=list(JOINTS))
    assert coordinator._create_task_from_config(other) is other
    assert KITEOS_TELEOP_IK in control_task_registry.available()
    assert issubclass(KiteDualOpenYamCoordinatorConfig, KiteCarrySettings)
    assert issubclass(KiteOpenArmCoordinatorConfig, KiteCarrySettings)
    with pytest.raises(ValidationError):
        KiteCarrySettings(carry_max_speed_m_s=0.0)


def test_every_kiteos_quest_setup_uses_the_carry_lock_task() -> None:
    from kiteos.blueprints.openarm import openarm_teleop_task
    from kiteos.blueprints.teleop import DUAL_OPENYAM_WEBXR_TASK

    assert DUAL_OPENYAM_WEBXR_TASK.type == openarm_teleop_task().type == KITEOS_TELEOP_IK
