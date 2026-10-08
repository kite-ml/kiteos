# Copyright 2026 Kite ML.
"""Carry lock for Quest teleop: keep a gripper level while it carries something that must not tilt.

A 96-well plate spills when it tilts, and an operator's wrist cannot hold a plate level while the
arm moves across a table. Locking a joint does not help: the gripper's tilt depends on every joint
of the arm. So the lock is in the IK target instead.

``KiteTeleopIKTask`` is the runtime's ``TeleopIKTask``, which turns each Quest controller's motion into a Pink IK
target for one gripper. KiteOS adds a carry lock per arm:

* **Toggle.** Click the thumbstick of that arm's controller, or call
  ``task_invoke(<task>, "set_carry_lock", {"hand": "left", "locked": True})`` from a script.
  The lock stays on when the grips are released, because the plate is still in the gripper.
* **Tilt held.** A locked gripper keeps the tilt it actually had when it was locked (its measured
  pose, which can lag the hand). The hand's rotation is ignored except its turn about the vertical,
  which still turns the gripper (``yaw: follow``); ``yaw: lock`` ignores that too. The hand moves the
  gripper relative to where the gripper was at the lock.
* **Gentle motion.** The gripper follows the hand's position with a bounded speed and
  acceleration, so a jerk of the hand does not slosh the wells. The turn is rate limited the same way.
* **Watchdog.** If the measured gripper tilts more than ``max_tilt_error_deg`` from the
  command (the arm cannot reach the hand's position level), the gripper stops following the
  hand's position. It settles where it is until the IK has it level again (half that error),
  then follows the hand again, so the arm stops at the edge of what it can reach level.
* **No jump on release.** The gripper continues from where it is and follows the hand's motion
  from then on.

Vertical is the robot base frame's +z, which assumes the robot stands on a level table.
Collection records the joint commands after the IK, so demonstrations recorded under the lock are level.
The lock is not a safety barrier: the fence still applies to every command, and a policy rollout
is not held level by it.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dimos.control.task import CoordinatorState
from dimos.control.tasks.pose_target_ik import FrameTargetSnapshot, PinkPoseTargetSolver
from dimos.control.tasks.teleop_ik_task.teleop_ik_task import (
    OperatorHand,
    TeleopHandBinding,
    TeleopIKTask,
    TeleopIKTaskConfig,
    TeleopIKTaskParams,
)
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.protocol.service.spec import BaseConfig
from dimos.teleop.webxr.controller_types import Buttons
from dimos.utils.logging_config import setup_logger

from kiteos.teleop.tasks import (
    CARRY_MAX_ACCEL_M_S2,
    CARRY_MAX_SPEED_M_S,
    CARRY_MAX_TILT_ERROR_DEG,
    CARRY_MAX_YAW_RATE_DEG_S,
    CARRY_YAW,
    CarryYaw,
)

if TYPE_CHECKING:
    from dimos.control.coordinator import TaskConfig
    from dimos.control.hardware_interface import ConnectedHardware, ConnectedWholeBody

logger = setup_logger()

# Each arm's lock toggles with the thumbstick press of its controller. It is free in every KiteOS
# setup: the grips are the deadman, the triggers the grippers, B/Y the episodes, A the policy.
CARRY_BUTTONS: Mapping[OperatorHand, str] = {
    OperatorHand.LEFT: "left_thumbstick",
    OperatorHand.RIGHT: "right_thumbstick",
}
# The turn about the vertical reaches its rate limit in a third of a second.
_TURN_ACCEL_PER_RATE = 3.0
_DEFAULT_DT = 0.01
_WATCHDOG_LOG_INTERVAL_S = 2.0


class CarrySettings(BaseConfig):
    """How a locked gripper follows the hand (the station file's ``carry:`` section)."""

    max_speed_m_s: float = Field(CARRY_MAX_SPEED_M_S, gt=0)
    max_accel_m_s2: float = Field(CARRY_MAX_ACCEL_M_S2, gt=0)
    max_yaw_rate_deg_s: float = Field(CARRY_MAX_YAW_RATE_DEG_S, gt=0)
    yaw: CarryYaw = CARRY_YAW
    max_tilt_error_deg: float = Field(CARRY_MAX_TILT_ERROR_DEG, gt=0)


# ---------------------------------------------------------------------------- geometry


def identity() -> Quaternion:
    return Quaternion(0.0, 0.0, 0.0, 1.0)


def about_vertical(angle: float) -> Quaternion:
    """A rotation by ``angle`` radians about +z."""
    return Quaternion(0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def turn_about_vertical(q: Quaternion) -> float | None:
    """The part of rotation ``q`` that turns about +z, in radians within [-pi, pi].

    This is the twist of a swing-twist split. It is None when undefined, which happens when ``q`` turns the vertical upside down.
    """
    w, z = float(q.w), float(q.z)
    if math.hypot(w, z) < 1e-6:
        return None
    if w < 0.0:
        w, z = -w, -z
    return 2.0 * math.atan2(z, w)


def tilt_between(a: Quaternion, b: Quaternion) -> float:
    """How far orientation ``a`` is tilted from ``b``, in radians.

    This is the angle by which ``a * b^-1`` moves the vertical, so a pure turn about the vertical counts as no tilt.
    """
    e = a * b.inverse()
    norm = e.x * e.x + e.y * e.y + e.z * e.z + e.w * e.w
    if norm <= 0.0:
        return math.pi
    up_z = 1.0 - 2.0 * (e.x * e.x + e.y * e.y) / norm
    return math.acos(max(-1.0, min(1.0, up_z)))


def wrap_angle(angle: float) -> float:
    """``angle`` in [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Follower:
    """Moves a value (a point, or an angle) toward a goal with bounded speed and acceleration.

    It brakes in time to stop at the goal, so it neither overshoots nor jerks when the goal stops.
    """

    def __init__(self, value: Sequence[float] | NDArray[np.float64], max_speed: float, max_accel: float) -> None:
        self.value = np.array(value, dtype=np.float64)
        self.velocity = np.zeros_like(self.value)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)

    def step(self, goal: Sequence[float] | NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        goal = np.asarray(goal, dtype=np.float64)
        error = goal - self.value
        distance = float(np.linalg.norm(error))
        if distance > 1e-12:
            speed = min(self.max_speed, math.sqrt(2.0 * self.max_accel * distance))
            desired = error * (speed / distance)
        else:
            desired = np.zeros_like(error)
        change = desired - self.velocity
        size = float(np.linalg.norm(change))
        limit = self.max_accel * dt
        if size > limit:
            change *= limit / size
        self.velocity = self.velocity + change
        step = self.velocity * dt
        if float(np.linalg.norm(step)) >= distance and float(np.dot(step, error)) > 0.0:
            self.value = goal.copy()  # arrives this tick
            self.velocity = np.zeros_like(self.value)
        else:
            self.value = self.value + step
        return self.value.copy()


# ---------------------------------------------------------------------------- the task


def _vector(v: Vector3) -> NDArray[np.float64]:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


@dataclass
class _Held:
    """What one lock holds, captured from the gripper's measured pose when the lock takes effect."""

    orientation: Quaternion  # the gripper's orientation then, before the turn about the vertical
    hand: Quaternion  # the hand's raw target orientation then (the turn's reference)
    position: Follower
    turn: Follower
    turn_goal: float = 0.0
    holding: bool = False  # the watchdog paused following the hand
    tilt_error_deg: float | None = None
    warned: bool = False  # this hold was logged (so its end is logged too)


@dataclass
class _Carry:
    """One arm's carry-lock state (guarded by ``KiteTeleopIKTask._carry_lock``)."""

    locked: bool = False
    held: _Held | None = None  # None until the lock has captured the gripper (or while free)
    # The gripper's target is the hand's target moved by these, so it never jumps when a lock
    # starts or ends; the turn offset is applied after the hand's rotation.
    offset_position: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    offset_orientation: Quaternion = field(default_factory=identity)
    last_position: NDArray[np.float64] | None = None
    last_orientation: Quaternion | None = None
    watchdog_logged_at: float = float("-inf")

    def reset_session(self) -> None:
        """A teleop session started or ended: the runtime re-captures its references."""
        self.held = None
        self.offset_position = np.zeros(3)
        self.offset_orientation = identity()
        self.last_position = self.last_orientation = None


class KiteTeleopIKTask(TeleopIKTask):
    """The runtime's two-hand Quest teleop IK task with a carry lock per arm (see the module doc)."""

    def __init__(
        self,
        name: str,
        config: TeleopIKTaskConfig,
        *,
        carry: CarrySettings | None = None,
        solver: PinkPoseTargetSolver | None = None,
        solver_type: type[PinkPoseTargetSolver] | None = None,
    ) -> None:
        self._carry_settings = carry or CarrySettings()
        self._carry_lock = threading.Lock()
        self._carry_frames = {binding.hand: binding.target_frame for binding in config.bindings}
        self._carry = {hand: _Carry() for hand in self._carry_frames}
        self._carry_pressed = dict.fromkeys(self._carry_frames, False)
        # Bumped at every teleop session boundary; a tick that straddles one is skipped.
        self._carry_generation = 0
        super().__init__(name, config, solver=solver, solver_type=solver_type)

    # ------------------------------------------------------------------ commands

    def on_teleop_buttons(self, msg: Buttons, t_now: float) -> bool:
        """The runtime's deadman handling, plus: a thumbstick press toggles that arm's lock."""
        with self._carry_lock:
            for hand in self._carry:
                pressed = bool(getattr(msg, CARRY_BUTTONS[hand]))
                if pressed and not self._carry_pressed[hand]:
                    self._set_locked(hand, not self._carry[hand].locked, "thumbstick")
                self._carry_pressed[hand] = pressed
        return super().on_teleop_buttons(msg, t_now)

    def set_carry_lock(self, hand: str, locked: bool) -> dict[str, Any]:
        """Lock (``locked=True``) or release one arm's carry lock; returns ``carry_status()``."""
        key = OperatorHand(hand)
        with self._carry_lock:
            if key not in self._carry:
                raise ValueError(f"this teleop task has no {key.value} hand (hands: {[h.value for h in self._carry]})")
            self._set_locked(key, bool(locked), "command")
        return self.carry_status()

    def carry_status(self) -> dict[str, Any]:
        """The settings and, per hand: locked, holding (watchdog), tilt error and turn (degrees)."""
        with self._carry_lock:
            hands = {hand.value: _status(state) for hand, state in self._carry.items()}
        return {"settings": self._carry_settings.model_dump(), "hands": hands}

    def _set_locked(self, hand: OperatorHand, locked: bool, source: str) -> None:
        state = self._carry[hand]
        if state.locked == locked:
            return
        state.locked = locked
        if locked:
            state.held = None  # captured from the gripper's measured pose at the next tick
            s = self._carry_settings
            turn = "follows the hand" if s.yaw == "follow" else "locked"
            logger.info(
                f"{hand.value} arm carry lock ON ({source}): tilt held, turn about the vertical {turn}, "
                f"at most {s.max_speed_m_s:g} m/s"
            )
        else:
            logger.info(f"{hand.value} arm carry lock OFF ({source})")

    # ------------------------------------------------------------------ targets

    def _clear_hand_session_locked(self) -> None:
        super()._clear_hand_session_locked()
        with self._carry_lock:
            self._carry_generation += 1
            for state in self._carry.values():
                state.reset_session()

    def _frame_target_snapshot(self, state: CoordinatorState) -> FrameTargetSnapshot | None:
        with self._carry_lock:
            generation = self._carry_generation
            locked_frames = [self._carry_frames[hand] for hand, carry in self._carry.items() if carry.locked]
        snapshot = super()._frame_target_snapshot(state)
        if snapshot is None:
            return None
        measured = (self.current_frame_poses(state, locked_frames) if locked_frames else None) or {}
        dt = state.dt if state.dt > 0.0 else _DEFAULT_DT
        with self._carry_lock:
            if generation != self._carry_generation:
                return None  # a teleop session started or ended during this tick
            targets = dict(snapshot.targets)
            for hand, frame in self._carry_frames.items():
                if frame in targets:
                    targets[frame] = self._carry_target(hand, targets[frame], measured.get(frame), dt, state.t_now)
        return FrameTargetSnapshot(
            targets=targets,
            last_update_time=snapshot.last_update_time,
            extra_joint_positions=snapshot.extra_joint_positions,
        )

    def _carry_target(
        self, hand: OperatorHand, raw: PoseStamped, measured: PoseStamped | None, dt: float, t_now: float
    ) -> PoseStamped:
        """This tick's target for one gripper, from the runtime's target for the hand (``raw``).

        ``measured`` is the gripper's live pose (only looked up while the arm is locked).
        """
        state = self._carry[hand]
        raw_position = _vector(raw.position)
        raw_orientation = raw.orientation
        if not state.locked and state.held is not None:
            # Released since the last tick: keep the gripper where it is, follow the hand from here.
            if state.last_position is not None and state.last_orientation is not None:
                state.offset_position = state.last_position - raw_position
                state.offset_orientation = raw_orientation.inverse() * state.last_orientation
            state.held = None
        if state.locked and state.held is None and measured is not None:
            state.held = self._capture(state, raw_position, raw_orientation, measured)
        held = state.held
        if held is None:  # free, or just locked and waiting one tick for the measured pose
            position = raw_position + state.offset_position
            orientation = raw_orientation * state.offset_orientation
        else:
            position, orientation = self._held_target(hand, state, held, raw_position, raw_orientation, measured, dt, t_now)
        state.last_position, state.last_orientation = position, orientation
        return PoseStamped(
            ts=raw.ts,
            frame_id=raw.frame_id,
            position=Vector3(float(position[0]), float(position[1]), float(position[2])),
            orientation=orientation,
        )

    def _capture(
        self, state: _Carry, raw_position: NDArray[np.float64], raw_orientation: Quaternion, measured: PoseStamped
    ) -> _Held:
        """Hold the gripper as it actually is; the hand moves it from there."""
        settings = self._carry_settings
        position = _vector(measured.position)
        state.offset_position = position - raw_position
        # The watchdog compares the gripper with the lock from this tick on, not with the free target.
        state.last_position, state.last_orientation = position, measured.orientation
        rate = math.radians(settings.max_yaw_rate_deg_s)
        return _Held(
            orientation=measured.orientation,
            hand=raw_orientation,
            position=Follower(position, settings.max_speed_m_s, settings.max_accel_m_s2),
            turn=Follower([0.0], rate, rate * _TURN_ACCEL_PER_RATE),
        )

    def _held_target(
        self,
        hand: OperatorHand,
        state: _Carry,
        held: _Held,
        raw_position: NDArray[np.float64],
        raw_orientation: Quaternion,
        measured: PoseStamped | None,
        dt: float,
        t_now: float,
    ) -> tuple[NDArray[np.float64], Quaternion]:
        settings = self._carry_settings
        goal = raw_position + state.offset_position
        if settings.yaw == "follow":
            turn = turn_about_vertical(raw_orientation * held.hand.inverse())
            if turn is not None:
                held.turn_goal += wrap_angle(turn - held.turn_goal)
        turn_goal = held.turn_goal if settings.yaw == "follow" else 0.0
        if measured is not None and state.last_orientation is not None:
            error = math.degrees(tilt_between(measured.orientation, state.last_orientation))
            held.tilt_error_deg = error
            if error > settings.max_tilt_error_deg and not held.holding:
                held.holding = True
                held.warned = t_now - state.watchdog_logged_at >= _WATCHDOG_LOG_INTERVAL_S
                if held.warned:  # at the edge of the arm's level reach this can repeat: log it at most every 2 s
                    state.watchdog_logged_at = t_now
                    logger.warning(
                        f"{hand.value} arm carry lock: gripper tilted {error:.1f} deg from the lock; "
                        "not following the hand until it is level again (out of level reach?)"
                    )
            elif error < settings.max_tilt_error_deg / 2.0 and held.holding:
                held.holding = False
                if held.warned:
                    logger.info(f"{hand.value} arm carry lock: level again, following the hand")
        if held.holding:
            # Settle where the gripper is (the IK then only has to level it), turn nowhere.
            goal = held.position.value if measured is None else _vector(measured.position)
            turn_goal = float(held.turn.value[0])
        position = held.position.step(goal, dt)
        orientation = about_vertical(float(held.turn.step([turn_goal], dt)[0])) * held.orientation
        return position, orientation


def _status(state: _Carry) -> dict[str, Any]:
    held = state.held
    return {
        "locked": state.locked,
        "holding": bool(held and held.holding),
        "tilt_error_deg": None if held is None or held.tilt_error_deg is None else round(held.tilt_error_deg, 2),
        "turn_deg": None if held is None else round(math.degrees(float(held.turn.value[0])), 1),
    }


# ---------------------------------------------------------------------------- factory


class KiteTeleopIKTaskParams(TeleopIKTaskParams):
    """The runtime's teleop IK task params, plus the carry-lock settings."""

    carry: CarrySettings = Field(default_factory=CarrySettings)


def create_task(
    cfg: TaskConfig,
    hardware: Mapping[str, ConnectedHardware | ConnectedWholeBody],
) -> KiteTeleopIKTask:
    """Build the task from a ``kiteos_teleop_ik`` ``TaskConfig`` (as the runtime's ``teleop_ik`` factory does)."""
    del hardware
    params = KiteTeleopIKTaskParams.model_validate(cfg.params)
    config = TeleopIKTaskConfig(
        joint_names=tuple(cfg.joint_names),
        robot_model=params.robot_model,
        bindings=tuple(TeleopHandBinding(hand=b.hand, target_frame=b.target_frame) for b in params.bindings),
        pink=params.pink,
        priority=cfg.priority,
        timeout=params.timeout,
        max_joint_velocity_rad_s=params.max_joint_velocity_rad_s,
        joint_velocity_limits_rad_s=params.joint_velocity_limits_rad_s,
        joint_command_filter_cutoff_hz=params.joint_command_filter_cutoff_hz,
        max_command_tracking_error_deg=params.max_command_tracking_error_deg,
        feedback_limit_tolerance=params.feedback_limit_tolerance,
        command_limit_margin=params.command_limit_margin,
    )
    return KiteTeleopIKTask(cfg.name, config, carry=params.carry, solver_type=params.solver_type)
