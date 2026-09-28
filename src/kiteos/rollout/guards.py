# Copyright 2026 Kite ML.
"""Safety checks for dual-arm policy action chunks (numpy only).

``kiteos.rollout.runtime`` runs every chunk through :func:`guard_chunk` before it
is sent to the coordinator. This module must stay free of dimOS, LeRobot and
torch: it is imported inside dimOS's isolated LeRobot environment and is unit
tested on its own.

Every check refuses the whole chunk by raising :class:`ChunkRejectedError`. The
LeRobot runtime then stops the rollout and cancels the trajectory, so the arms
hold where they are (fail closed). None of this runs on the coordinator's
control tick; the KiteOS fence in the coordinator is the final stage.

Units: arm joints in rad, grippers as a normalized opening (0 closed, 1 open).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRIPPER_RANGE: tuple[float, float] = (0.0, 1.0)
# Checkpoint action statistics are float32, so a target that is nominally 1.0
# can land a few ulps outside the range.
_GRIPPER_RANGE_TOLERANCE = 1e-6

# (joint name, measured value, limit) -> operator-facing explanation
_Describe = Callable[[str, float, float], str]


@dataclass(frozen=True)
class GuardLimits:
    """Per-chunk limits. Defaults match ``DualArmPolicyConfig``.

    Calibrated against the validated ACT checkpoint and the teleop data it was trained on: a
    Quest-trigger gripper command moves up to a full opening per frame (30 opening/s), so the
    gripper guards only catch garbage; what the arms and grippers really execute is capped by
    the joint-trajectory task (1.0 rad/s or opening/s per joint, explicit in the rollout
    blueprint) and then by the KiteOS fence.
    """

    # Largest distance between the measured arm pose and the chunk's first target (the rig's
    # validated setting; kite_rollout --max-start-jump 0.3).
    max_start_jump_rad: float = 0.3
    # Fastest arm joint motion allowed at execution rate, over the executed path (legacy
    # kite_rollout: 3.0).
    max_joint_speed_rad_s: float = 3.0
    # Largest arm joint change between two consecutive executed points,
    # independent of playback speed.
    max_step_delta_rad: float = 0.1
    # Largest gripper distance between the measured opening and the first target (the full range:
    # a policy may open or close at once, like the Quest trigger).
    max_gripper_start_jump: float = 1.0
    # Fastest gripper opening change allowed at execution rate (twice a full swing per frame at
    # 30 fps: only non-physical chunks trip it).
    max_gripper_speed: float = 60.0
    # Number of leading steps blended from the measured pose into the policy path.
    blend_steps: int = 10

    def __post_init__(self) -> None:
        for name in (
            "max_start_jump_rad",
            "max_joint_speed_rad_s",
            "max_step_delta_rad",
            "max_gripper_start_jump",
            "max_gripper_speed",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if not isinstance(self.blend_steps, int) or self.blend_steps < 1:
            raise ValueError(f"blend_steps must be an integer >= 1, got {self.blend_steps!r}")


class ChunkRejectedError(RuntimeError):
    """A policy chunk failed a KiteOS rollout guard and must not be executed."""

    def __init__(
        self,
        check: str,
        message: str,
        *,
        joint: str | None = None,
        value: float | None = None,
        limit: float | None = None,
    ) -> None:
        super().__init__(f"KiteOS rollout guard ({check}): {message}")
        self.check = check
        self.joint = joint
        self.value = value
        self.limit = limit


@dataclass(frozen=True)
class JointGroups:
    """Column indices of the arm and gripper joints in the policy's joint order."""

    names: tuple[str, ...]
    arm: NDArray[np.intp]
    gripper: NDArray[np.intp]


def joint_groups(joint_names: Sequence[str], gripper_joints: Sequence[str]) -> JointGroups:
    """Split ``joint_names`` into arm and gripper columns.

    Raises:
        ValueError: duplicate names, a gripper joint that is not in
            ``joint_names``, or no arm joints left.
    """
    names = tuple(joint_names)
    if len(set(names)) != len(names):
        raise ValueError("joint_names must not contain duplicates")
    grippers = tuple(gripper_joints)
    if len(set(grippers)) != len(grippers):
        raise ValueError("gripper_joints must not contain duplicates")
    unknown = [name for name in grippers if name not in names]
    if unknown:
        raise ValueError(f"gripper_joints are not in joint_names: {unknown}")
    arm = np.asarray([i for i, name in enumerate(names) if name not in grippers], dtype=np.intp)
    if arm.size == 0:
        raise ValueError("joint_names must contain at least one arm joint")
    gripper = np.asarray([names.index(name) for name in grippers], dtype=np.intp)
    return JointGroups(names=names, arm=arm, gripper=gripper)


@dataclass(frozen=True)
class GuardedChunk:
    """A chunk that passed every guard, ready to become a joint trajectory.

    ``start`` is the measured state (grippers clamped into the normalized range)
    and becomes the trajectory's first point; ``actions`` are the policy targets
    with their first steps blended from ``start``.
    """

    start: NDArray[np.float32]
    actions: NDArray[np.float32]


def blend_from_state(state: ArrayLike, actions: ArrayLike, steps: int) -> NDArray[np.float32]:
    """Blend the first ``steps`` actions linearly from ``state`` into the policy path.

    Step ``k`` (1-based) becomes ``state + (k / steps) * (actions[k-1] - state)``,
    so the offset between the measured pose and the policy's path is spread over
    ``steps`` points instead of executed at once. Later steps are unchanged.
    """
    state_arr = np.asarray(state, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    count = min(int(steps), len(actions_arr))
    blended = actions_arr.copy()
    if count <= 0:
        return blended
    alpha = (np.arange(1, count + 1, dtype=np.float32) / np.float32(count))[:, None]
    blended[:count] = (1.0 - alpha) * state_arr[None, :] + alpha * actions_arr[:count]
    return blended


def guard_chunk(
    state: ArrayLike,
    actions: ArrayLike,
    *,
    groups: JointGroups,
    fps: float,
    limits: GuardLimits,
) -> GuardedChunk:
    """Check one policy chunk against ``limits`` and return it blended from ``state``.

    Args:
        state: Measured joint positions, shape ``(J,)``, in ``groups.names`` order.
        actions: Policy targets, shape ``(N, J)``, one row per execution step.
        groups: Arm and gripper columns (see :func:`joint_groups`).
        fps: Execution rate in steps per second (policy fps times playback speed).
        limits: Guard limits.

    Raises:
        ChunkRejectedError: the chunk (or the observation it was computed from) is
            unsafe to execute. Checks, in order: shapes and finiteness, gripper
            target range, arm and gripper start jump, then over the executed
            path (measured pose followed by the blended chunk) the arm per-step
            delta, arm speed and gripper speed.
    """
    joint_count = len(groups.names)
    state_arr = np.asarray(state, dtype=np.float64)
    actions_arr = np.asarray(actions, dtype=np.float64)
    if state_arr.shape != (joint_count,):
        raise ChunkRejectedError(
            "shape", f"joint state has shape {state_arr.shape}, expected ({joint_count},)"
        )
    if actions_arr.ndim != 2 or actions_arr.shape[0] < 1 or actions_arr.shape[1] != joint_count:
        raise ChunkRejectedError(
            "shape", f"action chunk has shape {actions_arr.shape}, expected (steps, {joint_count})"
        )
    if not np.all(np.isfinite(state_arr)):
        raise ChunkRejectedError("non_finite", "joint state contains non-finite positions")
    if not np.all(np.isfinite(actions_arr)):
        raise ChunkRejectedError("non_finite", "policy chunk contains non-finite targets")
    if not math.isfinite(fps) or fps <= 0:
        raise ChunkRejectedError(
            "fps", f"execution rate must be a finite positive fps, got {fps!r}"
        )

    arm, gripper = groups.arm, groups.gripper
    low, high = GRIPPER_RANGE

    if gripper.size:
        targets = actions_arr[:, gripper]
        outside = (targets < low - _GRIPPER_RANGE_TOLERANCE) | (
            targets > high + _GRIPPER_RANGE_TOLERANCE
        )
        if np.any(outside):
            row, col = (int(i) for i in np.argwhere(outside)[0])
            value = float(targets[row, col])
            name = groups.names[gripper[col]]
            raise ChunkRejectedError(
                "gripper_range",
                f"policy target {value:.3f} on {name} is outside the normalized opening "
                f"range [{low:g}, {high:g}]",
                joint=name,
                value=value,
            )

    # A measured opening a hair outside [0, 1] must not propagate into commands.
    start = state_arr.copy()
    if gripper.size:
        start[gripper] = np.clip(start[gripper], low, high)

    _check_max(
        np.abs(actions_arr[0, arm] - start[arm]),
        arm,
        groups,
        limits.max_start_jump_rad,
        check="arm_start_jump",
        describe=lambda name, value, limit: (
            f"policy's first target is {value:.3f} rad from the current pose on {name} "
            f"(limit {limit:g} rad); refusing to jump. Move the arms near the training "
            "start pose."
        ),
    )
    _check_max(
        np.abs(actions_arr[0, gripper] - start[gripper]),
        gripper,
        groups,
        limits.max_gripper_start_jump,
        check="gripper_start_jump",
        describe=lambda name, value, limit: (
            f"policy's first target on {name} is {value:.3f} opening from the current "
            f"opening (limit {limit:g}); refusing to jump."
        ),
    )

    blended = blend_from_state(start, actions_arr, limits.blend_steps).astype(np.float64)
    if gripper.size:
        blended[:, gripper] = np.clip(blended[:, gripper], low, high)
    steps = np.abs(np.diff(np.vstack([start[None, :], blended]), axis=0)).max(axis=0)

    _check_max(
        steps[arm],
        arm,
        groups,
        limits.max_step_delta_rad,
        check="arm_step_delta",
        describe=lambda name, value, limit: (
            f"policy chunk moves {name} by {value:.3f} rad in one step "
            f"(limit {limit:g} rad per step); refusing to execute"
        ),
    )
    _check_max(
        steps[arm] * fps,
        arm,
        groups,
        limits.max_joint_speed_rad_s,
        check="arm_speed",
        describe=lambda name, value, limit: (
            f"policy chunk would move {name} at {value:.2f} rad/s "
            f"(limit {limit:g} rad/s); refusing to execute"
        ),
    )
    _check_max(
        steps[gripper] * fps,
        gripper,
        groups,
        limits.max_gripper_speed,
        check="gripper_speed",
        describe=lambda name, value, limit: (
            f"policy chunk would move {name} at {value:.2f} opening/s "
            f"(limit {limit:g} opening/s); refusing to execute"
        ),
    )

    return GuardedChunk(start=start.astype(np.float32), actions=blended.astype(np.float32))


def _check_max(
    values: NDArray[np.float64],
    columns: NDArray[np.intp],
    groups: JointGroups,
    limit: float,
    *,
    check: str,
    describe: _Describe,
) -> None:
    if values.size == 0:
        return
    worst = int(np.argmax(values))
    value = float(values[worst])
    if value > limit:
        name = groups.names[int(columns[worst])]
        raise ChunkRejectedError(
            check, describe(name, value, limit), joint=name, value=value, limit=limit
        )
