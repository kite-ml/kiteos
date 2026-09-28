# Copyright 2026 Kite ML.
"""Pure-numpy tests for kiteos.rollout.guards (no runtime, LeRobot or torch)."""

from __future__ import annotations

import numpy as np
import pytest

from kiteos.rollout.guards import (
    ChunkRejectedError,
    GuardLimits,
    blend_from_state,
    guard_chunk,
    joint_groups,
)

ARM = [f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 7)]
GRIPPERS = ["left_arm/gripper", "right_arm/gripper"]
JOINTS = [*ARM, *GRIPPERS]
GROUPS = joint_groups(JOINTS, GRIPPERS)
FPS = 30.0
LIMITS = GuardLimits()


def _state() -> np.ndarray:
    state = np.zeros(len(JOINTS), dtype=np.float32)
    state[:12] = np.linspace(-0.5, 0.5, 12)
    state[12:] = (0.8, 0.3)
    return state


def _hold_chunk(state: np.ndarray, steps: int = 20) -> np.ndarray:
    return np.repeat(state[None, :], steps, axis=0)


def _ramp(state: np.ndarray, column: int, per_step: float, *, hold: int = 10) -> np.ndarray:
    """Hold for ``hold`` steps (past the blend), then move one joint at ``per_step``."""
    actions = _hold_chunk(state, hold + 20)
    actions[hold:, column] = state[column] + per_step * np.arange(1, 21)
    return actions


def _reject(
    state: np.ndarray, actions: np.ndarray, *, fps: float = FPS, limits: GuardLimits = LIMITS
) -> ChunkRejectedError:
    with pytest.raises(ChunkRejectedError) as info:
        guard_chunk(state, actions, groups=GROUPS, fps=fps, limits=limits)
    return info.value


def test_defaults_match_the_calibrated_rollout_limits():
    # Calibrated on the validated ACT checkpoint and the teleop data (see guards.GuardLimits).
    assert LIMITS.max_joint_speed_rad_s == 3.0
    assert LIMITS.max_start_jump_rad == 0.3
    assert LIMITS.max_gripper_start_jump == 1.0
    assert LIMITS.max_gripper_speed == 60.0
    assert LIMITS.max_step_delta_rad == 0.1
    assert LIMITS.blend_steps == 10


@pytest.mark.parametrize(
    "field",
    [
        "max_start_jump_rad",
        "max_joint_speed_rad_s",
        "max_step_delta_rad",
        "max_gripper_start_jump",
        "max_gripper_speed",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_limits_reject_non_positive_or_non_finite(field, value):
    with pytest.raises(ValueError, match=field):
        GuardLimits(**{field: value})


@pytest.mark.parametrize("steps", [0, -3, 1.5])
def test_limits_reject_bad_blend_steps(steps):
    with pytest.raises(ValueError, match="blend_steps"):
        GuardLimits(blend_steps=steps)


def test_joint_groups_split_arm_and_gripper_columns():
    assert GROUPS.arm.tolist() == list(range(12))
    assert GROUPS.gripper.tolist() == [12, 13]
    assert GROUPS.names == tuple(JOINTS)


@pytest.mark.parametrize(
    ("names", "grippers", "message"),
    [
        (["a", "a"], [], "duplicates"),
        (["a", "g"], ["g", "g"], "duplicates"),
        (["a"], ["g"], "not in joint_names"),
        (["g"], ["g"], "at least one arm joint"),
    ],
)
def test_joint_groups_reject_inconsistent_names(names, grippers, message):
    with pytest.raises(ValueError, match=message):
        joint_groups(names, grippers)


def test_holding_chunk_passes_unchanged():
    state = _state()
    actions = _hold_chunk(state)
    chunk = guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    np.testing.assert_allclose(chunk.start, state)
    np.testing.assert_allclose(chunk.actions, actions)
    assert chunk.start.dtype == np.float32
    assert chunk.actions.dtype == np.float32


def test_blend_spreads_the_start_offset_over_blend_steps():
    state = np.zeros(3, dtype=np.float32)
    actions = np.ones((12, 3), dtype=np.float32)
    blended = blend_from_state(state, actions, 4)
    np.testing.assert_allclose(blended[:4, 0], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(blended[4:], actions[4:])
    # Short chunks blend over their own length.
    np.testing.assert_allclose(blend_from_state(state, actions[:2], 10)[:, 0], [0.5, 1.0])


def test_small_start_offset_is_blended_not_rejected():
    state = _state()
    actions = _hold_chunk(state)
    actions[:, 0] += 0.15
    chunk = guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    np.testing.assert_allclose(np.diff(chunk.actions[:10, 0]), 0.015, atol=1e-6)
    assert chunk.actions[0, 0] == pytest.approx(state[0] + 0.015, abs=1e-6)


def test_arm_start_jump_is_refused():
    state = _state()
    actions = _hold_chunk(state)
    actions[:, 3] += 0.35
    error = _reject(state, actions)
    assert error.check == "arm_start_jump"
    assert error.joint == "left_joint4"
    assert error.value == pytest.approx(0.35, abs=1e-6)
    assert error.limit == 0.3
    assert "KiteOS" in str(error)


def test_arm_start_jump_limit_is_configurable():
    state = _state()
    actions = _hold_chunk(state)
    actions[:, 3] += 0.25
    limits = GuardLimits(max_start_jump_rad=0.3, max_step_delta_rad=0.3, max_joint_speed_rad_s=10)
    guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=limits)


def test_gripper_start_jump_is_refused_and_configurable():
    state = _state()
    actions = _hold_chunk(state)
    actions[:, 13] = 0.9  # measured 0.3
    # A full-range jump (what the Quest trigger does) passes by default ...
    guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    # ... and a stricter limit refuses it.
    error = _reject(state, actions, limits=GuardLimits(max_gripper_start_jump=0.5, max_gripper_speed=100))
    assert error.check == "gripper_start_jump"
    assert error.joint == "right_arm/gripper"
    assert error.value == pytest.approx(0.6, abs=1e-6)
    guard_chunk(
        state,
        actions,
        groups=GROUPS,
        fps=FPS,
        limits=GuardLimits(max_gripper_start_jump=0.7, max_gripper_speed=100),
    )


def test_gripper_rate_is_refused_and_configurable():
    state = _state()
    # Left gripper measured 0.8: a full close in one step (Quest trigger, 30 opening/s) passes.
    snap = _hold_chunk(state)
    snap[1:, 12] = 0.0
    guard_chunk(state, snap, groups=GROUPS, fps=FPS, limits=LIMITS)
    # 0.1 per step = 3 opening/s is refused by a stricter limit.
    fast = _ramp(state, 12, -0.1)
    fast[:, 12] = np.clip(fast[:, 12], 0.0, 1.0)
    strict = GuardLimits(max_gripper_speed=2.0)
    error = _reject(state, fast, limits=strict)
    assert error.check == "gripper_speed"
    assert error.joint == "left_arm/gripper"
    assert error.value == pytest.approx(3.0, rel=1e-3)
    guard_chunk(state, fast, groups=GROUPS, fps=FPS, limits=GuardLimits(max_gripper_speed=3.1))


def test_blending_counts_toward_the_gripper_rate():
    state = _state()
    # The first target is only 0.4 from the measured opening, but the chunk keeps
    # closing, so the blended path is faster than 2 opening/s.
    actions = _hold_chunk(state)
    actions[:, 12] = np.clip(state[12] - 0.1 * np.arange(1, 21), 0.0, 1.0)
    assert _reject(state, actions, limits=GuardLimits(max_gripper_speed=2.0)).check == "gripper_speed"


def test_arm_speed_is_refused_above_the_limit():
    state = _state()
    # 0.07 rad per step at 30 fps = 2.1 rad/s (under the 0.1 rad per-step limit): passes the
    # 3.0 rad/s default (the validated ACT policy reaches 2.05 rad/s), refused at 2.0.
    actions = _ramp(state, 5, 0.07)
    guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    error = _reject(state, actions, limits=GuardLimits(max_joint_speed_rad_s=2.0))
    assert error.check == "arm_speed"
    assert error.joint == "left_joint6"
    assert error.value == pytest.approx(2.1, rel=1e-3)
    assert error.limit == 2.0
    # 0.06 rad per step = 1.8 rad/s passes.
    guard_chunk(state, _ramp(state, 5, 0.06), groups=GROUPS, fps=FPS, limits=LIMITS)


def test_arm_speed_scales_with_execution_rate():
    state = _state()
    actions = _ramp(state, 7, 0.06)
    guard_chunk(state, actions, groups=GROUPS, fps=15.0, limits=LIMITS)
    guard_chunk(state, actions, groups=GROUPS, fps=30.0, limits=LIMITS)
    error = _reject(state, actions, fps=60.0)
    assert error.check == "arm_speed"


def test_arm_step_delta_is_refused_independent_of_playback_speed():
    state = _state()
    # One 0.12 rad discontinuity mid-chunk; at 10 fps that is only 1.2 rad/s.
    actions = _hold_chunk(state)
    actions[15:, 8] += 0.12
    error = _reject(state, actions, fps=10.0)
    assert error.check == "arm_step_delta"
    assert error.joint == "right_joint3"
    assert error.value == pytest.approx(0.12, abs=1e-6)
    guard_chunk(
        state, actions, groups=GROUPS, fps=10.0, limits=GuardLimits(max_step_delta_rad=0.15)
    )


@pytest.mark.parametrize("value", [-0.05, 1.05])
def test_gripper_targets_outside_the_normalized_range_are_refused(value):
    state = _state()
    actions = _hold_chunk(state)
    actions[:, 12] = value
    limits = GuardLimits(max_gripper_start_jump=1.0, max_gripper_speed=100)
    error = _reject(state, actions, limits=limits)
    assert error.check == "gripper_range"
    assert error.joint == "left_arm/gripper"


def test_measured_gripper_slightly_out_of_range_is_clamped_in_the_trajectory():
    state = _state()
    state[12] = 1.004
    actions = _hold_chunk(state)
    actions[:, 12] = 1.0
    chunk = guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    assert chunk.start[12] == pytest.approx(1.0)
    assert np.all(chunk.actions[:, 12] <= 1.0)


@pytest.mark.parametrize("where", ["state", "actions"])
def test_non_finite_values_are_refused(where):
    state = _state()
    actions = _hold_chunk(state)
    if where == "state":
        state[2] = np.nan
    else:
        actions[4, 2] = np.inf
    assert _reject(state, actions).check == "non_finite"


@pytest.mark.parametrize("actions_shape", [(0, 14), (5, 13), (14,), (2, 5, 14)])
def test_wrong_action_shapes_are_refused(actions_shape):
    state = _state()
    assert _reject(state, np.zeros(actions_shape, dtype=np.float32)).check == "shape"


def test_wrong_state_shape_is_refused():
    assert _reject(np.zeros(12), np.zeros((3, 14))).check == "shape"


@pytest.mark.parametrize("fps", [0.0, -30.0, float("nan"), float("inf")])
def test_bad_execution_rate_is_refused(fps):
    state = _state()
    assert _reject(state, _hold_chunk(state), fps=fps).check == "fps"


def test_rejection_is_a_runtime_error_so_the_rollout_loop_stops():
    # dimos_lerobot's rollout loop catches Exception, records it and cancels the trajectory.
    assert issubclass(ChunkRejectedError, RuntimeError)


def test_inputs_are_not_modified():
    state = _state()
    state[12] = 1.004
    actions = _ramp(state, 0, 0.01)
    actions[:, 12] = 1.0
    state_before, actions_before = state.copy(), actions.copy()
    guard_chunk(state, actions, groups=GROUPS, fps=FPS, limits=LIMITS)
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(actions, actions_before)


def test_arm_only_policy_without_grippers():
    groups = joint_groups(ARM, [])
    state = np.zeros(12, dtype=np.float32)
    chunk = guard_chunk(state, np.zeros((4, 12)), groups=groups, fps=FPS, limits=LIMITS)
    assert chunk.actions.shape == (4, 12)



def test_validated_act_policy_first_chunks_pass_the_guards():
    """Regression: the review found the first chunk of every evaluated episode refused.

    Replays the checkpoint's saved open-loop eval chunks (clipped to the gripper range like the runtime)
    from the recorded states at 1.0x and 0.3x. Skips when the local checkpoint/dataset is absent.
    """
    import json
    from pathlib import Path

    traces = Path("~/teleop_stack/policies/act-close-box-step30000/eval/traces.json").expanduser()
    parquet = Path("~/teleop_stack/dimos/datasets/close-box-001/data/chunk-000/file-000.parquet").expanduser()
    if not traces.is_file() or not parquet.is_file():
        pytest.skip("local ACT eval traces / dataset not present")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(parquet).to_pandas()
    runs = json.loads(traces.read_text())["episodes"]
    for fps in (30.0, 9.0):
        rejected = 0
        for episode, run in runs.items():
            pred = np.asarray(run["pred"], dtype=np.float32)
            pred[:, 12:] = np.clip(pred[:, 12:], 0.0, 1.0)
            state = np.stack(table[table.episode_index == int(episode)]["observation.state"].to_numpy())
            guard_chunk(state[0], pred[:100], groups=GROUPS, fps=fps, limits=LIMITS)  # first chunk
            for start in range(100, len(pred) - 1, 100):
                try:
                    guard_chunk(state[start], pred[start : start + 100], groups=GROUPS, fps=fps, limits=LIMITS)
                except ChunkRejectedError:
                    rejected += 1
        assert rejected <= 1, (fps, rejected)  # legacy kite_rollout guards also refused 1 of 38
