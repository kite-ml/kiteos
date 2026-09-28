# Copyright 2026 Kite ML.
"""FencedWholeBodyAdapter: fence enforcement at the runtime's whole-body adapter boundary.

The first group drives the adapter with a fake inner adapter (nothing from the runtime needed beyond the
message types). The last test runs the runtime's real MockWholeBodyAdapter behind ConnectedWholeBody, the
exact path the coordinator's tick loop uses, and checks the arm never crosses the fence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.runtime
pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.hardware.whole_body.spec import (  # noqa: E402
    POS_STOP,
    IMUState,
    MotorCommand,
    MotorState,
    WholeBodyAdapter,
)
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402

import kiteos.safety.adapter as adapter_module  # noqa: E402
from kiteos.safety.adapter import (  # noqa: E402
    ARM_HOLD_KD,
    ARM_HOLD_KP,
    FencedWholeBodyAdapter,
    FenceEvent,
)
from kiteos.safety.fence import FenceChecker, FenceConfig  # noqa: E402
from kiteos.safety.kinematics import ARM_JOINTS, DUAL_OPENYAM_JOINTS  # noqa: E402
from kiteos.safety.zones import BoxZone, HalfSpaceZone  # noqa: E402

HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
# Forward reach: the left fingertip ends ~18 cm beyond a glass at x = 0.45.
REACH = [0.0, 2.0, 2.0, 0.0, 0.0, 0.0]
# Left arm too close to the glass (inside the 3 cm margin, not touching it).
INSIDE = [0.0, 1.2, 1.047, 0.0, 0.0, 0.0]
# Joint 2 back by 0.02 rad: moves the gripper away from the glass.
ESCAPE = [0.0, 1.18, 1.047, 0.0, 0.0, 0.0]
DEEPER = [0.0, 1.22, 1.047, 0.0, 0.0, 0.0]
GLASS = HalfSpaceZone(name="glass", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
# Placed so that only the ESCAPE pose (not INSIDE) comes within the margin of it.
SHELF = BoxZone(name="shelf", center=(0.1618, 0.31, 0.5028), half_extents=(0.01, 0.01, 0.01))
WB_KP = (*ARM_HOLD_KP, *ARM_HOLD_KP, 0.0, 0.0)
WB_KD = (*ARM_HOLD_KD, *ARM_HOLD_KD, 0.0, 0.0)
LEFT = slice(0, 6)
RIGHT = slice(6, 12)


def q14(
    left: Sequence[float] = HOME, right: Sequence[float] = HOME, grip=(0.0, 0.0)
) -> list[float]:
    return [*map(float, left), *map(float, right), *map(float, grip)]


def commands(q: Sequence[float], *, kp=WB_KP, kd=WB_KD, dq: float = 0.0) -> list[MotorCommand]:
    """Commands as the runtime's ConnectedWholeBody builds them (dq = 0, tau = 0, wb_config gains)."""
    return [
        MotorCommand(q=float(v), dq=dq, kp=float(p), kd=float(d), tau=0.0)
        for v, p, d in zip(q, kp, kd, strict=True)
    ]


def sent_q(cmds: Sequence[MotorCommand]) -> np.ndarray:
    return np.array([c.q for c in cmds])


class FakeClock:
    def __init__(self, step: float = 0.01) -> None:
        self.t = 100.0
        self.step = step

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float | None = None) -> None:
        self.t += self.step if dt is None else dt


class FakeInner:
    """Minimal whole-body adapter: records writes; motors follow commands like the runtime's mock."""

    def __init__(self, positions: Sequence[float]) -> None:
        self.states = [MotorState(q=float(q)) for q in positions]
        self.writes: list[list[MotorCommand]] = []
        self.lifecycle: list[str] = []
        self.connected = False
        self.write_result = True
        self.reads = 0

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def activate(self) -> bool:
        self.lifecycle.append("activate")
        return True

    def deactivate(self) -> bool:
        self.lifecycle.append("deactivate")
        return True

    def read_motor_states(self) -> list[MotorState]:
        self.reads += 1
        return list(self.states)

    def has_motor_states(self) -> bool:
        return self.connected

    def read_imu(self) -> IMUState:
        return IMUState(rpy=(0.1, 0.2, 0.3))

    def get_limits(self) -> None:
        return None

    def write_motor_commands(self, cmds: list[MotorCommand]) -> bool:
        self.writes.append(list(cmds))
        if self.write_result:
            self.states = [MotorState(q=c.q) for c in cmds]
        return self.write_result


class Rig:
    def __init__(
        self,
        config: FenceConfig,
        positions: Sequence[float] | None = None,
        *,
        activate: bool = True,
        allow_monitor: bool = True,
    ) -> None:
        self.clock = FakeClock()
        self.inner = FakeInner(positions if positions is not None else q14())
        self.events: list[FenceEvent] = []
        self.fenced = FencedWholeBodyAdapter(
            self.inner,
            joint_names=list(DUAL_OPENYAM_JOINTS),
            checker=FenceChecker(config),
            on_event=self.events.append,
            clock=self.clock,
            allow_monitor=allow_monitor,
        )
        assert self.fenced.connect()
        if activate:
            assert self.fenced.activate()

    def write(self, q: Sequence[float], **kw: Any) -> list[MotorCommand]:
        self.clock.tick()
        assert self.fenced.write_motor_commands(commands(q, **kw))
        return self.inner.writes[-1]

    def kinds(self, side: str | None = None) -> list[str]:
        return [e.kind for e in self.events if side is None or e.side == side]


def glass_config(**kw: Any) -> FenceConfig:
    kw.setdefault("max_joint_speed_rad_s", 1000.0)  # rate limit out of the way unless tested
    return FenceConfig(margin_m=0.03, zones=[GLASS], **kw)


# --------------------------------------------------------------------------- basics


def test_implements_the_runtime_protocol_and_delegates():
    rig = Rig(FenceConfig(), activate=False)
    assert isinstance(rig.fenced, WholeBodyAdapter)
    assert rig.fenced.is_connected() and rig.fenced.has_motor_states()
    assert rig.fenced.read_imu().rpy == (0.1, 0.2, 0.3)
    assert rig.fenced.get_limits() is None
    assert [s.q for s in rig.fenced.read_motor_states()] == q14()
    rig.fenced.disconnect()
    assert not rig.inner.connected


def test_joint_names_must_cover_both_arms():
    with pytest.raises(ValueError, match="missing arm joints"):
        FencedWholeBodyAdapter(
            FakeInner(q14()),
            joint_names=DUAL_OPENYAM_JOINTS[:6],
            checker=FenceChecker(FenceConfig()),
        )


def test_safe_command_passes_unchanged():
    rig = Rig(glass_config())
    cmds = commands(q14(right=[0.1, 1.0, 1.1, 0.0, 0.0, 0.0], grip=(0.1, 0.05)))
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(cmds)
    assert rig.inner.writes[-1] == cmds
    assert "hold" not in rig.kinds()
    np.testing.assert_allclose(rig.fenced.last_written("right"), [0.1, 1.0, 1.1, 0, 0, 0])


def test_command_into_half_space_holds_at_last_safe_target():
    rig = Rig(glass_config())
    out = rig.write(q14(left=REACH))
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)
    held = out[LEFT]
    assert [c.kp for c in held] == list(ARM_HOLD_KP)
    assert [c.kd for c in held] == list(ARM_HOLD_KD)
    assert all(c.dq == 0.0 and c.tau == 0.0 for c in held)
    hold = [e for e in rig.events if e.kind == "hold"]
    assert hold and hold[-1].side == "left" and hold[-1].zone == "glass"
    assert hold[-1].clearance_m is not None and hold[-1].clearance_m < 0.03
    assert rig.fenced.status()["arms"]["left"]["holding"] is True


def test_hold_keeps_the_callers_stiffer_gains():
    rig = Rig(glass_config())
    kp = (200.0, 0.0, 80.0, 50.0, 1.0, 10.0, *ARM_HOLD_KP, 0.0, 0.0)
    out = rig.write(q14(left=REACH), kp=kp)
    assert [c.kp for c in out[LEFT]] == [200.0, 80.0, 80.0, 50.0, 10.0, 10.0]


def test_arms_are_fenced_independently():
    rig = Rig(glass_config())
    right = [0.2, 1.0, 1.2, 0.0, 0.1, 0.0]
    out = rig.write(q14(left=REACH, right=right))
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)
    np.testing.assert_allclose(sent_q(out)[RIGHT], right)
    assert rig.kinds("right") == []


def test_gripper_opening_into_the_fence_is_refused_and_closing_always_passes():
    rig = Rig(glass_config())
    # Home is clear with the gripper closed (4.1 cm), but open fingers reach the glass (-0.9 cm):
    # the fence keeps the grippers at their last written opening instead of opening them.
    out = rig.write(q14(grip=(1.0, 0.5)))
    assert sent_q(out)[12:].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)
    hold = [e for e in rig.events if e.kind == "hold" and e.side == "left"]
    assert hold and "gripper opening refused" in (hold[-1].detail or "")
    assert hold[-1].zone == "glass" and hold[-1].new_episode
    assert rig.fenced.status()["arms"]["left"]["gripper_opening"] == 0.0
    # A small opening that stays clear (0.1 -> +0.5 cm of finger growth) passes, closing too.
    out = rig.write(q14(grip=(0.1, 0.1)))
    assert sent_q(out)[12:].tolist() == [0.1, 0.1]
    out = rig.write(q14(grip=(0.0, 0.0)))
    assert sent_q(out)[12:].tolist() == [0.0, 0.0]
    # Away from the glass, a full opening passes.
    rig2 = Rig(glass_config(), q14(left=[0.0, 0.6, 0.6, 0.0, 0.0, 0.0]))
    out = rig2.write(q14(left=[0.0, 0.6, 0.6, 0.0, 0.0, 0.0], grip=(1.0, 0.0)))
    assert sent_q(out)[12] == 1.0 and "hold" not in rig2.kinds("left")


# --------------------------------------------------------------------------- escape rule


def test_escape_out_of_a_violation_is_accepted():
    rig = Rig(glass_config(refuse_start_inside=False), q14(left=INSIDE))
    assert rig.kinds("left") == ["start_inside"]
    out = rig.write(q14(left=ESCAPE))
    np.testing.assert_allclose(sent_q(out)[LEFT], ESCAPE)
    assert rig.kinds("left")[-1] == "escape"


def test_moving_deeper_or_sideways_while_inside_is_held():
    rig = Rig(glass_config(refuse_start_inside=False), q14(left=INSIDE))
    out = rig.write(q14(left=DEEPER))
    np.testing.assert_allclose(sent_q(out)[LEFT], INSIDE)
    assert rig.kinds("left")[-1] == "hold"
    # Rolling the wrist keeps the fingers where they are: no strict improvement, so no escape.
    out = rig.write(q14(left=[*INSIDE[:5], 0.3]))
    np.testing.assert_allclose(sent_q(out)[LEFT], INSIDE)
    assert rig.kinds("left")[-1] == "hold"


def test_escape_cannot_enter_another_zone():
    shelf_only = FenceChecker(FenceConfig(margin_m=0.03, zones=[SHELF]))
    assert shelf_only.check_arm("left", INSIDE, 0.0).ok  # precondition: shelf clear at start
    assert not shelf_only.check_arm("left", ESCAPE, 0.0).ok  # ... but not at the escape pose

    config = FenceConfig(
        margin_m=0.03, zones=[GLASS, SHELF], refuse_start_inside=False, max_joint_speed_rad_s=1000
    )
    rig = Rig(config, q14(left=INSIDE))
    out = rig.write(q14(left=ESCAPE))
    np.testing.assert_allclose(sent_q(out)[LEFT], INSIDE)
    assert rig.kinds("left")[-1] == "hold"


# --------------------------------------------------------------------------- limits


def test_jump_is_rate_limited_along_the_requested_direction():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=2.5))
    target = [0.5, 1.047 + 0.25, 1.047, 0.0, 0.0, 0.0]
    out = rig.write(q14(left=target))  # clock advanced 10 ms -> 0.025 rad max step
    np.testing.assert_allclose(sent_q(out)[LEFT], [0.025, 1.047 + 0.0125, 1.047, 0, 0, 0])
    assert "rate_limit" in rig.kinds("left")
    # A stalled tick (5 s) cannot unlock a bigger step than the 20 ms cap allows.
    rig.clock.tick(5.0)
    assert rig.fenced.write_motor_commands(commands(q14(left=target)))
    step = sent_q(rig.inner.writes[-1])[LEFT] - sent_q(out)[LEFT]
    assert np.max(np.abs(step)) == pytest.approx(2.5 * 0.02)


def test_out_of_limit_target_is_clamped():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    upper = rig.fenced.checker.joint_limits["left"][0, 1]
    out = rig.write(q14(left=[3.5, *HOME[1:]]))
    assert out[0].q == pytest.approx(upper - 0.01)
    assert "limit_clamp" in rig.kinds("left")


def test_velocity_feedforward_is_bounded():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=2.5))
    out = rig.write(q14(), dq=40.0)
    assert {c.dq for c in out[:12]} == {2.5}
    assert {c.dq for c in out[12:]} == {40.0}  # grippers untouched


# --------------------------------------------------------------------------- fail closed


def test_non_finite_or_sentinel_target_holds():
    rig = Rig(glass_config())
    out = rig.write(q14(left=[0.0, 1.047, math.nan, 0.0, 0.0, 0.0], right=[0.1, *HOME[1:]]))
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)
    np.testing.assert_allclose(sent_q(out)[RIGHT], [0.1, *HOME[1:]])
    assert rig.kinds("left") == ["hold"]
    out = rig.write(q14(left=[POS_STOP, *HOME[1:]]))
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)


def test_exception_in_the_checker_holds_both_arms(monkeypatch):
    rig = Rig(glass_config())
    rig.write(q14(left=[0.05, *HOME[1:]], right=[-0.05, *HOME[1:]]))
    before = {side: rig.fenced.last_written(side) for side in ("left", "right")}
    log = MagicMock()
    monkeypatch.setattr(adapter_module, "logger", log)

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(rig.fenced.checker, "check_arm_motion", boom)
    for i in range(5):  # 50 ms of ticks (a moving target): one log line, five holds
        step = 0.1 + 0.001 * i
        out = rig.write(q14(left=[step, *HOME[1:]], right=[-step, *HOME[1:]], grip=(0.2, 0.3)))
        np.testing.assert_allclose(sent_q(out)[LEFT], before["left"])
        np.testing.assert_allclose(sent_q(out)[RIGHT], before["right"])
        assert sent_q(out)[12:].tolist() == [0.0, 0.0]  # grippers hold too (fail closed)
        assert all(c.kp >= kp for c, kp in zip(out[:12], WB_KP, strict=False))
    assert rig.kinds().count("error") == 5
    assert log.error.call_count == 1

    monkeypatch.setattr(rig.fenced.checker, "clamp_to_limits", boom)
    out = rig.write(q14(left=[0.2, *HOME[1:]]))
    np.testing.assert_allclose(sent_q(out)[LEFT], before["left"])


def test_write_before_any_state_is_refused():
    rig = Rig(FenceConfig(), activate=False)
    assert rig.fenced.write_motor_commands(commands(q14())) is False
    assert rig.inner.writes == []


def test_wrong_command_count_is_refused():
    rig = Rig(FenceConfig())
    assert rig.fenced.write_motor_commands(commands(q14())[:12]) is False
    assert rig.inner.writes == []
    assert rig.kinds() == ["error"]


def test_inner_rejection_keeps_the_last_written_target():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    rig.inner.write_result = False
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(commands(q14(left=[0.2, *HOME[1:]]))) is False
    np.testing.assert_allclose(rig.fenced.last_written("left"), HOME)


def test_inner_write_exception_never_escapes():
    rig = Rig(FenceConfig())
    rig.inner.write_motor_commands = MagicMock(side_effect=OSError("bus off"))
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(commands(q14())) is False
    assert rig.kinds()[-1] == "error"


def test_failing_event_handler_is_contained():
    rig = Rig(glass_config())
    rig.fenced._on_event = MagicMock(side_effect=RuntimeError("handler bug"))
    out = rig.write(q14(left=REACH))
    np.testing.assert_allclose(sent_q(out)[LEFT], HOME)


# --------------------------------------------------------------------------- estop / monitor


def test_estop_holds_both_arms_until_cleared():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=2.5))
    rig.fenced.latch_estop(True)
    assert rig.fenced.estopped and rig.kinds() == ["estop"]
    out = rig.write(q14(left=[0.1, *HOME[1:]], right=[-0.1, *HOME[1:]], grip=(0.7, 0.0)))
    np.testing.assert_allclose(sent_q(out)[:12], q14()[:12])
    assert sent_q(out)[12] == 0.0  # the grippers are frozen too
    assert rig.fenced.status()["estop"] is True

    rig.fenced.latch_estop(False)
    # The request seen during the E-STOP is stale: it stays refused after clearing ...
    out = rig.write(q14(left=[0.1, *HOME[1:]], right=[-0.1, *HOME[1:]]))
    np.testing.assert_allclose(sent_q(out)[:12], q14()[:12])
    # ... and a new request resumes, rate limited from the hold.
    out = rig.write(q14(left=[0.1001, *HOME[1:]]))
    assert out[0].q == pytest.approx(0.025)


def test_estop_holds_where_the_arm_is():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    rig.write(q14(left=[0.3, *HOME[1:]]))
    rig.inner.states[0] = MotorState(q=0.2)  # arm lags its target
    rig.fenced.read_motor_states()
    rig.fenced.latch_estop(True)
    out = rig.write(q14(left=[0.3, *HOME[1:]]))
    assert out[0].q == pytest.approx(0.2)


def test_monitor_mode_passes_through_and_reports():
    rig = Rig(glass_config(mode="monitor"))
    cmds = commands(q14(left=REACH))
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(cmds)
    assert rig.inner.writes[-1] == cmds
    hold = [e for e in rig.events if e.kind == "hold"]
    assert hold and not hold[-1].enforced
    assert rig.fenced.status()["mode"] == "monitor"


def test_monitor_mode_is_refused_when_not_allowed():
    with pytest.raises(ValueError, match="monitor"):
        Rig(glass_config(mode="monitor"), allow_monitor=False)
    rig = Rig(glass_config(), allow_monitor=False)
    with pytest.raises(ValueError, match="monitor"):
        rig.fenced.reload(glass_config(mode="monitor"))
    assert rig.fenced.mode == "enforce"


# --------------------------------------------------------------------------- activation


def test_activate_refuses_to_start_inside_a_zone():
    rig = Rig(glass_config(), q14(left=INSIDE), activate=False)
    assert rig.fenced.activate() is False
    assert rig.inner.lifecycle == ["activate", "deactivate"]
    assert rig.kinds("left") == ["start_inside"]
    assert "left arm" in (rig.fenced.refusal or "") and "glass" in (rig.fenced.refusal or "")


def test_activate_initialises_last_written_from_measured():
    rig = Rig(glass_config(), q14(left=[0.1, *HOME[1:]]), activate=False)
    assert rig.fenced.last_written("left") is None
    assert rig.fenced.activate() is True
    np.testing.assert_allclose(rig.fenced.last_written("left"), [0.1, *HOME[1:]])
    assert rig.kinds() == []


def test_activate_fails_closed_when_state_is_unreadable():
    rig = Rig(FenceConfig(), q14(left=[math.nan, *HOME[1:]]), activate=False)
    assert rig.fenced.activate() is False
    assert rig.inner.lifecycle == ["activate", "deactivate"]


# --------------------------------------------------------------------------- extras


def test_rewrite_applied_reports_what_was_written():
    rig = Rig(glass_config())
    assert rig.fenced.rewrite_applied(JointState(name=["left_joint1"], position=[1.0])) is None
    other = JointState(name=["base/vx"], position=[0.5])
    assert rig.fenced.rewrite_applied(other) is other  # not this adapter's joints
    rig.write(q14(left=REACH, grip=(0.1, 0.0)))
    names = [*ARM_JOINTS["left"], "left_arm/gripper", "unrelated"]
    msg = JointState(ts=12.5, frame_id="coordinator", name=names, position=[*REACH, 0.1, 9.0])
    out = rig.fenced.rewrite_applied(msg)
    assert out is not None and out is not msg
    assert out.name == names and out.ts == 12.5 and out.frame_id == "coordinator"
    np.testing.assert_allclose(out.position, [*HOME, 0.1, 9.0])
    # A refused gripper opening is recorded as the opening actually written.
    rig.write(q14(left=REACH, grip=(1.0, 0.0)))
    msg = JointState(ts=13.0, frame_id="coordinator", name=names, position=[*REACH, 1.0, 9.0])
    out = rig.fenced.rewrite_applied(msg)
    assert out is not None
    np.testing.assert_allclose(out.position, [*HOME, 0.1, 9.0])
    assert msg.position[1] == 2.0  # the original is not mutated


def test_reload_swaps_the_fence_atomically():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    rig.write(q14(left=[0.0, 1.06, 1.047, 0.0, 0.0, 0.0]))
    assert rig.kinds() == []
    rig.fenced.reload(glass_config())
    out = rig.write(q14(left=REACH))
    np.testing.assert_allclose(sent_q(out)[LEFT], [0.0, 1.06, 1.047, 0, 0, 0])
    assert rig.fenced.status()["zones"][0]["name"] == "glass"


def test_status_is_plain_yaml():
    rig = Rig(glass_config())
    rig.write(q14(left=REACH))
    status = rig.fenced.status()
    text = yaml.safe_dump(status)
    assert "glass" in text
    assert status["holds"] >= 1 and status["events"]["hold"] >= 1
    assert status["last_hold"]["zone"] == "glass"
    assert set(status["arms"]) == {"left", "right"}


def test_check_positions_and_waypoints():
    rig = Rig(glass_config())
    named = dict(zip(DUAL_OPENYAM_JOINTS, q14(left=REACH), strict=True))
    assert rig.fenced.check_positions(named)["ok"] is False
    named = dict(zip(DUAL_OPENYAM_JOINTS, q14(), strict=True))
    assert rig.fenced.check_positions(named)["ok"] is True

    names = ARM_JOINTS["left"]
    path = [HOME, [0.0, 1.5, 1.5, 0.0, 0.0, 0.0], REACH]
    report = rig.fenced.check_waypoints(names, path)
    assert not report.ok and not report.arms["left"].ok
    assert "waypoint" in (report.arms["left"].detail or "")
    assert report.arms["right"].ok and report.arms["right"].detail == "not moved by this motion"
    assert rig.fenced.check_waypoints(names, [HOME, [0.2, *HOME[1:]]]).ok
    # A gripper column opening the fingers at home is also caught.
    names14 = list(DUAL_OPENYAM_JOINTS)
    assert not rig.fenced.check_waypoints(names14, [q14(), q14(grip=(1.0, 0.0))]).ok
    # Malformed rows never raise.
    assert not rig.fenced.check_waypoints(names, [[0.0, 1.0]]).ok


# --------------------------------------------------------------------------- the runtime's mock, end to end


def test_runtime_mock_through_connected_whole_body_never_crosses_the_plane():
    from dimos.control.hardware_interface import ConnectedWholeBody
    from dimos.hardware.manipulators.spec import ControlMode
    from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
    from dimos.robot.manipulators.dual_openyam.config import dual_openyam_mock_hardware

    component = dual_openyam_mock_hardware()
    config = FenceConfig(margin_m=0.03, zones=[GLASS])  # default 2.5 rad/s rate limit
    inner = MockWholeBodyAdapter(dof=len(component.joints), **component.adapter_kwargs)
    clock = FakeClock(step=0.01)
    events: list[FenceEvent] = []
    fenced = FencedWholeBodyAdapter(
        inner,
        joint_names=component.joints,
        checker=FenceChecker(config),
        on_event=events.append,
        clock=clock,
    )
    assert fenced.connect() and fenced.activate()
    hardware = ConnectedWholeBody(fenced, component)
    referee = FenceChecker(config)  # independent checker on what the "motors" did
    arm_names = [*ARM_JOINTS["left"], *ARM_JOINTS["right"]]

    def drive(start: Sequence[float], end: Sequence[float], ticks: int) -> list[np.ndarray]:
        seen = []
        for alpha in np.linspace(0.0, 1.0, ticks):
            target = (1 - alpha) * np.asarray(start) + alpha * np.asarray(end)
            clock.tick()
            hardware.read_state()
            assert hardware.write_command(
                dict(zip(arm_names, [*target, *target], strict=True)), ControlMode.SERVO_POSITION
            )
            measured = [s.q for s in inner.read_motor_states()]
            report = referee.check(measured)
            assert report.ok, report.to_dict()
            seen.append(np.asarray(measured[:12]))
        return seen

    # Cross the plane fast (targets jump ahead of the rate limit) and slow.
    seen = drive(HOME, REACH, 40) + drive(REACH, REACH, 20)
    seen += drive(HOME, [0.0, 1.6, 1.3, 0.0, 0.0, 0.0], 400)
    steps = np.abs(np.diff(np.stack(seen), axis=0)).max()
    assert steps <= 2.5 * 0.01 + 1e-9
    assert any(e.kind == "hold" for e in events)
    held = np.asarray([s.q for s in inner.read_motor_states()][:6])
    assert held[1] < 1.2  # stopped short of the glass

    # Driving back is allowed and the arms return home.
    drive(held, HOME, 300)
    np.testing.assert_allclose([s.q for s in inner.read_motor_states()][:12], q14()[:12], atol=1e-9)


# --------------------------------------------------------------------------- review regressions

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
WALL = HalfSpaceZone(name="wall", point=(0.0, 0.50, 0.0), normal=(0.0, 1.0, 0.0))


def _body_clearance(checker: FenceChecker, zone: Any, q6: Sequence[float]) -> float:
    """Clearance of the arm with the gripper CLOSED (independent of the finger model)."""
    from kiteos.safety.zones import compile_zone

    compiled = compile_zone(zone, checker.frame_T)
    spheres = checker.spheres("left", q6, 0.0)
    return float(compiled.clearance(spheres.centers, spheres.radii).min())


def test_gripper_toggle_ratchet_cannot_walk_the_arm_through_the_glass():
    """Replay of the review's greedy attack: 161 rate-limited 10 ms commands with 6 gripper
    toggles. Under the old escape rule every arm step was accepted and the closed-gripper arm body
    ended 8.9 cm past the plane. Now the arm never even enters the margin."""
    config = FenceConfig(margin_m=0.03, zones=[GLASS], max_joint_speed_rad_s=2.5)
    rig = Rig(config)
    referee = FenceChecker(config)
    worst = math.inf
    for row in np.load(FIXTURES / "ratchet_gripper_toggle_m003_x045.npy"):
        out = rig.write(q14(left=row[:6], grip=(row[6], 0.0)))
        q = sent_q(out)
        worst = min(worst, _body_clearance(referee, GLASS, q[LEFT]))
        # What is physically commanded (arm + the gripper opening actually sent) stays clear.
        assert referee.check_arm("left", q[LEFT], q[12]).min_clearance_m > 0.03 - 1e-9
    assert worst > 0.03 - 1e-9
    assert "escape" not in rig.kinds("left")


def test_escape_cannot_trade_one_zone_against_another():
    """The review's 5-step 'escape' that drove a side wall from +1.5 cm to -2.7 cm while the
    glass improved by 0.4 cm: every step now holds (the wall pair may not lose clearance)."""
    seq = np.load(FIXTURES / "escape_trade_glass_wall.npy")
    rig = Rig(FenceConfig(max_joint_speed_rad_s=2.5), q14(left=seq[0]))
    config = FenceConfig(margin_m=0.03, zones=[GLASS, WALL], max_joint_speed_rad_s=2.5)
    rig.fenced.reload(config)
    referee = FenceChecker(config)
    before = _body_clearance(referee, WALL, seq[0])
    assert 0.0 < before < 0.03
    for q in seq[1:]:
        out = rig.write(q14(left=q))
        assert _body_clearance(referee, WALL, sent_q(out)[LEFT]) >= before - 1e-9
    assert "escape" not in rig.kinds("left")


def test_escape_keeps_every_pair_within_the_margin_non_decreasing():
    rig = Rig(glass_config(refuse_start_inside=False), q14(left=INSIDE))
    referee = FenceChecker(glass_config())
    samples = referee.motion_samples(np.array(INSIDE), np.array(ESCAPE))
    C = referee.pair_clearances("left", samples, 0.0)
    from kiteos.safety.fence import escape_path_ok

    assert escape_path_ok(C, 0.03)
    out = rig.write(q14(left=ESCAPE))
    np.testing.assert_allclose(sent_q(out)[LEFT], ESCAPE)
    # A synthetic path where one within-margin pair improves while another loses is refused.
    assert not escape_path_ok(np.array([[0.01, 0.02], [0.015, 0.019]]), 0.03)
    assert not escape_path_ok(np.array([[0.01, 0.05], [0.02, 0.029]]), 0.03)  # clear pair enters
    assert escape_path_ok(np.array([[0.01, 0.05], [0.02, 0.04]]), 0.03)
    assert not escape_path_ok(np.array([[0.01, math.nan]]), 0.03)


def test_gripper_cannot_open_while_the_arm_is_inside_and_closing_is_allowed():
    rig = Rig(glass_config(refuse_start_inside=False), q14(left=INSIDE, grip=(0.4, 0.0)))
    out = rig.write(q14(left=INSIDE, grip=(0.8, 0.0)))  # opening further while inside
    assert sent_q(out)[12] == pytest.approx(0.4)
    out = rig.write(q14(left=INSIDE, grip=(0.0, 0.0)))  # closing always passes
    assert sent_q(out)[12] == 0.0


def test_a_held_arm_does_not_move_later_when_only_the_gripper_is_commanded():
    """The runtime's ConnectedWholeBody re-sends the last PRE-fence arm target on partial writes. After a
    hold (gripper open), gripper-only commands closing the fingers used to make that stale target
    valid, and the arm drove 0.2 rad toward the glass with nothing commanding it."""
    from dimos.control.hardware_interface import ConnectedWholeBody
    from dimos.hardware.manipulators.spec import ControlMode
    from dimos.hardware.whole_body.mock.adapter import MockWholeBodyAdapter
    from dimos.robot.manipulators.dual_openyam.config import dual_openyam_mock_hardware

    glass = HalfSpaceZone(name="glass", point=(0.50, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    component = dual_openyam_mock_hardware()
    inner = MockWholeBodyAdapter(dof=len(component.joints), **component.adapter_kwargs)
    clock = FakeClock()
    fenced = FencedWholeBodyAdapter(
        inner,
        joint_names=component.joints,
        checker=FenceChecker(FenceConfig(margin_m=0.03, zones=[glass])),
        clock=clock,
    )
    assert fenced.connect() and fenced.activate()
    hw = ConnectedWholeBody(fenced, component)
    target = np.array(HOME) + np.array([0.0, 0.22, 0.0, 0.0, 0.0, 0.0])

    def tick(cmd: dict[str, float]) -> np.ndarray:
        clock.tick()
        hw.read_state()
        hw.write_command(cmd, ControlMode.SERVO_POSITION)
        return np.array([s.q for s in inner.read_motor_states()])

    tick({"left_arm/gripper": 1.0})
    for _ in range(100):
        held = tick(dict(zip(ARM_JOINTS["left"], target, strict=True)))
    assert held[1] < target[1] - 0.05
    for _ in range(100):
        after = tick({"left_arm/gripper": 0.0})
    np.testing.assert_allclose(after[:6], held[:6], atol=1e-12)
    assert after[12] == 0.0  # the gripper itself closed
    # A NEW arm request is evaluated again (and may now proceed toward the glass).
    moved = tick(dict(zip(ARM_JOINTS["left"], target + 1e-6, strict=True)))
    assert moved[1] > held[1]


def test_hold_episodes_are_flagged_once():
    rig = Rig(glass_config())
    rig.write(q14(left=REACH))
    rig.write(q14(left=[0.0, 2.0, 2.0, 0.0, 0.0, 0.01]))
    holds = [e for e in rig.events if e.kind == "hold"]
    assert [e.new_episode for e in holds] == [True, False]
    rig.write(q14(left=[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))  # accepted: the episode ends
    rig.write(q14(left=REACH))
    holds = [e for e in rig.events if e.kind == "hold"]
    assert [e.new_episode for e in holds] == [True, False, True]


def test_torque_feedforward_is_bounded():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    cmds = [MotorCommand(q=float(v), dq=0.0, kp=80.0, kd=5.0, tau=7.5) for v in q14()]
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(cmds)
    assert {c.tau for c in rig.inner.writes[-1][:12]} == {0.0}  # default: no feed-forward
    rig.fenced.reload(FenceConfig(max_joint_speed_rad_s=1000.0, max_feedforward_tau_nm=2.0))
    cmds = [MotorCommand(q=float(v), dq=0.0, kp=80.0, kd=5.0, tau=-7.5) for v in q14()]
    rig.clock.tick()
    assert rig.fenced.write_motor_commands(cmds)
    assert {c.tau for c in rig.inner.writes[-1][:12]} == {-2.0}


def test_routine_limit_clamp_is_silent():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    lower = rig.fenced.checker.joint_limits["left"][1, 0]  # joint 2 lower limit (0)
    # Teleop IK clamps to limit - 1e-4 itself: the fence's 1 cm band clamps silently.
    out = rig.write(q14(left=[0.0, lower + 1e-4, 1.047, 0.0, 0.0, 0.0]))
    assert out[1].q == pytest.approx(lower + 0.01)
    assert "limit_clamp" not in rig.kinds()
    rig.write(q14(left=[0.0, lower - 0.2, 1.047, 0.0, 0.0, 0.0]))
    assert "limit_clamp" in rig.kinds()


def test_invalid_gripper_command_holds_the_gripper():
    rig = Rig(FenceConfig(max_joint_speed_rad_s=1000.0))
    out = rig.write(q14(grip=(0.3, 0.0)))
    assert sent_q(out)[12] == pytest.approx(0.3)
    out = rig.write(q14(grip=(math.nan, 0.0)))
    assert sent_q(out)[12] == pytest.approx(0.3)
    assert rig.kinds("left")[-1] == "hold"
