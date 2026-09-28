# Copyright 2026 Kite ML.
"""KiteOS fence enforcement at the runtime's whole-body adapter boundary.

Every command source in a runtime ``ControlCoordinator`` (Quest teleop IK, policy trajectories, the
gripper tasks, the ``joint_command`` stream) ends in one ``write_motor_commands`` call per tick.
``FencedWholeBodyAdapter`` wraps the real (or mock) adapter at exactly that point and, per arm:

1. rejects non-finite or implausible targets (hold);
2. holds a *stale* target: an arm request that is bit-for-bit the one it last refused (the runtime
   re-sends the last pre-fence target of every joint a command does not mention) stays refused
   until a different request arrives, so a held arm never drives off by itself later;
3. clamps the target to the URDF limits minus ``joint_limit_margin_rad`` and rate-limits it
   against the last written target (``max_joint_speed_rad_s``);
4. accepts it if the swept joint-space motion from the last written target stays clear of every
   zone, or if it escapes a violation (``FenceChecker.is_escape_motion``: no zone/sphere pair
   within the margin loses clearance, nothing clear enters the margin); otherwise it HOLDS the
   arm at the last written target, which is known safe, with stiff gains and zero velocity /
   torque feed-forward. Accepted commands have their velocity feed-forward bounded by the speed
   limit and their torque feed-forward by ``max_feedforward_tau_nm`` (default 0).

Grippers are fenced too: closing always passes, but an opening increase passes only when the
arm (at its planned target, with the wider fingers) is clear of the fence; otherwise the gripper
holds its last written opening. An E-STOP latch holds both arms and both grippers. ``mode:
monitor`` never modifies a command and only reports. Any internal error holds both arms and both
grippers (fail closed); nothing here raises into the control tick.

The adapter runs on the coordinator's tick thread while the coordinator's non re-entrant
``_hardware_lock`` is held, so ``on_event`` callbacks must be quick and must never call coordinator
RPCs. See ``kiteos.safety.coordinator`` for the supervisor thread that acts on events.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.spec import IMUState, MotorCommand, MotorState, WholeBodyAdapter
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

from kiteos.safety.fence import ArmReport, FenceChecker, FenceConfig, FenceReport
from kiteos.safety.geometry import normalize_opening
from kiteos.safety.kinematics import ARM_JOINTS, ARM_SIDES, DOF_PER_ARM, GRIPPER_JOINTS

logger = setup_logger()

# Dual OpenYAM arm gains (dimos.robot.manipulators.dual_openyam.config _ARM_KP/_ARM_KD). A hold
# never uses less stiffness than these, so a caller sending zero gains cannot make a held arm limp.
ARM_HOLD_KP: tuple[float, ...] = (80.0, 80.0, 80.0, 10.0, 10.0, 10.0)
ARM_HOLD_KD: tuple[float, ...] = (5.0, 5.0, 5.0, 1.5, 1.5, 1.5)

# Bounds on the tick period used by the rate limiter: a stalled tick must not unlock a big jump.
RATE_DT_MIN_S = 0.001
RATE_DT_MAX_S = 0.02

# Arm joint ranges are within +/-3.7 rad; anything beyond this is a sentinel or garbage
# (e.g. the runtime's POS_STOP = 2.146e9), never something to clamp and chase.
MAX_PLAUSIBLE_TARGET_RAD = 4.0 * math.pi

LOG_INTERVAL_S = 1.0

# An opening change smaller than this is not an "opening increase" (float noise).
_OPENING_EPS = 1e-6
# A clamp of at most joint_limit_margin_rad + this is routine (teleop IK clamps to the URDF limit
# itself) and produces no limit_clamp event.
_LIMIT_EVENT_SLACK_RAD = 1e-3

FenceEventKind = Literal[
    "hold", "escape", "limit_clamp", "rate_limit", "estop", "error", "start_inside"
]
EVENT_KINDS: tuple[str, ...] = (
    "hold",
    "escape",
    "limit_clamp",
    "rate_limit",
    "estop",
    "error",
    "start_inside",
)


@dataclass(frozen=True)
class FenceEvent:
    """Something the fence did (or, in monitor mode, would have done).

    ``t`` comes from the adapter's clock (``time.monotonic`` by default). ``side`` is ``"left"``,
    ``"right"`` or ``"both"``. ``enforced`` is False when the event was only observed (monitor mode,
    or a start-inside check that was not configured to refuse).
    """

    t: float
    side: str
    kind: FenceEventKind
    zone: str | None = None
    sphere: str | None = None
    clearance_m: float | None = None
    detail: str | None = None
    enforced: bool = True
    # True for the first hold/error of a hold episode (the side was not holding on the previous
    # write); the coordinator cancels an executing trajectory at most once per episode.
    new_episode: bool = False

    def to_dict(self) -> dict[str, Any]:
        clearance = self.clearance_m
        return {
            "t": float(self.t),
            "side": self.side,
            "kind": self.kind,
            "zone": self.zone,
            "sphere": self.sphere,
            "clearance_m": float(clearance)
            if clearance is not None and math.isfinite(clearance)
            else None,
            "detail": self.detail,
            "enforced": self.enforced,
            "new_episode": self.new_episode,
        }


class _RateLimitedLog:
    """At most one log line per key per interval; counts what it suppressed."""

    def __init__(self, clock: Callable[[], float], interval_s: float = LOG_INTERVAL_S) -> None:
        self._clock = clock
        self._interval = interval_s
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def __call__(self, key: str, level: str, message: str) -> bool:
        try:
            now = float(self._clock())
            last = self._last.get(key)
            if last is not None and 0.0 <= now - last < self._interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return False
            self._last[key] = now
            suppressed = self._suppressed.pop(key, 0)
            if suppressed:
                message = f"{message} ({suppressed} similar messages suppressed)"
            getattr(logger, level)(message)
            return True
        except Exception:  # logging must never break the tick
            return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_list(q: np.ndarray | None) -> list[float] | None:
    return None if q is None else [float(v) for v in q]


def _request_key(cmds: Sequence[Any]) -> bytes | None:
    """Bit-exact key of a raw arm request (for stale-request holds); None if unreadable."""
    try:
        return np.array([float(c.q) for c in cmds], dtype=float).tobytes()
    except Exception:
        return None


def _max_opening(*values: float | None) -> float | None:
    """Most-open known value in [0, 1]; ``None`` (treated as fully open) if none is known."""
    known = [normalize_opening(v) for v in values if v is not None and _finite(v)]
    return max(known) if known else None


class _NoBaseline(RuntimeError):
    """An arm has no known-safe target yet (no state read since connect)."""


@dataclass
class _ArmPlan:
    commands: list[MotorCommand]
    q: np.ndarray  # arm target actually sent
    held: bool
    report: ArmReport | None
    escape: bool = False
    request_key: bytes | None = None  # raw requested arm target (for stale-request holds)
    grip: MotorCommand | None = None  # gripper command actually sent (None: no gripper joint)
    grip_held: bool = False


class FencedWholeBodyAdapter:
    """the runtime ``WholeBodyAdapter`` that enforces the KiteOS fence before the inner adapter.

    Args:
        inner: The adapter to protect (``MockWholeBodyAdapter``, the Damiao adapter, ...).
        joint_names: The component's joint order (``HardwareComponent.joints``); must contain the
            12 arm joints. Joints the fence does not know pass through untouched.
        checker: The fence to enforce; swap it at runtime with ``reload``.
        on_event: Called with each ``FenceEvent``. Runs on the tick thread under the coordinator's
            hardware lock: keep it quick and never call coordinator RPCs from it.
        clock: Monotonic time source (seconds), injectable for tests.
        allow_monitor: False refuses ``mode: monitor`` (the coordinator passes False on real
            hardware).
        hold_kp / hold_kd: Minimum per-arm-joint gains used while holding.
    """

    def __init__(
        self,
        inner: WholeBodyAdapter,
        *,
        joint_names: Sequence[str],
        checker: FenceChecker,
        on_event: Callable[[FenceEvent], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        allow_monitor: bool = True,
        hold_kp: Sequence[float] = ARM_HOLD_KP,
        hold_kd: Sequence[float] = ARM_HOLD_KD,
    ) -> None:
        names = [str(n) for n in joint_names]
        if len(set(names)) != len(names):
            raise ValueError("joint_names contains duplicates")
        index = {name: i for i, name in enumerate(names)}
        missing = [n for side in ARM_SIDES for n in ARM_JOINTS[side] if n not in index]
        if missing:
            raise ValueError(f"joint_names is missing arm joints {missing}")
        if len(hold_kp) != DOF_PER_ARM or len(hold_kd) != DOF_PER_ARM:
            raise ValueError(f"hold_kp and hold_kd need {DOF_PER_ARM} values each")
        if not all(_finite(v) and float(v) >= 0.0 for v in (*hold_kp, *hold_kd)):
            raise ValueError("hold gains must be finite and >= 0")
        self._check_mode_allowed(checker.config, allow_monitor)

        self._inner = inner
        self._joint_names = names
        self._index = index
        self._arm_idx: dict[str, list[int]] = {
            side: [index[n] for n in ARM_JOINTS[side]] for side in ARM_SIDES
        }
        self._grip_idx: dict[str, int | None] = {
            side: index.get(GRIPPER_JOINTS[side]) for side in ARM_SIDES
        }
        self._on_event = on_event
        self._clock = clock
        self._allow_monitor = allow_monitor
        self._hold_kp = tuple(float(v) for v in hold_kp)
        self._hold_kd = tuple(float(v) for v in hold_kd)

        # One re-entrant lock guards all state. Lock order: the coordinator's hardware lock may be
        # held when this is taken, never the other way round (nothing here calls the coordinator).
        self._lock = threading.RLock()
        self._checker = checker
        self._estop = False
        self._last_written: dict[str, np.ndarray | None] = {side: None for side in ARM_SIDES}
        self._measured: dict[str, np.ndarray | None] = {side: None for side in ARM_SIDES}
        self._measured_opening: dict[str, float | None] = {side: None for side in ARM_SIDES}
        # Last gripper opening actually written (initialised from the measured opening).
        self._commanded_opening: dict[str, float | None] = {side: None for side in ARM_SIDES}
        self._last_sent: list[float] | None = None
        self._last_write_t: float | None = None
        self._holding: dict[str, bool] = {side: False for side in ARM_SIDES}
        self._grip_held: dict[str, bool] = {side: False for side in ARM_SIDES}
        # Raw arm request that was last refused; the identical request stays refused.
        self._sticky: dict[str, bytes | None] = {side: None for side in ARM_SIDES}
        self._last_report: dict[str, ArmReport | None] = {side: None for side in ARM_SIDES}
        self._counts: dict[str, int] = {kind: 0 for kind in EVENT_KINDS}
        self._last_event: FenceEvent | None = None
        self._last_hold_event: FenceEvent | None = None
        self._refusal: str | None = None
        self._log = _RateLimitedLog(clock)

    # ------------------------------------------------------------------ WholeBodyAdapter protocol

    def connect(self) -> bool:
        return self._inner.connect()

    def disconnect(self) -> None:
        self._inner.disconnect()

    def is_connected(self) -> bool:
        return self._inner.is_connected()

    def has_motor_states(self) -> bool:
        return self._inner.has_motor_states()

    def read_imu(self) -> IMUState:
        return self._inner.read_imu()

    def get_limits(self) -> JointLimits | None:
        return self._inner.get_limits()

    def read_motor_states(self) -> list[MotorState]:
        """Delegate, and cache the measured arm state for the fence (never re-read in writes)."""
        states = self._inner.read_motor_states()
        try:
            self._cache_measured(states)
        except Exception as exc:
            self._log("measure", "warning", f"KiteOS fence could not cache motor states: {exc!r}")
        return states

    def write_motor_commands(self, commands: list[MotorCommand]) -> bool:
        """Fence, then forward. Returns the inner result, or False if nothing safe could be sent."""
        events: list[FenceEvent] = []
        try:
            with self._lock:
                return self._write_locked(commands, events)
        except Exception as exc:  # last line of defence: never raise into the tick
            self._log("fatal", "error", f"KiteOS fence write failed, nothing sent: {exc!r}")
            return False
        finally:
            self._dispatch(events)

    # ------------------------------------------------------------------ optional lifecycle

    def activate(self) -> bool:
        """Delegate, then read the measured state and (optionally) refuse to start inside a zone."""
        inner_activate = getattr(self._inner, "activate", None)
        if callable(inner_activate) and inner_activate() is False:
            return False
        events: list[FenceEvent] = []
        try:
            refusal = self._check_start(events)
        except Exception as exc:
            refusal = f"could not verify the start pose: {exc!r}"
        finally:
            self._dispatch(events)
        if refusal is None:
            return True
        self._refusal = refusal
        logger.error(f"KiteOS fence refused to activate the arms: {refusal}")
        self._deactivate_inner()
        return False

    def deactivate(self) -> bool:
        inner_deactivate = getattr(self._inner, "deactivate", None)
        if not callable(inner_deactivate):
            return True
        return inner_deactivate() is not False

    # ------------------------------------------------------------------ KiteOS extras

    @property
    def inner(self) -> WholeBodyAdapter:
        return self._inner

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def checker(self) -> FenceChecker:
        with self._lock:
            return self._checker

    @property
    def mode(self) -> str:
        return str(self.checker.config.mode)

    @property
    def estopped(self) -> bool:
        with self._lock:
            return self._estop

    @property
    def refusal(self) -> str | None:
        """Why the last ``activate`` was refused, if it was."""
        return self._refusal

    def last_written(self, side: str) -> np.ndarray | None:
        """Copy of the last arm target sent to the motors (``None`` before the first state read)."""
        with self._lock:
            q = self._last_written.get(side)
            return None if q is None else q.copy()

    def latch_estop(self, estopped: bool) -> None:
        """Latch or clear the E-STOP. While latched both arms and both grippers hold (also in
        monitor mode).

        On latching, each arm holds where it measurably is if that pose is clear of the fence,
        otherwise at its last written target; each gripper holds its measured opening. After
        clearing, an arm request identical to the one last seen during the E-STOP stays refused
        (stale target) until a new request arrives.
        """
        events: list[FenceEvent] = []
        try:
            with self._lock:
                was = self._estop
                self._estop = bool(estopped)
                if self._estop and not was:
                    for side in ARM_SIDES:
                        measured = self._measured[side]
                        if measured is None:
                            continue
                        report = self._checker.check_arm(
                            side, measured, self._current_opening(side)
                        )
                        if report.ok or self._last_written[side] is None:
                            self._last_written[side] = measured.copy()
                    for side in ARM_SIDES:
                        # The grippers freeze where they physically are, not where they were
                        # heading.
                        if self._measured_opening[side] is not None:
                            self._commanded_opening[side] = self._measured_opening[side]
                    events.append(self._event("both", "estop", detail="E-STOP latched"))
                elif was and not self._estop:
                    events.append(
                        self._event("both", "estop", detail="E-STOP cleared", enforced=False)
                    )
        finally:
            self._dispatch(events)

    def reload(self, config: FenceConfig) -> None:
        """Atomically swap in a new fence. Raises (and keeps the old fence) if it is invalid."""
        self._check_mode_allowed(config, self._allow_monitor)
        checker = FenceChecker(config)
        with self._lock:
            self._checker = checker
            self._last_report = {side: None for side in ARM_SIDES}
        logger.info(
            f"KiteOS fence reloaded: mode={config.mode}, margin={config.margin_m:.3f} m, "
            f"zones={[z.name for z in config.zones if z.enabled]}"
        )

    def check_positions(self, positions: Mapping[str, float]) -> dict[str, Any]:
        """Fence verdict for named joint positions (``FenceReport.to_dict()``)."""
        try:
            return self.checker.check(dict(positions)).to_dict()
        except Exception as exc:
            return {"ok": False, "arms": {}, "error": f"{type(exc).__name__}: {exc}"}

    def check_waypoints(
        self, joint_names: Sequence[str], waypoints: Sequence[Sequence[float]]
    ) -> FenceReport:
        """Check a joint-space path starting at the current arm targets.

        Joints missing from ``joint_names`` stay at their current target; gripper columns (if any)
        set the finger inflation. Routes checked: current -> waypoint 0 -> ... -> last, and
        current -> waypoint 1 -> ... -> last, the path the runtime's trajectory task takes when it
        anchors waypoint 0 at the current command. When an arm already starts within the margin,
        a route passes only as an escape (``FenceChecker.check_escape_path``). Never raises.
        """
        arms: dict[str, ArmReport] = {}
        try:
            with self._lock:
                checker = self._checker
                start = {
                    side: (
                        self._last_written[side]
                        if self._last_written[side] is not None
                        else self._measured[side]
                    )
                    for side in ARM_SIDES
                }
                start = {s: None if q is None else q.copy() for s, q in start.items()}
                opening_now = {side: self._current_opening(side) for side in ARM_SIDES}
            column = {str(name): i for i, name in enumerate(joint_names)}
            for side in ARM_SIDES:
                arms[side] = self._check_path(
                    checker, side, start[side], opening_now[side], column, waypoints
                )
        except Exception as exc:
            for side in ARM_SIDES:
                arms[side] = ArmReport(
                    side=side,
                    ok=False,
                    min_clearance_m=-math.inf,
                    zone=None,
                    sphere=None,
                    detail=f"fence error: {type(exc).__name__}: {exc}",
                )
        return FenceReport(ok=all(r.ok for r in arms.values()), arms=arms)

    def rewrite_applied(self, msg: JointState) -> JointState | None:
        """Replace commanded positions by what was actually written, so recordings are truthful.

        Called with the runtime's ``applied_joint_position_command`` message right after a successful
        write. Returns a new message, or ``None`` when no truthful message can be produced (the
        caller must then drop it rather than record pre-fence targets).
        """
        try:
            names = list(msg.name)
            positions = [float(v) for v in msg.position]
            if len(positions) != len(names):
                return None
            if not any(name in self._index for name in names):
                return msg  # none of this adapter's joints
            with self._lock:
                sent = self._last_sent
            if sent is None:
                return None
            for k, name in enumerate(names):
                i = self._index.get(name)
                if i is not None:
                    positions[k] = float(sent[i])
            return JointState(
                ts=msg.ts,
                frame_id=msg.frame_id,
                name=names,
                position=positions,
                velocity=list(msg.velocity),
                effort=list(msg.effort),
            )
        except Exception as exc:
            self._log("rewrite", "error", f"KiteOS could not rewrite the applied command: {exc!r}")
            return None

    def status(self) -> dict[str, Any]:
        """Plain-Python (YAML/JSON-safe) snapshot for RPCs and UIs."""
        with self._lock:
            config = self._checker.config
            arms: dict[str, Any] = {}
            for side in ARM_SIDES:
                report = self._last_report[side]
                arms[side] = {
                    "holding": self._holding[side],
                    "zones": self._checker.zone_names(side),
                    "last_report": report.to_dict() if report is not None else None,
                    "last_written": _as_list(self._last_written[side]),
                    "measured": _as_list(self._measured[side]),
                    "gripper_opening": self._current_opening(side),
                }
            return {
                "mode": config.mode,
                "station": config.station,
                "margin_m": float(config.margin_m),
                "max_joint_speed_rad_s": float(config.max_joint_speed_rad_s),
                "refuse_start_inside": bool(config.refuse_start_inside),
                "zones": [
                    {
                        "name": z.name,
                        "kind": z.kind,
                        "frame": z.frame,
                        "applies_to": z.applies_to,
                        "enabled": bool(z.enabled),
                    }
                    for z in config.zones
                ],
                "estop": self._estop,
                "holds": int(self._counts["hold"] + self._counts["error"]),
                "events": dict(self._counts),
                "last_event": self._last_event.to_dict() if self._last_event else None,
                "last_hold": self._last_hold_event.to_dict() if self._last_hold_event else None,
                "refusal": self._refusal,
                "arms": arms,
            }

    # ------------------------------------------------------------------ write pipeline

    def _write_locked(self, commands: list[MotorCommand], events: list[FenceEvent]) -> bool:
        try:
            count = len(commands)
        except TypeError:
            count = -1
        if count != len(self._joint_names):
            detail = f"expected {len(self._joint_names)} commands, got {count}"
            events.append(self._event("both", "error", detail=detail))
            self._log(
                "length",
                "error",
                f"KiteOS fence rejected a write with {count} commands "
                f"(expected {len(self._joint_names)}); nothing sent",
            )
            return False
        missing = [side for side in ARM_SIDES if self._last_written[side] is None]
        if missing:
            self._log(
                "baseline",
                "warning",
                f"KiteOS fence has no measured state for {missing} yet; nothing sent",
            )
            return False

        now = float(self._clock())
        checker = self._checker
        try:
            plans = self._plan(commands, now, checker, events)
        except Exception as exc:
            events.append(
                self._event(
                    "both",
                    "error",
                    detail=f"{type(exc).__name__}: {exc}",
                    new_episode=not all(self._holding.values()),
                )
            )
            self._log(
                "error",
                "error",
                f"KiteOS fence error, holding both arms and grippers (fail closed): {exc!r}",
            )
            plans = self._hold_plans(commands)
        out = self._assemble(commands, plans)

        try:
            accepted = bool(self._inner.write_motor_commands(out))
        except Exception as exc:
            events.append(self._event("both", "error", detail=f"inner write raised: {exc!r}"))
            self._log("inner", "error", f"Motor write raised under the KiteOS fence: {exc!r}")
            return False
        if accepted:
            for side, plan in plans.items():
                self._last_written[side] = plan.q.copy()
                self._holding[side] = plan.held
                self._grip_held[side] = plan.grip_held
                self._sticky[side] = plan.request_key if plan.held else None
                if plan.grip is not None and _finite(plan.grip.q):
                    self._commanded_opening[side] = float(plan.grip.q)
                if plan.report is not None:
                    self._last_report[side] = plan.report
            self._last_sent = [float(c.q) for c in out]
            self._last_write_t = now
        return accepted

    def _plan(
        self,
        commands: Sequence[MotorCommand],
        now: float,
        checker: FenceChecker,
        events: list[FenceEvent],
    ) -> dict[str, _ArmPlan]:
        config = checker.config
        if self._last_write_t is None:
            dt = RATE_DT_MAX_S
        else:
            dt = min(max(now - self._last_write_t, RATE_DT_MIN_S), RATE_DT_MAX_S)
        plans: dict[str, _ArmPlan] = {}
        for side in ARM_SIDES:
            last = self._last_written[side]
            if last is None:
                raise _NoBaseline(side)
            cmds = [commands[i] for i in self._arm_idx[side]]
            g = self._grip_idx[side]
            grip_cmd = commands[g] if g is not None else None
            plans[side] = self._plan_side(
                side, cmds, grip_cmd, last, checker, config, dt, events
            )
        return plans

    def _plan_side(
        self,
        side: str,
        cmds: Sequence[MotorCommand],
        grip_cmd: MotorCommand | None,
        last: np.ndarray,
        checker: FenceChecker,
        config: FenceConfig,
        dt: float,
        events: list[FenceEvent],
    ) -> _ArmPlan:
        enforce = config.mode == "enforce"
        measured_opening = self._measured_opening[side]
        base_opening = self._commanded_opening[side]
        if base_opening is None:
            base_opening = measured_opening
        new_episode = not (self._holding[side] or self._grip_held[side])
        request_key = _request_key(cmds)

        if self._estop:
            plan = self._hold(side, cmds, last, None)
            plan.request_key = request_key
            plan.grip, plan.grip_held = self._hold_gripper(grip_cmd, base_opening), True
            return plan

        grip_valid = grip_cmd is not None and all(
            _finite(v) for v in (grip_cmd.q, grip_cmd.dq, grip_cmd.kp, grip_cmd.kd, grip_cmd.tau)
        )
        requested_opening = float(grip_cmd.q) if grip_cmd is not None and grip_valid else None
        if grip_cmd is not None and not grip_valid:
            events.append(
                self._event(
                    side,
                    "hold",
                    detail="invalid gripper command (non-finite); gripper held",
                    enforced=enforce,
                    new_episode=new_episode,
                )
            )
        opening_up = requested_opening is not None and (
            base_opening is None or requested_opening > base_opening + _OPENING_EPS
        )

        if not opening_up:
            # Closing (or unchanged): always allowed. The fingers may lag a closing command, so
            # the arm is checked with the more open of commanded and measured.
            finger = _max_opening(
                requested_opening if requested_opening is not None else base_opening,
                measured_opening,
            )
            plan, arm_events = self._plan_arm(side, cmds, last, finger, checker, config, dt)
            plan.request_key = request_key
            events.extend(self._stamp(arm_events, new_episode))
            if grip_cmd is None:
                return plan
            if grip_valid or not enforce:
                plan.grip = grip_cmd
            else:
                plan.grip, plan.grip_held = self._hold_gripper(grip_cmd, base_opening), True
            return plan

        # Opening increase: only if the arm, where it will be this tick, clears the fence with
        # the wider fingers. Otherwise the gripper keeps its last written opening.
        wide = _max_opening(requested_opening, measured_opening)
        plan_wide, events_wide = self._plan_arm(side, cmds, last, wide, checker, config, dt)
        verdict: ArmReport | None
        if plan_wide.held:
            verdict = checker.check_arm(side, last, wide)
            opening_ok = verdict.ok
        else:
            verdict = plan_wide.report
            opening_ok = bool(verdict is not None and verdict.ok and not plan_wide.escape)
        if opening_ok or not enforce:
            plan_wide.request_key = request_key
            plan_wide.grip = grip_cmd
            events.extend(self._stamp(events_wide, new_episode))
            if not opening_ok:
                events.append(self._gripper_event(side, verdict, enforce, new_episode))
            return plan_wide
        narrow = _max_opening(base_opening, measured_opening)
        plan, arm_events = self._plan_arm(side, cmds, last, narrow, checker, config, dt)
        plan.request_key = request_key
        events.extend(self._stamp(arm_events, new_episode))
        events.append(self._gripper_event(side, verdict, enforce, new_episode))
        plan.grip, plan.grip_held = self._hold_gripper(grip_cmd, base_opening), True
        return plan

    def _plan_arm(
        self,
        side: str,
        cmds: Sequence[MotorCommand],
        last: np.ndarray,
        opening: float | None,
        checker: FenceChecker,
        config: FenceConfig,
        dt: float,
    ) -> tuple[_ArmPlan, list[FenceEvent]]:
        """Plan one arm for a given finger opening. Events are returned, not recorded."""
        events: list[FenceEvent] = []
        enforce = config.mode == "enforce"
        speed = float(config.max_joint_speed_rad_s)
        max_step = speed * dt
        tau_max = float(config.max_feedforward_tau_nm)

        requested = np.array([float(c.q) for c in cmds], dtype=float)
        fields_ok = all(_finite(v) for c in cmds for v in (c.q, c.dq, c.kp, c.kd, c.tau))
        if not fields_ok or np.any(np.abs(requested) > MAX_PLAUSIBLE_TARGET_RAD):
            events.append(
                self._event(
                    side,
                    "hold",
                    detail="invalid command (non-finite or implausible target)",
                    enforced=enforce,
                    record=False,
                )
            )
            if enforce:
                return self._hold(side, cmds, last, None), events
            passed = requested if np.all(np.isfinite(requested)) else last
            return _ArmPlan(list(cmds), passed.copy(), False, None), events

        sticky = self._sticky[side]
        if enforce and sticky is not None and requested.tobytes() == sticky:
            events.append(
                self._event(
                    side,
                    "hold",
                    detail="stale arm target: unchanged since the fence refused it",
                    record=False,
                )
            )
            return self._hold(side, cmds, last, None), events

        candidate = checker.clamp_to_limits(side, requested)
        excess = np.abs(candidate - requested)
        if np.any(excess > float(config.joint_limit_margin_rad) + _LIMIT_EVENT_SLACK_RAD):
            worst = int(np.argmax(excess))
            events.append(
                self._event(
                    side,
                    "limit_clamp",
                    detail=f"{ARM_JOINTS[side][worst]} target {requested[worst]:.4f} rad "
                    f"clamped to {candidate[worst]:.4f}",
                    enforced=enforce,
                    record=False,
                )
            )
        delta = candidate - last
        peak = float(np.max(np.abs(delta)))
        if peak > max_step:
            # Scale the whole step so the arm keeps moving along the requested line.
            candidate = last + delta * (max_step / peak)
            events.append(
                self._event(
                    side,
                    "rate_limit",
                    detail=f"step {peak:.4f} rad limited to {max_step:.4f} rad",
                    enforced=enforce,
                    record=False,
                )
            )

        report = checker.check_arm_motion(side, last, candidate, opening)
        accept = report.ok
        escape = False
        if not accept and checker.is_escape_motion(side, last, candidate, opening):
            accept = escape = True
            events.append(
                self._event(
                    side,
                    "escape",
                    zone=report.zone,
                    sphere=report.sphere,
                    clearance_m=report.min_clearance_m,
                    detail="moving out of a violation",
                    enforced=enforce,
                    record=False,
                )
            )
        elif not accept:
            events.append(
                self._event(
                    side,
                    "hold",
                    zone=report.zone,
                    sphere=report.sphere,
                    clearance_m=report.min_clearance_m,
                    detail=report.detail,
                    enforced=enforce,
                    record=False,
                )
            )

        if not enforce:
            return _ArmPlan(list(cmds), requested.copy(), False, report), events
        if not accept:
            return self._hold(side, cmds, last, report), events
        sent = [
            MotorCommand(
                q=float(candidate[k]),
                dq=float(min(max(c.dq, -speed), speed)),
                kp=float(c.kp),
                kd=float(c.kd),
                tau=float(min(max(c.tau, -tau_max), tau_max)),
            )
            for k, c in enumerate(cmds)
        ]
        return _ArmPlan(sent, candidate.copy(), False, report, escape=escape), events

    def _stamp(self, events: list[FenceEvent], new_episode: bool) -> list[FenceEvent]:
        """Record planned events (counters, last event); mark the first hold of an episode."""
        out = []
        for event in events:
            if new_episode and event.kind == "hold" and event.enforced:
                event = replace(event, new_episode=True)
            self._record(event)
            out.append(event)
        return out

    def _gripper_event(
        self, side: str, verdict: ArmReport | None, enforce: bool, new_episode: bool
    ) -> FenceEvent:
        clearance = None if verdict is None else verdict.min_clearance_m
        return self._event(
            side,
            "hold",
            zone=None if verdict is None else verdict.zone,
            sphere=None if verdict is None else verdict.sphere,
            clearance_m=clearance,
            detail="gripper opening refused: the wider fingers would be within the margin; "
            "gripper held at its last opening",
            enforced=enforce,
            new_episode=new_episode and enforce,
        )

    def _hold(
        self,
        side: str,
        cmds: Sequence[Any],
        last: np.ndarray,
        report: ArmReport | None,
    ) -> _ArmPlan:
        held = [
            MotorCommand(
                q=float(last[k]),
                dq=0.0,
                kp=self._gain(getattr(cmds[k], "kp", None), self._hold_kp[k]),
                kd=self._gain(getattr(cmds[k], "kd", None), self._hold_kd[k]),
                tau=0.0,
            )
            for k in range(DOF_PER_ARM)
        ]
        return _ArmPlan(held, last.copy(), True, report)

    def _hold_gripper(self, cmd: MotorCommand | None, opening: float | None) -> MotorCommand | None:
        """The gripper command that keeps it at ``opening`` (its last written opening)."""
        if cmd is None:
            return None
        if opening is None or not _finite(opening):
            opening = 0.0  # never forward a non-finite target; closed is the small finger model
        return MotorCommand(
            q=float(opening),
            dq=0.0,
            kp=float(cmd.kp) if _finite(cmd.kp) else 0.0,
            kd=float(cmd.kd) if _finite(cmd.kd) else 0.0,
            tau=0.0,
        )

    def _hold_plans(self, commands: Sequence[Any]) -> dict[str, _ArmPlan]:
        """Fail-closed plan: both arms and both grippers keep their last written targets."""
        plans: dict[str, _ArmPlan] = {}
        for side in ARM_SIDES:
            last = self._last_written[side]
            if last is None:
                raise _NoBaseline(side)
            cmds = [commands[i] for i in self._arm_idx[side]]
            plan = self._hold(side, cmds, last, None)
            plan.request_key = _request_key(cmds)
            g = self._grip_idx[side]
            if g is not None:
                opening = self._commanded_opening[side]
                if opening is None:
                    opening = self._measured_opening[side]
                plan.grip, plan.grip_held = self._hold_gripper(commands[g], opening), True
            plans[side] = plan
        return plans

    def _assemble(
        self, commands: Sequence[MotorCommand], plans: Mapping[str, _ArmPlan]
    ) -> list[MotorCommand]:
        out = list(commands)
        for side, plan in plans.items():
            for k, i in enumerate(self._arm_idx[side]):
                out[i] = plan.commands[k]
            g = self._grip_idx[side]
            if g is not None and plan.grip is not None:
                out[g] = plan.grip
        return out

    @staticmethod
    def _gain(value: Any, floor: float) -> float:
        return max(float(value), floor) if _finite(value) else floor

    # ------------------------------------------------------------------ helpers

    def _check_path(
        self,
        checker: FenceChecker,
        side: str,
        start: np.ndarray | None,
        opening_now: float | None,
        column: Mapping[str, int],
        waypoints: Sequence[Sequence[float]],
    ) -> ArmReport:
        arm_cols = [column.get(name) for name in ARM_JOINTS[side]]
        grip_col = column.get(GRIPPER_JOINTS[side])
        if all(c is None for c in arm_cols) and grip_col is None:
            return ArmReport(
                side=side,
                ok=True,
                min_clearance_m=math.inf,
                zone=None,
                sphere=None,
                detail="not moved by this motion",
            )
        if start is None:
            return ArmReport(
                side=side,
                ok=False,
                min_clearance_m=-math.inf,
                zone=None,
                sphere=None,
                detail="no measured arm state yet",
            )
        path: list[tuple[np.ndarray, float | None]] = []
        for wp in waypoints:
            q = start.copy()
            for k, c in enumerate(arm_cols):
                if c is not None:
                    q[k] = float(wp[c])
            opening = float(wp[grip_col]) if grip_col is not None else opening_now
            path.append((q, opening))
        # the runtime's trajectory task either follows current -> waypoint 0 -> ... or, when it anchors
        # waypoint 0 at its current command, current -> waypoint 1 -> ...: both routes must pass.
        routes: list[list[tuple[str, np.ndarray, np.ndarray, float | None]]] = []
        for first in (0, 1) if len(path) > 1 else (0,):
            route = []
            previous, previous_opening, label = start, opening_now, "current target"
            for i in range(first, len(path)):
                q, opening = path[i]
                route.append(
                    (f"{label} -> waypoint {i}", previous, q, _max_opening(previous_opening, opening))
                )
                previous, previous_opening, label = q, opening, f"waypoint {i}"
            routes.append(route)

        start_report = checker.check_arm(side, start, opening_now)
        if not start_report.ok and math.isfinite(start_report.min_clearance_m):
            # The arm is already within the margin (hot reload, new zone): accept only a path
            # that backs straight out of it (no zone/sphere pair gets closer) and ends clear.
            escape: ArmReport | None = None
            for route in routes:
                report = checker.check_escape_path(
                    side, [(qa, qb, opening) for _text, qa, qb, opening in route]
                )
                if not report.ok:
                    return report
                escape = escape or report
            assert escape is not None
            return escape

        worst: ArmReport | None = None
        checked: set[tuple[bytes, bytes, float | None]] = set()
        for route in routes:
            for text, qa, qb, opening in route:
                key = (qa.tobytes(), qb.tobytes(), opening)
                if key in checked:
                    continue  # the second route shares every segment after its first
                checked.add(key)
                report = checker.check_arm_motion(side, qa, qb, opening)
                if not report.ok:
                    detail = f"{text}: {report.detail}" if report.detail else text
                    return ArmReport(
                        side=side,
                        ok=False,
                        min_clearance_m=report.min_clearance_m,
                        zone=report.zone,
                        sphere=report.sphere,
                        detail=detail,
                    )
                if worst is None or report.min_clearance_m < worst.min_clearance_m:
                    worst = report
        assert worst is not None
        return worst

    def _check_start(self, events: list[FenceEvent]) -> str | None:
        """Read the measured state after activation; return a refusal reason or ``None``."""
        states = self._inner.read_motor_states()
        self._cache_measured(states)
        with self._lock:
            missing = [side for side in ARM_SIDES if self._measured[side] is None]
            if missing:
                return f"no valid measured joint state for {missing}"
            checker = self._checker
            config = checker.config
            for side in ARM_SIDES:
                measured = self._measured[side]
                assert measured is not None
                self._last_written[side] = measured.copy()
                self._holding[side] = False
                self._grip_held[side] = False
                self._sticky[side] = None
                self._commanded_opening[side] = self._measured_opening[side]
            # The arm is at its measured pose now: rate-limit the first write from this instant.
            self._last_write_t = float(self._clock())
            self._refusal = None
            refuse = config.refuse_start_inside and config.mode == "enforce"
            problems: list[str] = []
            for side in ARM_SIDES:
                measured = self._last_written[side]
                assert measured is not None
                report = checker.check_arm(side, measured, self._current_opening(side))
                self._last_report[side] = report
                if report.ok:
                    continue
                events.append(
                    self._event(
                        side,
                        "start_inside",
                        zone=report.zone,
                        sphere=report.sphere,
                        clearance_m=report.min_clearance_m,
                        detail=report.detail,
                        enforced=refuse,
                    )
                )
                clearance = (
                    f"{report.min_clearance_m * 100:+.1f} cm"
                    if math.isfinite(report.min_clearance_m)
                    else "unknown clearance"
                )
                problems.append(
                    f"{side} arm starts within the {config.margin_m * 100:.0f} cm margin of zone "
                    f"{report.zone!r} ({report.sphere}, {clearance})"
                    + (f": {report.detail}" if report.detail else "")
                )
            if not problems:
                return None
            message = "; ".join(problems)
            if refuse:
                return (
                    f"{message}. Power the arms down, move them clear by hand, and start again "
                    "(or fix the fence file)."
                )
            logger.warning(f"KiteOS fence: {message}; only moves out of the zone are allowed")
            return None

    def _cache_measured(self, states: Sequence[MotorState]) -> None:
        if len(states) != len(self._joint_names):
            return
        with self._lock:
            for side in ARM_SIDES:
                q = np.array([float(states[i].q) for i in self._arm_idx[side]], dtype=float)
                valid = bool(np.all(np.isfinite(q)))
                self._measured[side] = q if valid else None
                if valid and self._last_written[side] is None:
                    self._last_written[side] = q.copy()
                g = self._grip_idx[side]
                opening = float(states[g].q) if g is not None else None
                self._measured_opening[side] = opening if _finite(opening) else None
                if self._commanded_opening[side] is None and self._measured_opening[side] is not None:
                    self._commanded_opening[side] = self._measured_opening[side]

    def _current_opening(self, side: str) -> float | None:
        # The more open of commanded and measured: fingers may lag a closing command.
        return _max_opening(self._commanded_opening[side], self._measured_opening[side])

    def _deactivate_inner(self) -> None:
        try:
            deactivate = getattr(self._inner, "deactivate", None)
            if callable(deactivate):
                deactivate()
        except Exception as exc:
            logger.error(f"KiteOS: deactivating the arms after a refused start failed: {exc!r}")

    @staticmethod
    def _check_mode_allowed(config: FenceConfig, allow_monitor: bool) -> None:
        if config.mode == "monitor" and not allow_monitor:
            raise ValueError(
                "KiteOS refuses fence mode 'monitor' here (real hardware): it would only log "
                "violations. Use mode: enforce."
            )

    def _event(
        self,
        side: str,
        kind: FenceEventKind,
        *,
        zone: str | None = None,
        sphere: str | None = None,
        clearance_m: float | None = None,
        detail: str | None = None,
        enforced: bool = True,
        new_episode: bool = False,
        record: bool = True,
    ) -> FenceEvent:
        try:
            t = float(self._clock())
        except Exception:
            t = math.nan
        event = FenceEvent(
            t=t,
            side=side,
            kind=kind,
            zone=zone,
            sphere=sphere,
            clearance_m=clearance_m,
            detail=detail,
            enforced=enforced,
            new_episode=bool(new_episode and enforced and kind in ("hold", "error")),
        )
        if record:
            self._record(event)
        return event

    def _record(self, event: FenceEvent) -> None:
        with self._lock:
            self._counts[event.kind] = self._counts.get(event.kind, 0) + 1
            self._last_event = event
            if event.kind in ("hold", "error") and event.enforced:
                self._last_hold_event = event

    def _dispatch(self, events: Sequence[FenceEvent]) -> None:
        if not events or self._on_event is None:
            return
        for event in events:
            try:
                self._on_event(event)
            except Exception as exc:
                self._log("on_event", "error", f"KiteOS fence event handler failed: {exc!r}")
