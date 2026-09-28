# Copyright 2026 Kite ML.
"""KiteOS Dual OpenYAM coordinator: dimOS's ``DualOpenYamCoordinator`` behind the safety fence.

``KiteDualOpenYamCoordinator`` wraps the whole-body adapter in ``FencedWholeBodyAdapter`` (the only
point every command source passes through) and closes the gaps the fence cannot close alone:

* refuses real hardware without a fence file (``require_fence_on_hardware``), with a fence that
  leaves an arm without an enabled zone (unless the file says ``allow_no_zones: true``), with
  ``mode: monitor`` or with implausible rate/sweep settings; refuses any real adapter at all when
  started with ``expect_mock`` (the KiteOS CLI sets it for every run without ``--real``);
* rewrites ``applied_joint_position_command`` so recorded actions are what the fence applied; the
  hook is installed on the output port before dimOS creates its tick loop, so no sample escapes;
* ``set_estop`` latches the fence and cancels the executing trajectory (dimOS's trajectory task
  ignores E-STOP); ``execute_trajectory`` and ``task_invoke`` on a trajectory task's ``execute`` /
  ``on_joint_command`` reject trajectories that would cross the fence (a trajectory that starts
  inside the margin passes only if it backs straight out and ends clear);
* a daemon supervisor thread cancels the executing trajectory once per hold episode. Fence events
  arrive on the tick thread under the coordinator's non re-entrant hardware lock, where calling
  RPCs would deadlock, so the event handler only records and signals.

Runs inside a dimOS forkserver worker: the class is pickled by reference and this module is
imported there, so it has no import-time side effects beyond defining classes.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dimos.control.components import HardwareComponent
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JointTrajectoryTask,
    TrajectoryExecutionResult,
    TrajectoryExecutionStatus,
)
from dimos.core.core import rpc
from dimos.hardware.whole_body.spec import WholeBodyAdapter
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    DualOpenYamCoordinator,
    DualOpenYamCoordinatorConfig,
)
from dimos.utils.logging_config import setup_logger

from kiteos.safety.adapter import FencedWholeBodyAdapter, FenceEvent
from kiteos.safety.fence import (
    FenceChecker,
    FenceConfig,
    FenceReport,
    real_hardware_problems,
)
from kiteos.safety.kinematics import ARM_JOINTS, ARM_SIDES, GRIPPER_JOINTS

logger = setup_logger()

# The only adapter KiteOS treats as not-real hardware; anything else is assumed to move motors.
MOCK_ADAPTER_TYPE = "mock_whole_body"
# Enforced events after which an executing trajectory is cancelled.
_CANCEL_ON_KINDS = frozenset({"hold", "error"})
_SUPERVISOR_POLL_S = 0.25
_SUPERVISOR_JOIN_S = 2.0
_EVENT_LOG_INTERVAL_S = 1.0
_RECENT_EVENTS = 50
_RECENT_HOLDS = 20
# Trajectory-task commands that start motion and therefore go through the fence precheck.
_PRECHECKED_TASK_COMMANDS = frozenset({"execute", "on_joint_command"})


def resolve_fence_path(path: str | Path) -> Path:
    """Absolute fence path; relative paths are relative to the current working directory."""
    p = Path(path).expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


class KiteDualOpenYamCoordinatorConfig(DualOpenYamCoordinatorConfig):
    """``DualOpenYamCoordinatorConfig`` plus the KiteOS fence settings.

    ``fence_path`` is the fence YAML (``--controlcoordinator.fence-path``); ``None`` runs mock arms
    with an empty fence (joint limits and the rate limit still apply) and is refused on real
    hardware unless ``require_fence_on_hardware`` is turned off.
    """

    fence_path: str | None = None
    require_fence_on_hardware: bool = True
    # Set by the KiteOS CLI for every run without --real: refuse to build any real adapter, even
    # if CAN ports arrive through the environment, a dimOS config file or a .env file.
    expect_mock: bool = False


class KiteDualOpenYamCoordinator(DualOpenYamCoordinator):
    """Dual OpenYAM coordinator that enforces the KiteOS safety fence on every motor write."""

    config: KiteDualOpenYamCoordinatorConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fenced: dict[str, FencedWholeBodyAdapter] = {}
        self._fence_real: dict[str, bool] = {}
        self._fence_source: str | None = None
        self._fence_inbox: deque[FenceEvent] = deque(maxlen=4096)
        # Recent events coalesced by (side, kind, zone, enforced), most recent last:
        # key -> [latest event, count]. A flood of one kind cannot push the others out.
        self._recent_fence_events: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()
        self._recent_fence_holds: deque[FenceEvent] = deque(maxlen=_RECENT_HOLDS)
        self._fence_wakeup = threading.Event()
        self._fence_cancel = threading.Event()
        self._supervisor_stop = threading.Event()
        self._supervisor: threading.Thread | None = None
        self._trajectory_cancels = 0
        # Whether the executing trajectory was already cancelled in the current hold episode.
        self._episode_cancelled = False
        self._fence_log_last: dict[str, float] = {}
        self._fence_log_suppressed: dict[str, int] = {}

    # ------------------------------------------------------------------ hardware

    def _create_whole_body_adapter(self, component: HardwareComponent) -> WholeBodyAdapter:
        """Create dimOS's adapter for ``component`` and wrap it in the fence.

        The fence file is validated before the inner adapter exists, so a refusal never
        constructs a hardware adapter.
        """
        real = component.adapter_type != MOCK_ADAPTER_TYPE
        if real and self.config.expect_mock:
            raise RuntimeError(
                "KiteOS was started for MOCK arms (no --real), but dimOS resolved the real "
                f"'{component.adapter_type}' adapter: CAN ports came from the environment "
                "(CONTROLCOORDINATOR__*), ~/.config/dimos/config or the dimOS .env file. "
                "Refusing to power the arms. Remove those settings; use --real to drive the "
                "real arms."
            )
        config, source = self._load_fence(real=real)
        checker = FenceChecker(config)
        inner = super()._create_whole_body_adapter(component)
        fenced = FencedWholeBodyAdapter(
            inner,
            joint_names=list(component.joints),
            checker=checker,
            on_event=self._on_fence_event,
            allow_monitor=not real,
        )
        self._fenced[component.hardware_id] = fenced
        self._fence_real[component.hardware_id] = real
        self._fence_source = source
        zones = [z.name for z in config.zones if z.enabled]
        logger.info(
            f"KiteOS fence armed on {component.hardware_id} ({component.adapter_type}): "
            f"mode={config.mode}, margin={config.margin_m:.3f} m, zones={zones}, "
            f"file={source or 'none'}"
        )
        return fenced

    def _load_fence(self, *, real: bool, path: str | None = None) -> tuple[FenceConfig, str | None]:
        raw = path if path is not None else self.config.fence_path
        if raw is None or not str(raw).strip():
            if real and self.config.require_fence_on_hardware:
                raise RuntimeError(
                    "KiteOS refuses to drive real arms without a fence file. Pass "
                    "--controlcoordinator.fence-path <fence.yaml> (see docs/FENCE.md), or run "
                    "without CAN ports to use mock arms."
                )
            if real:
                logger.warning(
                    "KiteOS: real arms WITHOUT a fence file (require_fence_on_hardware=False); "
                    "only joint limits and the rate limit are enforced"
                )
            return FenceConfig(), None
        resolved = resolve_fence_path(str(raw))
        if not resolved.is_file():
            raise FileNotFoundError(f"KiteOS fence file not found: {resolved}")
        config = FenceConfig.load(resolved)
        if real:
            check_fence_for_real_hardware(config, str(resolved))
        return config, str(resolved)

    # ------------------------------------------------------------------ lifecycle

    @rpc
    def start(self) -> None:
        """Hook the applied-command publisher, start dimOS's coordinator and the supervisor.

        The rewrite is installed on the ``applied_joint_position_command`` port before dimOS
        builds its tick loop (which captures ``port.publish``), so not even the first tick can
        publish a pre-fence target. ``_wrap_applied_publisher`` then verifies the tick loop really
        uses it (and wraps the tick loop's callback as a fallback if a dimOS change bypassed it).
        """
        was_running = self._tick_loop is not None and self._tick_loop.is_running
        self._start_fence_supervisor()
        try:
            if not was_running:
                self._install_applied_rewrite()
            super().start()
            if not was_running:
                self._wrap_applied_publisher()
        except Exception:
            self._stop_fence_supervisor()
            raise

    @rpc
    def stop(self) -> None:
        self._stop_fence_supervisor()
        super().stop()

    def _rewrite_applied(self, msg: JointState) -> JointState | None:
        out: JointState | None = msg
        for fenced in list(self._fenced.values()):
            if out is None:
                break
            out = fenced.rewrite_applied(out)
        if out is None:
            self._rate_log(
                "applied", "warning", "KiteOS dropped an applied-command sample it could not verify"
            )
        return out

    def _fenced_publisher(self, original: Callable[[JointState], None]) -> Callable[[JointState], None]:
        def publish_applied(msg: JointState) -> None:
            out = self._rewrite_applied(msg)
            if out is not None:
                original(out)

        publish_applied._kiteos_fenced = True  # type: ignore[attr-defined]
        return publish_applied

    def _install_applied_rewrite(self) -> None:
        """Shadow ``applied_joint_position_command.publish`` with the rewriting publisher."""
        port = getattr(self, "applied_joint_position_command", None)
        current = getattr(port, "publish", None)
        if port is None or not callable(current):
            raise RuntimeError(
                "KiteOS cannot hook dimOS's applied_joint_position_command port (has the dimOS "
                "pin changed?). Refusing to run: recorded actions would not match what the fence "
                "applied."
            )
        if getattr(current, "_kiteos_fenced", False):
            return
        port.publish = self._fenced_publisher(current)

    def _wrap_applied_publisher(self) -> None:
        tick_loop = self._tick_loop
        # Private dimOS attribute (pinned commit); fail loudly rather than record pre-fence actions.
        if tick_loop is None or not hasattr(tick_loop, "_publish_command_callback"):
            if tick_loop is not None:
                tick_loop.stop()
            raise RuntimeError(
                "KiteOS cannot hook dimOS's applied-command publisher "
                "(TickLoop._publish_command_callback is missing; has the dimOS pin changed?). "
                "Refusing to run: recorded actions would not match what the fence applied."
            )
        original: Callable[[JointState], None] | None = tick_loop._publish_command_callback
        if original is None or getattr(original, "_kiteos_fenced", False):
            return
        logger.warning(
            "KiteOS: dimOS's tick loop did not pick up the fenced applied-command publisher; "
            "wrapping its callback after start (the first ticks may have been recorded unfenced)"
        )
        tick_loop._publish_command_callback = self._fenced_publisher(original)

    # ------------------------------------------------------------------ overridden RPCs

    @rpc
    def set_estop(self, estopped: bool) -> bool:
        """Latch the fence (arms hold), cancel any trajectory, then latch dimOS's tasks."""
        for fenced in list(self._fenced.values()):
            fenced.latch_estop(estopped)
        if estopped:
            try:
                result = self.cancel_trajectory()
                if result.cancelled:
                    self._trajectory_cancels += 1
            except Exception as exc:
                logger.error(f"KiteOS E-STOP: cancelling the trajectory failed: {exc!r}")
        return super().set_estop(estopped)

    @rpc
    def execute_trajectory(self, trajectory: JointTrajectory) -> TrajectoryExecutionResult:
        """Reject trajectories while E-STOP is latched or when they would cross the fence."""
        rejection = self._trajectory_gate(trajectory)
        if rejection is not None:
            return rejection
        return super().execute_trajectory(trajectory)

    @rpc
    def task_invoke(
        self, task_name: str, method: str, kwargs: dict[str, Any] | None = None
    ) -> Any:
        """dimOS ``task_invoke`` with the trajectory precheck on motion-starting commands.

        ``task_invoke("joint_trajectory", "execute", {"trajectory": ...})`` (a declared command)
        and ``on_joint_command`` on any trajectory task start motion exactly like
        ``execute_trajectory``, so they get the same E-STOP and fence precheck. Everything else is
        dimOS's ``task_invoke`` unchanged; the adapter still fences every tick regardless.
        """
        task = self._tasks.get(task_name)
        if isinstance(task, JointTrajectoryTask) and method in _PRECHECKED_TASK_COMMANDS:
            args = dict(kwargs or {})
            if method == "execute":
                rejection = self._trajectory_gate(args.get("trajectory"))
                if rejection is not None:
                    return rejection
            else:
                trajectory = _single_point_trajectory(args.get("msg"))
                if trajectory is not None:
                    rejection = self._trajectory_gate(trajectory)
                    if rejection is not None:
                        logger.warning(f"KiteOS rejected task_invoke on_joint_command: {rejection.message}")
                        return False
        return super().task_invoke(task_name, method, kwargs)

    def _trajectory_gate(self, trajectory: Any) -> TrajectoryExecutionResult | None:
        """Rejection (E-STOP latched, or the trajectory would cross the fence) or None."""
        fenced = list(self._fenced.values())
        if any(f.estopped for f in fenced):
            return TrajectoryExecutionResult(
                TrajectoryExecutionStatus.INVALID_TRAJECTORY,
                "KiteOS E-STOP is latched; clear it with set_estop(False) first",
            )
        if trajectory is not None and fenced:
            return self._fence_precheck(trajectory, fenced)
        return None

    def _fence_precheck(
        self, trajectory: JointTrajectory, fenced: list[FencedWholeBodyAdapter]
    ) -> TrajectoryExecutionResult | None:
        try:
            names = [str(n) for n in trajectory.joint_names]
            points = [list(p.positions) for p in trajectory.points]
        except Exception as exc:
            return TrajectoryExecutionResult(
                TrajectoryExecutionStatus.INVALID_TRAJECTORY,
                f"KiteOS fence could not read the trajectory: {exc!r}",
            )
        if not names or not points:
            return None  # structurally invalid; dimOS rejects it with its own message
        for adapter in fenced:
            report = adapter.check_waypoints(names, points)
            if not report.ok:
                return TrajectoryExecutionResult(
                    TrajectoryExecutionStatus.INVALID_TRAJECTORY, describe_rejection(report)
                )
        return None

    # ------------------------------------------------------------------ fence RPCs

    @rpc
    def fence_status(self) -> dict[str, Any]:
        """Fence mode, zones, E-STOP, hold counters and per-arm state (YAML/JSON-safe)."""
        hardware: dict[str, Any] = {}
        for hw_id, fenced in list(self._fenced.items()):
            hardware[hw_id] = {
                "real_hardware": self._fence_real.get(hw_id, True),
                "adapter": type(fenced.inner).__name__,
                **fenced.status(),
            }
        tick_loop = self._tick_loop
        return {
            "product": "KiteOS",
            "running": bool(tick_loop is not None and tick_loop.is_running),
            "fence_path": self._fence_source,
            "require_fence_on_hardware": bool(self.config.require_fence_on_hardware),
            "trajectory_cancels": int(self._trajectory_cancels),
            "hardware": hardware,
            "recent_events": [
                {**event.to_dict(), "repeats": int(count)}
                for event, count in list(self._recent_fence_events.values())[-10:]
            ],
            "recent_holds": [e.to_dict() for e in list(self._recent_fence_holds)[-10:]],
        }

    @rpc
    def fence_check(self, positions: dict[str, float]) -> dict[str, Any]:
        """Check named joint positions against the running fence (``FenceReport.to_dict()``)."""
        fenced = self._fence_for(positions)
        if fenced is None:
            return {"ok": False, "arms": {}, "error": "KiteOS fence is not running"}
        return fenced.check_positions(positions)

    @rpc
    def fence_reload(self, path: str | None = None) -> dict[str, Any]:
        """Load a fence file (default: the one in use) and swap it in atomically.

        Raises if the file is missing or invalid, or would put real hardware in monitor mode; the
        running fence is then left unchanged.
        """
        if not self._fenced:
            raise RuntimeError("KiteOS fence is not running; start the coordinator first")
        source = path if path is not None else (self._fence_source or self.config.fence_path)
        if source is None or not str(source).strip():
            raise ValueError("no fence file is configured; pass the path to reload")
        real = any(self._fence_real.values())
        config, resolved = self._load_fence(real=real, path=str(source))
        running = next(iter(self._fenced.values())).checker.config
        if config.station != running.station:
            message = (
                f"fence {resolved} is for station {config.station!r}, the running fence is for "
                f"{running.station!r}"
            )
            if real:
                raise ValueError(f"KiteOS refuses to reload: {message}")
            logger.warning(f"KiteOS fence reload: {message}")
        for line in describe_fence_change(running, config):
            logger.warning(f"KiteOS fence reload changes {line}")
        for fenced in list(self._fenced.values()):
            fenced.reload(config)
        self._fence_source = resolved
        current: dict[str, Any] = {}
        for hw_id, fenced in list(self._fenced.items()):
            current[hw_id] = fenced.check_positions(_current_targets(fenced))
        return {
            "ok": True,
            "path": resolved,
            "mode": config.mode,
            "margin_m": float(config.margin_m),
            "zones": [z.name for z in config.zones if z.enabled],
            "current": current,
        }

    def _fence_for(self, positions: Mapping[str, float]) -> FencedWholeBodyAdapter | None:
        names = set(positions)
        for fenced in self._fenced.values():
            if names & set(fenced.joint_names):
                return fenced
        return next(iter(self._fenced.values()), None)

    # ------------------------------------------------------------------ fence events

    def _on_fence_event(self, event: FenceEvent) -> None:
        # Tick thread, hardware lock held: record and signal only (no RPCs, no blocking).
        self._fence_inbox.append(event)
        recent = self._recent_fence_events
        key = (event.side, event.kind, event.zone, event.enforced)
        entry = recent.pop(key, None)
        recent[key] = [event, 1] if entry is None else [event, entry[1] + 1]
        while len(recent) > _RECENT_EVENTS:
            recent.popitem(last=False)
        if event.kind in _CANCEL_ON_KINDS and event.enforced:
            self._recent_fence_holds.append(event)
            # Cancel an executing trajectory once per hold episode: a stream that re-executes the
            # trajectory task every message must not be cancelled at the stream rate.
            if event.new_episode:
                self._episode_cancelled = False
            if not self._episode_cancelled:
                self._fence_cancel.set()
                self._fence_wakeup.set()

    def _start_fence_supervisor(self) -> None:
        if self._supervisor is not None and self._supervisor.is_alive():
            return
        self._supervisor_stop.clear()
        self._supervisor = threading.Thread(
            target=self._supervise_fence, name="KiteOS-FenceSupervisor", daemon=True
        )
        self._supervisor.start()

    def _stop_fence_supervisor(self) -> None:
        stop = getattr(self, "_supervisor_stop", None)
        thread = getattr(self, "_supervisor", None)
        if stop is None:
            return
        stop.set()
        self._fence_wakeup.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=_SUPERVISOR_JOIN_S)
        self._supervisor = None
        self._drain_fence_events()

    def _supervise_fence(self) -> None:
        while not self._supervisor_stop.is_set():
            self._fence_wakeup.wait(timeout=_SUPERVISOR_POLL_S)
            self._fence_wakeup.clear()
            try:
                self._drain_fence_events()
                if self._fence_cancel.is_set():
                    self._fence_cancel.clear()
                    self._cancel_after_hold()
            except Exception as exc:  # the supervisor must outlive any single failure
                logger.error(f"KiteOS fence supervisor error: {exc!r}")

    def _cancel_after_hold(self) -> None:
        """Cancel the executing trajectory, at most once per hold episode.

        Nothing happens while no trajectory executes; the next hold event of the same episode
        checks again, so a trajectory that starts during a hold is still cancelled (once).
        """
        if self._episode_cancelled:
            return
        task = self._trajectory_task
        if task is None or not task.is_active():
            return
        result = self.cancel_trajectory()
        if not result.cancelled:
            return
        self._episode_cancelled = True
        self._trajectory_cancels += 1
        self._rate_log(
            "cancel",
            "warning",
            "KiteOS fence held an arm; cancelled the executing trajectory "
            f"(cancel #{self._trajectory_cancels})",
        )

    def _drain_fence_events(self) -> None:
        while True:
            try:
                event = self._fence_inbox.popleft()
            except IndexError:
                return
            self._log_fence_event(event)

    def _log_fence_event(self, event: FenceEvent) -> None:
        level = "info" if event.kind in ("escape", "rate_limit", "limit_clamp") else "warning"
        if event.kind in ("start_inside", "error") and event.enforced:
            level = "error"
        where = f"{event.side} arm" if event.side in ARM_SIDES else "both arms"
        parts = [f"KiteOS fence {event.kind.replace('_', ' ')}: {where}"]
        if event.zone:
            clearance = (
                f", {event.clearance_m * 100:+.1f} cm"
                if event.clearance_m is not None and math.isfinite(event.clearance_m)
                else ""
            )
            parts.append(f"zone {event.zone!r} ({event.sphere}{clearance})")
        if event.detail:
            parts.append(event.detail)
        if not event.enforced:
            parts.append("[not enforced]")
        self._rate_log(f"{event.kind}:{event.side}", level, " - ".join(parts))

    def _rate_log(self, key: str, level: str, message: str) -> None:
        now = time.monotonic()
        last = self._fence_log_last.get(key)
        if last is not None and now - last < _EVENT_LOG_INTERVAL_S:
            self._fence_log_suppressed[key] = self._fence_log_suppressed.get(key, 0) + 1
            return
        self._fence_log_last[key] = now
        suppressed = self._fence_log_suppressed.pop(key, 0)
        if suppressed:
            message = f"{message} (+{suppressed} similar)"
        getattr(logger, level)(message)


def describe_rejection(report: FenceReport) -> str:
    """One line explaining why the fence rejected a motion."""
    reasons = []
    for side, arm in report.arms.items():
        if arm.ok:
            continue
        if arm.zone is not None and math.isfinite(arm.min_clearance_m):
            text = (
                f"{side} arm would be {arm.min_clearance_m * 100:+.1f} cm from zone "
                f"{arm.zone!r} ({arm.sphere}), inside the safety margin"
            )
        else:
            text = f"{side} arm cannot be verified"
        if arm.detail:
            text += f" [{arm.detail}]"
        reasons.append(text)
    return "KiteOS fence rejected the trajectory: " + ("; ".join(reasons) or "unknown reason")


def _current_targets(fenced: FencedWholeBodyAdapter) -> dict[str, float]:
    status = fenced.status()
    named: dict[str, float] = {}
    for side in ARM_SIDES:
        arm = status["arms"][side]
        q = arm.get("last_written")
        if q is not None:
            named.update({name: float(v) for name, v in zip(ARM_JOINTS[side], q, strict=True)})
        opening = arm.get("gripper_opening")
        if opening is not None:
            named[GRIPPER_JOINTS[side]] = float(opening)
    return named


def check_fence_for_real_hardware(config: FenceConfig, source: str) -> None:
    """Raise if ``config`` must not drive real arms (``real_hardware_problems``)."""
    problems = real_hardware_problems(config)
    if problems:
        raise RuntimeError(
            f"KiteOS refuses to drive real arms with fence {source}: " + "; ".join(problems) + "."
        )
    unprotected = config.unprotected_arms()
    if unprotected:
        logger.warning(
            f"KiteOS fence {source}: no enabled zone applies to {unprotected} "
            "(allow_no_zones: true); only joint limits and the rate limit protect them"
        )


def describe_fence_change(old: FenceConfig, new: FenceConfig) -> list[str]:
    """Human-readable settings that differ between two fences (zones are listed elsewhere)."""
    changes = []
    for key in (
        "mode",
        "margin_m",
        "gripper_stroke_m",
        "max_joint_speed_rad_s",
        "joint_limit_margin_rad",
        "interpolation_step_rad",
        "max_feedforward_tau_nm",
        "allow_no_zones",
    ):
        a, b = getattr(old, key), getattr(new, key)
        if a != b:
            changes.append(f"{key}: {a} -> {b}")
    for side in ARM_SIDES:
        a, b = old.mounts[side], new.mounts[side]
        if a != b:
            changes.append(f"mounts.{side}: {tuple(a.xyz)}/{tuple(a.rpy)} -> {tuple(b.xyz)}/{tuple(b.rpy)}")
    if old.extra_spheres != new.extra_spheres:
        changes.append(
            f"extra_spheres: {[x.name for x in old.extra_spheres]} -> "
            f"{[x.name for x in new.extra_spheres]}"
        )
    old_zones = [z.name for z in old.zones if z.enabled]
    new_zones = [z.name for z in new.zones if z.enabled]
    if old_zones != new_zones:
        changes.append(f"enabled zones: {old_zones} -> {new_zones}")
    return changes


def _single_point_trajectory(msg: Any) -> JointTrajectory | None:
    """The trajectory ``JointTrajectoryTask.on_joint_command`` would execute for ``msg``."""
    try:
        names = [str(n) for n in msg.name]
        positions = [float(v) for v in msg.position]
    except Exception:
        return None
    if not names or len(names) != len(positions):
        return None
    return JointTrajectory(joint_names=names, points=[TrajectoryPoint(positions=positions)])

