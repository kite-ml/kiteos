# Copyright 2026 Kite ML.
"""``kiteos fence show|validate|check|teach|test``: author and verify the safety fence.

Offline commands (show, validate, test) only read the fence file. ``check`` and ``teach`` attach
to the running stack with ``Dimos.connect()`` and talk to the ``ControlCoordinator`` RPCs
(``fence_status``, ``fence_check``, ``fence_reload``; joint positions from the
``coordinator_joint_state`` stream, falling back to ``get_joint_positions``). See docs/FENCE.md.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import numpy as np
import typer
from rich.table import Table

from kiteos.cli._common import (
    StationOption,
    console,
    die,
    load_station,
    note,
    prompt_text,
    status_line,
    warn,
)

if TYPE_CHECKING:
    from kiteos.safety.fence import ArmReport, FenceChecker, FenceConfig

app = typer.Typer(help="Author and verify the safety fence (keep-out zones).", no_args_is_help=True)

COORDINATOR = "ControlCoordinator"
JOINT_STATE_STREAM = "coordinator_joint_state"


def reference_poses(config: FenceConfig) -> dict[str, list[float]]:
    """Poses the arms normally start from (per robot); a zone that covers them stops the robot."""
    return {name: list(q) for name, q in config.profile.reference_poses.items()}


# The dual OpenYAM's (kept for callers of the old name).
REFERENCE_POSES: dict[str, list[float]] = {
    "home": [0.0, 1.047, 1.047, 0.0, 0.0, 0.0],
    "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
# Default robot reference for orienting a taught plane: between the arm bases, 25 cm up.
ROBOT_REF_LIFT_M = 0.25

FenceOption = Annotated[
    Optional[Path],
    typer.Option("--fence", "-f", help="Fence file (default: the station's fence).", show_default=False),
]


# --------------------------------------------------------------------------- loading


def fence_file(fence: Path | None, station: Path | None) -> Path:
    if fence is not None:
        return fence.expanduser().resolve()
    return load_station(station).fence_path()


def load_fence(path: Path) -> FenceConfig:
    from kiteos.safety.fence import FenceConfig, FenceConfigError

    if not path.is_file():
        die(f"fence file not found: {path}\nStart from config/fence.example.yaml.", code=2)
    try:
        return FenceConfig.load(path)
    except (FenceConfigError, OSError) as exc:
        die(str(exc), code=2)


def zone_geometry(zone: Any) -> str:
    def vec(v: Sequence[float]) -> str:
        return "(" + ", ".join(f"{x:g}" for x in v) + ")"

    if zone.kind == "half_space":
        return f"point {vec(zone.point)} normal {vec(zone.normal)}"
    text = f"center {vec(zone.center)} half {vec(zone.half_extents)}"
    if any(zone.rpy):
        text += f" rpy {vec(zone.rpy)}"
    return text


def zones_table(config: FenceConfig, title: str = "Zones") -> Table:
    table = Table(title=title, title_justify="left", show_edge=False)
    for column in ("zone", "kind", "frame", "applies to", "enabled", "geometry (m)"):
        table.add_column(column)
    for zone in config.zones:
        enabled = "[ok]yes[/ok]" if zone.enabled else "[muted]no[/muted]"
        table.add_row(zone.name, zone.kind, zone.frame, zone.applies_to, enabled, zone_geometry(zone))
    return table


def fmt_clearance(report: ArmReport) -> str:
    value = report.min_clearance_m
    if math.isinf(value) and value > 0:
        return "no zone applies"
    if not math.isfinite(value):
        return report.detail or "error"
    return f"{value * 100:+.1f} cm to {report.zone} ({report.sphere})"


# --------------------------------------------------------------------------- show / validate


@app.command("show")
def show(fence: FenceOption = None, station: StationOption = None) -> None:
    """Print the fence: mode, margins, mounts, extra spheres and zones."""
    path = fence_file(fence, station)
    config = load_fence(path)
    console.print(f"[kite]Fence[/kite] {path}")
    console.print(
        f"station {config.station or '-'} · mode [kite]{config.mode}[/kite] · margin {config.margin_m * 100:.1f} cm · "
        f"gripper stroke {config.gripper_stroke_m * 100:.1f} cm"
    )
    console.print(
        f"joint speed limit {config.max_joint_speed_rad_s:g} rad/s · joint limit margin "
        f"{config.joint_limit_margin_rad:g} rad · sweep step {config.interpolation_step_rad:g} rad · "
        f"refuse start inside: {'yes' if config.refuse_start_inside else 'no'}"
    )
    for side, mount in config.mounts.items():
        console.print(f"mount {side}: xyz {tuple(mount.xyz)} rpy {tuple(mount.rpy)}")
    if config.extra_spheres:
        for sphere in config.extra_spheres:
            console.print(
                f"extra sphere {sphere.name}: {sphere.side} {sphere.link} center {tuple(sphere.center)} r {sphere.radius:g}"
            )
    if config.zones:
        console.print(zones_table(config))
    enabled = sum(z.enabled for z in config.zones)
    if enabled == 0:
        warn("no zone is enabled: only joint limits and the speed limit are enforced")


@dataclass
class Finding:
    level: str
    name: str
    detail: str = ""
    fix: str | None = None


def validate_fence(config: FenceConfig, *, station_name: str | None = None) -> list[Finding]:
    """Sanity checks beyond the schema. FAIL findings block a real run."""
    from kiteos.safety.fence import MAX_REAL_JOINT_SPEED_RAD_S, FenceChecker

    findings: list[Finding] = [Finding("PASS", "schema", "valid")]
    enabled = [z for z in config.zones if z.enabled]
    unprotected = config.unprotected_arms()
    if enabled:
        findings.append(Finding("PASS", "zones", f"{len(enabled)} enabled: {', '.join(z.name for z in enabled)}"))
    else:
        findings.append(
            Finding(
                "WARN" if config.allow_no_zones else "FAIL",
                "zones",
                "none enabled: only joint limits and the speed limit are enforced"
                + (" (allow_no_zones: true, for teaching)" if config.allow_no_zones else ""),
                "teach the glass/wall with `kiteos fence teach`, then `kiteos fence test`"
                + ("" if config.allow_no_zones else "; --real refuses a fence without zones"),
            )
        )
    if enabled and unprotected:
        covered = [s for s in ("left", "right") if s not in unprotected]
        findings.append(
            Finding(
                "WARN" if config.allow_no_zones else "FAIL",
                "arm coverage",
                f"no enabled zone applies to the {' and '.join(unprotected)} arm "
                f"(zones cover only the {' and '.join(covered)} arm)",
                f"teach it: `kiteos fence teach --arms {unprotected[0]}` (or give the zone "
                "`applies_to: both`); --real refuses this unless allow_no_zones: true",
            )
        )
    if config.mode != "enforce":
        findings.append(
            Finding(
                "WARN",
                "mode",
                f"{config.mode}: violations are only logged",
                "use mode: enforce; --real refuses monitor",
            )
        )
    if config.margin_m < 0.03:
        findings.append(Finding("WARN", "margin", f"{config.margin_m * 100:.1f} cm is tight", "use at least 3 cm"))
    if config.max_joint_speed_rad_s > MAX_REAL_JOINT_SPEED_RAD_S:
        findings.append(
            Finding(
                "FAIL",
                "speed limit",
                f"{config.max_joint_speed_rad_s:g} rad/s",
                f"--real refuses more than {MAX_REAL_JOINT_SPEED_RAD_S:g} rad/s (normal: 2.5)",
            )
        )
    if station_name and config.station and config.station != station_name:
        findings.append(
            Finding(
                "WARN",
                "station",
                f"fence is for {config.station!r}, station is {station_name!r}",
                "--real refuses a fence from another station; check its zones, then fix `station:`",
            )
        )
    wrist = config.profile.wrist_links
    cameraless = [
        side
        for side in ("left", "right")
        if not any(x.side in ("both", side) and x.link in wrist for x in config.extra_spheres)
    ]
    if cameraless:
        findings.append(
            Finding(
                "WARN",
                "wrist cameras",
                f"no extra sphere on the {' and '.join(cameraless)} {'/'.join(wrist)}: the wrist camera "
                "and its cable are not fenced",
                "measure the camera and add it under extra_spheres (docs/FENCE.md)",
            )
        )
    checker = FenceChecker(config)
    poses = reference_poses(config)
    for pose_name, q in poses.items():
        for side in ("left", "right"):
            closed = checker.check_arm(side, q, 0.0)
            wide = checker.check_arm(side, q, None)
            if closed.ok and wide.ok:
                continue
            if closed.ok:
                findings.append(
                    Finding(
                        "FAIL",
                        f"{side} arm at {pose_name} pose",
                        f"clear with the gripper closed ({fmt_clearance(closed)}), but violates it "
                        f"with the gripper open ({fmt_clearance(wide)})",
                        "the gripper could never open there (the fence refuses the opening); move "
                        "the zone away from the workspace",
                    )
                )
            else:
                findings.append(
                    Finding(
                        "FAIL",
                        f"{side} arm at {pose_name} pose",
                        f"violates the fence even with the gripper closed: {fmt_clearance(closed)}",
                        "the arm could not start or return here; move the zone away from the workspace",
                    )
                )
    if all(f.level != "FAIL" for f in findings[1:]) and enabled:
        names = " and ".join(poses) + (" poses are" if len(poses) > 1 else " pose is")
        findings.append(Finding("PASS", "reference poses", f"{names} clear for both arms, gripper closed and open"))
    return findings


@app.command("validate")
def validate(fence: FenceOption = None, station: StationOption = None) -> None:
    """Validate the fence file (schema, enabled zones, start poses clear). Exit 1 on FAIL."""
    station_name = None
    if fence is None:
        config_station = load_station(station)
        path = config_station.fence_path()
        station_name = config_station.station
    else:
        path = fence.expanduser().resolve()
    config = load_fence(path)
    console.print(f"[kite]Validating[/kite] {path}")
    findings = validate_fence(config, station_name=station_name)
    for finding in findings:
        status_line(finding.level, finding.name, finding.detail, finding.fix)  # type: ignore[arg-type]
    if any(f.level == "FAIL" for f in findings):
        raise typer.Exit(1)


# --------------------------------------------------------------------------- live stack


class LiveStack:
    """RPC access to the running ControlCoordinator via ``Dimos.connect()``."""

    def __init__(self, timeout: float = 5.0) -> None:
        from dimos.porcelain.dimos import Dimos

        self._driver = Dimos.connect(timeout=timeout)
        try:
            self.coordinator = self._driver.get_module(COORDINATOR)
        except Exception:
            self._driver.stop()
            raise

    def joint_positions(self) -> dict[str, float]:
        """Measured joint positions from the published joint state (no extra bus read)."""
        try:
            msg = self.coordinator.peek_stream(JOINT_STATE_STREAM, 1.0)
        except Exception:
            msg = None
        names = getattr(msg, "name", None)
        values = getattr(msg, "position", None)
        if names and values and len(names) == len(values):
            return {str(n): float(v) for n, v in zip(names, values, strict=True)}
        return {str(k): float(v) for k, v in dict(self.coordinator.get_joint_positions()).items()}

    def fence_status(self) -> dict[str, Any]:
        return dict(self.coordinator.fence_status())

    def fence_check(self, positions: Mapping[str, float]) -> dict[str, Any]:
        return dict(self.coordinator.fence_check(dict(positions)))

    def fence_reload(self, path: str | None = None) -> dict[str, Any]:
        return dict(self.coordinator.fence_reload(path))

    def close(self) -> None:
        self._driver.stop()


def connect_live() -> LiveStack:
    """Factory used by the live commands (tests replace it)."""
    from kiteos.cli._common import attach_to_run

    attach_to_run("fence")
    try:
        return LiveStack()
    except ImportError as exc:
        die(f"the KiteOS runtime is not importable here ({exc}); run kiteos from the runtime virtualenv (see `kiteos doctor`)", code=2)
    except Exception as exc:
        die(f"cannot reach the running stack ({exc}); start it with `kiteos teleop|collect start|rollout start`")


def parse_positions(text: str, dof: int = 6) -> dict[str, float] | list[float]:
    """``--positions``: JSON object name->value, JSON list, or comma-separated values: both arms
    (2 x ``dof``), optionally followed by both grippers (dual OpenYAM: 12/14, OpenArm: 14/16)."""
    text = text.strip()
    value = json.loads(text) if text.startswith(("{", "[")) else [float(v) for v in text.split(",") if v.strip()]
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    if isinstance(value, list) and len(value) in (2 * dof, 2 * dof + 2):
        return [float(v) for v in value]
    raise ValueError(f"positions must be a JSON object of joint names, or {2 * dof}/{2 * dof + 2} values")


def as_named(positions: Mapping[str, float] | Sequence[float], robot: Any = None) -> dict[str, float]:
    """Joint name -> value; a list is in the robot's joint order (default: the dual OpenYAM)."""
    from kiteos.robots import get_robot

    if isinstance(positions, Mapping):
        return dict(positions)
    names = get_robot(robot).joint_names
    return {name: float(v) for name, v in zip(names[: len(positions)], positions, strict=True)}


def print_report(title: str, report: Mapping[str, Any]) -> bool:
    ok = bool(report.get("ok"))
    console.print(f"{title}: " + ("[ok]clear[/ok]" if ok else "[fail]VIOLATION[/fail]"))
    for side, arm in dict(report.get("arms", {})).items():
        arm = dict(arm)
        clearance = arm.get("min_clearance_m")
        text = (
            "no zone applies"
            if clearance is None and arm.get("ok")
            else (
                f"{clearance * 100:+.1f} cm to {arm.get('zone')} ({arm.get('sphere')})"
                if clearance is not None
                else "error"
            )
        )
        style = "ok" if arm.get("ok") else "fail"
        detail = f"  [muted]{arm['detail']}[/muted]" if arm.get("detail") else ""
        console.print(f"  [{style}]{side:<5}[/{style}] {text}{detail}")
    return ok


@app.command("check")
def check(
    positions: Annotated[
        Optional[str],
        typer.Option(
            "--positions",
            help="Check these instead of the live pose (JSON object/list, or both arms' joint values [+ grippers]).",
        ),
    ] = None,
    offline: Annotated[
        bool, typer.Option("--offline", help="Evaluate with the local fence file; no running stack needed.")
    ] = False,
    fence: FenceOption = None,
    station: StationOption = None,
) -> None:
    """Show the running fence status and check the current (or given) pose. Exit 1 on violation."""
    from kiteos.safety.fence import FenceChecker

    path = fence_file(fence, station)
    robot = load_fence(path).profile if path.is_file() else load_station(station).profile
    parsed = None
    if positions is not None:
        try:
            parsed = parse_positions(positions, robot.dof)
        except (ValueError, json.JSONDecodeError) as exc:
            die(f"--positions: {exc}", code=2)

    if offline:
        if parsed is None:
            die("--offline needs --positions", code=2)
        config = load_fence(path)
        report = FenceChecker(config).check(as_named(parsed, robot)).to_dict()
        ok = print_report(f"local fence {path.name}", report)
        raise typer.Exit(0 if ok else 1)

    live = connect_live()
    try:
        status = live.fence_status()
        console.print("[kite]Running fence[/kite]")
        console.print(json.dumps(status, indent=2, default=str))
        named = as_named(parsed, robot) if parsed is not None else live.joint_positions()
        ok = print_report("running stack", live.fence_check(named))
        if path.is_file():
            try:
                local = FenceChecker(load_fence(path)).check(named).to_dict()
                if bool(local.get("ok")) != ok:
                    warn(
                        f"{path.name} gives a different verdict than the running stack, which runs an "
                        "older fence; restart the stack to apply the file"
                    )
            except typer.Exit:
                warn(f"local fence file {path} is invalid")
    finally:
        live.close()
    raise typer.Exit(0 if ok else 1)


# --------------------------------------------------------------------------- teach


def mount_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """World-from-base transform, URDF convention Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def frame_poses(config: FenceConfig) -> dict[str, np.ndarray]:
    """World pose of each zone frame from the fence mounts."""
    from kiteos.robots import WORLD_ALIAS

    poses = {WORLD_ALIAS: np.eye(4), config.profile.world_frame: np.eye(4)}
    for side in ("left", "right"):
        mount = config.mounts[side]
        poses[f"{side}_base"] = mount_matrix(mount.xyz, mount.rpy)
    return poses


def to_frame(point_world: np.ndarray, frame_pose: np.ndarray) -> np.ndarray:
    R, t = frame_pose[:3, :3], frame_pose[:3, 3]
    return R.T @ (np.asarray(point_world, dtype=float) - t)


def default_robot_ref(config: FenceConfig) -> np.ndarray:
    """A point well inside the workspace (world): between the arm bases, lifted 25 cm."""
    left = np.array(config.mounts["left"].xyz, dtype=float)
    right = np.array(config.mounts["right"].xyz, dtype=float)
    return np.asarray((left + right) / 2.0 + np.array([0.0, 0.0, ROBOT_REF_LIFT_M]))


def backup_file(path: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    return backup


def upsert_zone(config: FenceConfig, zone: Any) -> FenceConfig:
    """Config with ``zone`` replacing the zone of the same name in place, or appended."""
    from kiteos.safety.fence import FenceConfig

    zones = list(config.zones)
    for i, existing in enumerate(zones):
        if existing.name == zone.name:
            zones[i] = zone
            break
    else:
        zones.append(zone)
    data = config.model_dump(mode="json")
    data["zones"] = [z.model_dump(mode="json") for z in zones]
    return FenceConfig.model_validate(data)


def write_fence(config: FenceConfig, path: Path, note_text: str) -> Path | None:
    """Back up the current file (if any) and replace it atomically with ``config``."""
    backup = backup_file(path) if path.exists() else None
    header = f"# {note_text}\n" + (f"# Previous version (with its comments): {backup.name}\n" if backup else "")
    tmp = path.with_name(f".{path.name}.kiteos-tmp")
    try:
        config.dump(tmp)
        tmp.write_text(header + tmp.read_text())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return backup


def preview_zone(
    config: FenceConfig,
    zone: Any,
    points: np.ndarray,
    current: Mapping[str, float] | None,
    standoff: float = 0.0,
) -> bool:
    """Show the fitted zone, its fit quality and whether the reference and current poses stay clear."""
    clear = True
    from kiteos.safety.fence import FenceChecker
    from kiteos.safety.teach import TILT_CHECK_RADIUS_M, plane_residuals, plane_tilt_uncertainty

    to_zone = plane_residuals(points, zone.point, zone.normal) if zone.kind == "half_space" else np.zeros(0)
    console.print(f"[kite]Proposed zone[/kite] {zone.name}: {zone.kind} in {zone.frame}, applies to {zone.applies_to}")
    console.print(f"  {zone_geometry(zone)}")
    if to_zone.size:
        # The zone plane is the best-fit plane (through the points' centroid) moved by the standoff
        # toward the robot, so the fit residuals are the distances to the zone minus their mean.
        standoff = float(np.mean(to_zone))
        residuals = to_zone - standoff
        console.print(
            f"  fit: {len(points)} points, rms {float(np.sqrt(np.mean(residuals**2))) * 1000:.1f} mm, "
            f"worst {float(np.max(np.abs(residuals))) * 1000:.1f} mm; "
            f"zone plane {standoff * 1000:.1f} mm in front of the touched surface"
        )
        console.print(
            "  per-point residuals (mm): " + ", ".join(f"{round(r * 1000, 1) + 0.0:+.1f}" for r in residuals)
        )
        tilt = plane_tilt_uncertainty(points)
        drift = 2.0 * TILT_CHECK_RADIUS_M * tilt
        style = "fail" if drift > abs(standoff) else "ok"
        console.print(
            f"  tilt uncertainty {math.degrees(tilt):.2f} deg: the plane may be off by up to "
            f"[{style}]{drift * 1000:.1f} mm[/{style}] {TILT_CHECK_RADIUS_M * 100:.0f} cm from the touched "
            f"points (standoff {standoff * 1000:.1f} mm)"
        )
        if drift > abs(standoff):
            clear = False
            warn(
                "the fit is not precise enough for this standoff: away from the touched points the "
                "zone may sit BEHIND the real glass. Touch more (and more widely spread) points."
            )
    trial = upsert_zone(config, zone.model_copy(update={"enabled": True}))
    checker = FenceChecker(trial)
    for pose_name, q in reference_poses(trial).items():
        for side in ("left", "right"):
            if not any(z.enabled and z.applies(side) for z in trial.zones):
                console.print(f"  {side} arm at {pose_name}: [fail]NO ZONE APPLIES to this arm[/fail]")
                continue
            report = checker.check_arm(side, q, None)
            style = "ok" if report.ok else "fail"
            console.print(f"  {side} arm at {pose_name}: [{style}]{fmt_clearance(report)}[/{style}]")
            clear = clear and report.ok
    for side in trial.unprotected_arms():
        warn(f"after this change the {side} arm has NO keep-out zone; teach it too (`--arms {side}`)")
    if current is not None:
        now = checker.check(current)
        for side, arm in now.arms.items():
            style = "ok" if arm.ok else "warn"
            console.print(f"  {side} arm now: [{style}]{fmt_clearance(arm)}[/{style}]")
    return clear


def default_zone_name(arms: str) -> str:
    """``glass_window`` for both arms, ``glass_window_<side>`` for one: per-arm teach runs never
    overwrite each other's zone."""
    return "glass_window" if arms == "both" else f"glass_window_{arms}"


def replacement_conflict(config: FenceConfig, zone_name: str, frame: str, applies: str) -> str | None:
    """Why replacing the existing zone ``zone_name`` would silently change what it protects."""
    for existing in config.zones:
        if existing.name == zone_name and (existing.frame != frame or existing.applies_to != applies):
            return (
                f"zone {zone_name!r} already exists in frame {existing.frame} for "
                f"{existing.applies_to} arm(s); the new one would be in {frame} for {applies}, "
                "leaving the other arm unprotected"
            )
    return None


@app.command("teach")
def teach(
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            help="Zone name (default: glass_window, or glass_window_<side> with --arms left|right).",
        ),
    ] = None,
    arms: Annotated[str, typer.Option("--arms", help="Arms used to touch the surface: both, left or right.")] = "both",
    frame: Annotated[
        Optional[str],
        typer.Option(
            "--frame",
            help="Zone frame: the robot's world frame (dual_openyam_base, openarm_body_link0), left_base or "
            "right_base (default: per --arms).",
        ),
    ] = None,
    applies_to: Annotated[
        Optional[str],
        typer.Option(
            "--applies-to",
            help="both, left or right (default: the frame's arm, else both; required for --arms both with a per-arm frame).",
        ),
    ] = None,
    standoff: Annotated[
        float,
        typer.Option("--standoff", min=0.0, help="Move the plane this far (m) toward the robot (absorbs FK error)."),
    ] = 0.02,
    min_points: Annotated[
        int, typer.Option("--min-points", min=2, help="Points to touch per arm (at least 4 per fit).")
    ] = 4,
    replace: Annotated[
        bool, typer.Option("--replace", help="Allow replacing a same-name zone with another frame/arm.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Accept a plane fit with a large residual.")] = False,
    fence: FenceOption = None,
    station: StationOption = None,
) -> None:
    """Teach a flat keep-out surface (glass, wall) by touching it with the fingertips."""
    from kiteos.safety.kinematics import DualArmKinematics
    from kiteos.safety.teach import MIN_FIT_POINTS, TeachError, plane_zone_from_points

    if arms not in ("both", "left", "right"):
        die("--arms must be both, left or right", code=2)
    sides = ["left", "right"] if arms == "both" else [arms]
    name = name or default_zone_name(arms)
    path = fence_file(fence, station)
    config = load_fence(path)
    robot = config.profile
    world = robot.world_frame
    zone_frame = frame or (world if arms == "both" else f"{arms}_base")
    if zone_frame == "world":
        zone_frame = world
    if zone_frame not in (world, "left_base", "right_base"):
        die(f"--frame must be {world} (or world), left_base or right_base", code=2)
    if arms == "both" and zone_frame != world and applies_to is None:
        die(
            f"--arms both with --frame {zone_frame}: say which arms the zone protects with "
            "--applies-to (both, left or right), or teach each arm in its own frame "
            "(`--arms left`, then `--arms right`)",
            code=2,
        )
    # A zone in one arm's base frame is only trustworthy for that arm (the spacing may be unmeasured).
    applies = applies_to or (zone_frame.removesuffix("_base") if zone_frame != world else "both")
    if applies not in ("both", "left", "right"):
        die("--applies-to must be both, left or right", code=2)
    touched = set(sides)
    if applies != "both" and {applies} != touched:
        warn(f"the zone will protect the {applies} arm, but it is touched with {' and '.join(sides)}")
    per_arm_min = max(min_points, MIN_FIT_POINTS if len(sides) == 1 else 2)

    conflict = replacement_conflict(config, name, zone_frame, applies)
    if conflict and not replace:
        die(f"{conflict}.\nUse another --name, or pass --replace if that is intended.", code=2)
    poses = frame_poses(config)
    kinematics = DualArmKinematics(mounts={s: poses[f"{s}_base"] for s in ("left", "right")}, robot=robot)

    live = connect_live()
    points_world: list[np.ndarray] = []
    try:
        console.print(
            f"[kite]Teaching {name}[/kite] in frame {zone_frame} for the {applies} arm(s). The arms stay under the\n"
            "operator's control: move them with the Quest (or by hand if gravity-compensated) and touch the\n"
            "surface with the FINGERTIP (the end of the closed fingers). Touch at least 4 points (5 is better),\n"
            "spread over the surface at least 20 cm apart, and hold still while each one is recorded."
        )
        for side in sides:
            count = 0
            while True:
                hint = "Enter to record" if count < per_arm_min else "Enter to record, 'done' to finish this arm"
                answer = prompt_text(f"{side} arm point {count + 1} ({hint}, 'q' to abort)")
                if answer is None or answer.strip().lower() in ("q", "quit", "abort"):
                    die("teach aborted; the fence file was not changed", code=1)
                if answer.strip().lower() == "done":
                    if count >= per_arm_min:
                        break
                    warn(f"need at least {per_arm_min} points for the {side} arm")
                    continue
                try:
                    joints = live.joint_positions()
                except Exception as exc:
                    warn(f"cannot read the joint state ({exc}); try again")
                    continue
                missing = [j for j in robot.arm_joints[side] if j not in joints]
                if missing:
                    warn(f"joint state is missing {missing}; is the stack running (for the {robot.title})?")
                    continue
                q = np.array([joints[j] for j in robot.arm_joints[side]], dtype=float)
                tip = kinematics.fingertip(side, q)
                points_world.append(tip)
                count += 1
                console.print(f"  recorded {side} fingertip at ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m (world)")

        frame_pose = poses[zone_frame]
        points = np.array([to_frame(p, frame_pose) for p in points_world])
        robot_ref = to_frame(default_robot_ref(config), frame_pose)
        try:
            zone = plane_zone_from_points(
                points,
                robot_ref,
                name,
                frame=zone_frame,  # type: ignore[arg-type]  # validated above
                applies_to=applies,  # type: ignore[arg-type]
                standoff_m=standoff,
                force=force,
            )
        except TeachError as exc:
            die(f"cannot fit a zone: {exc}", code=1)

        try:
            current: dict[str, float] | None = live.joint_positions()
        except Exception:
            current = None
        clear = preview_zone(config, zone, points, current, standoff=standoff)
        if not clear:
            warn("check the points before enabling this zone (see above)")
        answer = prompt_text("Write this zone to the fence file? [y/N]")
        if answer is None or answer.strip().lower() not in ("y", "yes"):
            die("not written; the fence file was not changed", code=1)
        enable_answer = prompt_text("Enable it now? [Y/n]")
        enabled = enable_answer is not None and enable_answer.strip().lower() in ("", "y", "yes")
        new_config = upsert_zone(config, zone.model_copy(update={"enabled": enabled}))
        if new_config.allow_no_zones and not new_config.unprotected_arms():
            # Both arms are covered now: the teaching opt-in is no longer needed.
            new_config = new_config.model_copy(update={"allow_no_zones": False})
            note("both arms have a keep-out zone now: allow_no_zones switched off")
        backup = write_fence(
            new_config,
            path,
            f"Zone {name!r} taught by `kiteos fence teach` on {datetime.now().isoformat(timespec='seconds')}.",
        )
        console.print(f"[ok]wrote[/ok] {path}" + (f" (backup: {backup.name})" if backup else ""))
        try:
            result = live.fence_reload(str(path))
            console.print(f"running stack reloaded the fence: {result}")
        except Exception as exc:
            warn(f"the running stack did not reload the fence ({exc}); restart it to apply the zone")
    finally:
        live.close()
    note("next: `kiteos fence test --dataset <recorded dataset>` to confirm normal task motion is not blocked")


# --------------------------------------------------------------------------- test


@dataclass
class SweepRow:
    zone: str
    enabled: bool
    blocked: dict[str, float | None] = field(default_factory=dict)


def sweep(config: FenceConfig, samples: int = 2000, seed: int = 0) -> tuple[list[SweepRow], dict[str, float]]:
    """Fraction of uniformly sampled arm configurations (within URDF limits) each zone blocks.

    Every zone is evaluated alone and enabled (so a disabled zone can be tried before enabling
    it); the second result is the fraction blocked by all currently enabled zones together.
    Grippers count as fully open (the conservative, largest finger spheres).
    """
    from kiteos.safety.fence import FenceChecker

    base = FenceChecker(config)
    rng = np.random.default_rng(seed)
    qs: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        limits = base.joint_limits[side]
        qs[side] = rng.uniform(limits[:, 0], limits[:, 1], size=(samples, base.dof))

    def blocked_fraction(checker: FenceChecker, side: str) -> float:
        return float(np.mean([not checker.check_arm(side, q, None).ok for q in qs[side]]))

    rows = []
    for zone in config.zones:
        single = upsert_zone(config.model_copy(update={"zones": []}), zone.model_copy(update={"enabled": True}))
        checker = FenceChecker(single)
        row = SweepRow(zone=zone.name, enabled=zone.enabled)
        for side in ("left", "right"):
            row.blocked[side] = blocked_fraction(checker, side) if zone.applies_to in ("both", side) else None
        rows.append(row)
    combined = {side: blocked_fraction(base, side) for side in ("left", "right")}
    return rows, combined


@dataclass
class HeldFrame:
    episode: int
    frame: int
    side: str
    stream: str
    zone: str | None
    sphere: str | None
    clearance_m: float
    detail: str | None = None


@dataclass
class ReplayResult:
    dataset: Path
    frames: int = 0
    episodes: int = 0
    held: list[HeldFrame] = field(default_factory=list)

    @property
    def held_action_frames(self) -> int:
        return len({(h.episode, h.frame) for h in self.held if h.stream == "action"})


def _feature_order(info: Mapping[str, Any], key: str, joints: Sequence[str] | None = None) -> list[int] | None:
    """Indices that reorder a dataset vector into the robot's joint order (None = already).

    ``joints`` defaults to the dual OpenYAM's ``DUAL_OPENYAM_JOINTS``."""
    from kiteos.safety.kinematics import DUAL_OPENYAM_JOINTS

    order = list(joints) if joints is not None else list(DUAL_OPENYAM_JOINTS)
    names = info.get("features", {}).get(key, {}).get("names")
    if not names or list(names) == order:
        return None
    missing = [n for n in order if n not in names]
    if missing:
        raise ValueError(f"{key} lacks joints {missing}")
    return [list(names).index(n) for n in order]


def iter_dataset(
    root: Path, joints: Sequence[str] | None = None
) -> Iterator[tuple[int, int, np.ndarray | None, np.ndarray | None]]:
    """(episode, frame, state, action) for every frame of a LeRobot v3 dataset, in order; the
    vectors are in ``joints`` order (default: the dual OpenYAM's 14 joints)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - pyarrow ships with the runtime
        raise RuntimeError(f"reading datasets needs pyarrow ({exc})") from exc

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset (no meta/info.json): {root}")
    info = json.loads(info_path.read_text())
    orders = {key: _feature_order(info, key, joints) for key in ("observation.state", "action")}
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    for path in files:
        table = pq.read_table(path)
        columns = set(table.column_names)
        n = table.num_rows
        episodes = table.column("episode_index").to_pylist() if "episode_index" in columns else [0] * n
        frames = table.column("frame_index").to_pylist() if "frame_index" in columns else list(range(n))
        vectors: dict[str, np.ndarray | None] = {}
        for key in ("observation.state", "action"):
            if key in columns:
                arr = np.asarray(table.column(key).to_pylist(), dtype=float)
                order = orders[key]
                vectors[key] = arr[:, order] if order is not None else arr
            else:
                vectors[key] = None
        for i in range(n):
            state = vectors["observation.state"]
            action = vectors["action"]
            yield (
                int(episodes[i]),
                int(frames[i]),
                None if state is None else state[i],
                None if action is None else action[i],
            )


def replay_dataset(
    config: FenceConfig,
    root: Path,
    *,
    stride: int = 1,
    motion: bool = True,
    progress: Callable[[int], None] | None = None,
) -> ReplayResult:
    """Run a dataset's recorded states and actions through the fence.

    An action frame is *held* when its target violates the fence, or (``motion``) when the
    straight joint-space motion from the previous action of the same episode does, exactly as
    the adapter judges each command. State violations mean the arm physically went there.
    """
    from kiteos.safety.fence import FenceChecker

    checker = FenceChecker(config)
    robot, dof = config.profile, checker.dof
    result = ReplayResult(dataset=root)
    previous: dict[str, np.ndarray] = {}
    last_episode: int | None = None
    episodes: set[int] = set()
    sides = [
        (side, slice(0, dof) if side == "left" else slice(dof, 2 * dof), 2 * dof if side == "left" else 2 * dof + 1)
        for side in ("left", "right")
    ]
    for count, (episode, frame, state, action) in enumerate(iter_dataset(root, robot.joint_names)):
        if episode != last_episode:
            previous.clear()
            last_episode = episode
        episodes.add(episode)
        if count % stride:
            continue
        result.frames += 1
        if progress is not None and result.frames % 2000 == 0:
            progress(result.frames)
        for side, cols, grip in sides:
            for stream, vec in (("state", state), ("action", action)):
                if vec is None:
                    continue
                q6 = vec[cols]
                opening = float(np.clip(vec[grip], 0.0, 1.0)) if vec.shape[0] > grip else None
                report = checker.check_arm(side, q6, opening)
                if report.ok and stream == "action" and motion and side in previous:
                    report = checker.check_arm_motion(side, previous[side], q6, opening)
                if stream == "action":
                    previous[side] = q6
                if not report.ok:
                    result.held.append(
                        HeldFrame(
                            episode,
                            frame,
                            side,
                            stream,
                            report.zone,
                            report.sphere,
                            report.min_clearance_m,
                            report.detail,
                        )
                    )
    result.episodes = len(episodes)
    return result


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


@app.command("test")
def test(
    dataset: Annotated[
        Optional[list[Path]],
        typer.Option("--dataset", "-d", help="LeRobot v3 dataset to replay through the fence (repeatable)."),
    ] = None,
    samples: Annotated[int, typer.Option("--samples", min=100, help="Random configurations per arm.")] = 2000,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for the sweep.")] = 0,
    stride: Annotated[int, typer.Option("--stride", min=1, help="Replay every Nth frame.")] = 1,
    no_motion: Annotated[
        bool, typer.Option("--no-motion", help="Replay: skip the swept check between successive actions.")
    ] = False,
    examples: Annotated[int, typer.Option("--examples", min=0, help="Held frames to list per dataset.")] = 10,
    fence: FenceOption = None,
    station: StationOption = None,
) -> None:
    """Offline test: joint-space sweep per zone, and replay of recorded datasets.

    Exit 1 when a replayed dataset has frames the fence would hold (a zone that blocks
    normal task motion must be fixed before going live).
    """
    path = fence_file(fence, station)
    config = load_fence(path)
    console.print(f"[kite]Fence test[/kite] {path} · margin {config.margin_m * 100:.1f} cm")

    rows, combined = sweep(config, samples=samples, seed=seed)
    table = Table(
        title=f"Joint-space sweep ({samples} random configurations per arm)", title_justify="left", show_edge=False
    )
    for column in ("zone", "enabled", "left blocked", "right blocked"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.zone,
            "yes" if row.enabled else "[muted]no[/muted]",
            _pct(row.blocked["left"]),
            _pct(row.blocked["right"]),
        )
    table.add_row("[kite]all enabled zones[/kite]", "", _pct(combined["left"]), _pct(combined["right"]))
    console.print(table)
    if not rows:
        warn("no zones defined: the sweep blocks nothing")

    failed = False
    for root in dataset or []:
        root = root.expanduser().resolve()
        try:
            result = replay_dataset(
                config,
                root,
                stride=stride,
                motion=not no_motion,
                progress=lambda n: console.print(f"  [muted]{n} frames ...[/muted]"),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            die(str(exc), code=2)
        held = result.held_action_frames
        state_hits = sum(1 for h in result.held if h.stream == "state")
        style = "fail" if held else "ok"
        console.print(
            f"[kite]{root.name}[/kite]: {result.frames} frames, {result.episodes} episodes -> "
            f"[{style}]{held} action frames held[/{style}]"
            + (f" ({held / result.frames * 100:.2f}%)" if result.frames else "")
            + (f", [warn]{state_hits} recorded states inside the fence[/warn]" if state_hits else "")
        )
        if result.held:
            by_zone = Counter(h.zone or "error" for h in result.held)
            console.print("  by zone: " + ", ".join(f"{z}: {n}" for z, n in by_zone.most_common()))
            listing = Table(show_edge=False)
            for column in (
                "episode",
                "frame",
                "arm",
                "stream",
                "zone",
                "sphere",
                f"clearance (need > {config.margin_m * 100:g} cm)",
            ):
                listing.add_column(column)
            for h in result.held[:examples]:
                clearance = f"{h.clearance_m * 100:+.1f} cm" if math.isfinite(h.clearance_m) else (h.detail or "error")
                listing.add_row(
                    str(h.episode), str(h.frame), h.side, h.stream, h.zone or "-", h.sphere or "-", clearance
                )
            console.print(listing)
            failed = failed or held > 0 or state_hits > 0
    if failed:
        console.print("[fail]The fence would interfere with recorded task motion.[/fail] Move or shrink the zone.")
        raise typer.Exit(1)
