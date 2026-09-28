# Copyright 2026 Kite ML.
"""KiteOS safety fence: configuration, checker and reports.

``FenceChecker`` answers one question for the control tick: would this arm configuration (or the
straight joint-space motion to it) bring any covering sphere within ``margin_m`` of a zone? The
check is pure numpy, vectorised over interpolated samples, and never raises: any internal error
produces a failing report so callers hold position (fail closed).

See docs/FENCE.md for what the fence does and does not guarantee.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationError,
    field_validator,
    model_validator,
)

from kiteos.yamlio import load_yaml_unique
from kiteos.safety.geometry import DEFAULT_GRIPPER_STROKE_M, SphereModel, Spheres, normalize_opening
from kiteos.safety.kinematics import (
    ARM_JOINTS,
    ARM_SIDES,
    DOF_PER_ARM,
    GRIPPER_JOINTS,
    LINK_NAMES,
    DualArmKinematics,
    Side,
    pose_matrix,
)
from kiteos.safety.zones import ZONE_KINDS, CompiledZone, Zone, compile_zone, frame_transforms

Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
MIN_MARGIN_M = 0.02
# A single swept check never evaluates more samples than this; longer motions fail closed.
MAX_MOTION_SAMPLES = 4096
# Plausibility bounds: a units typo (mm for m) or a coarse sweep silently weakens the fence.
MAX_INTERPOLATION_STEP_RAD = 0.02
MAX_EXTRA_SPHERE_OFFSET_M = 0.3
MAX_EXTRA_SPHERE_RADIUS_M = 0.2
MAX_MOUNT_OFFSET_M = 2.0
# Refused on real hardware (kiteos.safety.coordinator, kiteos.cli.launch): above this the
# final-stage rate limit no longer bounds anything the runtime tasks command (teleop 2.0 rad/s).
MAX_REAL_JOINT_SPEED_RAD_S = 3.0
# Upper bound for max_feedforward_tau_nm (the Damiao adapter adds gravity compensation itself).
MAX_FEEDFORWARD_TAU_NM = 20.0
# Escape rule tolerances (see ``escape_path_ok``).
ESCAPE_MIN_GAIN_M = 1e-6
ESCAPE_DEPTH_TOLERANCE_M = 1e-9

_YAML_HEADER = "# KiteOS fence configuration. Reference: docs/FENCE.md\n"


class FenceConfigError(ValueError):
    """A fence config file could not be read or failed validation."""


class Mount(BaseModel):
    """Pose of one arm base in ``dual_openyam_base``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    xyz: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)

    @field_validator("xyz")
    @classmethod
    def _plausible_offset(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if float(np.linalg.norm(v)) > MAX_MOUNT_OFFSET_M:
            raise ValueError(
                f"mount is {float(np.linalg.norm(v)):.2f} m from the robot origin (limit "
                f"{MAX_MOUNT_OFFSET_M:g} m); values are metres, not millimetres"
            )
        return v

    def matrix(self) -> np.ndarray:
        """4x4 world-from-arm-base transform."""
        return pose_matrix(self.xyz, self.rpy)


class ExtraSphere(BaseModel):
    """An additional covering sphere rigidly attached to an arm link (e.g. a wrist camera)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    side: Literal["left", "right", "both"] = "both"
    link: str
    center: Vec3  # in the link frame, metres
    radius: float = Field(gt=0, le=MAX_EXTRA_SPHERE_RADIUS_M, allow_inf_nan=False)

    @field_validator("center")
    @classmethod
    def _plausible_center(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if float(np.linalg.norm(v)) > MAX_EXTRA_SPHERE_OFFSET_M:
            raise ValueError(
                f"centre is {float(np.linalg.norm(v)):.2f} m from the link origin (limit "
                f"{MAX_EXTRA_SPHERE_OFFSET_M:g} m); values are metres, not millimetres"
            )
        return v

    @field_validator("link")
    @classmethod
    def _known_link(cls, v: str) -> str:
        if v not in LINK_NAMES:
            raise ValueError(
                f"unknown link {v!r}; use one of: {', '.join(LINK_NAMES)} (no side prefix)"
            )
        return v


def _default_mounts() -> dict[Side, Mount]:
    return {"left": Mount(xyz=(0.0, 0.31, 0.0)), "right": Mount(xyz=(0.0, -0.31, 0.0))}


class FenceConfig(BaseModel):
    """The fence file (``config/fence.yaml``)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    version: Literal[1] = 1
    station: str = ""
    mode: Literal["enforce", "monitor"] = "enforce"  # monitor = log only; refused on real hardware
    margin_m: float = Field(0.05, ge=MIN_MARGIN_M, allow_inf_nan=False)
    gripper_stroke_m: float = Field(DEFAULT_GRIPPER_STROKE_M, ge=0, allow_inf_nan=False)
    mounts: dict[Side, Mount] = Field(default_factory=_default_mounts)
    extra_spheres: list[ExtraSphere] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)
    max_joint_speed_rad_s: float = Field(2.5, gt=0, allow_inf_nan=False)
    joint_limit_margin_rad: float = Field(0.01, ge=0, allow_inf_nan=False)
    interpolation_step_rad: float = Field(
        0.01, gt=0, le=MAX_INTERPOLATION_STEP_RAD, allow_inf_nan=False
    )
    refuse_start_inside: bool = True
    # Largest feed-forward torque (Nm) passed to an arm motor; the Damiao adapter adds its own
    # gravity compensation, so the default forwards none.
    max_feedforward_tau_nm: float = Field(
        0.0, ge=0.0, le=MAX_FEEDFORWARD_TAU_NM, allow_inf_nan=False
    )
    # Real hardware refuses a fence that leaves an arm without an enabled zone unless this is
    # true (only for teaching the first zone; `kiteos fence teach` clears it once both arms
    # are covered).
    allow_no_zones: bool = False

    @field_validator("mounts")
    @classmethod
    def _both_mounts(cls, v: dict[str, Mount]) -> dict[str, Mount]:
        missing = [side for side in ARM_SIDES if side not in v]
        if missing:
            raise ValueError(f"mounts must define both 'left' and 'right' (missing {missing})")
        return v

    @model_validator(mode="after")
    def _unique_names(self) -> FenceConfig:
        for label, names in (
            ("zone", [z.name for z in self.zones]),
            ("extra sphere", [s.name for s in self.extra_spheres]),
        ):
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"duplicate {label} names: {dupes}")
        return self

    def unprotected_arms(self) -> list[str]:
        """Arms that no enabled zone applies to."""
        return [side for side in ARM_SIDES if not any(z.enabled and z.applies(side) for z in self.zones)]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> FenceConfig:
        """Load and validate a YAML fence file; raises ``FenceConfigError`` with clear details.

        An empty (or comment-only) file is an error, never "a fence with no zones", and so is a
        duplicated key anywhere in the file.
        """
        p = Path(path).expanduser()
        try:
            data = load_yaml_unique(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise FenceConfigError(f"{p}: invalid YAML: {exc}") from exc
        if data is None:
            raise FenceConfigError(
                f"{p}: the fence file is empty (or only comments). Restore it from its backup "
                "(fence.yaml.bak-*) or config/fence.example.yaml; KiteOS never treats an empty "
                "file as a fence."
            )
        if not isinstance(data, Mapping):
            raise FenceConfigError(
                f"{p}: expected a mapping at the top level, got {type(data).__name__}"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise FenceConfigError(
                f"{p}: invalid fence config:\n{format_validation_error(exc)}"
            ) from exc

    def dump(self, path: str | os.PathLike[str]) -> None:
        """Write the config as YAML (atomic replace, so a running reload never sees half a file)."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        text = _YAML_HEADER + self.to_yaml()
        fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=p.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.model_dump(mode="json"), sort_keys=False, default_flow_style=None
        )

    def frame_transforms(self) -> dict[str, np.ndarray]:
        """World pose of each zone frame (``dual_openyam_base``, ``left_base``, ``right_base``)."""
        return frame_transforms({side: self.mounts[side].matrix() for side in ARM_SIDES})


def real_hardware_problems(config: FenceConfig) -> list[str]:
    """Why ``config`` must not drive real arms (empty list: it may). Shared by the coordinator
    and the CLI so both refuse the same things."""
    problems = []
    if config.mode != "enforce":
        problems.append(f"fence mode is {config.mode!r}; only 'enforce' drives real arms (monitor only logs)")
    unprotected = config.unprotected_arms()
    if unprotected and not config.allow_no_zones:
        arms = " and ".join(unprotected)
        problems.append(
            f"no enabled zone applies to the {arms} arm{'s' if len(unprotected) > 1 else ''}; teach "
            "the glass for both arms (`kiteos fence teach`), or set `allow_no_zones: true` in the "
            "fence file while teaching the first zone"
        )
    if config.max_joint_speed_rad_s > MAX_REAL_JOINT_SPEED_RAD_S:
        problems.append(
            f"max_joint_speed_rad_s {config.max_joint_speed_rad_s:g} rad/s is above the "
            f"{MAX_REAL_JOINT_SPEED_RAD_S:g} rad/s allowed on real hardware"
        )
    return problems


_ERROR_HINTS = {
    "margin_m": f"KiteOS never allows less than {MIN_MARGIN_M * 100:.0f} cm of margin",
    "interpolation_step_rad": "joint-space step between swept samples, e.g. 0.01",
}


def format_validation_error(exc: ValidationError) -> str:
    """One readable line per pydantic error: ``  - zones[0].normal: <message>``."""
    kinds = ", ".join(ZONE_KINDS)
    lines = []
    for err in exc.errors():
        raw_loc = tuple(err.get("loc", ()))
        parts: list[str] = []
        for item in raw_loc:
            if isinstance(item, int):
                parts.append(f"[{item}]")
            elif item in ZONE_KINDS and parts and parts[-1].startswith("["):
                continue  # discriminator tag inserted by pydantic, not a user key
            else:
                parts.append(("." if parts else "") + str(item))
        loc = "".join(parts) or "config"
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        if err.get("type") == "union_tag_not_found":
            msg = f"missing 'kind' (one of: {kinds})"
        elif err.get("type") == "union_tag_invalid":
            given = err.get("input")
            kind = given.get("kind") if isinstance(given, Mapping) else None
            msg = f"unknown kind {kind!r} (one of: {kinds})"
        hint = _ERROR_HINTS.get(str(raw_loc[0])) if raw_loc else None
        lines.append(f"  - {loc}: {msg}" + (f" ({hint})" if hint else ""))
    return "\n".join(lines)


@dataclass
class ArmReport:
    """Fence verdict for one arm.

    ``min_clearance_m`` is the smallest sphere-to-zone clearance, margin NOT subtracted (``inf``
    when no zone applies, ``-inf`` on an internal error). ``ok`` iff it is ``> margin_m``.
    ``zone``/``sphere`` name the closest pair even when ``ok``.
    """

    side: str
    ok: bool
    min_clearance_m: float
    zone: str | None
    sphere: str | None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "ok": self.ok,
            "min_clearance_m": self.min_clearance_m
            if math.isfinite(self.min_clearance_m)
            else None,
            "zone": self.zone,
            "sphere": self.sphere,
            "detail": self.detail,
        }


@dataclass
class FenceReport:
    """Fence verdict for both arms."""

    ok: bool
    arms: dict[str, ArmReport] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "arms": {side: report.to_dict() for side, report in self.arms.items()},
        }


def escape_path_ok(
    clearances: np.ndarray, margin: float, *, tol: float = ESCAPE_DEPTH_TOLERANCE_M
) -> bool:
    """Whether a path keeps every (zone, sphere) pair from getting worse while it is too close.

    ``clearances`` is ``(M, P)``: the clearance of each of ``P`` zone/sphere pairs at ``M``
    consecutive samples of a densely sampled path. Between every two consecutive samples, a pair
    that is clear of the margin must stay clear of it, and a pair within the margin must not lose
    clearance (beyond float noise). So no part of the arm ever gets closer to any zone than it
    was while it is too close, and nothing that was clear enters the margin: an arm can back out
    of a violation, but cannot trade one sphere or zone against another. Non-finite values are
    never an escape.
    """
    C = np.asarray(clearances, dtype=float)
    if C.ndim != 2 or C.shape[0] < 1 or not np.all(np.isfinite(C)):
        return False
    before, after = C[:-1], C[1:]
    ok = np.where(before > margin, after > margin, after >= before - tol)
    return bool(np.all(ok))


def _fail(side: str, detail: str) -> ArmReport:
    return ArmReport(
        side=side, ok=False, min_clearance_m=-math.inf, zone=None, sphere=None, detail=detail
    )


def _as_q6(q6: Any) -> np.ndarray | None:
    try:
        q = np.asarray(q6, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if q.shape != (DOF_PER_ARM,) or not np.all(np.isfinite(q)):
        return None
    return q


@dataclass(frozen=True, eq=False)
class _Arm:
    side: str
    base_T: np.ndarray
    spheres: SphereModel
    link_idx: np.ndarray
    zones: tuple[CompiledZone, ...]


class FenceChecker:
    """Evaluates arm configurations and motions against a ``FenceConfig``.

    Immutable after construction; to change the fence build a new checker and swap it in.
    Every ``check*`` method is safe to call from the control tick: it never raises.
    """

    def __init__(self, config: FenceConfig, urdf_path: str | None = None) -> None:
        self.config = config
        mounts: dict[str, np.ndarray] = {side: config.mounts[side].matrix() for side in ARM_SIDES}
        self.kinematics = DualArmKinematics(mounts=mounts, urdf_path=urdf_path)
        self.joint_limits: dict[str, np.ndarray] = self.kinematics.joint_limits
        self.frame_T = frame_transforms(mounts)
        self._margin = float(config.margin_m)
        self._stroke = float(config.gripper_stroke_m)
        self._step = float(config.interpolation_step_rad)

        m = float(config.joint_limit_margin_rad)
        self._clamp_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, limits in self.joint_limits.items():
            lo, hi = limits[:, 0] + m, limits[:, 1] - m
            mid = (limits[:, 0] + limits[:, 1]) / 2.0
            self._clamp_bounds[side] = (np.minimum(lo, mid), np.maximum(hi, mid))

        self._arms: dict[str, _Arm] = {}
        for side in ARM_SIDES:
            chain = self.kinematics.chain(side)
            model = SphereModel.for_arm(side, config.extra_spheres)
            zones = tuple(compile_zone(z, self.frame_T) for z in config.zones if z.applies(side))
            self._arms[side] = _Arm(
                side=side,
                base_T=self.kinematics.mounts[side],
                spheres=model,
                link_idx=model.link_indices(chain.link_index),
                zones=zones,
            )

    # ---------------------------------------------------------------- queries

    @property
    def margin_m(self) -> float:
        return self._margin

    def zone_names(self, side: str) -> list[str]:
        """Names of the enabled zones that constrain ``side``."""
        arm = self._arms.get(side)
        return [z.name for z in arm.zones] if arm else []

    def spheres(
        self, side: str, q6: Sequence[float] | np.ndarray, gripper_opening: float | None = None
    ) -> Spheres:
        """World-frame covering spheres of one arm (for display and diagnostics)."""
        arm = self._arms[side]
        q = np.asarray(q6, dtype=float).reshape(1, DOF_PER_ARM)
        poses = self.kinematics.chain(side).link_poses_batch(q, arm.base_T)
        return Spheres(
            centers=arm.spheres.centers_from_poses(poses, arm.link_idx)[0],
            radii=arm.spheres.radii(gripper_opening, self._stroke),
            labels=list(arm.spheres.labels),
        )

    def fingertip(self, side: str, q6: Sequence[float] | np.ndarray) -> np.ndarray:
        """World position of the finger end (see ``DualArmKinematics.fingertip``)."""
        return self.kinematics.fingertip(side, q6)

    def clamp_to_limits(self, side: str, q6: Sequence[float] | np.ndarray) -> np.ndarray:
        """Clip ``q6`` to the URDF limits shrunk by ``joint_limit_margin_rad``.

        Non-finite entries stay non-finite (callers reject them first). Raises ``ValueError`` on an
        unknown side or a wrong shape; the adapter treats that as an error and holds.
        """
        bounds = self._clamp_bounds.get(side)
        if bounds is None:
            raise ValueError(f"side must be one of {ARM_SIDES}, got {side!r}")
        q = np.asarray(q6, dtype=float).reshape(-1)
        if q.shape != (DOF_PER_ARM,):
            raise ValueError(f"expected {DOF_PER_ARM} joint values, got {q.shape[0]}")
        return np.clip(q, bounds[0], bounds[1])

    # ---------------------------------------------------------------- checks

    def check_arm(
        self, side: str, q6: Sequence[float] | np.ndarray, gripper_opening: float | None = None
    ) -> ArmReport:
        """``ok`` iff every sphere clears > ``margin_m`` from each zone applying to ``side``."""
        try:
            arm = self._arms.get(side)
            if arm is None:
                return _fail(str(side), f"unknown side {side!r}")
            q = _as_q6(q6)
            if q is None:
                return _fail(side, "joint positions must be 6 finite values")
            return self._evaluate(arm, q[None, :], gripper_opening)
        except Exception as exc:  # fail closed: the tick must never see an exception
            return _fail(str(side), f"fence error: {type(exc).__name__}: {exc}")

    def check_arm_motion(
        self,
        side: str,
        q6_from: Sequence[float] | np.ndarray,
        q6_to: Sequence[float] | np.ndarray,
        gripper_opening: float | None = None,
    ) -> ArmReport:
        """Worst case over the straight joint-space motion ``q6_from -> q6_to``.

        Samples include both endpoints and no joint moves more than ``interpolation_step_rad``
        between consecutive samples.
        """
        try:
            arm = self._arms.get(side)
            if arm is None:
                return _fail(str(side), f"unknown side {side!r}")
            qa, qb = _as_q6(q6_from), _as_q6(q6_to)
            if qa is None or qb is None:
                return _fail(side, "joint positions must be 6 finite values")
            samples = self.motion_samples(qa, qb)
            if samples is None:
                return _fail(side, "motion too large to verify; command smaller steps")
            return self._evaluate(arm, samples, gripper_opening)
        except Exception as exc:
            return _fail(str(side), f"fence error: {type(exc).__name__}: {exc}")

    def motion_samples(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray | None:
        """Samples (M, 6) from ``q_from`` to ``q_to`` inclusive; ``None`` if there are too many."""
        delta = q_to - q_from
        span = float(np.max(np.abs(delta)))
        # Aim a hair below the step so rounding in the interpolation can never exceed it.
        n = max(1, math.ceil(span / (self._step * (1.0 - 1e-9))))
        if n + 1 > MAX_MOTION_SAMPLES:
            return None
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        samples = q_from + delta * t
        samples[0] = q_from
        samples[-1] = q_to
        return samples

    def pair_clearances(
        self, side: str, samples: np.ndarray, gripper_opening: float | None = None
    ) -> np.ndarray:
        """Clearance ``(M, Z*N)`` of every (zone, sphere) pair at each of ``M`` samples ``(M, 6)``.

        Pair ``z*N + n`` is zone ``z`` (order of ``zone_names``) and sphere ``n``. ``(M, 0)`` when
        no zone applies. Raises on bad input (callers fail closed).
        """
        arm = self._arms[side]
        q = np.asarray(samples, dtype=float).reshape(-1, DOF_PER_ARM)
        if not arm.zones:
            return np.zeros((q.shape[0], 0))
        poses = self.kinematics.chain(side).link_poses_batch(q, arm.base_T)
        centers = arm.spheres.centers_from_poses(poses, arm.link_idx)  # (M, N, 3)
        radii = arm.spheres.radii(normalize_opening(gripper_opening), self._stroke)
        return np.concatenate([zone.clearance(centers, radii) for zone in arm.zones], axis=1)

    def pair_label(self, side: str, pair: int) -> tuple[str, str]:
        """(zone name, sphere label) of a column of ``pair_clearances``."""
        arm = self._arms[side]
        n = len(arm.spheres.labels)
        return arm.zones[pair // n].name, arm.spheres.labels[pair % n]

    def is_escape_motion(
        self,
        side: str,
        q6_from: Sequence[float] | np.ndarray,
        q6_to: Sequence[float] | np.ndarray,
        gripper_opening: float | None = None,
    ) -> bool:
        """Whether ``q6_from -> q6_to`` backs an arm that is too close out of trouble.

        True only if the start is within the margin of some zone, the densely sampled motion
        satisfies ``escape_path_ok`` (no pair within the margin loses clearance, no clear pair
        enters the margin) and at least one pair within the margin gains clearance. Never raises.
        """
        try:
            qa, qb = _as_q6(q6_from), _as_q6(q6_to)
            if qa is None or qb is None or side not in self._arms:
                return False
            samples = self.motion_samples(qa, qb)
            if samples is None:
                return False
            C = self.pair_clearances(side, samples, gripper_opening)
            if C.shape[1] == 0 or not escape_path_ok(C, self._margin):
                return False
            within = C[0] <= self._margin
            gained = (C[-1] - C[0]) > ESCAPE_MIN_GAIN_M
            return bool(np.any(within) and np.any(within & gained))
        except Exception:
            return False

    def check_escape_path(
        self,
        side: str,
        segments: Sequence[tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray, float | None]],
    ) -> ArmReport:
        """Judge a multi-segment path that STARTS inside the margin (the trajectory precheck).

        ``segments`` are ``(q_from, q_to, gripper_opening)`` in order. ``ok`` iff every segment's
        dense samples, concatenated, satisfy ``escape_path_ok`` and the final pose (with the last
        segment's opening) is clear of the fence. Everything else fails. Never raises.
        """
        try:
            arm = self._arms.get(side)
            if arm is None:
                return _fail(str(side), f"unknown side {side!r}")
            if not segments:
                return _fail(side, "empty path")
            rows = []
            for q_from, q_to, opening in segments:
                qa, qb = _as_q6(q_from), _as_q6(q_to)
                if qa is None or qb is None:
                    return _fail(side, "joint positions must be 6 finite values")
                samples = self.motion_samples(qa, qb)
                if samples is None:
                    return _fail(side, "motion too large to verify; command smaller steps")
                rows.append(self.pair_clearances(side, samples, opening))
            C = np.vstack(rows)
            if C.shape[1] == 0:
                return ArmReport(side=side, ok=True, min_clearance_m=math.inf, zone=None, sphere=None)
            worst = int(np.argmin(C))
            m, pair = divmod(worst, C.shape[1])
            zone, sphere = self.pair_label(side, pair)
            if not escape_path_ok(C, self._margin):
                bad = np.where(C[:-1] > self._margin, C[1:] <= self._margin, C[1:] < C[:-1] - ESCAPE_DEPTH_TOLERANCE_M)
                row, col = (int(v) for v in np.argwhere(bad)[0]) if np.any(bad) else (0, pair)
                bad_zone, bad_sphere = self.pair_label(side, col)
                return ArmReport(
                    side=side,
                    ok=False,
                    min_clearance_m=float(C[row + 1, col]) if np.any(bad) else float(C.flat[worst]),
                    zone=bad_zone,
                    sphere=bad_sphere,
                    detail=(
                        "starts inside the margin and does not move straight out of it: "
                        f"{bad_sphere} gets closer to {bad_zone!r} at sample {row + 1} of "
                        f"{C.shape[0] - 1}"
                    ),
                )
            q_end, opening_end = segments[-1][1], segments[-1][2]
            end = self.check_arm(side, q_end, opening_end)
            if not end.ok:
                return ArmReport(
                    side=side,
                    ok=False,
                    min_clearance_m=end.min_clearance_m,
                    zone=end.zone,
                    sphere=end.sphere,
                    detail="starts inside the margin and does not end clear of it",
                )
            return ArmReport(
                side=side,
                ok=True,
                min_clearance_m=float(C.flat[worst]),
                zone=zone,
                sphere=sphere,
                detail="escape: moves out of a violation without getting closer anywhere",
            )
        except Exception as exc:
            return _fail(str(side), f"fence error: {type(exc).__name__}: {exc}")

    def check(
        self,
        q14_or_by_name: Mapping[str, float] | Sequence[float] | np.ndarray,
        gripper_openings: Mapping[str, float] | None = None,
    ) -> FenceReport:
        """Check both arms.

        Accepts a 14-vector (or the 12 arm joints) in ``DUAL_OPENYAM_JOINTS`` order, or a mapping
        joint name -> value. Gripper openings come from ``gripper_openings`` (keyed by side or by
        gripper joint name), else from the input; unknown openings count as fully open.
        """
        arms: dict[str, ArmReport] = {}
        try:
            q_by_side, openings = self._split(q14_or_by_name, gripper_openings)
            for side in ARM_SIDES:
                q6 = q_by_side.get(side)
                if q6 is None or isinstance(q6, str):
                    arms[side] = _fail(side, q6 or "no joint positions")
                else:
                    arms[side] = self.check_arm(side, q6, openings.get(side))
        except Exception as exc:
            for side in ARM_SIDES:
                arms[side] = _fail(side, f"fence error: {type(exc).__name__}: {exc}")
        return FenceReport(ok=all(r.ok for r in arms.values()), arms=arms)

    # ---------------------------------------------------------------- internals

    def _split(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        gripper_openings: Mapping[str, float] | None,
    ) -> tuple[dict[str, np.ndarray | str], dict[str, float | None]]:
        q_by_side: dict[str, np.ndarray | str] = {}
        openings: dict[str, float | None] = {side: None for side in ARM_SIDES}
        if isinstance(values, Mapping):
            for side in ARM_SIDES:
                missing = [name for name in ARM_JOINTS[side] if name not in values]
                if missing:
                    q_by_side[side] = f"missing joints {missing}"
                else:
                    q_by_side[side] = np.array([float(values[name]) for name in ARM_JOINTS[side]])
                if GRIPPER_JOINTS[side] in values:
                    openings[side] = float(values[GRIPPER_JOINTS[side]])
        else:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.shape[0] not in (2 * DOF_PER_ARM, 2 * DOF_PER_ARM + 2):
                reason = f"expected 12 or 14 joint values, got {arr.shape[0]}"
                return {side: reason for side in ARM_SIDES}, openings
            q_by_side["left"] = arr[:DOF_PER_ARM]
            q_by_side["right"] = arr[DOF_PER_ARM : 2 * DOF_PER_ARM]
            if arr.shape[0] == 2 * DOF_PER_ARM + 2:
                openings["left"] = float(arr[12])
                openings["right"] = float(arr[13])
        if gripper_openings:
            for side in ARM_SIDES:
                for key in (side, GRIPPER_JOINTS[side]):
                    if key in gripper_openings:
                        openings[side] = float(gripper_openings[key])
        return q_by_side, openings

    def _evaluate(self, arm: _Arm, samples: np.ndarray, gripper_opening: float | None) -> ArmReport:
        if not arm.zones:
            return ArmReport(
                side=arm.side, ok=True, min_clearance_m=math.inf, zone=None, sphere=None
            )
        poses = self.kinematics.chain(arm.side).link_poses_batch(samples, arm.base_T)
        centers = arm.spheres.centers_from_poses(poses, arm.link_idx)  # (M, N, 3)
        radii = arm.spheres.radii(normalize_opening(gripper_opening), self._stroke)
        n_spheres = centers.shape[1]
        worst = math.inf
        worst_zone = worst_sphere = None
        worst_sample = 0
        for zone in arm.zones:
            c = zone.clearance(centers, radii)  # (M, N)
            k = int(np.argmin(c))  # a NaN anywhere wins argmin, so it cannot hide
            value = float(c.flat[k])
            if math.isnan(value):
                return _fail(arm.side, f"fence error: non-finite clearance for zone {zone.name!r}")
            if value < worst:
                worst = value
                worst_zone = zone.name
                worst_sample, sphere_i = divmod(k, n_spheres)
                worst_sphere = arm.spheres.labels[sphere_i]
        ok = worst > self._margin
        detail = None
        if not ok and samples.shape[0] > 1:
            detail = (
                f"violation at sample {worst_sample} of {samples.shape[0] - 1} along the motion"
            )
        return ArmReport(
            side=arm.side,
            ok=ok,
            min_clearance_m=worst,
            zone=worst_zone,
            sphere=worst_sphere,
            detail=detail,
        )
