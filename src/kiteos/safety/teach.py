# Copyright 2026 Kite ML.
"""Teach helpers: turn touched fingertip points into fence zones (pure numpy).

Typical glass-window flow (see docs/FENCE.md):

1. Touch at least 4 (5 recommended) well-spread points on the glass with the fingertip and record
   ``DualArmKinematics.fingertip(side, q)`` for each (world frame, or convert with
   ``points_in_frame`` for a per-arm zone).
2. ``plane_zone_from_points(points, robot_ref, "glass", standoff_m=0.02)`` fits the plane,
   rejects a poor fit (rms > 1 cm), orients the normal away from the robot and moves the plane
   ``standoff_m`` toward the robot. The checker's ``margin_m`` still applies on top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from kiteos.safety.kinematics import invert_transform
from kiteos.safety.zones import AppliesTo, BoxZone, Frame, HalfSpaceZone, KeepInBox

DEFAULT_MAX_RMS_M = 0.01
# A plane fit needs a redundant point: with 3 points the residual is always 0, so a missed touch
# (fingertip hovering) tilts the plane without any warning.
MIN_FIT_POINTS = 4
# RMS spread of the points along BOTH in-plane principal directions (m); below this a small
# error in one touch tilts the plane a lot over the rest of the workspace.
DEFAULT_MIN_SPREAD_M = 0.05
# How far from the touched patch the tilt uncertainty is evaluated (m, about the arm's reach).
TILT_CHECK_RADIUS_M = 0.5
# The robot reference must be at least this far from the fitted plane to pick a side reliably.
DEFAULT_MIN_REF_DISTANCE_M = 0.02
_DECIMALS = 6  # micrometre rounding keeps the YAML readable


class TeachError(ValueError):
    """Touched points cannot produce a trustworthy zone."""


class PlaneFit(NamedTuple):
    point: np.ndarray  # (3,) centroid of the points, on the plane
    normal: np.ndarray  # (3,) unit normal; sign arbitrary until oriented
    rms_m: float  # RMS point-to-plane distance


def _as_points(points: Sequence[Sequence[float]] | np.ndarray, min_count: int) -> np.ndarray:
    try:
        P = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TeachError(f"points must be numeric xyz triples: {exc}") from exc
    if P.ndim != 2 or P.shape[1] != 3:
        raise TeachError(f"points must have shape (N, 3), got {P.shape}")
    if P.shape[0] < min_count:
        raise TeachError(f"need at least {min_count} points, got {P.shape[0]}")
    if not np.all(np.isfinite(P)):
        raise TeachError("points contain non-finite values")
    return P


def _as_vec3(value: Sequence[float] | np.ndarray, what: str) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise TeachError(f"{what} must be 3 finite numbers")
    return v


def _rounded(v: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (round(float(c), _DECIMALS) + 0.0 for c in v)  # + 0.0 folds -0.0 into 0.0
    return (x, y, z)


def fit_plane(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    min_spread_m: float = DEFAULT_MIN_SPREAD_M,
) -> PlaneFit:
    """Least-squares plane through >= 4 well-spread points (SVD of the centred points)."""
    P = _as_points(points, MIN_FIT_POINTS)
    centroid = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - centroid, full_matrices=False)
    n = P.shape[0]
    spread = float(s[1]) / math.sqrt(n)
    if spread < min_spread_m:
        raise TeachError(
            f"points are (nearly) collinear or too close together: they spread only "
            f"{spread * 1000:.1f} mm (RMS) across their main direction (need "
            f"{min_spread_m * 1000:.0f} mm); touch points spread over the surface, e.g. the corners "
            "of a quadrilateral at least 20 cm across"
        )
    rms = float(s[2]) / math.sqrt(n) if s.shape[0] > 2 else 0.0
    normal = vt[2] / np.linalg.norm(vt[2])
    return PlaneFit(point=centroid, normal=normal, rms_m=rms)


def plane_tilt_uncertainty(points: Sequence[Sequence[float]] | np.ndarray) -> float:
    """1-sigma uncertainty (rad) of a fitted plane's normal, from its residuals and spread.

    Least-squares slope error: residual sigma (n - 3 degrees of freedom) over sqrt(n) times the
    RMS spread along the weaker in-plane direction. Multiply by a distance from the touched patch
    to get how far the plane may be off there.
    """
    P = _as_points(points, MIN_FIT_POINTS)
    n = P.shape[0]
    centroid = P.mean(axis=0)
    _, s, _ = np.linalg.svd(P - centroid, full_matrices=False)
    spread = float(s[1]) / math.sqrt(n)
    sigma = float(s[2]) / math.sqrt(n - 3) if s.shape[0] > 2 else 0.0
    if spread <= 0.0:
        return math.inf
    return sigma / (math.sqrt(n) * spread)


def plane_residuals(
    points: Sequence[Sequence[float]] | np.ndarray,
    point: Sequence[float] | np.ndarray,
    normal: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Signed distance (N,) of each point to the plane, positive on the ``normal`` side."""
    P = _as_points(points, 1)
    n = _as_vec3(normal, "normal")
    n = n / np.linalg.norm(n)
    return (P - _as_vec3(point, "point")) @ n


def orient_normal_away_from(
    point: Sequence[float] | np.ndarray,
    normal: Sequence[float] | np.ndarray,
    robot_ref: Sequence[float] | np.ndarray,
    *,
    min_distance_m: float = DEFAULT_MIN_REF_DISTANCE_M,
) -> np.ndarray:
    """Unit normal pointing away from ``robot_ref``, i.e. into the forbidden side."""
    p = _as_vec3(point, "point")
    n = _as_vec3(normal, "normal")
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise TeachError("normal must be non-zero")
    n = n / norm
    d = float(n @ (_as_vec3(robot_ref, "robot_ref") - p))
    if abs(d) < min_distance_m:
        raise TeachError(
            f"robot reference is only {abs(d) * 1000:.1f} mm from the plane, so its side is "
            "ambiguous; use a point well inside the workspace (e.g. the arm base)"
        )
    return n if d < 0 else -n


def plane_zone_from_points(
    points: Sequence[Sequence[float]] | np.ndarray,
    robot_ref: Sequence[float] | np.ndarray,
    name: str,
    frame: Frame = "dual_openyam_base",
    applies_to: AppliesTo = "both",
    standoff_m: float = 0.0,
    *,
    max_rms_m: float = DEFAULT_MAX_RMS_M,
    force: bool = False,
) -> HalfSpaceZone:
    """Half-space zone from points touched on a flat surface (glass, wall, table edge).

    ``points`` and ``robot_ref`` must be in ``frame``. The normal points away from ``robot_ref``
    and the plane is moved ``standoff_m`` toward the robot. Unless ``force``, a fit is rejected
    when its rms residual is above ``max_rms_m``, or when its tilt uncertainty could move the
    plane by more than the standoff (at least 1 cm) ``TILT_CHECK_RADIUS_M`` away from the touched
    points (2 sigma): there the zone could lie behind the real surface.
    """
    if not (math.isfinite(standoff_m) and standoff_m >= 0.0):
        raise TeachError(f"standoff_m must be >= 0, got {standoff_m}")
    fit = fit_plane(points)
    if fit.rms_m > max_rms_m and not force:
        worst = float(np.max(np.abs(plane_residuals(points, fit.point, fit.normal))))
        raise TeachError(
            f"points are not on one plane: rms residual {fit.rms_m * 1000:.1f} mm "
            f"(worst {worst * 1000:.1f} mm) exceeds {max_rms_m * 1000:.1f} mm; re-touch the "
            "surface or pass force=True if you accept the fit"
        )
    drift = 2.0 * TILT_CHECK_RADIUS_M * plane_tilt_uncertainty(points)
    allowed = max(standoff_m, 0.01)
    if drift > allowed and not force:
        raise TeachError(
            f"the fitted plane is not certain enough: {TILT_CHECK_RADIUS_M * 100:.0f} cm from the "
            f"touched points it may be off by {drift * 1000:.0f} mm, more than the "
            f"{allowed * 1000:.0f} mm standoff, so the zone could sit behind the real surface. "
            "Touch more points (5-8), spread wider, and hold still at each touch"
        )
    normal = orient_normal_away_from(fit.point, fit.normal, robot_ref)
    point = fit.point - normal * standoff_m
    if float(normal @ (_as_vec3(robot_ref, "robot_ref") - point)) >= 0.0:
        raise TeachError(
            "standoff_m moves the plane past the robot reference; use a smaller standoff"
        )
    return HalfSpaceZone(
        name=name,
        frame=frame,
        applies_to=applies_to,
        point=_rounded(point),
        normal=_rounded(normal),
    )


def box_zone_from_corners(
    corners: Sequence[Sequence[float]] | np.ndarray,
    name: str,
    frame: Frame = "dual_openyam_base",
    applies_to: AppliesTo = "both",
    pad_m: float = 0.0,
    *,
    keep_in: bool = False,
) -> BoxZone | KeepInBox:
    """Axis-aligned box (in ``frame``) spanning >= 2 touched corner points.

    ``pad_m`` makes the zone safer: a keep-out box grows by it, a keep-in box shrinks by it.
    """
    if not (math.isfinite(pad_m) and pad_m >= 0.0):
        raise TeachError(f"pad_m must be >= 0, got {pad_m}")
    P = _as_points(corners, 2)
    lo, hi = P.min(axis=0), P.max(axis=0)
    half = (hi - lo) / 2.0 + (-pad_m if keep_in else pad_m)
    flat = [axis for axis, h in zip("xyz", half, strict=True) if h <= 0.0]
    if flat:
        raise TeachError(
            f"box has no size along {', '.join(flat)}; touch corners spread along every axis"
            + ("" if keep_in else " or add pad_m")
        )
    center = (lo + hi) / 2.0
    cls = KeepInBox if keep_in else BoxZone
    return cls(
        name=name,
        frame=frame,
        applies_to=applies_to,
        center=_rounded(center),
        half_extents=_rounded(half),
    )


def points_in_frame(
    points_world: Sequence[Sequence[float]] | np.ndarray, frame_pose: np.ndarray
) -> np.ndarray:
    """Express world points (N, 3) in a frame whose world pose is ``frame_pose`` (4x4)."""
    P = _as_points(points_world, 1)
    T = invert_transform(np.asarray(frame_pose, dtype=float))
    return P @ T[:3, :3].T + T[:3, 3]
