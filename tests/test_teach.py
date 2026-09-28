# Copyright 2026 Kite ML.
"""Teach helpers (kiteos.safety.teach)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kiteos.safety.fence import FenceChecker, FenceConfig
from kiteos.safety.kinematics import pose_matrix
from kiteos.safety.teach import (
    PlaneFit,
    TeachError,
    box_zone_from_corners,
    fit_plane,
    orient_normal_away_from,
    plane_residuals,
    plane_zone_from_points,
    points_in_frame,
)
from kiteos.safety.zones import BoxZone, HalfSpaceZone, KeepInBox

HOME = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
FORWARD_REACH = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0])
ROBOT_REF = (0.0, 0.0, 0.1)  # between the arm bases, dual_openyam_base

# Four fingertip touches on a glass pane at x = 0.45 (y, z spread ~30 cm).
GLASS_POINTS = np.array(
    [[0.45, 0.15, 0.20], [0.45, -0.15, 0.22], [0.45, -0.12, 0.45], [0.45, 0.14, 0.43]]
)


def test_fit_exact_plane() -> None:
    fit = fit_plane(GLASS_POINTS)
    assert isinstance(fit, PlaneFit)
    point, normal, rms = fit  # unpacks as (point, unit normal, rms residual)
    np.testing.assert_allclose(point, GLASS_POINTS.mean(axis=0))
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert abs(normal @ (1, 0, 0)) == pytest.approx(1.0)
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_fit_tilted_noisy_plane() -> None:
    rng = np.random.default_rng(7)
    n_true = np.array([1.0, 0.2, -0.1])
    n_true /= np.linalg.norm(n_true)
    u = np.cross(n_true, (0, 0, 1))
    u /= np.linalg.norm(u)
    v = np.cross(n_true, u)
    uv = rng.uniform(-0.2, 0.2, size=(40, 2))
    noise = rng.normal(0.0, 0.002, size=40)
    pts = np.array([0.5, 0.0, 0.3]) + uv[:, :1] * u + uv[:, 1:] * v + noise[:, None] * n_true
    fit = fit_plane(pts)
    assert abs(fit.normal @ n_true) == pytest.approx(1.0, abs=1e-3)
    assert fit.rms_m == pytest.approx(0.002, rel=0.3)
    res = plane_residuals(pts, fit.point, fit.normal)
    assert math.sqrt(np.mean(res**2)) == pytest.approx(fit.rms_m)


@pytest.mark.parametrize(
    ("points", "match"),
    [
        ([[0.45, 0, 0.2], [0.45, 0.1, 0.2]], "at least 4"),
        ([[0.45, 0, 0.2], [0.45, 0.2, 0.2], [0.45, 0.1, 0.4]], "at least 4"),  # 3: rms is always 0
        ([[0.45, 0, 0.2], [0.45, 0.1, 0.2], [0.45, 0.2, 0.2], [0.45, 0.3, 0.2]], "collinear"),
        ([[0.45, 0, 0.2]] * 4, "collinear"),
        ([[0.45, 0, 0.2], [0.45, 0.1, np.nan], [0.45, 0.2, 0.3], [0.45, 0.0, 0.4]], "non-finite"),
        ([[0.45, 0], [0.45, 0.1], [0.45, 0.2]], "shape"),
    ],
)
def test_fit_rejects_bad_points(points, match: str) -> None:
    with pytest.raises(TeachError, match=match):
        fit_plane(points)


def test_orient_normal_points_away_from_robot() -> None:
    p = np.array([0.45, 0.0, 0.3])
    np.testing.assert_allclose(orient_normal_away_from(p, (1, 0, 0), ROBOT_REF), (1, 0, 0))
    np.testing.assert_allclose(orient_normal_away_from(p, (-3, 0, 0), ROBOT_REF), (1, 0, 0))
    # a robot on the other side of the plane flips it
    np.testing.assert_allclose(orient_normal_away_from(p, (1, 0, 0), (1.0, 0.0, 0.3)), (-1, 0, 0))
    with pytest.raises(TeachError, match="ambiguous"):
        orient_normal_away_from(p, (1, 0, 0), (0.45, 0.5, 0.0))
    with pytest.raises(TeachError):
        orient_normal_away_from(p, (0, 0, 0), ROBOT_REF)


def test_plane_zone_from_points_glass() -> None:
    zone = plane_zone_from_points(GLASS_POINTS[::-1], ROBOT_REF, "glass_window", standoff_m=0.02)
    assert isinstance(zone, HalfSpaceZone)
    assert (
        zone.name == "glass_window"
        and zone.frame == "dual_openyam_base"
        and zone.applies_to == "both"
    )
    np.testing.assert_allclose(zone.normal, (1.0, 0.0, 0.0), atol=1e-6)
    assert zone.point[0] == pytest.approx(0.43)  # glass minus 2 cm standoff


def test_plane_zone_options_and_errors() -> None:
    zone = plane_zone_from_points(
        GLASS_POINTS, (0, 0, 0.1), "g", frame="left_base", applies_to="left"
    )
    assert zone.frame == "left_base" and zone.applies_to == "left"
    with pytest.raises(TeachError, match="standoff_m"):
        plane_zone_from_points(GLASS_POINTS, ROBOT_REF, "g", standoff_m=-0.01)
    with pytest.raises(TeachError, match="past the robot"):
        plane_zone_from_points(GLASS_POINTS, ROBOT_REF, "g", standoff_m=0.5)


def test_plane_zone_rejects_poor_fit_unless_forced() -> None:
    bumpy = GLASS_POINTS.copy()
    bumpy[:, 0] += [0.02, -0.02, 0.02, -0.02]  # +/- 2 cm: not one plane
    with pytest.raises(TeachError, match="rms residual"):
        plane_zone_from_points(bumpy, ROBOT_REF, "g")
    zone = plane_zone_from_points(bumpy, ROBOT_REF, "g", force=True)
    assert zone.normal[0] > 0.9
    # a looser rms threshold alone is not enough: the 4-point fit is too uncertain (tilt)
    with pytest.raises(TeachError, match="not certain enough"):
        plane_zone_from_points(bumpy, ROBOT_REF, "g", max_rms_m=0.03)


def test_taught_plane_behaves_like_the_glass() -> None:
    taught = plane_zone_from_points(GLASS_POINTS, ROBOT_REF, "glass_window")
    exact = HalfSpaceZone(name="glass_window", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    a = FenceChecker(FenceConfig(margin_m=0.03, zones=[taught]))
    b = FenceChecker(FenceConfig(margin_m=0.03, zones=[exact]))
    for q in (HOME, FORWARD_REACH):
        ra, rb = a.check_arm("left", q, 0.0), b.check_arm("left", q, 0.0)
        assert ra.ok == rb.ok
        assert ra.min_clearance_m == pytest.approx(rb.min_clearance_m, abs=1e-6)


def test_teach_from_fingertips_in_arm_frame() -> None:
    """Record fingertips in world, express them in left_base, and fence per arm."""
    checker = FenceChecker(FenceConfig())
    rng = np.random.default_rng(11)
    tips = []
    for _ in range(5):
        q = HOME + rng.uniform(-0.3, 0.3, size=6)
        tips.append(checker.fingertip("left", q))
    tips = np.array(tips)
    # pretend the glass is the vertical plane through the furthest touched point
    x0 = tips[:, 0].max() + 0.01
    touches = np.column_stack([np.full(5, x0), tips[:, 1], tips[:, 2]])
    local = points_in_frame(touches, checker.frame_T["left_base"])
    np.testing.assert_allclose(local, touches - (0.0, 0.31, 0.0), atol=1e-12)
    zone = plane_zone_from_points(
        local, (0.0, 0.0, 0.1), "glass_left", frame="left_base", applies_to="left"
    )
    np.testing.assert_allclose(zone.point[0], x0, atol=1e-6)
    np.testing.assert_allclose(zone.normal, (1, 0, 0), atol=1e-6)


def test_points_in_rotated_frame() -> None:
    T = pose_matrix((0.1, 0.2, 0.0), (0.0, 0.0, math.pi / 2))
    world = np.array([[0.1, 1.2, 0.5]])  # 1 m along the frame's +x (world +y)
    np.testing.assert_allclose(points_in_frame(world, T), [[1.0, 0.0, 0.5]], atol=1e-12)


def test_box_zone_from_corners() -> None:
    corners = [[0.2, -0.1, 0.0], [0.4, 0.1, 0.3], [0.3, 0.0, 0.1]]
    box = box_zone_from_corners(corners, "fixture")
    assert isinstance(box, BoxZone)
    np.testing.assert_allclose(box.center, (0.3, 0.0, 0.15))
    np.testing.assert_allclose(box.half_extents, (0.1, 0.1, 0.15))
    padded = box_zone_from_corners(
        corners, "fixture", pad_m=0.02, frame="right_base", applies_to="right"
    )
    np.testing.assert_allclose(padded.half_extents, (0.12, 0.12, 0.17))
    assert padded.frame == "right_base" and padded.applies_to == "right"
    cell = box_zone_from_corners(corners, "cell", pad_m=0.02, keep_in=True)
    assert isinstance(cell, KeepInBox)
    np.testing.assert_allclose(cell.half_extents, (0.08, 0.08, 0.13))


def test_box_zone_from_corners_errors() -> None:
    flat = [[0.2, -0.1, 0.1], [0.4, 0.1, 0.1]]
    with pytest.raises(TeachError, match="along z"):
        box_zone_from_corners(flat, "b")
    assert box_zone_from_corners(flat, "b", pad_m=0.01).half_extents[2] == pytest.approx(0.01)
    with pytest.raises(TeachError, match="at least 2"):
        box_zone_from_corners([[0, 0, 0]], "b")
    with pytest.raises(TeachError, match="pad_m"):
        box_zone_from_corners(flat, "b", pad_m=-1.0)
    with pytest.raises(TeachError, match="along x"):
        box_zone_from_corners([[0, 0, 0], [0.05, 1, 1]], "b", pad_m=0.03, keep_in=True)



# --------------------------------------------------------------------------- review regressions


def test_a_thin_sliver_of_points_is_rejected() -> None:
    """Review: a 30 cm x 1.5 cm sliver with 2 mm noise was accepted and tilted 7.6 degrees."""
    sliver = [[0.47, 0.15, 0.30], [0.47, 0.45, 0.30], [0.472, 0.30, 0.315], [0.47, 0.25, 0.305]]
    with pytest.raises(TeachError, match="collinear or too close"):
        fit_plane(sliver)


def test_a_missed_touch_is_caught_with_four_points() -> None:
    """Review: with 3 points a touch recorded 4 cm short of the glass tilted the plane behind the
    real glass with rms 0. With a fourth touch the residual exposes it."""
    glass_x = 0.47
    points = [
        [glass_x, 0.20, 0.30],
        [glass_x, 0.40, 0.30],
        [glass_x - 0.04, 0.30, 0.45],  # Enter pressed before contact
        [glass_x, 0.30, 0.15],
    ]
    with pytest.raises(TeachError, match="rms residual|not certain enough"):
        plane_zone_from_points(points, ROBOT_REF, "glass", standoff_m=0.02)
    # Five careful touches (1.5 mm noise) over ~20 cm pass.
    rng = np.random.default_rng(3)
    good = np.array([[glass_x, y, z] for y, z in ((0.2, 0.2), (0.4, 0.2), (0.2, 0.4), (0.4, 0.4), (0.3, 0.3))])
    good[:, 0] += rng.normal(0.0, 0.0015, size=5)
    zone = plane_zone_from_points(good, ROBOT_REF, "glass", standoff_m=0.02)
    assert zone.normal[0] > 0.99


def test_tilt_uncertainty_grows_with_noise_and_shrinks_with_spread() -> None:
    from kiteos.safety.teach import plane_tilt_uncertainty

    assert plane_tilt_uncertainty(GLASS_POINTS) == pytest.approx(0.0, abs=1e-9)
    noisy = GLASS_POINTS.copy()
    noisy[:, 0] += [0.002, -0.002, 0.002, -0.002]
    wide = noisy.copy()
    wide[:, 1:] *= 2.0
    assert plane_tilt_uncertainty(noisy) > plane_tilt_uncertainty(wide) > 0.0
