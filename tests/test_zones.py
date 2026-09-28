# Copyright 2026 Kite ML.
"""Zone models and signed clearance (kiteos.safety.zones)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from kiteos.safety.kinematics import pose_matrix
from kiteos.safety.zones import (
    BoxZone,
    HalfSpaceZone,
    KeepInBox,
    Zone,
    clearance,
    compile_zone,
    frame_transforms,
)

WORLD = {"dual_openyam_base": np.eye(4)}
DEFAULT_FRAMES = frame_transforms(
    {"left": pose_matrix((0, 0.31, 0)), "right": pose_matrix((0, -0.31, 0))}
)


def _c(zone, centers, radii, frames=WORLD) -> np.ndarray:
    return clearance(zone, np.asarray(centers, dtype=float), np.asarray(radii, dtype=float), frames)


# ------------------------------------------------------------------------------ half-space


def test_half_space_basic() -> None:
    glass = HalfSpaceZone(name="glass", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    centers = [(0.30, 0.0, 0.2), (0.40, 1.0, -3.0), (0.45, 0.0, 0.0), (0.50, 0.0, 0.0)]
    got = _c(glass, centers, [0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(got, [0.10, 0.0, -0.05, -0.10], atol=1e-12)


def test_half_space_straddling_sphere_is_penetrating() -> None:
    glass = HalfSpaceZone(name="glass", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    # centre 1 cm before the glass, radius 3 cm: the sphere pokes 2 cm through
    assert _c(glass, [(0.44, 0.0, 0.0)], [0.03])[0] == pytest.approx(-0.02)


def test_half_space_normal_is_normalised_and_tilted() -> None:
    tilted = HalfSpaceZone(name="wall", point=(0.5, 0.0, 0.0), normal=(2.0, 2.0, 0.0))
    got = _c(tilted, [(0.0, 0.0, 0.0)], [0.0])[0]
    assert got == pytest.approx(0.5 / math.sqrt(2.0))


def test_half_space_normal_sign_matters() -> None:
    # normal toward -x: everything with x < 0.45 is forbidden
    zone = HalfSpaceZone(name="back", point=(0.45, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))
    np.testing.assert_allclose(_c(zone, [(0.30, 0, 0), (0.60, 0, 0)], [0.0, 0.0]), [-0.15, 0.15])


def test_half_space_in_arm_frame_uses_mount() -> None:
    world = HalfSpaceZone(name="w", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    per_arm = HalfSpaceZone(
        name="a", frame="left_base", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0)
    )
    pts = np.random.default_rng(0).uniform(-1, 1, size=(20, 3))
    np.testing.assert_allclose(
        _c(world, pts, 0.02), _c(per_arm, pts, 0.02, DEFAULT_FRAMES), atol=1e-12
    )
    # a mount 10 cm forward and yawed 90 degrees moves and turns the zone with the arm
    frames = frame_transforms(
        {"left": pose_matrix((0.1, 0.3, 0.0), (0, 0, math.pi / 2)), "right": np.eye(4)}
    )
    # in world, the arm-frame +x is +y and the plane passes through (0.1, 0.3 + 0.45)
    got = _c(per_arm, [(0.1, 0.30, 0.0), (5.0, 0.80, 0.0)], [0.0, 0.0], frames)
    np.testing.assert_allclose(got, [0.45, -0.05], atol=1e-12)


def test_missing_frame_pose_is_an_error() -> None:
    zone = HalfSpaceZone(name="a", frame="right_base", point=(0, 0, 0), normal=(1, 0, 0))
    with pytest.raises(ValueError, match="right_base"):
        _c(zone, [(0, 0, 0)], [0.0])


# ------------------------------------------------------------------------------ boxes


BOX = BoxZone(name="fixture", center=(0.5, 0.0, 0.2), half_extents=(0.1, 0.2, 0.05))


def test_box_face_edge_corner_and_inside() -> None:
    centers = [
        (0.7, 0.0, 0.2),  # 10 cm beyond the +x face
        (0.7, 0.3, 0.2),  # beyond the +x/+y edge: diagonal (0.1, 0.1)
        (0.7, 0.3, 0.35),  # beyond a corner: diagonal (0.1, 0.1, 0.1)
        (0.5, 0.0, 0.2),  # at the centre: 5 cm from the nearest (z) face
        (0.55, 0.0, 0.2),  # inside, 5 cm from +x face and 5 cm from z faces
    ]
    got = _c(BOX, centers, [0.0] * 5)
    np.testing.assert_allclose(
        got, [0.1, math.sqrt(0.02), math.sqrt(0.03), -0.05, -0.05], atol=1e-12
    )
    got_r = _c(BOX, centers[:1], [0.04])
    assert got_r[0] == pytest.approx(0.06)


def test_box_straddling_sphere() -> None:
    # centre exactly on the top face: penetration equals the radius
    assert _c(BOX, [(0.5, 0.0, 0.25)], [0.03])[0] == pytest.approx(-0.03)
    # centre 1 cm above the top face with a 3 cm radius: 2 cm penetration
    assert _c(BOX, [(0.5, 0.0, 0.26)], [0.03])[0] == pytest.approx(-0.02)


def test_rotated_box() -> None:
    # 45 degrees about z: the box's +x axis points along world (1, 1, 0)/sqrt(2)
    box = BoxZone(
        name="rot", center=(0.0, 0.0, 0.0), half_extents=(0.2, 0.05, 0.05), rpy=(0, 0, math.pi / 4)
    )
    d = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    along = _c(box, [0.3 * d], [0.0])[0]
    assert along == pytest.approx(0.1)  # 30 cm along the long axis, half extent 20 cm
    # (0.18, 0, 0) is inside the unrotated box but outside the rotated one
    p = np.array([0.18, 0.0, 0.0])
    local = np.array([p @ d, p @ np.array([-d[1], d[0], 0.0]), 0.0])
    want = math.hypot(max(abs(local[0]) - 0.2, 0.0), abs(local[1]) - 0.05)
    assert _c(box, [p], [0.0])[0] == pytest.approx(want)
    assert want > 0


def test_rotated_box_about_pitch() -> None:
    # pitch 90 degrees: the box's x axis maps to world -z
    box = BoxZone(
        name="p", center=(0, 0, 0), half_extents=(0.3, 0.05, 0.05), rpy=(0, math.pi / 2, 0)
    )
    np.testing.assert_allclose(
        _c(box, [(0, 0, -0.4), (0.4, 0, 0)], [0.0, 0.0]), [0.1, 0.35], atol=1e-12
    )


def test_box_in_arm_frame() -> None:
    box = BoxZone(
        name="b", frame="right_base", center=(0.3, 0.0, 0.1), half_extents=(0.05, 0.05, 0.05)
    )
    # right_base is at y = -0.31: the box centre sits at world (0.3, -0.31, 0.1)
    assert _c(box, [(0.3, -0.31, 0.1)], [0.0], DEFAULT_FRAMES)[0] == pytest.approx(-0.05)
    assert _c(box, [(0.3, 0.0, 0.1)], [0.0], DEFAULT_FRAMES)[0] == pytest.approx(0.26)


def test_keep_in_box() -> None:
    cell = KeepInBox(name="cell", center=(0.3, 0.0, 0.3), half_extents=(0.3, 0.6, 0.3))
    centers = [
        (0.3, 0.0, 0.3),
        (0.55, 0.0, 0.3),
        (0.60, 0.0, 0.3),
        (0.65, 0.0, 0.3),
        (0.3, 0.0, 0.58),
    ]
    got = _c(cell, centers, [0.05] * 5)
    np.testing.assert_allclose(got, [0.25, 0.0, -0.05, -0.10, -0.03], atol=1e-12)


def test_batched_shapes() -> None:
    glass = HalfSpaceZone(name="glass", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
    centers = np.zeros((4, 7, 3))
    centers[..., 0] = np.linspace(0.0, 0.6, 4)[:, None]
    radii = np.full(7, 0.05)
    for zone in (glass, BOX, KeepInBox(name="k", center=(0, 0, 0), half_extents=(1, 1, 1))):
        got = compile_zone(zone, WORLD).clearance(centers, radii)
        assert got.shape == (4, 7)
        np.testing.assert_allclose(
            got[:, 0], [_c(zone, [centers[m, 0]], [0.05])[0] for m in range(4)]
        )


# ------------------------------------------------------------------------------ models


def test_zone_union_parses_by_kind() -> None:
    adapter = TypeAdapter(Zone)
    z = adapter.validate_python(
        {"kind": "keep_in_box", "name": "c", "center": [0, 0, 0], "half_extents": [1, 1, 1]}
    )
    assert isinstance(z, KeepInBox)
    z = adapter.validate_python(
        {"kind": "half_space", "name": "g", "point": [0, 0, 0], "normal": [0, 0, 1]}
    )
    assert isinstance(z, HalfSpaceZone)
    assert z.frame == "dual_openyam_base" and z.applies_to == "both" and z.enabled


def test_zone_validation_errors() -> None:
    with pytest.raises(ValidationError, match="non-zero"):
        HalfSpaceZone(name="g", point=(0, 0, 0), normal=(0, 0, 0))
    with pytest.raises(ValidationError, match="half_extents must all be > 0"):
        BoxZone(name="b", center=(0, 0, 0), half_extents=(0.1, 0.0, 0.1))
    with pytest.raises(ValidationError, match="half_extents must all be > 0"):
        KeepInBox(name="b", center=(0, 0, 0), half_extents=(0.1, -0.1, 0.1))
    with pytest.raises(ValidationError):
        BoxZone(name="b", center=(0, 0, 0), half_extents=(0.1, 0.1, 0.1), size=3)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        BoxZone(name="b", frame="table", center=(0, 0, 0), half_extents=(0.1, 0.1, 0.1))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        HalfSpaceZone(name="g", point=(0, float("nan"), 0), normal=(1, 0, 0))
    with pytest.raises(ValidationError):
        HalfSpaceZone(name="", point=(0, 0, 0), normal=(1, 0, 0))


def test_applies_to_and_enabled() -> None:
    both = HalfSpaceZone(name="g", point=(0, 0, 0), normal=(1, 0, 0))
    left = both.model_copy(update={"applies_to": "left"})
    off = both.model_copy(update={"enabled": False})
    assert both.applies("left") and both.applies("right")
    assert left.applies("left") and not left.applies("right")
    assert not off.applies("left") and not off.applies("right")
