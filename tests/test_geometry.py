# Copyright 2026 Kite ML.
"""Covering spheres (kiteos.safety.geometry)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from kiteos.safety.geometry import (
    DEFAULT_GRIPPER_STROKE_M,
    LINK_SPHERES,
    SphereModel,
    arm_spheres,
    normalize_opening,
)
from kiteos.safety.kinematics import DualArmKinematics

HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]

# CONTRACTS.md sphere table, in declaration order.
CONTRACT_SPHERES = [
    ("link1", (-0.0315, -0.003, -0.0092), 0.065),
    ("link2", (0, -0.0334, 0.0235), 0.070),
    ("link2", (0, -0.0323, 0.132), 0.070),
    ("link2", (0, -0.0334, 0.2406), 0.070),
    ("link3", (-0.0276, -0.0349, -0.0044), 0.080),
    ("link3", (-0.0624, -0.0349, -0.0572), 0.077),
    ("link3", (-0.0608, -0.033, -0.1491), 0.061),
    ("link3", (-0.06, -0.0344, -0.2327), 0.061),
    ("link4", (-0.0184, 0.03, -0.0134), 0.066),
    ("link4", (-0.0694, 0.0339, -0.0667), 0.051),
    ("link5", (0.0345, 0, -0.0011), 0.054),
    ("gripper", (0, 0, -0.0157), 0.044),
    ("gripper", (0, 0, -0.0681), 0.068),
    ("gripper", (0, 0, -0.1206), 0.046),
]


@dataclass
class Extra:
    name: str
    link: str
    center: tuple[float, float, float]
    radius: float
    side: str = "both"


@pytest.fixture(scope="module")
def kin() -> DualArmKinematics:
    return DualArmKinematics()


def test_sphere_table_matches_contract() -> None:
    flat = [(link, c, r) for link, spheres in LINK_SPHERES.items() for c, r in spheres]
    assert len(flat) == len(CONTRACT_SPHERES) == 14
    for (link, c, r), (want_link, want_c, want_r) in zip(flat, CONTRACT_SPHERES, strict=True):
        assert link == want_link
        np.testing.assert_allclose(c, want_c)
        assert r == pytest.approx(want_r)


def test_labels_and_closed_radii(kin: DualArmKinematics) -> None:
    spheres = arm_spheres("left", kin.link_poses("left", HOME), 0.0, DEFAULT_GRIPPER_STROKE_M, [])
    assert spheres.labels[0] == "left.link1[0]"
    assert spheres.labels[5] == "left.link3[1]"
    assert spheres.labels[-1] == "left.gripper[2]"
    assert spheres.centers.shape == (14, 3)
    np.testing.assert_allclose(spheres.radii, [r for *_, r in CONTRACT_SPHERES])


@pytest.mark.parametrize(
    ("opening", "grow"),
    [
        (0.0, 0.0),
        (0.5, 0.025),
        (1.0, 0.05),
        (None, 0.05),
        (2.0, 0.05),
        (-1.0, 0.0),
        (float("nan"), 0.05),
    ],
)
def test_finger_inflation(kin: DualArmKinematics, opening: float | None, grow: float) -> None:
    spheres = arm_spheres("right", kin.link_poses("right", HOME), opening, 0.05, [])
    base = np.array([r for *_, r in CONTRACT_SPHERES])
    expected = base.copy()
    expected[-2:] += grow  # only the two finger spheres grow
    np.testing.assert_allclose(spheres.radii, expected)


def test_normalize_opening() -> None:
    assert normalize_opening(None) == 1.0
    assert normalize_opening(float("inf")) == 1.0
    assert normalize_opening("bad") == 1.0  # type: ignore[arg-type]
    assert normalize_opening(0.3) == pytest.approx(0.3)


def test_centers_follow_link_frames(kin: DualArmKinematics) -> None:
    poses = kin.link_poses("left", HOME)
    spheres = arm_spheres("left", poses, 1.0, 0.05, [])
    for i, (link, c, _) in enumerate(CONTRACT_SPHERES):
        T = poses[link]
        np.testing.assert_allclose(
            spheres.centers[i], T[:3, :3] @ np.asarray(c) + T[:3, 3], atol=1e-12
        )


def test_fingertip_is_covered(kin: DualArmKinematics) -> None:
    rng = np.random.default_rng(5)
    lim = kin.joint_limits["left"]
    for q in rng.uniform(lim[:, 0], lim[:, 1], size=(20, 6)):
        spheres = arm_spheres("left", kin.link_poses("left", q), 0.0, 0.05, [])
        tip = kin.fingertip("left", q)
        assert np.any(np.linalg.norm(spheres.centers - tip, axis=1) <= spheres.radii)


def test_extra_spheres(kin: DualArmKinematics) -> None:
    extras = [
        Extra("camera", "gripper", (0.0, 0.06, -0.03), 0.035),
        Extra("left_cable", "link5", (0.0, 0.0, 0.05), 0.02, side="left"),
        Extra("right_cable", "link5", (0.0, 0.0, 0.05), 0.02, side="right"),
    ]
    poses = kin.link_poses("left", HOME)
    spheres = arm_spheres("left", poses, 1.0, 0.05, extras)
    assert spheres.labels[14:] == ["left.camera", "left.left_cable"]
    grip = poses["gripper"]
    np.testing.assert_allclose(spheres.centers[14], grip[:3, :3] @ (0.0, 0.06, -0.03) + grip[:3, 3])
    assert spheres.radii[14] == pytest.approx(0.035)  # extras are never inflated by the stroke
    right = arm_spheres("right", kin.link_poses("right", HOME), 1.0, 0.05, extras)
    assert right.labels[14:] == ["right.camera", "right.right_cable"]


def test_extra_sphere_validation() -> None:
    with pytest.raises(ValueError, match="unknown link"):
        SphereModel.for_arm("left", [Extra("cam", "left_gripper", (0, 0, 0), 0.03)])
    with pytest.raises(ValueError, match="radius"):
        SphereModel.for_arm("left", [Extra("cam", "gripper", (0, 0, 0), 0.0)])
    with pytest.raises(ValueError, match="side"):
        SphereModel.for_arm("middle")


def test_batched_centers_match(kin: DualArmKinematics) -> None:
    model = SphereModel.for_arm("left", [Extra("camera", "gripper", (0.0, 0.06, -0.03), 0.035)])
    chain = kin.chain("left")
    idx = model.link_indices(chain.link_index)
    Q = np.array([HOME, [0.2, 1.3, 0.9, 0.1, -0.2, 0.5]])
    batch = model.centers_from_poses(kin.link_poses_batch("left", Q), idx)
    for m, q in enumerate(Q):
        np.testing.assert_allclose(batch[m], model.centers(kin.link_poses("left", q)), atol=1e-14)
