# Copyright 2026 Kite ML.
"""Forward kinematics of the dual OpenYAM (kiteos.safety.kinematics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kiteos.safety.kinematics import (
    ARM_JOINT_NAMES,
    ARM_JOINTS,
    ARM_SIDES,
    DUAL_OPENYAM_JOINTS,
    FINGERTIP_Z_M,
    GRIPPER_JOINTS,
    LINK_NAMES,
    DualArmKinematics,
    UrdfError,
    as_transform,
    default_urdf_path,
    parse_urdf_joints,
    pose_matrix,
    read_urdf_text,
    rpy_to_matrix,
)

HOME = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]

# FK test vectors from CONTRACTS.md (left arm, world frame, default mounts).
FK_VECTORS = [
    (
        [0.0] * 6,
        {
            "link3": (-0.244, 0.2751, 0.1135),
            "gripper": (0.1106, 0.31, 0.1735),
            "grasp_frame": (0.2106, 0.31, 0.1735),
            "fingertip": (0.2574, 0.31, 0.1735),
        },
    ),
    (
        [0.0, 1.047, 1.047, 0.0, 0.0, 0.0],
        {
            "gripper": (0.2426, 0.31, 0.4021),
            "grasp_frame": (0.3426, 0.31, 0.4021),
            "fingertip": (0.3894, 0.31, 0.4021),
        },
    ),
    (
        [0.5, 1.2, 0.8, -0.3, 0.4, 1.0],
        {
            "grasp_frame": (0.311, 0.4197, 0.1913),
            "fingertip": (0.3486, 0.4195, 0.1635),
        },
    ),
]

# Chain table from CONTRACTS.md: name -> (xyz, rpy, axis, limits).
CHAIN_TABLE = {
    "joint1": (
        (-2.49778e-07, -9.174e-13, 0.068),
        (0, 1.5708, 3.1415963267949),
        (-1, 0, 0),
        (-2.61799, 3.14159),
    ),
    "joint2": ((-0.0455, -0.0339, -0.02), (0, 0, -3.14159), (0, 1, 0), (0, 3.66519)),
    "joint3": ((4.08431e-07, -0.0688, 0.264), (0, 0, -3.14159), (0, 1, 0), (0, 3.14159)),
    "joint4": ((-0.0600003, -0.0688, -0.244999), (0, 0, 0), (0, 1, 0), (-1.69297, 1.5708)),
    "joint5": ((-0.0405003, 0.0338995, -0.0739989), (0, 0, 0), (1, 0, 0), (-1.5708, 1.5708)),
    "joint6": ((0.0404996, 2.39858e-07, -0.0356), (0, 0, 0), (0, 0, -1), (-2.0944, 2.0944)),
}


@pytest.fixture(scope="module")
def kin() -> DualArmKinematics:
    return DualArmKinematics()


def _point(kin: DualArmKinematics, side: str, q: list[float], what: str) -> np.ndarray:
    if what == "fingertip":
        return kin.fingertip(side, q)
    return kin.link_poses(side, q)[what][:3, 3]


def test_joint_name_lists() -> None:
    assert ARM_SIDES == ("left", "right")
    assert ARM_JOINTS["left"] == [f"left_joint{i}" for i in range(1, 7)]
    assert ARM_JOINTS["right"] == [f"right_joint{i}" for i in range(1, 7)]
    assert ARM_JOINT_NAMES == ARM_JOINTS["left"] + ARM_JOINTS["right"]
    assert GRIPPER_JOINTS == {"left": "left_arm/gripper", "right": "right_arm/gripper"}
    assert len(DUAL_OPENYAM_JOINTS) == 14
    assert DUAL_OPENYAM_JOINTS[12:] == ["left_arm/gripper", "right_arm/gripper"]


@pytest.mark.dimos
def test_joint_names_match_dimos() -> None:
    joints = pytest.importorskip("dimos.robot.manipulators.dual_openyam.joints")
    assert DUAL_OPENYAM_JOINTS == list(joints.DUAL_OPENYAM_JOINTS)
    assert ARM_JOINT_NAMES == list(joints.DUAL_OPENYAM_ARM_JOINTS)
    assert list(GRIPPER_JOINTS.values()) == list(joints.DUAL_OPENYAM_GRIPPER_JOINTS)


@pytest.mark.parametrize(("q", "expected"), FK_VECTORS)
def test_fk_vectors_left(kin: DualArmKinematics, q: list[float], expected: dict) -> None:
    for what, xyz in expected.items():
        np.testing.assert_allclose(_point(kin, "left", q, what), xyz, atol=1e-3, err_msg=what)


@pytest.mark.parametrize(("q", "expected"), FK_VECTORS)
def test_fk_vectors_right_is_left_shifted(
    kin: DualArmKinematics, q: list[float], expected: dict
) -> None:
    for what, xyz in expected.items():
        want = np.asarray(xyz) - (0.0, 0.62, 0.0)
        np.testing.assert_allclose(_point(kin, "right", q, what), want, atol=1e-3, err_msg=what)


def test_chain_matches_contract_table() -> None:
    joints = parse_urdf_joints(read_urdf_text())
    for side in ARM_SIDES:
        for short, (xyz, rpy, axis, limits) in CHAIN_TABLE.items():
            j = joints[f"{side}_{short}"]
            assert j.type == "revolute"
            np.testing.assert_allclose(j.xyz, xyz, atol=1e-6)
            np.testing.assert_allclose(rpy_to_matrix(j.rpy), rpy_to_matrix(rpy), atol=1e-5)
            np.testing.assert_allclose(j.axis, axis, atol=1e-6)
            np.testing.assert_allclose((j.lower, j.upper), limits, atol=1e-6)
    np.testing.assert_allclose(joints["left_arm_fixed_joint"].xyz, (0, 0.31, 0))
    np.testing.assert_allclose(joints["right_arm_fixed_joint"].xyz, (0, -0.31, 0))


def test_joint_limits_shape_and_values(kin: DualArmKinematics) -> None:
    for side in ARM_SIDES:
        limits = kin.joint_limits[side]
        assert limits.shape == (6, 2)
        np.testing.assert_allclose(limits, [v[3] for v in CHAIN_TABLE.values()], atol=1e-6)
        assert kin.chain(side).joint_names == ARM_JOINTS[side]


def test_link_names_and_base(kin: DualArmKinematics) -> None:
    poses = kin.link_poses("left", HOME)
    assert set(poses) == set(LINK_NAMES)
    np.testing.assert_allclose(poses["base"], pose_matrix((0, 0.31, 0)))
    for T in poses.values():
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)


def test_fingertip_is_gripper_offset(kin: DualArmKinematics) -> None:
    rng = np.random.default_rng(3)
    lim = kin.joint_limits["left"]
    for q in rng.uniform(lim[:, 0], lim[:, 1], size=(10, 6)):
        grip = kin.link_poses("left", q)["gripper"]
        want = grip[:3, :3] @ np.array([0.0, 0.0, FINGERTIP_Z_M]) + grip[:3, 3]
        np.testing.assert_allclose(kin.fingertip("left", q), want, atol=1e-12)
        tl, tr = kin.link_poses("left", q)["tip_left"], kin.link_poses("left", q)["tip_right"]
        # midpoint of the finger frames lies on the same gripper axis as the fingertip
        mid_local = np.linalg.inv(grip) @ np.r_[(tl[:3, 3] + tr[:3, 3]) / 2, 1.0]
        np.testing.assert_allclose(mid_local[:2], 0.0, atol=1e-6)


def test_home_gripper_points_forward(kin: DualArmKinematics) -> None:
    grip = kin.link_poses("left", HOME)["gripper"]
    np.testing.assert_allclose(grip[:3, :3] @ (0, 0, -1), (1, 0, 0), atol=1e-3)


def test_batch_matches_single(kin: DualArmKinematics) -> None:
    rng = np.random.default_rng(4)
    lim = kin.joint_limits["right"]
    Q = rng.uniform(lim[:, 0], lim[:, 1], size=(7, 6))
    batch = kin.link_poses_batch("right", Q)
    index = kin.chain("right").link_index
    for m, q in enumerate(Q):
        single = kin.link_poses("right", q)
        for name, T in single.items():
            np.testing.assert_allclose(batch[m, index[name]], T, atol=1e-14)


def test_custom_mounts(kin: DualArmKinematics) -> None:
    mount = pose_matrix((0.05, 0.40, 0.02), (0.0, 0.0, 0.3))
    moved = DualArmKinematics(mounts={"left": mount})
    np.testing.assert_allclose(moved.mounts["right"], kin.mounts["right"])
    ident = DualArmKinematics(mounts={"left": np.eye(4)})
    for name, T in moved.link_poses("left", HOME).items():
        np.testing.assert_allclose(T, mount @ ident.link_poses("left", HOME)[name], atol=1e-12)
    shifted = DualArmKinematics(mounts={"left": [0.0, 0.5, 0.0]})
    np.testing.assert_allclose(
        shifted.fingertip("left", HOME),
        kin.fingertip("left", HOME) + np.array([0.0, 0.19, 0.0]),
        atol=1e-12,
    )


def test_mount_validation() -> None:
    with pytest.raises(ValueError, match="unknown mount sides"):
        DualArmKinematics(mounts={"middle": np.eye(4)})
    with pytest.raises(ValueError):
        as_transform(np.ones((4, 4)))
    with pytest.raises(ValueError):
        as_transform([1.0, 2.0])
    bad = np.eye(4)
    bad[0, 0] = -1.0  # reflection
    with pytest.raises(ValueError):
        as_transform(bad)


def test_unknown_side(kin: DualArmKinematics) -> None:
    with pytest.raises(ValueError, match="side"):
        kin.link_poses("middle", HOME)


def test_urdf_path_argument(tmp_path: Path) -> None:
    path = tmp_path / "copy.urdf"
    path.write_text(read_urdf_text())
    k = DualArmKinematics(urdf_path=str(path))
    np.testing.assert_allclose(
        k.fingertip("left", HOME), DualArmKinematics().fingertip("left", HOME)
    )
    assert Path(default_urdf_path()).is_file()


def test_broken_urdf_rejected(tmp_path: Path) -> None:
    text = read_urdf_text().replace('name="left_joint3"', 'name="left_joint3b"')
    path = tmp_path / "broken.urdf"
    path.write_text(text)
    with pytest.raises(UrdfError, match="left arm movable joints"):
        DualArmKinematics(urdf_path=str(path))
    (tmp_path / "junk.urdf").write_text("<robot")
    with pytest.raises(UrdfError, match="XML"):
        DualArmKinematics(urdf_path=str(tmp_path / "junk.urdf"))


def test_matches_pinocchio_on_random_configurations(kin: DualArmKinematics) -> None:
    pin = pytest.importorskip("pinocchio")
    model = pin.buildModelFromUrdf(default_urdf_path())
    data = model.createData()
    assert model.nq == 12
    rng = np.random.default_rng(2026)
    lim = {side: kin.joint_limits[side] for side in ARM_SIDES}
    worst = 0.0
    for _ in range(50):
        q_side = {side: rng.uniform(lim[side][:, 0], lim[side][:, 1]) for side in ARM_SIDES}
        by_name = {f"{side}_joint{i + 1}": q_side[side][i] for side in ARM_SIDES for i in range(6)}
        q = np.zeros(model.nq)
        for jid in range(1, model.njoints):
            q[model.joints[jid].idx_q] = by_name[model.names[jid]]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for side in ARM_SIDES:
            for name, T in kin.link_poses(side, q_side[side]).items():
                ref = data.oMf[model.getFrameId(f"{side}_{name}")].homogeneous
                worst = max(worst, float(np.abs(T - ref).max()))
    assert worst < 1e-9, worst
