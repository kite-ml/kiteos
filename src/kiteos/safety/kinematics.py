# Copyright 2026 Kite ML.
"""Pure-numpy forward kinematics for the dual-arm robots KiteOS fences.

Each arm chain is parsed from the robot's vendored URDF (``kiteos/data``; see
``kiteos.robots``): ``dual_openyam.urdf`` (a copy of the runtime
``data/dual_openyam_abc_box_v2/dual_openyam.urdf``) or ``openarm_v20.urdf`` (the expanded
OpenArm v2.0 xacro). Every joint contributes

    T_child = T_parent @ Trans(xyz) @ Rz(yaw) @ Ry(pitch) @ Rx(roll) @ AxisAngle(axis, q)

which is the URDF convention and matches pinocchio to machine precision.

World frames: ``dual_openyam_base`` (+x forward, the gripper approach direction at home, +y
toward the left arm, +z up, z = 0 at the base mounting surface) and ``openarm_body_link0`` (the
OpenArm body: +x forward, +y left, +z up, z = 0 at the bottom of the body). Each arm base is placed
in the world by a mount transform; the URDF mounts are the default, and callers can pass measured
mounts instead.

The module-level names (``ARM_JOINTS``, ``DOF_PER_ARM``, ...) describe the dual OpenYAM, the
default robot; pass ``robot=`` for another one.

This module must not import the runtime or pinocchio: it runs inside the control tick.
"""

from __future__ import annotations

import functools
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np

from kiteos.robots import DUAL_OPENYAM, OPENYAM_FINGERTIP_Z_M, RobotProfile, get_robot

Side = Literal["left", "right"]
ARM_SIDES: tuple[Side, Side] = ("left", "right")
# The dual OpenYAM (the default robot); other robots: ``kiteos.robots``.
DOF_PER_ARM = DUAL_OPENYAM.dof
ARM_JOINTS: dict[str, list[str]] = {side: list(DUAL_OPENYAM.arm_joints[side]) for side in ARM_SIDES}
ARM_JOINT_NAMES: list[str] = [*ARM_JOINTS["left"], *ARM_JOINTS["right"]]
GRIPPER_JOINTS: dict[str, str] = dict(DUAL_OPENYAM.gripper_joints)
# Must stay identical to dimos.robot.manipulators.dual_openyam.joints.DUAL_OPENYAM_JOINTS
# (asserted by tests); redefined here so the safety core never imports the runtime.
DUAL_OPENYAM_JOINTS: list[str] = DUAL_OPENYAM.joint_names

WORLD_FRAME = DUAL_OPENYAM.world_frame
# Link names without the side prefix. The chain parser requires all of them.
LINK_NAMES: tuple[str, ...] = DUAL_OPENYAM.required_links
FINGERTIP_Z_M = OPENYAM_FINGERTIP_Z_M


class UrdfError(ValueError):
    """The URDF does not describe the arm chain KiteOS expects for the robot."""


@dataclass(frozen=True)
class UrdfJoint:
    """One ``<joint>`` element, with URDF defaults applied."""

    name: str
    type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None
    velocity: float | None


# --------------------------------------------------------------------------------------------
# Small transform helpers (shared by zones, fence and teach)
# --------------------------------------------------------------------------------------------


def rpy_to_matrix(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """Rotation matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` for URDF ``rpy = (roll, pitch, yaw)``."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def pose_matrix(
    xyz: Sequence[float] | np.ndarray, rpy: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """4x4 homogeneous transform from a translation and URDF roll-pitch-yaw angles."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def as_transform(value: np.ndarray | Sequence[float]) -> np.ndarray:
    """Coerce a mount to a 4x4 transform: accepts a 4x4 matrix or a 3-vector translation."""
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return pose_matrix(arr)
    if arr.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform or an xyz 3-vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("transform bottom row must be (0, 0, 0, 1)")
    R = arr[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
        raise ValueError("transform rotation must be a proper rotation matrix")
    return arr.copy()


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# --------------------------------------------------------------------------------------------
# URDF loading
# --------------------------------------------------------------------------------------------


def default_urdf_path(robot: str | RobotProfile | None = None) -> str:
    """Filesystem path of a robot's vendored URDF (the package is always installed as a directory)."""
    return str(resources.files("kiteos").joinpath("data", get_robot(robot).urdf_resource))


@functools.lru_cache(maxsize=None)
def _vendored_urdf_text(resource: str) -> str:
    return resources.files("kiteos").joinpath("data", resource).read_text(encoding="utf-8")


def read_urdf_text(urdf_path: str | Path | None = None, robot: str | RobotProfile | None = None) -> str:
    """URDF XML text: the robot's vendored copy by default, or the file at ``urdf_path``."""
    if urdf_path is None:
        return _vendored_urdf_text(get_robot(robot).urdf_resource)
    return Path(urdf_path).expanduser().read_text(encoding="utf-8")


def _floats3(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None or not text.strip():
        return default
    parts = text.split()
    if len(parts) != 3:
        raise UrdfError(f"expected 3 numbers, got {text!r}")
    x, y, z = (float(v) for v in parts)
    return (x, y, z)


def _opt_float(element: ET.Element | None, attr: str) -> float | None:
    if element is None:
        return None
    raw = element.get(attr)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def parse_urdf_joints(xml_text: str) -> dict[str, UrdfJoint]:
    """Parse every ``<joint>`` of a URDF, keyed by joint name."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise UrdfError(f"URDF is not valid XML: {exc}") from exc
    joints: dict[str, UrdfJoint] = {}
    for el in root.findall("joint"):
        name = el.get("name")
        parent = el.find("parent")
        child = el.find("child")
        if not name or parent is None or child is None:
            raise UrdfError("URDF joint without name, parent or child")
        origin = el.find("origin")
        axis = el.find("axis")
        limit = el.find("limit")
        joints[name] = UrdfJoint(
            name=name,
            type=el.get("type", ""),
            parent=parent.get("link", ""),
            child=child.get("link", ""),
            xyz=_floats3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            rpy=_floats3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
            axis=_floats3(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            lower=_opt_float(limit, "lower"),
            upper=_opt_float(limit, "upper"),
            velocity=_opt_float(limit, "velocity"),
        )
    return joints


# --------------------------------------------------------------------------------------------
# Per-arm chain
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Step:
    parent: int  # index into link_names
    child: int
    origin: np.ndarray  # (4,4) fixed joint origin
    q_index: int  # -1 for fixed joints
    K: np.ndarray | None  # skew(axis), for Rodrigues
    K2: np.ndarray | None


class ArmChain:
    """Kinematic chain of one arm, parsed from the URDF.

    Link names are reported without the side prefix (``"link3"``, not ``"left_link3"``).
    """

    def __init__(
        self, side: str, joints: Mapping[str, UrdfJoint], robot: str | RobotProfile | None = None
    ) -> None:
        if side not in ARM_SIDES:
            raise ValueError(f"side must be one of {ARM_SIDES}, got {side!r}")
        self.robot = get_robot(robot)
        self.side = side
        self.dof = self.robot.dof
        prefix = self.robot.link_prefix[side]
        root = self.robot.chain_root[side]
        expected = list(self.robot.arm_joints[side])

        children: dict[str, list[UrdfJoint]] = {}
        mount_joint: UrdfJoint | None = None
        for joint in joints.values():
            children.setdefault(joint.parent, []).append(joint)
            if joint.child == root:
                mount_joint = joint
        if root not in children:
            raise UrdfError(f"URDF has no joints below link {root!r}")

        order: list[UrdfJoint] = []
        queue = [root]
        seen = {root}
        while queue:
            link = queue.pop(0)
            for joint in children.get(link, []):
                if joint.child in seen:
                    raise UrdfError(f"URDF kinematic loop at link {joint.child!r}")
                seen.add(joint.child)
                order.append(joint)
                queue.append(joint.child)

        movable = [j for j in order if j.type != "fixed"]
        names = [j.name for j in movable]
        if names != expected:
            raise UrdfError(f"{side} arm movable joints are {names}, expected {expected}")
        for joint in movable:
            if joint.type != "revolute" or joint.lower is None or joint.upper is None:
                raise UrdfError(f"{joint.name} must be a revolute joint with position limits")
            if not joint.lower < joint.upper:
                raise UrdfError(f"{joint.name} has an empty limit range")

        def short(link: str) -> str:
            if not link.startswith(prefix):
                raise UrdfError(f"link {link!r} below {root!r} lacks the {prefix!r} prefix")
            return link[len(prefix) :]

        self.link_names: tuple[str, ...] = (short(root), *(short(j.child) for j in order))
        missing = [n for n in self.robot.required_links if n not in self.link_names]
        if missing:
            raise UrdfError(f"{side} arm is missing links {missing}")
        self.link_index: dict[str, int] = {n: i for i, n in enumerate(self.link_names)}

        steps: list[_Step] = []
        for joint in order:
            q_index = names.index(joint.name) if joint.type != "fixed" else -1
            K = K2 = None
            if q_index >= 0:
                axis = np.asarray(joint.axis, dtype=float)
                norm = float(np.linalg.norm(axis))
                if norm < 1e-12:
                    raise UrdfError(f"{joint.name} has a zero axis")
                K = _skew(axis / norm)
                K2 = K @ K
            steps.append(
                _Step(
                    parent=self.link_index[short(joint.parent)],
                    child=self.link_index[short(joint.child)],
                    origin=pose_matrix(joint.xyz, joint.rpy),
                    q_index=q_index,
                    K=K,
                    K2=K2,
                )
            )
        self._steps: tuple[_Step, ...] = tuple(steps)

        self.joint_names: list[str] = list(names)
        self.joint_limits: np.ndarray = np.array([[j.lower, j.upper] for j in movable], dtype=float)
        self.velocity_limits: np.ndarray = np.array(
            [j.velocity if j.velocity is not None else np.inf for j in movable], dtype=float
        )
        self.urdf_mount: np.ndarray = (
            pose_matrix(mount_joint.xyz, mount_joint.rpy) if mount_joint is not None else np.eye(4)
        )

        tip = self.robot.fingertip
        self.fingertip_link: str = tip.link
        if tip.link not in self.link_index:
            raise UrdfError(f"{side} arm has no fingertip link {tip.link!r}")
        if tip.between is not None:
            by_child = {short(j.child): j for j in order}
            tips = [by_child[n] for n in tip.between]
            if any(short(t.parent) != tip.link for t in tips):
                raise UrdfError(f"{side} finger frames must be children of the {tip.link}")
            mid = (np.asarray(tips[0].xyz) + np.asarray(tips[1].xyz)) / 2.0
            self.fingertip_local: np.ndarray = np.array([mid[0], mid[1], float(tip.z or 0.0), 1.0])
        else:
            from kiteos.robots import fingertip_points

            points = fingertip_points(self.robot)
            if not points or side not in points:
                raise UrdfError(f"{self.robot.key} has no fingertip point for the {side} arm")
            self.fingertip_local = np.array([*points[side], 1.0], dtype=float)

    def link_poses_batch(self, q: np.ndarray, base_T: np.ndarray) -> np.ndarray:
        """World poses of every link for a batch of configurations.

        ``q`` is (M, dof) or (dof,); returns (M, L, 4, 4) indexed by ``self.link_index``.
        """
        Q = np.asarray(q, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        if Q.ndim != 2 or Q.shape[1] != self.dof:
            raise ValueError(f"expected q of shape (M, {self.dof}), got {Q.shape}")
        m = Q.shape[0]
        out = np.empty((m, len(self.link_names), 4, 4))
        out[:, 0] = base_T
        sin = np.sin(Q)
        one_minus_cos = 1.0 - np.cos(Q)
        eye3 = np.eye(3)
        for step in self._steps:
            T = out[:, step.parent] @ step.origin
            if step.q_index >= 0:
                s = sin[:, step.q_index, None, None]
                c = one_minus_cos[:, step.q_index, None, None]
                R = eye3 + s * step.K + c * step.K2
                T[:, :3, :3] = T[:, :3, :3] @ R
            out[:, step.child] = T
        return out

    def link_poses(self, q: np.ndarray, base_T: np.ndarray) -> dict[str, np.ndarray]:
        """World pose (4x4) of every link for one configuration, keyed by unprefixed link name."""
        poses = self.link_poses_batch(np.asarray(q, dtype=float).reshape(1, self.dof), base_T)[0]
        return {name: poses[i] for i, name in enumerate(self.link_names)}

    def fingertip_batch(self, q: np.ndarray, base_T: np.ndarray) -> np.ndarray:
        """Fingertip positions (M, 3) for a batch of configurations."""
        link = self.link_poses_batch(q, base_T)[:, self.link_index[self.fingertip_link]]
        return (link @ self.fingertip_local)[:, :3]


class DualArmKinematics:
    """Forward kinematics of both arms of a robot, in its world frame.

    ``mounts`` maps side -> arm base pose in the world (4x4, or an xyz 3-vector). Sides left out
    use the URDF mount (dual OpenYAM: (0, +/-0.31, 0); OpenArm: (0, +/-0.031, 0.698)).
    """

    def __init__(
        self,
        mounts: Mapping[str, np.ndarray] | None = None,
        urdf_path: str | None = None,
        robot: str | RobotProfile | None = None,
    ) -> None:
        self.robot = get_robot(robot)
        self.dof = self.robot.dof
        joints = parse_urdf_joints(read_urdf_text(urdf_path, self.robot))
        self.chains: dict[str, ArmChain] = {side: ArmChain(side, joints, self.robot) for side in ARM_SIDES}
        mounts = dict(mounts or {})
        unknown = set(mounts) - set(ARM_SIDES)
        if unknown:
            raise ValueError(f"unknown mount sides {sorted(unknown)}; expected {ARM_SIDES}")
        self.mounts: dict[str, np.ndarray] = {
            side: as_transform(mounts[side])
            if side in mounts
            else self.chains[side].urdf_mount.copy()
            for side in ARM_SIDES
        }

    def chain(self, side: str) -> ArmChain:
        try:
            return self.chains[side]
        except KeyError:
            raise ValueError(f"side must be one of {ARM_SIDES}, got {side!r}") from None

    @property
    def joint_limits(self) -> dict[str, np.ndarray]:
        return {side: chain.joint_limits.copy() for side, chain in self.chains.items()}

    def link_poses(self, side: str, q: np.ndarray | Sequence[float]) -> dict[str, np.ndarray]:
        """World pose of every link of one arm, keyed by unprefixed link name."""
        return self.chain(side).link_poses(np.asarray(q, dtype=float), self.mounts[side])

    def link_poses_batch(self, side: str, q: np.ndarray) -> np.ndarray:
        """(M, L, 4, 4) world link poses; index links with ``self.chain(side).link_index``."""
        return self.chain(side).link_poses_batch(q, self.mounts[side])

    def fingertip(self, side: str, q: np.ndarray | Sequence[float]) -> np.ndarray:
        """World position (3,) of the finger end (dual OpenYAM: gripper-frame (0, 0, -0.1468),
        between the pads; OpenArm: the closed fingers' distal end, see ``kiteos.robots``)."""
        chain = self.chain(side)
        return chain.fingertip_batch(np.asarray(q, dtype=float).reshape(1, self.dof), self.mounts[side])[0]
