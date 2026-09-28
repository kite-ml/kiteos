# Copyright 2026 Kite ML.
"""The robots KiteOS drives with its own setups and safety fence: one profile per robot.

A profile is everything robot-specific that the fence, the KiteOS setups and the CLI need:

* the arm chains (vendored URDF, joint names, link prefix, where each chain starts, the link
  names the fence relies on and the fingertip point);
* the covering spheres of every arm link, the finger spheres that grow with the gripper opening
  and the gripper stroke;
* the world frame, the default arm mounts and the minimum gains a held arm keeps;
* the runtime adapter that drives the real arms and the KiteOS setup names.

Profiles: ``dual-openyam`` (the Kite station, two OpenYAM arms) and ``openarm`` (Enactic
OpenArm v2.0, a bimanual robot with two 7-joint arms, https://github.com/enactic/openarm).

Import-light on purpose: the fence runs this inside the control tick, and the CLI imports it
without the runtime.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Literal, get_args

RobotKey = Literal["dual-openyam", "openarm"]
ROBOT_KEYS: tuple[str, ...] = get_args(RobotKey)
DEFAULT_ROBOT: RobotKey = "dual-openyam"
SIDES: tuple[str, str] = ("left", "right")
# Zone frame naming shared by every robot: "world" is the robot's world frame, "<side>_base" the
# base of one arm (placed by the fence config's mount).
WORLD_ALIAS = "world"
SETUP_KINDS: tuple[str, ...] = ("teleop", "collection", "rollout", "rollout_recorded")
MOCK_ADAPTER = "mock_whole_body"

Vec3 = tuple[float, float, float]
SphereTable = Mapping[str, tuple[tuple[Vec3, float], ...]]  # link -> ((centre, radius), ...)


@dataclass(frozen=True)
class Fingertip:
    """Where the finger end is, in the frame of ``link`` (unprefixed name).

    Either a fixed ``point`` per side, or (dual OpenYAM) the midpoint of two finger frames
    (``between``) in x/y with ``z`` taken from the meshes, because those URDF finger frames sit
    at the finger roots.
    """

    link: str
    point: Mapping[str, Vec3] | None = None
    between: tuple[str, str] | None = None
    z: float | None = None


@dataclass(frozen=True)
class RobotProfile:
    """Everything robot-specific about one robot KiteOS fences (see the module doc)."""

    key: str
    title: str
    urdf_resource: str  # file in kiteos/data
    world_frame: str  # the URDF frame zones call "world"
    dof: int  # revolute joints per arm
    arm_joints: Mapping[str, tuple[str, ...]]  # side -> URDF joint names = component joint names
    gripper_joints: Mapping[str, str]  # side -> component gripper joint
    link_prefix: Mapping[str, str]  # side -> prefix of that arm's URDF links
    chain_root: Mapping[str, str]  # side -> full URDF name of the link each arm chain starts at
    required_links: tuple[str, ...]  # unprefixed links the chain must have
    fingertip: Fingertip
    default_mounts: Mapping[str, tuple[Vec3, Vec3]]  # side -> (xyz, rpy) of the arm base in the world
    hold_kp: tuple[float, ...]
    hold_kd: tuple[float, ...]
    real_adapter: str  # runtime adapter type that drives the real arms
    robot_type: str  # dataset robot_type
    setups: Mapping[str, str]  # setup kind -> KiteOS setup name (entry point, without "kiteos.")
    wrist_links: tuple[str, ...] = ()  # links a wrist camera hangs from (for the "unfenced camera" warning)
    reference_poses: Mapping[str, tuple[float, ...]] = field(default_factory=dict)  # per-arm start poses
    spheres_resource: str | None = None  # JSON sphere table in kiteos/data (else ``inline_spheres``)
    inline_spheres: SphereTable | None = None
    inline_finger_spheres: tuple[tuple[str, int], ...] = ()
    inline_stroke_m: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def sides(self) -> tuple[str, str]:
        return SIDES

    @property
    def arm_joint_names(self) -> list[str]:
        return [*self.arm_joints["left"], *self.arm_joints["right"]]

    @property
    def joint_names(self) -> list[str]:
        """Component order: left arm, right arm, left gripper, right gripper."""
        return [*self.arm_joint_names, self.gripper_joints["left"], self.gripper_joints["right"]]

    @property
    def zone_frames(self) -> tuple[str, ...]:
        return (WORLD_ALIAS, self.world_frame, *(f"{side}_base" for side in SIDES))

    def link_spheres(self, side: str) -> SphereTable:
        """Covering spheres of one arm, by unprefixed link name (centres in the link frame)."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        if self.inline_spheres is not None:
            return self.inline_spheres
        return _sphere_file(self.spheres_resource)["spheres"][side]

    @property
    def finger_spheres(self) -> tuple[tuple[str, int], ...]:
        """(link, index) of the spheres that cover the fingers and grow with the opening."""
        if self.inline_spheres is not None:
            return self.inline_finger_spheres
        data = _sphere_file(self.spheres_resource)
        links = data["finger_links"]
        spheres = data["spheres"]["left"]
        return tuple((link, i) for link in links for i in range(len(spheres[link])))

    @property
    def gripper_stroke_m(self) -> float:
        """How far any finger point moves from closed to fully open (finger sphere growth)."""
        if self.inline_stroke_m is not None:
            return self.inline_stroke_m
        return float(_sphere_file(self.spheres_resource)["stroke_m"])

    def fingertip_point(self, side: str) -> Vec3 | None:
        if self.fingertip.point is not None:
            return self.fingertip.point[side]
        return None

    def setup_name(self, kind: str) -> str:
        try:
            return self.setups[kind]
        except KeyError:
            raise ValueError(f"unknown setup kind {kind!r}; one of {', '.join(SETUP_KINDS)}") from None


@functools.lru_cache(maxsize=None)
def _sphere_file(resource: str | None) -> dict:
    """Parsed sphere table: {"spheres": {side: {link: ((c, r), ...)}}, "finger_links", "stroke_m", ...}."""
    if resource is None:
        raise ValueError("robot profile has no sphere table")
    raw = json.loads(resources.files("kiteos").joinpath("data", resource).read_text(encoding="utf-8"))
    spheres = {
        side: {
            link: tuple(((float(s[0]), float(s[1]), float(s[2])), float(s[3])) for s in rows)
            for link, rows in raw["spheres"][side].items()
        }
        for side in SIDES
    }
    fingertip = raw.get("fingertip", {})
    return {
        "spheres": spheres,
        "finger_links": tuple(raw["gripper"]["finger_links"]),
        "stroke_m": float(raw["gripper"]["stroke_m"]),
        "fingertip": {side: tuple(float(v) for v in fingertip.get("point", {}).get(side, ())) for side in SIDES},
        "source": raw.get("source", {}),
    }


# --------------------------------------------------------------------------- dual OpenYAM

# Covering spheres of the OpenYAM links. The URDF has no collision geometry, so each link is
# covered by spheres fitted to its visual meshes (every vertex and 40k surface samples inside;
# radii include a +5 mm pad). Both arms use the same table.
OPENYAM_LINK_SPHERES: SphereTable = {
    "link1": (((-0.0315, -0.003, -0.0092), 0.065),),
    "link2": (
        ((0.0, -0.0334, 0.0235), 0.070),
        ((0.0, -0.0323, 0.132), 0.070),
        ((0.0, -0.0334, 0.2406), 0.070),
    ),
    "link3": (
        ((-0.0276, -0.0349, -0.0044), 0.080),
        ((-0.0624, -0.0349, -0.0572), 0.077),
        ((-0.0608, -0.033, -0.1491), 0.061),
        ((-0.06, -0.0344, -0.2327), 0.061),
    ),
    "link4": (
        ((-0.0184, 0.03, -0.0134), 0.066),
        ((-0.0694, 0.0339, -0.0667), 0.051),
    ),
    "link5": (((0.0345, 0.0, -0.0011), 0.054),),
    "gripper": (
        ((0.0, 0.0, -0.0157), 0.044),
        ((0.0, 0.0, -0.0681), 0.068),
        ((0.0, 0.0, -0.1206), 0.046),
    ),
}
OPENYAM_FINGER_SPHERES: tuple[tuple[str, int], ...] = (("gripper", 1), ("gripper", 2))
OPENYAM_GRIPPER_STROKE_M = 0.05
# The finger meshes end 0.1468 m along the gripper -z axis (fingers closed), 4.68 cm beyond the
# grasp frame. The URDF finger frames sit at the finger roots, so this comes from the meshes.
OPENYAM_FINGERTIP_Z_M = -0.1468

DUAL_OPENYAM = RobotProfile(
    key="dual-openyam",
    title="Dual OpenYAM",
    urdf_resource="dual_openyam.urdf",
    world_frame="dual_openyam_base",
    dof=6,
    arm_joints={side: tuple(f"{side}_joint{i}" for i in range(1, 7)) for side in SIDES},
    gripper_joints={"left": "left_arm/gripper", "right": "right_arm/gripper"},
    link_prefix={side: f"{side}_" for side in SIDES},
    chain_root={side: f"{side}_base" for side in SIDES},
    required_links=("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right", "grasp_frame"),
    fingertip=Fingertip(link="gripper", between=("tip_left", "tip_right"), z=OPENYAM_FINGERTIP_Z_M),
    # The URDF places the bases at y = +/-0.31 m (from a simulation scene); measure your rig.
    default_mounts={"left": ((0.0, 0.31, 0.0), (0.0, 0.0, 0.0)), "right": ((0.0, -0.31, 0.0), (0.0, 0.0, 0.0))},
    # Runtime dual OpenYAM arm gains: a hold never uses less stiffness than these.
    hold_kp=(80.0, 80.0, 80.0, 10.0, 10.0, 10.0),
    hold_kd=(5.0, 5.0, 5.0, 1.5, 1.5, 1.5),
    real_adapter="dual_openyam_damiao",
    robot_type="dual_openyam",
    wrist_links=("gripper", "link5"),
    reference_poses={"home": (0.0, 1.047, 1.047, 0.0, 0.0, 0.0), "zero": (0.0,) * 6},
    setups={
        "teleop": "dual-openyam-teleop",
        "collection": "dual-openyam-collection",
        "rollout": "dual-openyam-rollout",
        "rollout_recorded": "dual-openyam-rollout-recorded",
    },
    inline_spheres=OPENYAM_LINK_SPHERES,
    inline_finger_spheres=OPENYAM_FINGER_SPHERES,
    inline_stroke_m=OPENYAM_GRIPPER_STROKE_M,
)

# --------------------------------------------------------------------------- OpenArm v2.0

OPENARM_SPHERES_RESOURCE = "openarm_v20_spheres.json"

OPENARM = RobotProfile(
    key="openarm",
    title="OpenArm (bimanual)",
    urdf_resource="openarm_v20.urdf",
    world_frame="openarm_body_link0",
    dof=7,
    arm_joints={side: tuple(f"openarm_{side}_joint{i}" for i in range(1, 8)) for side in SIDES},
    gripper_joints={"left": "left_arm/gripper", "right": "right_arm/gripper"},
    link_prefix={side: f"openarm_{side}_" for side in SIDES},
    chain_root={side: f"openarm_{side}_base_link" for side in SIDES},
    required_links=(
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "ee_base_link",
        "ee_link1",
        "ee_link2",
        "grasp_frame",
    ),
    fingertip=Fingertip(link="ee_base_link"),  # point per side from the sphere file
    # The URDF mounts both arm bases on the body at z = 0.698 m, 3.1 cm either side of the centre.
    default_mounts={"left": ((0.0, 0.031, 0.698), (0.0, 0.0, 0.0)), "right": ((0.0, -0.031, 0.698), (0.0, 0.0, 0.0))},
    # Runtime OpenArm MIT gains (measured on v1.0 hardware, the v2.0 starting point).
    hold_kp=(100.0, 100.0, 80.0, 80.0, 60.0, 60.0, 60.0),
    hold_kd=(1.5, 1.5, 1.0, 1.0, 0.8, 0.8, 0.8),
    real_adapter="openarm_damiao",
    robot_type="openarm",
    wrist_links=("ee_base_link", "link6"),
    # The runtime's OpenArm home is all zeros (arms hanging along the body).
    reference_poses={"zero": (0.0,) * 7},
    setups={
        "teleop": "openarm-teleop",
        "collection": "openarm-collection",
        "rollout": "openarm-rollout",
        "rollout_recorded": "openarm-rollout-recorded",
    },
    spheres_resource=OPENARM_SPHERES_RESOURCE,
)

ROBOTS: dict[str, RobotProfile] = {DUAL_OPENYAM.key: DUAL_OPENYAM, OPENARM.key: OPENARM}


def get_robot(key: str | RobotProfile | None = None) -> RobotProfile:
    """The profile for ``key`` (default: the dual OpenYAM)."""
    if isinstance(key, RobotProfile):
        return key
    try:
        return ROBOTS[key or DEFAULT_ROBOT]
    except KeyError:
        raise ValueError(f"unknown robot {key!r}; KiteOS knows {', '.join(ROBOTS)}") from None


def robot_for_setup(name: str) -> RobotProfile | None:
    """The robot a KiteOS setup name (with or without "kiteos.") belongs to."""
    bare = name.removeprefix("kiteos.")
    for robot in ROBOTS.values():
        if bare in robot.setups.values():
            return robot
    return None


def fingertip_points(robot: RobotProfile) -> Mapping[str, Vec3] | None:
    """Per-side fingertip points for robots whose sphere file defines them."""
    if robot.fingertip.point is not None:
        return robot.fingertip.point
    if robot.spheres_resource is None:
        return None
    points = _sphere_file(robot.spheres_resource)["fingertip"]
    return {side: points[side] for side in SIDES if len(points[side]) == 3}  # type: ignore[misc]
