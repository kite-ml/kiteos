# Copyright 2026 Kite ML.
"""Fence zones and their signed clearance to the arm spheres.

Three zone kinds, each expressed in one of three frames:

* ``half_space``  -- everything on the side the ``normal`` points to is forbidden (a glass wall:
  the normal points from the robot toward the glass and beyond).
* ``box``         -- a forbidden solid box (a fixture, a monitor, the space above a conveyor).
* ``keep_in_box`` -- the allowed volume; every sphere must stay fully inside it.

Frames: ``world`` (the robot's world frame, also by its own name: ``dual_openyam_base`` for the
dual OpenYAM, ``openarm_body_link0`` for the OpenArm), ``left_base`` and ``right_base`` (an arm's
own base, placed by the fence config's mount). A zone without a frame is in ``dual_openyam_base``
on the dual OpenYAM (for older fence files) and in ``world`` on other robots; the fence config
checks every zone's frame against its robot. Box orientation uses URDF roll-pitch-yaw
(``Rz(yaw) @ Ry(pitch) @ Rx(roll)``) relative to the zone frame.

Clearance is the signed distance from a sphere's surface to the forbidden region, in metres:
> 0 free distance, <= 0 penetration. The safety margin is NOT included; the fence checker adds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from kiteos.robots import ROBOTS, WORLD_ALIAS
from kiteos.safety.kinematics import WORLD_FRAME, pose_matrix

# A zone frame name; which ones a fence may use depends on its robot (``RobotProfile.zone_frames``).
Frame = str
AppliesTo = Literal["both", "left", "right"]
FRAMES: tuple[str, ...] = tuple(dict.fromkeys(f for robot in ROBOTS.values() for f in robot.zone_frames))
Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]

_MIN_NORMAL = 1e-9


class _ZoneBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Declared first so dumped YAML reads "kind, name, ..."; each subclass narrows it.
    kind: str
    name: str = Field(min_length=1)
    frame: Frame = WORLD_FRAME
    applies_to: AppliesTo = "both"
    enabled: bool = True

    @field_validator("frame")
    @classmethod
    def _known_frame(cls, v: str) -> str:
        if v not in FRAMES:
            raise ValueError(f"unknown frame {v!r}; use one of: {', '.join(FRAMES)}")
        return v

    def applies(self, side: str) -> bool:
        """Whether this zone is enabled and constrains ``side``."""
        return self.enabled and self.applies_to in ("both", side)


class HalfSpaceZone(_ZoneBase):
    """Forbidden half-space ``{x : normal . (x - point) > 0}``.

    ``normal`` points INTO the forbidden side (from the robot toward the glass and beyond). It need
    not be unit length; it is normalised when checked.
    """

    kind: Literal["half_space"] = "half_space"
    point: Vec3
    normal: Vec3

    @field_validator("normal")
    @classmethod
    def _normal_nonzero(cls, v: Vec3) -> Vec3:
        if float(np.linalg.norm(v)) < _MIN_NORMAL:
            raise ValueError(
                "normal must be a non-zero vector pointing INTO the forbidden side "
                "(from the robot toward the glass)"
            )
        return v


class _BoxBase(_ZoneBase):
    center: Vec3
    half_extents: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)

    @field_validator("half_extents")
    @classmethod
    def _extents_positive(cls, v: Vec3) -> Vec3:
        if any(e <= 0.0 for e in v):
            raise ValueError(
                f"half_extents must all be > 0 metres (got {tuple(v)}); they are HALF the box size"
            )
        return v


class BoxZone(_BoxBase):
    """Forbidden solid box, centred at ``center`` with ``half_extents`` along its own axes."""

    kind: Literal["box"] = "box"


class KeepInBox(_BoxBase):
    """Allowed volume: every sphere must stay fully inside this box."""

    kind: Literal["keep_in_box"] = "keep_in_box"


Zone = Annotated[HalfSpaceZone | BoxZone | KeepInBox, Field(discriminator="kind")]
ZONE_KINDS: tuple[str, ...] = ("half_space", "box", "keep_in_box")


def frame_transforms(mounts: Mapping[str, np.ndarray], world_frame: str = WORLD_FRAME) -> dict[str, np.ndarray]:
    """World pose of every zone frame, from the per-side arm mounts (4x4)."""
    out = {WORLD_ALIAS: np.eye(4), world_frame: np.eye(4)}
    for side, T in mounts.items():
        out[f"{side}_base"] = np.asarray(T, dtype=float)
    return out


def _frame_pose(zone: _ZoneBase, frame_T: Mapping[str, np.ndarray]) -> np.ndarray:
    T = frame_T.get(zone.frame)
    if T is None:
        if WORLD_ALIAS not in frame_T and zone.frame in (WORLD_FRAME, WORLD_ALIAS):
            return np.eye(4)  # a bare frame table (no robot): the world frame is the identity
        raise ValueError(
            f"zone {zone.name!r} uses frame {zone.frame!r} but no pose is known for it"
        )
    return np.asarray(T, dtype=float)


def _box_sdf(local: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Signed distance from points (..., 3) in the box frame to the box surface (< 0 inside)."""
    q = np.abs(local) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


@dataclass(frozen=True, eq=False)
class CompiledZone:
    """A zone resolved into world-frame arrays, ready for repeated clearance queries."""

    name: str
    kind: str
    normal: np.ndarray | None = None  # half-space: unit normal (world)
    offset: float = 0.0  # half-space: normal . point
    center: np.ndarray | None = None  # box: centre (world)
    rotation: np.ndarray | None = None  # box: world-from-box rotation (3x3)
    half_extents: np.ndarray | None = None

    def clearance(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Signed clearance (...,) for sphere centres (..., 3) and broadcastable radii."""
        centers = np.asarray(centers, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if self.kind == "half_space":
            assert self.normal is not None
            return (self.offset - centers @ self.normal) - radii
        assert (
            self.center is not None and self.rotation is not None and self.half_extents is not None
        )
        local = (centers - self.center) @ self.rotation  # R^T (c - centre), row-vector form
        sdf = _box_sdf(local, self.half_extents)
        if self.kind == "box":
            return sdf - radii
        return -sdf - radii  # keep_in_box


def compile_zone(
    zone: HalfSpaceZone | BoxZone | KeepInBox, frame_T: Mapping[str, np.ndarray]
) -> CompiledZone:
    """Resolve a zone's frame into world coordinates."""
    T = _frame_pose(zone, frame_T)
    R, t = T[:3, :3], T[:3, 3]
    if isinstance(zone, HalfSpaceZone):
        n = R @ np.asarray(zone.normal, dtype=float)
        n = n / np.linalg.norm(n)
        p = R @ np.asarray(zone.point, dtype=float) + t
        return CompiledZone(name=zone.name, kind=zone.kind, normal=n, offset=float(n @ p))
    if isinstance(zone, (BoxZone, KeepInBox)):
        box_T = T @ pose_matrix(zone.center, zone.rpy)
        return CompiledZone(
            name=zone.name,
            kind=zone.kind,
            center=box_T[:3, 3].copy(),
            rotation=box_T[:3, :3].copy(),
            half_extents=np.asarray(zone.half_extents, dtype=float),
        )
    raise TypeError(f"unsupported zone type {type(zone).__name__}")


def clearance(
    zone: HalfSpaceZone | BoxZone | KeepInBox,
    centers_world: np.ndarray,
    radii: np.ndarray,
    frame_T: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Per-sphere signed clearance in metres: > 0 free distance, <= 0 penetration.

    ``frame_T`` maps frame name -> world pose (4x4); see ``frame_transforms``. The margin is not
    included. ``centers_world`` may have any leading shape (..., 3).
    """
    return compile_zone(zone, frame_T).clearance(centers_world, radii)
