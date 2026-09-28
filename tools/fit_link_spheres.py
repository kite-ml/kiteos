#!/usr/bin/env python3
# Copyright 2026 Kite ML.
"""Fit the OpenArm v2.0 covering spheres used by the KiteOS safety fence.

Reads the expanded OpenArm URDF (xacro already run, finger joints fixed at q = 0) and writes
``src/kiteos/data/openarm_v20_spheres.json``, which ``kiteos.robots`` loads: for each arm side
and link, a few spheres (centre in the link frame, radius including a 5 mm pad) whose union
contains the link's collision AND visual meshes, plus the gripper stroke and fingertip point.
Run it with the KiteOS runtime interpreter (trimesh, pycollada, scipy).

Method, per side and link
-------------------------
1. Place. Every <collision> and <visual> mesh of the link goes into the link frame exactly as
   the URDF places it: v_link = R(rpy) @ (scale * v_mesh) + xyz. The left arm uses scale
   "1 -1 1" (a mirror) on most links, so each side is fitted from its own geometry. Collada
   meshes are loaded with their node transforms (trimesh + pycollada).
2. Subdivide. Each mesh is split until no triangle edge is longer than 4 mm. That only splits
   triangles (the surface is unchanged) but makes every triangle short.
3. Slabs. The points are projected on the principal axis (area-weighted PCA of the surface)
   and split into K equal slabs, each widened by 4 mm on both sides. A triangle spans at most
   4 mm along the axis, so its three corners all lie in the slab that holds its centroid.
4. Spheres. Each slab gets its minimum enclosing sphere (trimesh.nsphere.minimum_nsphere,
   polished by a small convex solve because trimesh only tries furthest-site Voronoi vertices
   and misses optima fixed by 2 or 3 points) plus the 5 mm pad. A sphere is convex, so it
   holds every triangle whose corners it holds: the union covers the whole surface exactly.
5. Choose K. K grows from 1 while the volume of the union of the spheres (counted on a 1.5 mm
   grid) drops by more than 10 %, up to 6. Both sides get the same K per link, because KiteOS
   numbers the finger spheres from the left table.
6. Verify. Every mesh vertex and 40,000 surface samples per mesh (fixed seed) must be inside:
   max over points of min over spheres of (distance - radius) <= 0. A point not held with the
   full pad grows its nearest sphere (never needed with the construction above; reported).
   Centres are rounded to 0.1 mm, radii recomputed from the rounded centres and rounded up,
   and the check is rerun on the numbers written, together with the exact check that every
   subdivided triangle lies inside one sphere shrunk by the pad.

Gripper
-------
The expanded URDF fixes ``finger_joint1`` / ``finger_joint2`` at q = 0. The unfixed xacro
expansion makes them revolute about -x (joint1) and +x (joint2) through the same origins, with
joint2 mimicking joint1 (multiplier 1) and limits [-0.7854, 0] rad on the right and
[0, 0.7854] on the left (the left gripper is the mirror image, so the xacro negates them).
q = 0 is CLOSED: the finger meshes touch there and their gap is smallest over the range (the
tool checks this). So the finger spheres are fitted at q = 0 in the finger link frames as the
URDF places them, and KiteOS grows them by ``stroke_m * opening``.

A finger point at distance rho from its joint axis moves by the chord 2 rho sin(|q| / 2) when
the finger turns by q. KiteOS scales the growth linearly with the opening, so ``stroke_m`` is
the arc bound rho_max * q_max: it is at least the largest chord (the largest displacement of
any finger point over the range, reached fully open) and, since chord <= arc, stroke_m * |q| /
q_max also bounds every intermediate opening. Both are checked on a dense sweep of q with both
fingers moving per the mimic.

Fingertip: the fingers point along -z of ee_base_link (the grasp frame sits at z = -0.138).
z is the most distal finger vertex at q = 0; x, y is the midpoint of the two fingers' distal
ends, each the centre of the bounding box of that finger's vertices within 1 mm of its most
distal point.

Usage (from the kiteos repo, in the runtime venv; package:// meshes resolve through
--package-root or, if omitted, the dimOS openarm_description cache)::

    python tools/fit_link_spheres.py --urdf src/kiteos/data/openarm_v20.urdf \\
        --out src/kiteos/data/openarm_v20_spheres.json

``--vendor-urdf PATH`` also writes the vendored URDF (mesh paths rewritten to
package://openarm_description/..., provenance comment on top) from an expansion whose mesh
paths are absolute paths into the checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, cKDTree
from trimesh.remesh import subdivide_to_size

SIDES = ("left", "right")
LINKS = (
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
)
PALM_LINK = "ee_base_link"
FINGER_LINKS = ("ee_link1", "ee_link2")
PACKAGE = "openarm_description"
SOURCE = {
    "url": "https://github.com/enactic/openarm_description",
    "ref": "1fba2cbc05001f05b4514120b70130b4ac06f409",
    "xacro": "assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro",
    "xacro_args": {"robot_preset": "default_bimanual", "emit_grasp_frame": "true"},
}

PAD_M = 0.005  # added to every radius
MAX_EDGE_M = 0.004  # subdivision edge length; also the slab overlap (must not be smaller)
MAX_SPHERES = 6
MIN_VOLUME_GAIN = 0.10  # add a sphere only while the union volume drops by more than this
VOXEL_M = 0.0015  # grid step of the union volume
SURFACE_SAMPLES = 40_000  # verification samples per mesh
SEED = 0
Q_SAMPLES = 2001  # finger joint sweep (displacements)
GAP_SAMPLES = 91  # finger joint sweep (gap between the fingers)
TIP_BAND_M = 0.001  # a finger's distal end: its vertices within this of its most distal point
DISTAL = np.array([0.0, 0.0, -1.0])  # finger direction in ee_base_link
TOL_M = 1e-9  # float noise allowed in the pad checks

# Finger joints as the unfixed xacro expansion at SOURCE emits them (pinch_gripper
# config/joint/joint_axes.yaml, joint_limits.yaml and joint_mimics.yaml; the left gripper is
# the mirror image, so the xacro negates its limits). Per finger link: (axis in the joint
# frame, mimic multiplier, mimic offset) relative to finger_joint1, whose limits are given.
# The joint origins are read from the URDF. --finger-urdf reads all of this from an unfixed
# expansion instead.
FINGER_JOINTS = {
    side: {
        "limits": limits,
        "fingers": {"ee_link1": ((-1.0, 0.0, 0.0), 1.0, 0.0), "ee_link2": ((1.0, 0.0, 0.0), 1.0, 0.0)},
    }
    for side, limits in (("left", (0.0, 0.7854)), ("right", (-0.7854, 0.0)))
}

VENDOR_COMMENT = f"""<!--
  Generated from enactic/openarm_description @ {SOURCE["ref"]}
  ({SOURCE["xacro"]}, robot_preset default_bimanual, emit_grasp_frame true,
  finger joints fixed at q = 0, i.e. closed).
  Copyright Enactic, Inc. Licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0).
  Vendored for the KiteOS fence: only the mesh paths were rewritten, to
  package://{PACKAGE}/...; covering spheres in openarm_v20_spheres.json
  (tools/fit_link_spheres.py).
-->"""


# ----------------------------------------------------------------------------- URDF and meshes


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed axes x, y, z) as a rotation matrix: Rz(yaw) Ry(pitch) Rx(roll)."""
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def origin_of(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """(xyz, rotation) of an element's <origin> (identity when absent)."""
    origin = element.find("origin")
    get = origin.get if origin is not None else (lambda _key: None)
    return _floats(get("xyz"), (0, 0, 0)), rpy_matrix(_floats(get("rpy"), (0, 0, 0)))


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the unit vector ``axis`` (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def default_package_root() -> Path | None:
    """The dimOS cache of openarm_description, when dimOS is importable."""
    try:
        from dimos.robot.manipulators.openarm.config import OPENARM_DESCRIPTION_ROOT
    except ImportError:
        return None
    return Path(OPENARM_DESCRIPTION_ROOT)


def resolve_mesh(filename: str, package_root: Path | None) -> Path:
    if filename.startswith("package://"):
        package, _, relative = filename.removeprefix("package://").partition("/")
        if package != PACKAGE or package_root is None:
            raise SystemExit(f"cannot resolve {filename}: pass --package-root (the {PACKAGE} checkout)")
        return package_root / relative
    return Path(filename.removeprefix("file://"))


@dataclass
class Mesh:
    """One <collision> or <visual> mesh of a link, in the link frame."""

    label: str
    vertices: np.ndarray  # (n, 3)
    faces: np.ndarray  # (f, 3)
    samples: np.ndarray  # (SURFACE_SAMPLES, 3) verification samples on the surface


_MESH_CACHE: dict[Path, tuple[trimesh.Trimesh, np.ndarray]] = {}


def _load(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    """A mesh file (Collada node transforms applied) and its surface samples, in the mesh frame."""
    if path not in _MESH_CACHE:
        mesh = trimesh.load(path, force="mesh")
        samples, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=SEED)
        _MESH_CACHE[path] = (mesh, samples)
    return _MESH_CACHE[path]


def link_meshes(link: ET.Element, kinds: tuple[str, ...], package_root: Path | None) -> list[Mesh]:
    """The link's meshes placed in the link frame: v_link = R(rpy) @ (scale * v) + xyz."""
    meshes = []
    for kind in kinds:
        for geometry in link.findall(kind):
            mesh_element = geometry.find("geometry/mesh")
            if mesh_element is None:
                raise SystemExit(f"{link.get('name')}: {kind} geometry is not a mesh")
            path = resolve_mesh(mesh_element.get("filename", ""), package_root)
            scale = _floats(mesh_element.get("scale"), (1, 1, 1))
            xyz, rotation = origin_of(geometry)
            mesh, samples = _load(path)

            def place(points: np.ndarray) -> np.ndarray:
                return (np.asarray(points) * scale) @ rotation.T + xyz

            label = f"{kind} {path.name}"
            meshes.append(Mesh(label, place(mesh.vertices), np.asarray(mesh.faces), place(samples)))
    if not meshes:
        raise SystemExit(f"{link.get('name')}: no mesh to cover")
    return meshes


# ----------------------------------------------------------------------------- sphere fitting


@dataclass
class Sphere:
    center: np.ndarray
    radius: float  # pad included
    members: np.ndarray  # (m, 3) points this sphere must hold with the full pad


def principal_axis(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Longest principal axis of the surface (triangle centroids weighted by area)."""
    triangles = np.concatenate([vertices[faces] for vertices, faces in parts])
    area = trimesh.triangles.area(triangles)
    centroids = triangles.mean(axis=1)
    offsets = centroids - area @ centroids / area.sum()
    covariance = (offsets * area[:, None]).T @ offsets
    axis = np.linalg.eigh(covariance)[1][:, -1]
    # Orient it alike on both arms (the left arm mirrors y, which keeps x and z): the larger of
    # its x and z components is made positive, so both sides list their slabs in the same order.
    key = axis[0] if abs(axis[0]) >= abs(axis[2]) else axis[2]
    return axis if key >= 0 else -axis


def enclosing_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum enclosing sphere (no pad): trimesh's, polished by a convex solve on the hull."""
    hull = points[ConvexHull(points).vertices]
    start, _ = trimesh.nsphere.minimum_nsphere(hull)

    def radius_sq(center: np.ndarray) -> float:
        return float(((hull - center) ** 2).sum(axis=1).max())

    # min t subject to |p - c|^2 <= t for every hull point
    solved = minimize(
        lambda x: x[3],
        np.r_[start, radius_sq(start)],
        jac=lambda x: np.array([0.0, 0.0, 0.0, 1.0]),
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: x[3] - ((hull - x[:3]) ** 2).sum(axis=1),
                "jac": lambda x: np.c_[2.0 * (hull - x[:3]), np.ones(len(hull))],
            }
        ],
        options={"maxiter": 200, "ftol": 1e-14},
    )
    center = min((start, solved.x[:3]), key=radius_sq)
    return center, float(np.sqrt(((points - center) ** 2).sum(axis=1).max()))


def slab_spheres(points: np.ndarray, axis: np.ndarray, k: int) -> list[Sphere]:
    """K equal slabs along ``axis`` (widened by MAX_EDGE_M), one padded sphere per slab."""
    along = points @ axis
    edges = np.linspace(along.min(), along.max(), k + 1)
    spheres = []
    for low, high in zip(edges[:-1], edges[1:]):
        members = points[(along >= low - MAX_EDGE_M) & (along <= high + MAX_EDGE_M)]
        center, radius = enclosing_sphere(members)
        spheres.append(Sphere(center, radius + PAD_M, members))
    return spheres


def union_volume(spheres: list[Sphere]) -> float:
    """Volume of the union of the spheres, counted on a grid anchored at the frame origin
    (so a mirrored fit counts the same)."""
    centers = np.array([s.center for s in spheres])
    radii = np.array([s.radius for s in spheres])
    low = np.floor((centers - radii[:, None]).min(axis=0) / VOXEL_M)
    high = np.ceil((centers + radii[:, None]).max(axis=0) / VOXEL_M)
    axes = [(np.arange(a, b) + 0.5) * VOXEL_M for a, b in zip(low, high)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.zeros(len(grid), dtype=bool)
    for center, radius in zip(centers, radii):
        inside |= ((grid - center) ** 2).sum(axis=1) <= radius * radius
    return float(inside.sum()) * VOXEL_M**3


def fits_while_shrinking(points: np.ndarray, axis: np.ndarray) -> list[list[Sphere]]:
    """Fits for K = 1, 2, ... as long as each extra sphere shrinks the union by > 10 %."""
    fits: list[list[Sphere]] = []
    volumes: list[float] = []
    for k in range(1, MAX_SPHERES + 1):
        fit = slab_spheres(points, axis, k)
        volume = union_volume(fit)
        if volumes and volume > (1.0 - MIN_VOLUME_GAIN) * volumes[-1]:
            break
        fits.append(fit)
        volumes.append(volume)
    return fits


def excess(points: np.ndarray, spheres: list[Sphere]) -> tuple[np.ndarray, np.ndarray]:
    """Per point: min over spheres of (distance - radius), and the sphere reaching it."""
    centers = np.array([s.center for s in spheres])
    radii = np.array([s.radius for s in spheres])
    gaps = np.linalg.norm(points[:, None, :] - centers[None], axis=2) - radii[None]
    return gaps.min(axis=1), gaps.argmin(axis=1)


def hold(points: np.ndarray, spheres: list[Sphere]) -> int:
    """Grow the nearest sphere of every point not held with the full pad. Returns the count."""
    worst, nearest = excess(points, spheres)
    loose = worst > -PAD_M + TOL_M
    for index in np.unique(nearest[loose]):
        sphere = spheres[index]
        extra = points[loose & (nearest == index)]
        sphere.members = np.vstack([sphere.members, extra])
        sphere.radius = max(sphere.radius, float(np.linalg.norm(extra - sphere.center, axis=1).max()) + PAD_M)
    return int(loose.sum())


def rounded(spheres: list[Sphere]) -> list[Sphere]:
    """Centres to 0.1 mm; radii recomputed from the rounded centres (pad kept), rounded up."""
    out = []
    for sphere in spheres:
        center = np.round(sphere.center, 4) + 0.0  # + 0.0 turns -0.0 into 0.0
        need = float(np.linalg.norm(sphere.members - center, axis=1).max()) + PAD_M
        out.append(Sphere(center, math.ceil(need * 1e4) / 1e4, sphere.members))
    return out


def surface_held(parts: list[tuple[np.ndarray, np.ndarray]], spheres: list[Sphere]) -> bool:
    """Exact surface check: every subdivided triangle has its three corners inside one sphere
    shrunk by the pad, hence lies inside it (convexity), hence the whole surface is covered."""
    centers = np.array([s.center for s in spheres])
    shrunk = np.array([s.radius for s in spheres]) - PAD_M + TOL_M
    for vertices, faces in parts:
        inside = np.linalg.norm(vertices[:, None, :] - centers[None], axis=2) <= shrunk[None]
        if not inside[faces].all(axis=1).any(axis=1).all():
            return False
    return True


@dataclass
class LinkGeometry:
    meshes: list[Mesh]
    parts: list[tuple[np.ndarray, np.ndarray]]  # subdivided (vertices, faces) per mesh
    points: np.ndarray  # all subdivided vertices
    axis: np.ndarray

    @classmethod
    def of(cls, meshes: list[Mesh]) -> LinkGeometry:
        parts = [subdivide_to_size(m.vertices, m.faces, max_edge=MAX_EDGE_M, max_iter=30) for m in meshes]
        return cls(meshes, parts, np.vstack([v for v, _ in parts]), principal_axis(parts))

    @property
    def check_points(self) -> np.ndarray:
        """Every vertex and the surface samples of every mesh."""
        return np.vstack([m.vertices for m in self.meshes] + [m.samples for m in self.meshes])


@dataclass
class LinkFit:
    spheres: list[Sphere]
    points: int
    max_excess_m: float
    grown: int
    surface_exact: bool


def finish(geometry: LinkGeometry, spheres: list[Sphere]) -> LinkFit:
    """Verify (growing if ever needed), round, and verify the rounded spheres again."""
    check = geometry.check_points
    grown = hold(check, spheres)
    final = rounded(spheres)
    worst, _ = excess(check, final)
    exact = surface_held(geometry.parts, final)
    if worst.max() > 0.0 or not exact:
        raise SystemExit(f"coverage failed after rounding (max excess {worst.max():.6f} m, exact {exact})")
    return LinkFit(final, len(check), float(worst.max()), grown, exact)


# ----------------------------------------------------------------------------- gripper


@dataclass
class FingerJoint:
    xyz: np.ndarray  # joint origin in ee_base_link
    rotation: np.ndarray
    axis: np.ndarray  # unit, joint frame
    multiplier: float  # finger angle = multiplier * q + offset, q = finger_joint1
    offset: float

    def place(self, points: np.ndarray, q: float) -> np.ndarray:
        """Finger-link points in ee_base_link with the finger joint at q (of finger_joint1)."""
        turn = axis_rotation(self.axis, self.multiplier * q + self.offset)
        return points @ (self.rotation @ turn).T + self.xyz


def finger_spec_from_urdf(robot: ET.Element) -> dict:
    """FINGER_JOINTS read from an expansion whose finger joints are revolute."""
    joints = {j.get("name"): j for j in robot.findall("joint")}
    spec = {}
    for side in SIDES:
        driver = joints[f"openarm_{side}_finger_joint1"]
        limit = driver.find("limit")
        fingers = {}
        for i, link in enumerate(FINGER_LINKS, start=1):
            joint = joints[f"openarm_{side}_finger_joint{i}"]
            if joint.get("type") != "revolute":
                raise SystemExit(f"--finger-urdf: {joint.get('name')} is {joint.get('type')}, not revolute")
            mimic = joint.find("mimic")
            if mimic is not None and mimic.get("joint") != driver.get("name"):
                raise SystemExit(f"--finger-urdf: {joint.get('name')} mimics {mimic.get('joint')}")
            fingers[link] = (
                tuple(_floats(joint.find("axis").get("xyz"), (1, 0, 0))),
                float(mimic.get("multiplier", 1.0)) if mimic is not None else 1.0,
                float(mimic.get("offset", 0.0)) if mimic is not None else 0.0,
            )
        spec[side] = {"limits": (float(limit.get("lower")), float(limit.get("upper"))), "fingers": fingers}
    return spec


def finger_joints(robot: ET.Element, side: str, spec: dict) -> dict[str, FingerJoint]:
    joints = {j.get("name"): j for j in robot.findall("joint")}
    out = {}
    for i, link in enumerate(FINGER_LINKS, start=1):
        joint = joints[f"openarm_{side}_finger_joint{i}"]
        if (joint.find("parent").get("link"), joint.find("child").get("link")) != (
            f"openarm_{side}_{PALM_LINK}",
            f"openarm_{side}_{link}",
        ):
            raise SystemExit(f"{joint.get('name')} does not join {PALM_LINK} to {link}")
        xyz, rotation = origin_of(joint)
        axis, multiplier, offset = spec["fingers"][link]
        axis = np.asarray(axis, dtype=float)
        out[link] = FingerJoint(xyz, rotation, axis / np.linalg.norm(axis), multiplier, offset)
    return out


@dataclass
class Gripper:
    limits: tuple[float, float]
    closed_q: float
    open_q: float
    gap_closed_m: float
    gap_open_m: float
    chord_max_m: float  # largest displacement of any finger point, closed -> any q
    arc_m: float  # rho_max * q_max: bounds the displacement linearly in the opening
    fingertip: np.ndarray  # ee_base_link


def analyse_gripper(
    joints: dict[str, FingerJoint], fingers: dict[str, np.ndarray], limits: tuple[float, float]
) -> Gripper:
    """Closed pose, finger displacement bounds and fingertip of one side.

    ``fingers[link]`` holds every collision and visual vertex of the finger in its link frame.
    """
    low, high = limits

    # The pinching parts: the distal half of each finger (chosen at q = 0, then carried along).
    pinch = {}
    for link in FINGER_LINKS:
        along = fingers[link] @ joints[link].rotation.T @ DISTAL  # link origin = joint axis
        pinch[link] = fingers[link][along > 0.5 * along.max()]

    def gap(q: float) -> float:
        """Smallest distance between the two fingers' distal halves with the joints at q."""
        first, second = (joints[link].place(pinch[link], q) for link in FINGER_LINKS)
        return float(cKDTree(second).query(first)[0].min())

    sweep = np.linspace(low, high, GAP_SAMPLES)
    gaps = np.array([gap(q) for q in sweep])
    closed_q = float(sweep[gaps.argmin()])
    if closed_q != 0.0:
        raise SystemExit(
            f"the fingers close at q = {closed_q}, not at the q = 0 the URDF fixes: "
            "the finger spheres would need the closed-pose transform"
        )
    open_q = high if abs(high - closed_q) > abs(low - closed_q) else low
    span = abs(open_q - closed_q)

    # Displacement of every finger point from the closed pose, over a dense sweep of q.
    qs = np.linspace(low, high, Q_SAMPLES)
    displacement = np.zeros(len(qs))
    arc = 0.0
    for link in FINGER_LINKS:
        joint, points = joints[link], fingers[link]
        radial = points - np.outer(points @ joint.axis, joint.axis)  # link frame: axis through 0
        arc = max(arc, float(np.linalg.norm(radial, axis=1).max()) * abs(joint.multiplier) * span)
        rest = joint.place(points, closed_q)
        for i, q in enumerate(qs):
            moved = float(np.linalg.norm(joint.place(points, q) - rest, axis=1).max())
            displacement[i] = max(displacement[i], moved)
    chord = float(displacement.max())
    opening = np.abs(qs - closed_q) / span
    moving = opening > 0
    # KiteOS grows the finger spheres by stroke * opening: that must cover every sampled q.
    ratio = float((displacement[moving] / (arc * opening[moving])).max())
    if chord > arc or ratio > 1.0 + 1e-9:
        raise SystemExit(f"finger displacement exceeds the arc bound (chord {chord}, ratio {ratio})")

    rest = {link: joints[link].place(fingers[link], closed_q) for link in FINGER_LINKS}
    depth = max(float((points @ DISTAL).max()) for points in rest.values())
    ends = []
    for points in rest.values():
        along = points @ DISTAL
        band = points[along >= along.max() - TIP_BAND_M]
        ends.append((band.min(axis=0) + band.max(axis=0)) / 2.0)
    middle = (ends[0] + ends[1]) / 2.0
    tip = middle - (middle @ DISTAL) * DISTAL + depth * DISTAL
    return Gripper(limits, closed_q, open_q, float(gaps.min()), float(gap(open_q)), chord, arc, tip)


# ----------------------------------------------------------------------------- output


def r4(value: float) -> float:
    return round(float(value), 4) + 0.0


def dump_json(document: dict) -> str:
    """Two-space JSON with every list of numbers or strings on one line."""
    scalar = r'(?:"[^"\n]*"|-?[0-9][0-9.eE+-]*)'
    flat_list = rf"\[\s*{scalar}(?:\s*,\s*{scalar})*\s*\]"
    text = json.dumps(document, indent=2)
    return re.sub(flat_list, lambda m: json.dumps(json.loads(m.group(0))), text) + "\n"


def vendor_urdf(text: str, package_root: Path) -> str:
    """The URDF with absolute mesh paths into the checkout rewritten to package:// URIs and the
    provenance comment on top (after the XML declaration, if any). Nothing else changes."""
    root = os.path.realpath(package_root)

    def to_package(match: re.Match) -> str:
        path = match.group(2)
        if not path.startswith("package://"):
            relative = os.path.relpath(os.path.realpath(path), root)
            if relative.startswith(".."):
                raise SystemExit(f"--vendor-urdf: {path} is not inside {package_root}")
            path = f"package://{PACKAGE}/{relative}"
        return f"{match.group(1)}{path}{match.group(3)}"

    body = re.sub(r'(<mesh\b[^>]*?\bfilename=")([^"]*)(")', to_package, text)
    declaration = re.match(r"\s*<\?xml[^>]*\?>\s*?\n?", body)
    if declaration:
        head, rest = body[: declaration.end()].rstrip("\n"), body[declaration.end() :]
        return f"{head}\n{VENDOR_COMMENT}\n{rest}"
    return f"{VENDOR_COMMENT}\n{body}"


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--urdf", type=Path, required=True, help="expanded OpenArm URDF, finger joints fixed")
    parser.add_argument("--out", type=Path, required=True, help="sphere JSON to write")
    parser.add_argument("--package-root", type=Path, help=f"{PACKAGE} checkout, for package:// meshes")
    parser.add_argument("--finger-urdf", type=Path, help="unfixed expansion to read the finger joints from")
    parser.add_argument("--vendor-urdf", type=Path, help="also write the vendored URDF here")
    args = parser.parse_args()

    text = args.urdf.read_text(encoding="utf-8")
    robot = ET.fromstring(text)
    package_root = args.package_root
    if package_root is None and ("package://" in text or args.vendor_urdf):
        package_root = default_package_root()
    kinds = ("collision", "visual")
    if importlib.util.find_spec("collada") is None:
        kinds = ("collision",)
        print("WARNING: pycollada is missing, fitting the collision meshes only", file=sys.stderr)
    finger_spec = FINGER_JOINTS
    if args.finger_urdf is not None:
        finger_spec = finger_spec_from_urdf(ET.parse(args.finger_urdf).getroot())

    links = {element.get("name"): element for element in robot.findall("link")}
    geometry = {
        (side, link): LinkGeometry.of(link_meshes(links[f"openarm_{side}_{link}"], kinds, package_root))
        for side in SIDES
        for link in LINKS
    }

    fits: dict[tuple[str, str], LinkFit] = {}
    for link in LINKS:
        candidates = {
            side: fits_while_shrinking(geometry[side, link].points, geometry[side, link].axis) for side in SIDES
        }
        k = max(len(c) for c in candidates.values())  # the same count on both sides
        for side in SIDES:
            g = geometry[side, link]
            accepted = candidates[side]
            spheres = accepted[k - 1] if len(accepted) >= k else slab_spheres(g.points, g.axis, k)
            fits[side, link] = fit = finish(g, spheres)
            radii = ",".join(f"{s.radius * 1000:.1f}" for s in fit.spheres)
            print(
                f"{side:5} {link:12} K={len(fit.spheres)} r(mm)={radii:26} max_excess={fit.max_excess_m:+.5f} "
                f"points={fit.points} surface_exact={fit.surface_exact} grown={fit.grown}"
            )

    grippers = {}
    for side in SIDES:
        joints = finger_joints(robot, side, finger_spec[side])
        fingers = {link: np.vstack([m.vertices for m in geometry[side, link].meshes]) for link in FINGER_LINKS}
        grippers[side] = g = analyse_gripper(joints, fingers, finger_spec[side]["limits"])
        print(
            f"{side:5} gripper: limits {g.limits}, closed q = {g.closed_q} (distal gap {g.gap_closed_m:.4f} m, "
            f"{g.gap_open_m:.4f} m at q = {g.open_q:+g}), largest displacement {g.chord_max_m:.5f} m, "
            f"arc bound {g.arc_m:.5f} m, fingertip {np.round(g.fingertip, 5)}"
        )
    stroke = math.ceil(max(g.arc_m for g in grippers.values()) * 1e4) / 1e4
    chord = max(g.chord_max_m for g in grippers.values())
    span = max(abs(g.open_q - g.closed_q) for g in grippers.values())
    opens = {side: f"{grippers[side].open_q:+g} rad" for side in SIDES}

    sources = "collision and visual meshes" if len(kinds) == 2 else "collision meshes (pycollada missing)"
    document = {
        "robot": "openarm",
        "model": "openarm_v20",
        "source": SOURCE,
        "method": (
            f"Per link, the {sources}, placed in the link frame as the URDF places them and subdivided "
            f"to {MAX_EDGE_M * 1000:g} mm edges, are split into K equal slabs along their principal axis "
            f"overlapping by {MAX_EDGE_M * 1000:g} mm, each slab gets its minimum enclosing sphere plus the "
            f"{PAD_M * 1000:g} mm pad, K (at most {MAX_SPHERES}) grows while the union volume drops by more "
            f"than {MIN_VOLUME_GAIN:.0%}, and every vertex, {SURFACE_SAMPLES:,} surface samples per mesh and "
            "every subdivided triangle are verified inside the spheres."
        ),
        "pad_m": PAD_M,
        "gripper": {
            "finger_links": list(FINGER_LINKS),
            "stroke_m": stroke,
            "closed_note": (
                "Closed is finger q = 0, the pose this URDF fixes (the finger meshes touch; the left joints "
                f"open towards {opens['left']}, the right towards {opens['right']}), so the finger spheres need "
                f"no transform; stroke_m is the arc bound rho_max * {span:g} rad (the largest finger-point "
                f"displacement is {chord:.4f} m), so stroke_m * opening bounds every finger point at every "
                f"opening |q| / {span:g}."
            ),
        },
        "fingertip": {
            "link": PALM_LINK,
            "point": {side: [r4(v) for v in grippers[side].fingertip] for side in SIDES},
        },
        "spheres": {
            side: {
                link: [[*map(r4, s.center), r4(s.radius)] for s in fits[side, link].spheres]
                for link in LINKS
            }
            for side in SIDES
        },
        "coverage": {
            side: {
                link: {"points": fits[side, link].points, "max_excess_m": r4(fits[side, link].max_excess_m)}
                for link in LINKS
            }
            for side in SIDES
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump_json(document), encoding="utf-8")
    print(f"wrote {args.out}: stroke_m {stroke} (largest displacement {chord:.5f} m)")

    if args.vendor_urdf:
        if package_root is None:
            raise SystemExit("--vendor-urdf needs --package-root")
        args.vendor_urdf.write_text(vendor_urdf(text, package_root), encoding="utf-8")
        print(f"wrote {args.vendor_urdf}")


if __name__ == "__main__":
    main()
