# Copyright 2026 Kite ML.
"""The manipulation module (planner + viser viewer) with the fence zones in its scene.

``KiteManipulationModule`` is the runtime's ``ManipulationModule`` that, once planning has
started, adds every enabled keep-out zone of the fence file (``--manipulationmodule.fence-path``,
passed by the KiteOS CLI) to its planning world. The viser viewer then draws the zones next to
the robot, and the planner keeps its plans out of them. A half-space (a glass wall) is drawn as a
pane ``SLAB_DEPTH_M`` deep on its forbidden side, centred on the arms and wide enough to cover
everything they reach (``REACH_M`` around each arm base); a box as itself; a keep-in box cannot be
a solid obstacle and is left out. The zones are the ones the stack started with: a fence changed
later (``kiteos fence teach``) shows after a restart.

This is a view and a planning aid only: the fence itself is enforced by the coordinator.
"""

from __future__ import annotations

import math

import numpy as np
from dimos.core.core import rpc
from dimos.manipulation.manipulation_module import ManipulationModule, ManipulationModuleConfig
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.logging_config import setup_logger

from kiteos.blueprints.viewer_theme import install_kiteos_viewer_theme

logger = setup_logger()

SLAB_DEPTH_M = 0.1
REACH_M = 0.8  # covers the reach of either robot's arm from its base
ZONE_PREFIX = "kiteos_fence_"


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix."""
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def _frame_with_x(normal: np.ndarray) -> np.ndarray:
    """A rotation whose x axis is ``normal`` (unit)."""
    x = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def zone_boxes(fence) -> list[tuple[str, np.ndarray, tuple[float, float, float, float], tuple[float, float, float]]]:  # type: ignore[no-untyped-def]
    """(name, centre, quaternion xyzw, box size) in the world frame for every drawable enabled zone."""
    from kiteos.safety.zones import compile_zone

    frames = fence.frame_transforms()
    bases = np.array([frames[f"{side}_base"][:3, 3] for side in ("left", "right")])
    middle = bases.mean(axis=0)
    width = 2.0 * (float(np.max(np.linalg.norm(bases - middle, axis=1))) + REACH_M)
    out = []
    for zone in fence.zones:
        if not zone.enabled or zone.kind == "keep_in_box":
            continue
        compiled = compile_zone(zone, frames)
        if compiled.kind == "half_space":
            n = np.asarray(compiled.normal, dtype=float)
            on_plane = middle - n * (float(n @ middle) - compiled.offset)  # the arms' middle, on the plane
            center = on_plane + n * (SLAB_DEPTH_M / 2.0)
            rotation = _frame_with_x(n)
            size = (SLAB_DEPTH_M, width, width)
        else:
            center = np.asarray(compiled.center, dtype=float)
            rotation = np.asarray(compiled.rotation, dtype=float)
            size = tuple(float(2.0 * e) for e in compiled.half_extents)  # type: ignore[union-attr]
        out.append((zone.name, center, _quaternion(rotation), size))  # type: ignore[arg-type]
    return out


class KiteManipulationModuleConfig(ManipulationModuleConfig):
    """``ManipulationModuleConfig`` plus the fence file whose zones are drawn."""

    fence_path: str | None = None


class KiteManipulationModule(ManipulationModule):
    """``ManipulationModule`` in the KiteOS look, with the fence zones as planning obstacles (drawn in viser)."""

    config: KiteManipulationModuleConfig

    @rpc
    def start(self) -> None:
        try:
            install_kiteos_viewer_theme()  # before the viewer starts
        except Exception as exc:  # a look must never stop the stack
            logger.warning(f"KiteOS could not theme the viewer: {exc!r}")
        super().start()
        try:
            self._add_fence_zones()
        except Exception as exc:  # a view aid must never stop the stack
            logger.warning(f"KiteOS could not show the fence zones in the viewer: {exc!r}")

    def _add_fence_zones(self) -> None:
        path = self.config.fence_path
        if not path:
            return
        from kiteos.safety.fence import FenceConfig

        fence = FenceConfig.load(path)
        shown = []
        for name, center, (x, y, z, w), size in zone_boxes(fence):
            pose = Pose(position=Vector3(*(float(v) for v in center)), orientation=Quaternion(x, y, z, w))
            if self.add_obstacle(f"{ZONE_PREFIX}{name}", pose, "box", list(size)):
                shown.append(name)
        if shown:
            logger.info(f"KiteOS fence zones in the viewer and the planning world: {shown}")
