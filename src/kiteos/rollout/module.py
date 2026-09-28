# Copyright 2026 Kite ML.
"""Host-side contract for the KiteOS dual-arm LeRobot policy.

``DualArmPolicyModule`` is dimOS's ``LeRobotPolicyModule`` with three camera
inputs and the KiteOS chunk guards configured. The implementation
(``kiteos.rollout.runtime``) runs in dimOS's isolated LeRobot environment and is
referenced here only by its import string; never import it from host code.

Operator controls (inherited from dimOS's LeRobot runtime): the Quest A button
toggles the rollout, and holding EITHER grip stops it. Holding both grips also
engages arm teleop in the rollout blueprint. A rollout refuses to start while a
grip is held.
"""

from __future__ import annotations

import os
from pathlib import Path

from dimos.core.stream import In
from dimos.imitation.policy.lerobot.module import (
    LeRobotPolicyModule,
    LeRobotPolicyModuleConfig,
)
from dimos.msgs.sensor_msgs.Image import Image
from pydantic import Field, model_validator

# Blueprint stream name -> policy input feature, as recorded by DUAL_OPENYAM_COLLECTION.
CAMERA_FEATURES: dict[str, str] = {
    "left_wrist_image": "observation.images.left_wrist",
    "right_wrist_image": "observation.images.right_wrist",
    "top_image": "observation.images.top",
}

RUNTIME_IMPLEMENTATION = "kiteos.rollout.runtime:DualArmPolicyRuntime"


class DualArmPolicyConfig(LeRobotPolicyModuleConfig):
    """Policy configuration plus the KiteOS chunk guards.

    Every guard refuses the whole chunk and stops the rollout (see
    ``kiteos.rollout.guards``). The upper bounds keep a mistyped flag from
    disabling a guard; the KiteOS fence in the coordinator still applies after
    these checks.
    """

    gripper_joints: list[str] = Field(default_factory=list)
    # Largest distance (rad) between the measured pose and a chunk's first arm target
    # (0.3 is the setting validated on the rig with the ACT close-box policy).
    max_start_jump_rad: float = Field(default=0.3, gt=0, le=0.5)
    # Fastest arm joint motion (rad/s) at execution rate (legacy kite_rollout: 3.0). The
    # trajectory task executes at most 1.0 rad/s whatever the chunk asks for.
    max_joint_speed_rad_s: float = Field(default=3.0, gt=0, le=3.0)
    # Largest arm joint change (rad) between two executed steps, at any playback speed.
    max_step_delta_rad: float = Field(default=0.1, gt=0, le=0.3)
    # Largest distance (opening) between the measured opening and the first gripper target;
    # 1.0 (the full range) because Quest-trained policies open/close in one step.
    max_gripper_start_jump: float = Field(default=1.0, gt=0, le=1.0)
    # Fastest gripper motion (opening/s) at execution rate; Quest-trigger data reaches
    # 30 opening/s, so this only rejects non-physical chunks.
    max_gripper_speed: float = Field(default=60.0, gt=0, le=100.0)
    # Blend the first N steps of each chunk from the measured pose into the policy's path,
    # so the offset between them is spread out instead of executed in one step.
    blend_steps: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _gripper_joints_are_policy_joints(self) -> DualArmPolicyConfig:
        if len(set(self.gripper_joints)) != len(self.gripper_joints):
            raise ValueError("gripper_joints must not contain duplicates")
        unknown = [name for name in self.gripper_joints if name not in self.joint_names]
        if unknown:
            raise ValueError(f"gripper_joints are not in joint_names: {unknown}")
        if len(self.gripper_joints) >= len(self.joint_names):
            raise ValueError("joint_names must contain at least one arm joint")
        return self


class DualArmPolicyModule(LeRobotPolicyModule):
    """Three cameras + 14-joint state in, 12 arm joints + 2 grippers out.

    Either Quest grip stops the policy; the Quest A button toggles it.
    """

    implementation = RUNTIME_IMPLEMENTATION
    config: DualArmPolicyConfig

    left_wrist_image: In[Image]
    right_wrist_image: In[Image]
    top_image: In[Image]

    def _runtime_env(self) -> dict[str, str]:
        # The isolated runtime imports kiteos.rollout.*; make that work even when
        # `dimos run` was started without the KiteOS CLI setting PYTHONPATH.
        return with_kiteos_pythonpath(super()._runtime_env())


def kiteos_import_root() -> str | None:
    """Directory whose presence on ``sys.path`` makes ``import kiteos`` work.

    Returns the parent of the ``kiteos`` package for a source or editable
    install (``<repo>/src``), or ``None`` when that parent is a
    ``site-packages`` directory: exporting the dimOS virtualenv's site-packages
    into the isolated LeRobot environment would shadow its own torch and numpy.
    """
    import kiteos

    package_file = getattr(kiteos, "__file__", None)
    if not package_file:
        return None
    root = Path(package_file).resolve().parent.parent
    if root.name in {"site-packages", "dist-packages"}:
        return None
    return str(root)


def with_kiteos_pythonpath(env: dict[str, str], root: str | None = None) -> dict[str, str]:
    """Return ``env`` with the KiteOS import root first on ``PYTHONPATH``.

    Args:
        env: Environment for the isolated runtime process (not modified).
        root: Import root to add; defaults to :func:`kiteos_import_root`.
    """
    result = dict(env)
    root = root if root is not None else kiteos_import_root()
    if root is None:
        return result
    parts = [part for part in result.get("PYTHONPATH", "").split(os.pathsep) if part]
    result["PYTHONPATH"] = os.pathsep.join([root, *(part for part in parts if part != root)])
    return result
