# Copyright 2026 Kite ML.
"""KiteOS Dual OpenYAM policy rollout: 3 cameras, fenced coordinator, policy, Quest takeover.

Port of ``kite_rollout/blueprint.py``. Run with::

    kiteos-runtime run kiteos.dual-openyam-rollout \\
        --policy.policy-path <checkpoint> --policy.task "<task>" [--policy.device cuda|mps|cpu] \\
        [--policy.fps 15] --left-wrist.hardware.camera-index <dev> \\
        --right-wrist.hardware.camera-index <dev> --top.hardware.camera-index <dev>

``--policy.policy-path`` and ``--policy.task`` are required at run time;
``--policy.fps`` is the execution rate (30 times the playback speed). Pass the
station's camera devices: the built-in defaults are the Kite lab Mac's indices
(left_wrist 0, top 1, right_wrist 2) and are wrong on any other host. Without CAN
ports the coordinator drives mock arms. The host must export ``PYTHONPATH`` to
the KiteOS sources, or install KiteOS editable, so that the runtime's isolated LeRobot
environment can import ``kiteos.rollout.runtime``.

Control ownership: the joint-trajectory task spans all 14 joints at priority 10,
so the policy drives the grippers too. The Quest gripper tasks from the teleop
blueprint are left out because once engaged they would override the policy's
gripper targets. Holding EITHER Quest grip stops the policy; holding both grips
also engages arm teleop (priority 20), during which the grippers hold their last
commanded opening. The Quest A button toggles the rollout.
"""

from collections.abc import Mapping
from typing import Any

from dimos.control.tasks.trajectory_task.trajectory_task import joint_trajectory_task
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.recorder import collection_recorder
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_GRIPPER_JOINTS,
    DUAL_OPENYAM_JOINTS,
)
from dimos.robot.manipulators.dual_openyam.learning import DUAL_OPENYAM_COLLECTION
from kiteos.blueprints.collection import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EPISODES_INSTANCE_NAME,
    RECORDER_INSTANCE_NAME,
    CameraDevice,
    camera_blueprints,
    camera_remappings,
)
from kiteos.blueprints.teleop import (
    ARM_TELEOP_CONTROLLER_REMAPPINGS,
    COORDINATOR_INSTANCE_NAME,
    DUAL_OPENYAM_WEBXR_TASK,
    arm_teleop_blueprint,
    manipulation_blueprint,
)
from kiteos.rollout.module import DualArmPolicyModule
from kiteos.safety.coordinator import KiteDualOpenYamCoordinator

POLICY_INSTANCE_NAME = "policy"
ROBOT_TYPE = "dual_openyam"
# Execution rate at 1x playback; the rate the dataset was recorded at.
POLICY_FPS = float(CAMERA_FPS)
# What the joint-trajectory task really executes: every joint (grippers included, in opening/s)
# moves at most this fast whatever the policy's chunk asks for. The runtime's default, made explicit.
TRAJECTORY_VELOCITY_LIMIT = 1.0


def policy_blueprint(
    *,
    policy_path: str | None = None,
    task: str | None = None,
    device: str | None = None,
    fps: float = POLICY_FPS,
    joint_names: list[str] | None = None,
    gripper_joints: list[str] | None = None,
    robot_type: str = ROBOT_TYPE,
    **guards: Any,
) -> Blueprint:
    """The dual-arm policy module (default: the dual OpenYAM's joints and robot type).

    Values left as ``None`` are not pinned, so ``policy_path`` and ``task``
    must then come from ``--policy.policy-path`` / ``--policy.task``, and the
    device from the checkpoint unless ``--policy.device`` is given.

    Args:
        guards: ``DualArmPolicyConfig`` guard overrides such as
            ``max_start_jump_rad`` (validated when the blueprint is parsed).
    """
    pinned = {
        key: value
        for key, value in (("policy_path", policy_path), ("task", task), ("device", device))
        if value is not None
    }
    return DualArmPolicyModule.blueprint(
        instance_name=POLICY_INSTANCE_NAME,
        joint_names=list(joint_names if joint_names is not None else DUAL_OPENYAM_JOINTS),
        gripper_joints=list(gripper_joints if gripper_joints is not None else DUAL_OPENYAM_GRIPPER_JOINTS),
        fps=fps,
        robot_type=robot_type,
        image_width=CAMERA_WIDTH,
        image_height=CAMERA_HEIGHT,
        **pinned,
        **guards,
    )


def build_kiteos_dual_openyam_rollout(
    *,
    visualization: ManipulationVisualizationConfig | None = None,
    camera_indices: Mapping[str, CameraDevice] | None = None,
    record: bool = False,
    **policy: Any,
) -> Blueprint:
    """Compose the fenced rollout stack.

    Args:
        visualization: Manipulation viewer; defaults to viser on 127.0.0.1.
        camera_indices: Optional role -> device overrides; normally passed as
            ``--<role>.hardware.camera-index`` flags from the station config.
        record: Also compose the collection recorder and episode monitor
            (instances ``recorder`` and ``episodes``, Quest B / Y) so rollouts are
            recorded in the collection format. Then ``--recorder.recording`` and
            ``--episodes.task`` are required at run time.
        policy: Forwarded to :func:`policy_blueprint`.
    """
    parts = [
        policy_blueprint(**policy),
        arm_teleop_blueprint(),
        KiteDualOpenYamCoordinator.blueprint(
            instance_name=COORDINATOR_INSTANCE_NAME,
            tasks=[
                DUAL_OPENYAM_WEBXR_TASK,
                joint_trajectory_task(
                    list(DUAL_OPENYAM_JOINTS),
                    priority=10,
                    start_position_tolerance=0.05,
                    velocity_limits={name: TRAJECTORY_VELOCITY_LIMIT for name in DUAL_OPENYAM_JOINTS},
                ),
            ],
        ),
        manipulation_blueprint(visualization),
        *camera_blueprints(camera_indices),
    ]
    if record:
        parts += [
            collection_recorder(
                profile=DUAL_OPENYAM_COLLECTION, instance_name=RECORDER_INSTANCE_NAME
            ),
            EpisodeMonitorModule.blueprint(instance_name=EPISODES_INSTANCE_NAME),
        ]
    return autoconnect(*parts).remappings([*ARM_TELEOP_CONTROLLER_REMAPPINGS, *camera_remappings()])


dual_openyam_rollout = build_kiteos_dual_openyam_rollout()
# Same stack plus the collection recorder (entry point dual-openyam-rollout-recorded;
# `kiteos rollout start --record`).
dual_openyam_rollout_recorded = build_kiteos_dual_openyam_rollout(record=True)
