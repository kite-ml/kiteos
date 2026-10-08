# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Kite ML for KiteOS.

"""KiteOS OpenArm setups: Quest teleop, data collection and policy rollout, all fenced.

The OpenArm (Enactic OpenArm v2.0, https://github.com/enactic/openarm) is a bimanual robot with
two 7-joint arms and pinch grippers. These setups mirror the runtime's ``teleop_webxr_openarm``
(``dimos/robot/manipulators/openarm/blueprints/teleop.py``) with ``KiteOpenArmCoordinator``, so
every command passes the KiteOS safety fence, and add the KiteOS pieces the runtime has no
OpenArm version of: three-camera collection with the recorder and episode monitor, and the
dual-arm policy rollout with Quest takeover.

* ``kiteos.openarm-teleop``: Quest teleop of both arms, viser viewer.
* ``kiteos.openarm-collection``: teleop + cameras ``left_wrist``, ``right_wrist``, ``top`` + recorder.
* ``kiteos.openarm-rollout`` and ``kiteos.openarm-rollout-recorded``: a LeRobot policy over the
  16 joints (7 + 7 arm joints, 2 grippers), Quest takeover, optionally recorded.

Run them with ``kiteos teleop`` / ``collect start`` / ``rollout start`` on a station whose
``robot`` is ``openarm``. Without CAN ports (``--controlcoordinator.left-can-port`` /
``--controlcoordinator.right-can-port``) the coordinator drives mock arms. The robot model comes
from ``enactic/openarm_description`` at the commit the runtime pins (fetched into the runtime's
asset cache on first use). The viser viewer and the Quest WebXR server bind 127.0.0.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import (
    JOINT_TRAJECTORY_TASK_NAME,
    joint_trajectory_task,
)
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.profile import CollectionFeature, CollectionProfile
from dimos.imitation.collection.recorder import collection_recorder
from dimos.imitation.dataprep.core import QualityConfig, SyncConfig
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.openarm.config import (
    OPENARM_ARM_JOINTS,
    OPENARM_GRIPPER_JOINTS,
    OPENARM_JOINTS,
    openarm_bimanual_model_config,
)
from dimos.robot.manipulators.openarm.teleop_ik import OpenArmPinkPoseTargetSolver

from kiteos.blueprints.collection import (
    CAMERA_ROLES,
    EPISODES_INSTANCE_NAME,
    RECORDER_INSTANCE_NAME,
    CameraDevice,
    camera_blueprints,
    camera_remappings,
)
from kiteos.blueprints.manipulation import KiteManipulationModule
from kiteos.blueprints.rollout import TRAJECTORY_VELOCITY_LIMIT, policy_blueprint
from kiteos.blueprints.teleop import (
    ARM_TELEOP_CONTROLLER_REMAPPINGS,
    ARM_TELEOP_GRIPPER_REMAPPINGS,
    COORDINATOR_INSTANCE_NAME,
    MANIPULATION_INSTANCE_NAME,
    arm_teleop_blueprint,
    default_visualization,
)
from kiteos.robots import OPENARM
from kiteos.safety.openarm_coordinator import OPENARM_TELEOP_TASK_NAME, KiteOpenArmCoordinator
from kiteos.teleop.tasks import KITEOS_TELEOP_IK

ROBOT_TYPE = OPENARM.robot_type

# As the runtime's OpenArm teleop: slower shoulder/elbow than wrist joints.
OPENARM_ARM_VELOCITY_PROFILE_RAD_S = (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
OPENARM_JOINT_VELOCITY_LIMITS_RAD_S: dict[str, float] = dict(
    zip(OPENARM_ARM_JOINTS, OPENARM_ARM_VELOCITY_PROFILE_RAD_S * 2, strict=True)
)
OPENARM_WEBXR_PINK = PinkKinematicsConfig(
    dt=0.01,
    position_cost=8.0,
    orientation_cost=2.0,
    posture_cost=0.01,
    joint_limit_posture_margin=0.3,
    lm_damping=0.01,
    gain=0.25,
)


def openarm_teleop_task() -> TaskConfig:
    """Both Quest controllers drive one bimanual IK task (the coordinator adds the model).

    The KiteOS type: the runtime's teleop IK task plus the carry lock (``kiteos.teleop.carry``).
    """
    return TaskConfig(
        name=OPENARM_TELEOP_TASK_NAME,
        type=KITEOS_TELEOP_IK,
        priority=20,
        joint_names=list(OPENARM_ARM_JOINTS),
        params={
            "bindings": [
                {"hand": "left", "target_frame": "openarm_left_grasp_frame"},
                {"hand": "right", "target_frame": "openarm_right_grasp_frame"},
            ],
            "solver_type": OpenArmPinkPoseTargetSolver,
            "pink": OPENARM_WEBXR_PINK,
            "timeout": 0.5,
            "max_command_tracking_error_deg": 10.0,
            "max_joint_velocity_rad_s": 2.0,
            "joint_velocity_limits_rad_s": OPENARM_JOINT_VELOCITY_LIMITS_RAD_S,
            "joint_command_filter_cutoff_hz": 5.0,
        },
    )


def openarm_gripper_tasks() -> list[TaskConfig]:
    """Quest-trigger gripper tasks, one per arm, at teleop priority."""
    return [
        TaskConfig(
            name=f"{side}_arm_gripper",
            type="gripper",
            joint_names=[OPENARM_GRIPPER_JOINTS[i]],
            priority=20,
            stream_bind={"gripper_command": f"{side}_gripper_command"},
        )
        for i, side in enumerate(("left", "right"))
    ]


def openarm_trajectory_task(*, priority: int = 10) -> TaskConfig:
    """The joint-trajectory task over all 16 joints (planner and scripted motion)."""
    return TaskConfig(
        name=JOINT_TRAJECTORY_TASK_NAME,
        type="trajectory",
        joint_names=list(OPENARM_JOINTS),
        priority=priority,
        params={"start_position_tolerance": 0.05},
    )


class KiteOpenArmManipulationModule(KiteManipulationModule):
    """The KiteOS manipulation module owning the OpenArm model (as the runtime's OpenArm teleop
    does), with the fence zones in its scene."""

    def _initialize_planning(self) -> None:
        self.config.model = openarm_bimanual_model_config()
        super()._initialize_planning()


def openarm_manipulation_blueprint(visualization: ManipulationVisualizationConfig | None = None) -> Blueprint:
    """The OpenArm model in viser (127.0.0.1 by default) and the planner."""
    return KiteOpenArmManipulationModule.blueprint(
        instance_name=MANIPULATION_INSTANCE_NAME,
        model=openarm_bimanual_model_config(),
        kinematics=OPENARM_WEBXR_PINK,
        visualization=visualization if visualization is not None else default_visualization(),
    )


def build_kiteos_openarm_teleop(*, visualization: ManipulationVisualizationConfig | None = None) -> Blueprint:
    """Quest teleop of both OpenArm arms on the fenced KiteOS coordinator."""
    return autoconnect(
        arm_teleop_blueprint(),
        KiteOpenArmCoordinator.blueprint(
            instance_name=COORDINATOR_INSTANCE_NAME,
            tasks=[openarm_teleop_task(), *openarm_gripper_tasks(), openarm_trajectory_task(priority=10)],
        ),
        openarm_manipulation_blueprint(visualization),
    ).remappings([*ARM_TELEOP_CONTROLLER_REMAPPINGS, *ARM_TELEOP_GRIPPER_REMAPPINGS])


# What the recorder writes: three cameras and the 16 joints, with the actions the fence applied.
OPENARM_COLLECTION = CollectionProfile(
    name="openarm-quest",
    robot_type=ROBOT_TYPE,
    observations={
        **{
            f"observation.images.{camera}": CollectionFeature(
                stream=f"{camera}_image",
                message_type=Image,
                field="data",
                dtype="video",
                shape=(480, 640, 3),
                names=["height", "width", "channels"],
            )
            for camera in CAMERA_ROLES
        },
        "observation.state": CollectionFeature(
            stream="coordinator_joint_state",
            message_type=JointState,
            field="position",
            dtype="float32",
            shape=(len(OPENARM_JOINTS),),
            names=list(OPENARM_JOINTS),
        ),
    },
    actions={
        "action": CollectionFeature(
            stream="applied_joint_position_command",
            source_kind="joint_position_updates",
            message_type=JointState,
            field="position",
            dtype="float32",
            shape=(len(OPENARM_JOINTS),),
            names=list(OPENARM_JOINTS),
        ),
    },
    sync=SyncConfig(anchor="observation.images.left_wrist", rate_hz=30, tolerance_ms=20),
    quality=QualityConfig(mode="strict"),
)


def build_kiteos_openarm_collection(
    *,
    visualization: ManipulationVisualizationConfig | None = None,
    camera_indices: Mapping[str, CameraDevice] | None = None,
) -> Blueprint:
    """Teleop, three cameras, the collection recorder and the episode monitor."""
    return autoconnect(
        build_kiteos_openarm_teleop(visualization=visualization),
        *camera_blueprints(camera_indices),
        collection_recorder(profile=OPENARM_COLLECTION, instance_name=RECORDER_INSTANCE_NAME),
        EpisodeMonitorModule.blueprint(instance_name=EPISODES_INSTANCE_NAME),
    ).remappings(camera_remappings())


def build_kiteos_openarm_rollout(
    *,
    visualization: ManipulationVisualizationConfig | None = None,
    camera_indices: Mapping[str, CameraDevice] | None = None,
    record: bool = False,
    **policy: Any,
) -> Blueprint:
    """The fenced OpenArm rollout: policy over 16 joints, Quest takeover, three cameras.

    Control ownership is the dual OpenYAM rollout's: the trajectory task spans all 16 joints at
    priority 10 (the policy drives the grippers too), the Quest IK task (priority 20) takes over
    while both grips are held, and either grip stops the policy.
    """
    parts = [
        policy_blueprint(
            joint_names=list(OPENARM_JOINTS),
            gripper_joints=list(OPENARM_GRIPPER_JOINTS),
            robot_type=ROBOT_TYPE,
            **policy,
        ),
        arm_teleop_blueprint(),
        KiteOpenArmCoordinator.blueprint(
            instance_name=COORDINATOR_INSTANCE_NAME,
            tasks=[
                openarm_teleop_task(),
                joint_trajectory_task(
                    list(OPENARM_JOINTS),
                    priority=10,
                    start_position_tolerance=0.05,
                    velocity_limits={name: TRAJECTORY_VELOCITY_LIMIT for name in OPENARM_JOINTS},
                ),
            ],
        ),
        openarm_manipulation_blueprint(visualization),
        *camera_blueprints(camera_indices),
    ]
    if record:
        parts += [
            collection_recorder(profile=OPENARM_COLLECTION, instance_name=RECORDER_INSTANCE_NAME),
            EpisodeMonitorModule.blueprint(instance_name=EPISODES_INSTANCE_NAME),
        ]
    return autoconnect(*parts).remappings([*ARM_TELEOP_CONTROLLER_REMAPPINGS, *camera_remappings()])


openarm_teleop = build_kiteos_openarm_teleop()
openarm_collection = build_kiteos_openarm_collection()
openarm_rollout = build_kiteos_openarm_rollout()
openarm_rollout_recorded = build_kiteos_openarm_rollout(record=True)
