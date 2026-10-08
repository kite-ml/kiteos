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

"""KiteOS coupled WebXR teleoperation for the complete Dual OpenYAM.

A copy of the runtime ``build_dual_openyam_webxr``
(``dimos/robot/manipulators/dual_openyam/blueprints/teleop.py``) whose
coordinator is ``KiteDualOpenYamCoordinator``, so every command passes the
KiteOS safety fence before it reaches the motors. The teleop IK task is rebuilt
here instead of importing the runtime's private module names.

Run with ``kiteos teleop`` (it calls ``kiteos-runtime run kiteos.dual-openyam-teleop``). Without CAN ports
(``--controlcoordinator.left-can-port`` / ``--controlcoordinator.right-can-port``)
the coordinator drives mock arms. The viser viewer binds 127.0.0.1; expose it
with ``--manipulationmodule.visualization.host 0.0.0.0``. The Quest WebXR server
also binds 127.0.0.1 (``KiteArmTeleopModule``; the Quest reaches it over USB
through ``kiteos quest proxy``). The Quest IK task is the KiteOS one with the carry lock
(``kiteos.teleop.carry``: a thumbstick press keeps that gripper level).
"""

from dataclasses import replace

from dimos.control.coordinator import TaskConfig
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.manipulation.planning.kinematics.config import PinkKinematicsConfig
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.manipulation.visualization.viser.config import ViserVisualizationConfig
from dimos.robot.manipulators.common.blueprints import teleop_ik_task
from dimos.robot.manipulators.dual_openyam.blueprints.basic import (
    dual_openyam_trajectory_task,
)
from dimos.robot.manipulators.dual_openyam.config import (
    dual_openyam_hardware,
    dual_openyam_model_config,
)
from dimos.robot.manipulators.dual_openyam.joints import (
    DUAL_OPENYAM_ARM_JOINTS,
    DUAL_OPENYAM_GRIPPER_JOINTS,
)
from dimos.robot.manipulators.dual_openyam.teleop_ik import (
    DualOpenYamPinkPoseTargetSolver,
)
from kiteos.blueprints.manipulation import KiteManipulationModule
from kiteos.blueprints.webxr import INSTANCE_NAME as ARM_TELEOP_INSTANCE_NAME
from kiteos.blueprints.webxr import KiteArmTeleopModule
from kiteos.safety.coordinator import KiteDualOpenYamCoordinator
from kiteos.teleop.tasks import KITEOS_TELEOP_IK

COORDINATOR_INSTANCE_NAME = "ControlCoordinator"
MANIPULATION_INSTANCE_NAME = "manipulationmodule"  # keeps the --manipulationmodule.* flags
DUAL_OPENYAM_WEBXR_TASK_NAME = "teleop_dual_openyam"
VISER_HOST = "127.0.0.1"

DUAL_OPENYAM_WEBXR_PINK = PinkKinematicsConfig(
    dt=0.01,
    position_cost=8.0,
    orientation_cost=2.0,
    posture_cost=0.01,
    joint_limit_posture_margin=0.3,
    lm_damping=0.01,
    gain=1.0,
)
DUAL_OPENYAM_WEBXR_MODEL = dual_openyam_model_config()
# The runtime's teleop IK task as the KiteOS type (the same task plus the carry lock).
DUAL_OPENYAM_WEBXR_TASK = teleop_ik_task(
    dual_openyam_hardware(),
    robot_model=DUAL_OPENYAM_WEBXR_MODEL,
    name=DUAL_OPENYAM_WEBXR_TASK_NAME,
    joint_names=DUAL_OPENYAM_ARM_JOINTS,
    priority=20,
    solver_type=DualOpenYamPinkPoseTargetSolver,
    bindings=[
        {
            "hand": "left",
            "target_frame": "left_grasp_frame",
        },
        {
            "hand": "right",
            "target_frame": "right_grasp_frame",
        },
    ],
    params={
        "pink": DUAL_OPENYAM_WEBXR_PINK,
        "timeout": 0.5,
        "max_command_tracking_error_deg": 10.0,
        "max_joint_velocity_rad_s": 2.0,
        "joint_command_filter_cutoff_hz": 30.0,
    },
)
DUAL_OPENYAM_WEBXR_TASK = replace(DUAL_OPENYAM_WEBXR_TASK, type=KITEOS_TELEOP_IK)

# Quest controller poses drive the IK task's per-arm Cartesian targets.
ARM_TELEOP_CONTROLLER_REMAPPINGS: list[tuple[str, str, str]] = [
    (ARM_TELEOP_INSTANCE_NAME, "left_controller_output", "left_cartesian_command"),
    (ARM_TELEOP_INSTANCE_NAME, "right_controller_output", "right_cartesian_command"),
]
ARM_TELEOP_GRIPPER_REMAPPINGS: list[tuple[str, str, str]] = [
    (ARM_TELEOP_INSTANCE_NAME, "left_gripper_command", "left_gripper_command"),
    (ARM_TELEOP_INSTANCE_NAME, "right_gripper_command", "right_gripper_command"),
]


def arm_teleop_blueprint() -> Blueprint:
    """Quest arm input; its WebXR server listens on 127.0.0.1 only (see kiteos.blueprints.webxr)."""
    return KiteArmTeleopModule.blueprint(instance_name=ARM_TELEOP_INSTANCE_NAME)


def default_visualization() -> ViserVisualizationConfig:
    """Viser viewer bound to the local machine only."""
    return ViserVisualizationConfig(host=VISER_HOST)


def dual_openyam_gripper_tasks() -> list[TaskConfig]:
    """Quest-trigger gripper tasks, one per arm, at teleop priority."""
    return [
        TaskConfig(
            name="left_arm_gripper",
            type="gripper",
            joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[0]],
            priority=20,
            stream_bind={"gripper_command": "left_gripper_command"},
        ),
        TaskConfig(
            name="right_arm_gripper",
            type="gripper",
            joint_names=[DUAL_OPENYAM_GRIPPER_JOINTS[1]],
            priority=20,
            stream_bind={"gripper_command": "right_gripper_command"},
        ),
    ]


def manipulation_blueprint(
    visualization: ManipulationVisualizationConfig | None = None,
) -> Blueprint:
    """ManipulationModule for the dual-arm model (viser on 127.0.0.1 by default), with the fence
    zones in its scene (``--manipulationmodule.fence-path``)."""
    return KiteManipulationModule.blueprint(
        instance_name=MANIPULATION_INSTANCE_NAME,
        model=DUAL_OPENYAM_WEBXR_MODEL,
        kinematics=DUAL_OPENYAM_WEBXR_PINK,
        visualization=visualization if visualization is not None else default_visualization(),
    )


def build_kiteos_dual_openyam_webxr(
    *, visualization: ManipulationVisualizationConfig | None = None
) -> Blueprint:
    """Compose dual-arm Quest teleop on the fenced KiteOS coordinator.

    Args:
        visualization: Manipulation viewer; defaults to viser on 127.0.0.1.
    """
    return autoconnect(
        arm_teleop_blueprint(),
        KiteDualOpenYamCoordinator.blueprint(
            instance_name=COORDINATOR_INSTANCE_NAME,
            tasks=[
                DUAL_OPENYAM_WEBXR_TASK,
                *dual_openyam_gripper_tasks(),
                dual_openyam_trajectory_task(priority=10),
            ],
        ),
        manipulation_blueprint(visualization),
    ).remappings([*ARM_TELEOP_CONTROLLER_REMAPPINGS, *ARM_TELEOP_GRIPPER_REMAPPINGS])


dual_openyam_teleop = autoconnect(build_kiteos_dual_openyam_webxr())
