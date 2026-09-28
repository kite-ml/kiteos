"""Dual OpenYAM policy rollout: 3 cameras + dual-arm coordinator + policy + Quest takeover.

Without CAN ports the coordinator uses simulated (mock) arms. The joint-trajectory task
spans all 14 joints so the policy drives the grippers too; the always-active gripper
tasks from the collection blueprint are left out because they would override it. Holding
both Quest grips engages teleop (priority 20) and stops the policy.
"""

from dimos.control.coordinator import TaskConfig
from dimos.control.tasks.trajectory_task.trajectory_task import JOINT_TRAJECTORY_TASK_NAME
from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.manipulation.manipulation_module import ManipulationModule
from dimos.robot.manipulators.dual_openyam.blueprints.basic import DualOpenYamCoordinator
from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
    _dual_openyam_webxr_model,
    _dual_openyam_webxr_pink,
    _dual_openyam_webxr_task,
)
from dimos.robot.manipulators.dual_openyam.joints import DUAL_OPENYAM_GRIPPER_JOINTS, DUAL_OPENYAM_JOINTS
from dimos.teleop.webxr.extensions import ArmTeleopModule
from kite_rollout.module import DualArmPolicyModule

WIDTH, HEIGHT, FPS = 640, 480, 30


def _camera(name: str, index: int) -> Blueprint:
    return CameraModule.blueprint(
        instance_name=name,
        hardware=WebcamConfig(camera_index=index, width=WIDTH, height=HEIGHT, fps=FPS,
                              frame_id_prefix=f"{name}_image"),
        frame_id=f"{name}_camera_link",
    )


def build_dual_openyam_rollout(
    checkpoint: str,
    *,
    task: str = "close the box",
    left_can_port: str | None = None,
    right_can_port: str | None = None,
    camera_indices: tuple[int, int, int] = (0, 2, 1),  # left_wrist, right_wrist, top
    speed: float = 1.0,
    device: str | None = None,
    max_start_jump_rad: float = 0.2,
) -> Blueprint:
    left, right, top = camera_indices
    return autoconnect(
        DualArmPolicyModule.blueprint(
            instance_name="policy",
            policy_path=checkpoint,
            task=task,
            device=device,
            joint_names=list(DUAL_OPENYAM_JOINTS),
            gripper_joints=list(DUAL_OPENYAM_GRIPPER_JOINTS),
            max_start_jump_rad=max_start_jump_rad,
            fps=FPS * speed,  # execution rate: <30 plays the policy slower
            robot_type="dual_openyam",
            image_width=WIDTH,
            image_height=HEIGHT,
        ),
        ArmTeleopModule.blueprint(),
        DualOpenYamCoordinator.blueprint(
            instance_name="ControlCoordinator",
            left_can_port=left_can_port,
            right_can_port=right_can_port,
            tasks=[
                _dual_openyam_webxr_task,
                TaskConfig(
                    name=JOINT_TRAJECTORY_TASK_NAME,
                    type="trajectory",
                    joint_names=list(DUAL_OPENYAM_JOINTS),
                    priority=10,
                    params={"start_position_tolerance": 0.05},
                ),
            ],
        ),
        ManipulationModule.blueprint(
            model=_dual_openyam_webxr_model,
            kinematics=_dual_openyam_webxr_pink,
            visualization={"backend": "viser"},
        ),
        _camera("left_wrist", left),
        _camera("right_wrist", right),
        _camera("top", top),
    ).remappings(
        [
            (ArmTeleopModule, "left_controller_output", "left_cartesian_command"),
            (ArmTeleopModule, "right_controller_output", "right_cartesian_command"),
            ("left_wrist", "color_image", "left_wrist_image"),
            ("left_wrist", "camera_info", "left_wrist_camera_info"),
            ("left_wrist", "tf", "left_wrist_tf"),
            ("right_wrist", "color_image", "right_wrist_image"),
            ("right_wrist", "camera_info", "right_wrist_camera_info"),
            ("right_wrist", "tf", "right_wrist_tf"),
            ("top", "color_image", "top_image"),
            ("top", "camera_info", "top_camera_info"),
            ("top", "tf", "top_tf"),
        ]
    )
