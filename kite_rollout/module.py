"""Host-side contract: dimOS's LeRobot policy module with three camera inputs.

The implementation (runtime.py) runs in dimOS's isolated LeRobot environment; it is found
through PYTHONPATH, which the isolated process inherits (see run.py).
"""

from pydantic import Field

from dimos.core.stream import In
from dimos.imitation.policy.lerobot.module import LeRobotPolicyModule, LeRobotPolicyModuleConfig
from dimos.msgs.sensor_msgs.Image import Image

# Blueprint stream name -> policy input feature (names used in the collected dataset).
CAMERA_FEATURES = {
    "left_wrist_image": "observation.images.left_wrist",
    "right_wrist_image": "observation.images.right_wrist",
    "top_image": "observation.images.top",
}


class DualArmPolicyConfig(LeRobotPolicyModuleConfig):
    gripper_joints: list[str] = Field(default_factory=list)
    # Refuse a chunk whose first arm target is further than this from the measured pose.
    max_start_jump_rad: float = Field(default=0.2, gt=0)
    # Refuse a chunk that would move any arm joint faster than this (at execution speed).
    max_joint_speed_rad_s: float = Field(default=3.0, gt=0)
    # Blend the first N steps of each chunk from the measured pose into the policy's path,
    # so the offset between them is spread out instead of executed in one step.
    blend_steps: int = Field(default=10, ge=1)


class DualArmPolicyModule(LeRobotPolicyModule):
    """Three cameras + 14-joint state in, 12 arm joints + 2 grippers out."""

    implementation = "kite_rollout.runtime:DualArmPolicyRuntime"
    config: DualArmPolicyConfig

    left_wrist_image: In[Image]
    right_wrist_image: In[Image]
    top_image: In[Image]
