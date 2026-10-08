# Copyright 2026 Kite ML.
"""KiteOS OpenArm coordinator: the runtime's teleop coordinator for the OpenArm behind the fence.

``KiteOpenArmCoordinator`` does what the runtime's ``OpenArmTeleopCoordinator``
(``dimos/robot/manipulators/openarm/blueprints/teleop.py``) does in ``_setup_from_config``:
mock hardware unless both CAN ports are set (``openarm_damiao`` with both), and the OpenArm model
for the Quest IK task ``teleop_openarm``. It is rebuilt here on the runtime's
``TeleopControlCoordinator`` instead of subclassing that class, because the module that defines
it imports the WebXR server (fastapi): a policy's reference to this coordinator is unpickled in
the runtime's isolated LeRobot environment, which has no fastapi, and a failed import there
leaves the rollout without its coordinator (tests/test_openarm.py keeps this module free of it).
"""

from __future__ import annotations

from dataclasses import replace

from dimos.control.coordinator import ControlCoordinatorConfig
from dimos.control.teleop_coordinator import TeleopControlCoordinator
from dimos.robot.manipulators.openarm.config import openarm_bimanual_model_config, openarm_hardware

from kiteos.robots import OPENARM
from kiteos.safety.coordinator import FencedCoordinatorMixin, KiteCarrySettings, KiteFenceSettings

# The runtime's OPENARM_WEBXR_TASK_NAME (asserted equal by the tests): the IK task that gets the model.
OPENARM_TELEOP_TASK_NAME = "teleop_openarm"


class KiteOpenArmCoordinatorConfig(ControlCoordinatorConfig, KiteFenceSettings, KiteCarrySettings):
    """Both OpenArm CAN ports (or none, for mock arms) plus the KiteOS fence and carry-lock settings."""

    left_can_port: str | None = None
    right_can_port: str | None = None


class KiteOpenArmCoordinator(FencedCoordinatorMixin, TeleopControlCoordinator):
    """OpenArm coordinator that enforces the KiteOS safety fence on every motor write."""

    ROBOT = OPENARM
    config: KiteOpenArmCoordinatorConfig

    def _setup_from_config(self) -> None:
        self.config.tasks = [
            replace(task, params={**task.params, "robot_model": openarm_bimanual_model_config()})
            if task.name == OPENARM_TELEOP_TASK_NAME
            else task
            for task in self.config.tasks
        ]
        self.config.hardware = [
            openarm_hardware(left_can_port=self.config.left_can_port, right_can_port=self.config.right_can_port)
        ]
        super()._setup_from_config()
