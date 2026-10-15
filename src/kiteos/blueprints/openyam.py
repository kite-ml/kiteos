# Copyright 2026 Kite ML.
"""KiteOS one-arm OpenYAM setups: the runtime's one-arm Quest teleop and collection, nearly unchanged.

The runtime already drives the one-arm OpenYAM (``teleop_webxr_openyam`` and
``openyam_quest_collection`` in ``dimos/robot/manipulators/openyam/blueprints``). These setups reuse
its coordinator, hardware, Quest IK task, gripper task, planning model, viewer, recorder and episode
buttons. They change only what the Kite rig needs:

* the Quest WebXR server binds 127.0.0.1 (``KiteArmTeleopModule``), where the runtime's binds every
  interface with no login. The Quest reaches it over USB through ``kiteos quest proxy``;
* the recording adds the station's top camera next to the wrist camera;
* the viewer has the KiteOS look (``kiteos.blueprints.viewer_theme``);
* the CAN port is a coordinator option (``--controlcoordinator.can-port``; mock arms without one), so
  ``kiteos run`` can pass it (see ``kiteos.safety.openyam_coordinator``).

* ``kiteos.openyam-teleop``: Quest teleop of the arm with the right controller (grip = deadman,
  trigger = gripper).
* ``kiteos.openyam-collection``: teleop + cameras ``wrist`` and ``top`` + the recorder + the episode
  monitor (Quest B starts/saves an episode, Y discards it).

The KiteOS fence does not cover the one-arm OpenYAM yet. Start these with ``kiteos run``: mock arms
by default, ``--real`` with the typed station name.
"""

from __future__ import annotations

from collections.abc import Mapping

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.profile import CollectionFeature, CollectionProfile
from dimos.imitation.collection.recorder import collection_recorder
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.manipulators.openyam.blueprints.teleop import (
    OPENYAM_QUEST_KINEMATICS,
    OPENYAM_QUEST_MODEL,
    openyam_quest_tasks,
)
from dimos.robot.manipulators.openyam.collection import OPENYAM_QUEST_COLLECTION

from kiteos.blueprints.cameras import KiteCameraModule
from kiteos.blueprints.manipulation import KiteManipulationModule
from kiteos.blueprints.collection import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EPISODES_INSTANCE_NAME,
    RECORDER_INSTANCE_NAME,
    CameraDevice,
    camera_remappings,
)
from kiteos.blueprints.teleop import (
    ARM_TELEOP_INSTANCE_NAME,
    COORDINATOR_INSTANCE_NAME,
    MANIPULATION_INSTANCE_NAME,
    arm_teleop_blueprint,
)
from kiteos.safety.openyam_coordinator import KiteOpenYamCoordinator

OPENYAM_CAMERA_ROLES: tuple[str, ...] = ("wrist", "top")
# The Kite lab Mac with the one-arm rig (top camera first in OpenCV's order); `kiteos run` passes
# the devices by flag (--wrist.hardware.camera-index, --top.hardware.camera-index).
DEFAULT_OPENYAM_CAMERA_INDICES: dict[str, CameraDevice] = {"wrist": 1, "top": 0}

# The runtime's one-arm Quest profile (wrist camera, 7 joints, the actions the arm accepted) plus
# the top camera.
OPENYAM_COLLECTION: CollectionProfile = OPENYAM_QUEST_COLLECTION.model_copy(
    update={
        "name": "openyam-quest-top",
        "observations": {
            **OPENYAM_QUEST_COLLECTION.observations,
            "observation.images.top": CollectionFeature(
                stream="top_image",
                message_type=Image,
                field="data",
                dtype="video",
                shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                names=["height", "width", "channels"],
            ),
        },
    }
)


def build_kiteos_openyam_teleop() -> Blueprint:
    """The runtime's ``teleop_webxr_openyam`` with the loopback Quest server and the CAN-port coordinator."""
    return autoconnect(
        arm_teleop_blueprint(),
        KiteOpenYamCoordinator.blueprint(instance_name=COORDINATOR_INSTANCE_NAME, tasks=openyam_quest_tasks()),
        KiteManipulationModule.blueprint(  # the runtime's viewer and planner in the KiteOS look
            instance_name=MANIPULATION_INSTANCE_NAME,
            model=OPENYAM_QUEST_MODEL,
            kinematics=OPENYAM_QUEST_KINEMATICS,
            visualization={"backend": "viser"},
        ),
    ).remappings([(ARM_TELEOP_INSTANCE_NAME, "right_controller_output", "right_cartesian_command")])


def openyam_camera(role: str, camera_index: CameraDevice) -> Blueprint:
    """One 640x480@30 camera named ``role`` (a webcam, or the synthetic ``mock`` test pattern)."""
    return KiteCameraModule.blueprint(
        instance_name=role,
        hardware=WebcamConfig(
            camera_index=camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            frame_id_prefix=f"{role}_image",
        ),
        frame_id=f"{role}_camera_link",
    )


def build_kiteos_openyam_collection(camera_indices: Mapping[str, CameraDevice] | None = None) -> Blueprint:
    """Teleop, the wrist and top cameras, the recorder and the episode monitor."""
    indices = {**DEFAULT_OPENYAM_CAMERA_INDICES, **(camera_indices or {})}
    return autoconnect(
        build_kiteos_openyam_teleop(),
        *(openyam_camera(role, indices[role]) for role in OPENYAM_CAMERA_ROLES),
        collection_recorder(profile=OPENYAM_COLLECTION, instance_name=RECORDER_INSTANCE_NAME),
        EpisodeMonitorModule.blueprint(instance_name=EPISODES_INSTANCE_NAME),
    ).remappings(camera_remappings(OPENYAM_CAMERA_ROLES))


openyam_teleop = build_kiteos_openyam_teleop()
openyam_collection = build_kiteos_openyam_collection()
