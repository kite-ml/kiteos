"""Single OpenYAM arm Quest collection: wrist + top camera, recorder, episode monitor.

The runtime's ``openyam_quest_collection``
(``dimos/robot/manipulators/openyam/blueprints/learning_quest_collection.py``) records one
wrist camera only. This is the same composition with a second camera, ``top``, and a profile
that records both (``observation.images.wrist`` and ``observation.images.top``, like the dual
OpenYAM dataset's camera names). dimOS is used unmodified.

``teleop_webxr_openyam`` picks the real or mock adapter and the CAN port from
``global_config`` when it is imported, so import this module only after setting them
(kite_collect.run does).
"""

from pathlib import Path

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.profile import CollectionFeature, CollectionProfile
from dimos.imitation.collection.recorder import collection_recorder
from dimos.imitation.dataprep.core import QualityConfig, SyncConfig
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.manipulators.openyam.blueprints.teleop import teleop_webxr_openyam
from dimos.robot.manipulators.openyam.joints import OPENYAM_JOINTS

WIDTH, HEIGHT, FPS = 640, 480, 30
CAMERAS = ("wrist", "top")


def _image(role: str) -> CollectionFeature:
    return CollectionFeature(
        stream=f"{role}_image",
        message_type=Image,
        field="data",
        dtype="video",
        shape=(HEIGHT, WIDTH, 3),
        names=["height", "width", "channels"],
    )


def _joints(stream: str, **kwargs: object) -> CollectionFeature:
    return CollectionFeature(
        stream=stream,
        message_type=JointState,
        field="position",
        dtype="float32",
        shape=(len(OPENYAM_JOINTS),),
        names=list(OPENYAM_JOINTS),
        **kwargs,
    )


# OPENYAM_QUEST_COLLECTION (dimos/robot/manipulators/openyam/collection.py) plus the top camera.
OPENYAM_QUEST_2CAM_COLLECTION = CollectionProfile(
    name="openyam-quest-2cam",
    robot_type="openyam",
    observations={
        **{f"observation.images.{role}": _image(role) for role in CAMERAS},
        "observation.state": _joints("coordinator_joint_state"),
    },
    actions={
        "action": _joints("applied_joint_position_command", source_kind="joint_position_updates"),
    },
    sync=SyncConfig(anchor="observation.images.wrist", rate_hz=FPS, tolerance_ms=20),
    quality=QualityConfig(mode="strict"),
)


def _camera(role: str, index: int) -> Blueprint:
    return CameraModule.blueprint(
        instance_name=role,
        hardware=WebcamConfig(camera_index=index, width=WIDTH, height=HEIGHT, fps=FPS,
                              frame_id_prefix=f"{role}_image"),
        frame_id=f"{role}_camera_link",
    )


def build_openyam_2cam_collection(*, recording: Path, task: str, wrist_index: int, top_index: int) -> Blueprint:
    return autoconnect(
        teleop_webxr_openyam,
        _camera("wrist", wrist_index),
        _camera("top", top_index),
        collection_recorder(profile=OPENYAM_QUEST_2CAM_COLLECTION, recording=recording, format="mcap"),
        EpisodeMonitorModule.blueprint(instance_name="episodes", task=task),
    ).remappings(
        [
            (role, stream, f"{role}_{suffix}")
            for role in CAMERAS
            for stream, suffix in (("color_image", "image"), ("camera_info", "camera_info"), ("tf", "tf"))
        ]
    )
