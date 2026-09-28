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

"""KiteOS Dual OpenYAM Quest data collection.

Mirrors the runtime ``dual_openyam_quest_collection``
(``dimos/robot/manipulators/dual_openyam/blueprints/learning_collection.py``) on
top of the KiteOS teleop blueprint, so collection runs through the fenced
``KiteDualOpenYamCoordinator`` and records the actions the fence actually applied.

Run with ``kiteos collect start`` (it calls ``kiteos-runtime run kiteos.dual-openyam-collection --recorder.recording <dir>
--episodes.task "<task>"`` plus the station's camera devices
(``--left-wrist.hardware.camera-index``, ``--right-wrist.hardware.camera-index``,
``--top.hardware.camera-index``: an index or a ``/dev/...`` path). The KiteOS CLI
always passes the station's devices; the built-in defaults are the Kite lab Mac's
AVFoundation indices (left_wrist 0, top 1, right_wrist 2, as legacy kite_rollout)
so a bare ``kiteos-runtime run`` there does not swap views, and they are wrong on any other
host. Unlike the runtime blueprint, viser and the WebXR server bind 127.0.0.1.
"""

from collections.abc import Mapping

from dimos.core.coordination.blueprints import Blueprint, autoconnect
from dimos.hardware.sensors.camera.webcam import WebcamConfig
from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule
from dimos.imitation.collection.recorder import collection_recorder
from dimos.manipulation.visualization.config import ManipulationVisualizationConfig
from dimos.robot.manipulators.dual_openyam.learning import DUAL_OPENYAM_COLLECTION

from kiteos.blueprints.cameras import KiteCameraModule
from kiteos.blueprints.teleop import build_kiteos_dual_openyam_webxr

CameraDevice = int | str

CAMERA_ROLES: tuple[str, ...] = ("left_wrist", "right_wrist", "top")
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# The Kite lab rig's macOS mapping (station.yaml darwin values, legacy kite_rollout (0, 2, 1)).
# Other hosts must pass their devices by flag (the KiteOS CLI always does).
DEFAULT_CAMERA_INDICES: dict[str, CameraDevice] = {"left_wrist": 0, "right_wrist": 2, "top": 1}
RECORDER_INSTANCE_NAME = "recorder"
EPISODES_INSTANCE_NAME = "episodes"


def camera_blueprint(role: str, camera_index: CameraDevice | None = None) -> Blueprint:
    """One 640x480@30 camera named ``role`` whose image topic is ``<role>_image``.

    A webcam, or the synthetic test pattern for the device ``mock`` (see kiteos.blueprints.cameras).
    """
    if role not in CAMERA_ROLES:
        raise ValueError(f"unknown camera role {role!r}; expected one of {CAMERA_ROLES}")
    return KiteCameraModule.blueprint(
        instance_name=role,
        hardware=WebcamConfig(
            camera_index=DEFAULT_CAMERA_INDICES[role] if camera_index is None else camera_index,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            frame_id_prefix=f"{role}_image",
        ),
        frame_id=f"{role}_camera_link",
    )


def camera_blueprints(
    camera_indices: Mapping[str, CameraDevice] | None = None,
) -> list[Blueprint]:
    """The three station cameras, in :data:`CAMERA_ROLES` order."""
    indices = dict(camera_indices or {})
    unknown = set(indices) - set(CAMERA_ROLES)
    if unknown:
        raise ValueError(f"unknown camera roles: {sorted(unknown)}")
    return [camera_blueprint(role, indices.get(role)) for role in CAMERA_ROLES]


def camera_remappings() -> list[tuple[str, str, str]]:
    """Give each camera's streams role-prefixed names (``left_wrist_image`` ...)."""
    return [
        (role, stream, f"{role}_{suffix}")
        for role in CAMERA_ROLES
        for stream, suffix in (
            ("color_image", "image"),
            ("camera_info", "camera_info"),
            ("tf", "tf"),
        )
    ]


def build_kiteos_dual_openyam_collection(
    *,
    visualization: ManipulationVisualizationConfig | None = None,
    camera_indices: Mapping[str, CameraDevice] | None = None,
) -> Blueprint:
    """Compose teleop, three cameras, the collection recorder and the episode monitor.

    Args:
        visualization: Manipulation viewer; defaults to viser on 127.0.0.1.
        camera_indices: Optional role -> device overrides baked into the blueprint.
    """
    return autoconnect(
        build_kiteos_dual_openyam_webxr(visualization=visualization),
        *camera_blueprints(camera_indices),
        collection_recorder(profile=DUAL_OPENYAM_COLLECTION, instance_name=RECORDER_INSTANCE_NAME),
        EpisodeMonitorModule.blueprint(instance_name=EPISODES_INSTANCE_NAME),
    ).remappings(camera_remappings())


dual_openyam_collection = build_kiteos_dual_openyam_collection()
