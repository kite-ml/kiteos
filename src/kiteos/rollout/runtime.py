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

"""Isolated-environment implementation of ``kiteos.rollout.module.DualArmPolicyModule``.

This module runs only inside dimOS's isolated LeRobot environment
(``native/python/lerobot``), started by the host-side ``DualArmPolicyModule``.
It imports torch, LeRobot and ``dimos_lerobot`` at import time, which exist only
there, so host code refers to it by import string and never imports it.
KiteOS reaches that environment through ``PYTHONPATH``.

It extends dimos_lerobot's ``LeRobotPolicyRuntime``, which keeps the rollout
loop, checkpoint-range clipping, trajectory execution, stop/cancel and the
Quest controls: A toggles the rollout and EITHER grip stops it. This subclass
changes the observation to three cameras and passes every chunk through
``kiteos.rollout.guards`` (start jump, per-step delta and speed for the arms;
range, start jump and speed for the grippers) before it reaches the
coordinator. A rejected chunk raises, which stops the rollout and cancels the
trajectory.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

from dimos.core.core import rpc
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.utils.logging_config import setup_logger
from dimos_lerobot.runtime import (
    _ACTION_FEATURE,
    _STATE_FEATURE,
    LeRobotPolicyRuntime,
    _LoadedPolicy,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference
import numpy as np
from numpy.typing import NDArray
from reactivex.disposable import Disposable
import torch

from kiteos.rollout.guards import GuardLimits, guard_chunk, joint_groups
from kiteos.rollout.module import CAMERA_FEATURES, DualArmPolicyConfig, DualArmPolicyModule

logger = setup_logger()

Images = dict[str, NDArray[np.uint8]]


class DualArmPolicyRuntime(LeRobotPolicyRuntime, DualArmPolicyModule):
    """Three-camera LeRobot runtime with KiteOS chunk guards."""

    config: DualArmPolicyConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_images: dict[str, tuple[NDArray[np.uint8], float]] = {}
        self._joint_groups = joint_groups(self.config.joint_names, self.config.gripper_joints)
        self._guard_limits = GuardLimits(
            max_start_jump_rad=self.config.max_start_jump_rad,
            max_joint_speed_rad_s=self.config.max_joint_speed_rad_s,
            max_step_delta_rad=self.config.max_step_delta_rad,
            max_gripper_start_jump=self.config.max_gripper_start_jump,
            max_gripper_speed=self.config.max_gripper_speed,
            blend_steps=self.config.blend_steps,
        )

    @rpc
    def start(self) -> None:
        super().start()
        for stream, feature in CAMERA_FEATURES.items():
            handler = partial(self._on_camera_image, feature)
            self.register_disposable(Disposable(getattr(self, stream).subscribe(handler)))

    def _on_camera_image(self, feature: str, image: Image) -> None:
        if image.format != ImageFormat.RGB or image.data.dtype != np.uint8:
            logger.warning("Ignoring non-uint8 RGB policy image", feature=feature)
            return
        expected = (self.config.image_height, self.config.image_width, 3)
        if image.data.shape != expected:
            logger.warning(
                "Ignoring policy image with unexpected shape",
                feature=feature,
                shape=image.data.shape,
                expected=expected,
            )
            return
        with self._lock:
            self._latest_images[feature] = (np.ascontiguousarray(image.data), image.ts)

    # The base runtime passes one image from _snapshot_observation to _predict;
    # here that value is the dict of all three camera images.
    def _snapshot_observation(  # type: ignore[override]
        self, now: float
    ) -> tuple[Images, NDArray[np.float32], float]:
        # Callers hold self._lock (same contract as the base runtime).
        max_age = self.config.max_observation_age_s
        images: Images = {}
        for feature in CAMERA_FEATURES.values():
            latest = self._latest_images.get(feature)
            if latest is None:
                raise RuntimeError(f"no image received for {feature}")
            data, ts = latest
            if now - ts > max_age:
                raise RuntimeError(f"{feature} is stale by {now - ts:.2f}s")
            images[feature] = data.copy()

        state = self._latest_joint_state
        if state is None:
            raise RuntimeError("no coordinator joint state has been received")
        if now - state.ts > max_age:
            raise RuntimeError(f"joint state is stale by {now - state.ts:.2f}s")
        positions = dict(zip(state.name, state.position, strict=False))
        missing = [name for name in self.config.joint_names if name not in positions]
        if missing:
            raise RuntimeError(f"joint state is missing configured joints: {missing}")
        vector = np.asarray([positions[name] for name in self.config.joint_names], dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("joint state contains non-finite positions")
        return images, vector, state.ts

    def _validate_features(self, policy_config: PreTrainedConfig) -> None:
        inputs = policy_config.input_features or {}
        outputs = policy_config.output_features or {}
        missing = {*CAMERA_FEATURES.values(), _STATE_FEATURE} - set(inputs)
        if missing:
            raise ValueError(
                "Policy is incompatible with the KiteOS dual-arm runtime; "
                f"missing input features: {sorted(missing)}"
            )
        if _ACTION_FEATURE not in outputs:
            raise ValueError(f"Policy has no {_ACTION_FEATURE!r} output feature")
        if getattr(policy_config, "temporal_ensemble_coeff", None) is not None:
            raise ValueError("Policies using temporal ensembling are not supported")
        expected_image = (3, self.config.image_height, self.config.image_width)
        for feature in CAMERA_FEATURES.values():
            shape = tuple(inputs[feature].shape)
            if shape != expected_image:
                raise ValueError(f"Policy {feature} shape {shape} does not match {expected_image}")
        joints = len(self.config.joint_names)
        state_shape = tuple(inputs[_STATE_FEATURE].shape)
        if state_shape[:1] != (joints,):
            raise ValueError(
                f"Policy state dimension {state_shape} does not match {joints} configured joints"
            )
        action_shape = tuple(outputs[_ACTION_FEATURE].shape)
        if action_shape[:1] != (joints,):
            raise ValueError(
                f"Policy action dimension {action_shape} does not match {joints} configured joints"
            )

    def _predict(  # type: ignore[override]
        self,
        loaded_policy: _LoadedPolicy,
        image: Images,
        state: NDArray[np.float32],
        *,
        task: str,
    ) -> NDArray[np.float32]:
        observation: dict[str, NDArray[Any]] = {**image, _STATE_FEATURE: state}
        amp = loaded_policy.device.type == "cuda" and loaded_policy.use_amp
        with torch.inference_mode(), torch.autocast(device_type="cuda") if amp else nullcontext():
            prepared = prepare_observation_for_inference(
                observation, loaded_policy.device, task=task, robot_type=self.config.robot_type
            )
            prepared = loaded_policy.preprocessor(prepared)
            predict = getattr(loaded_policy.policy, "predict_action_chunk", None)
            if not callable(predict):
                raise TypeError("Policy does not provide predict_action_chunk()")
            action_chunk = loaded_policy.postprocessor(predict(prepared))
        return np.asarray(action_chunk.to("cpu").numpy(), dtype=np.float32)

    def _trajectory(
        self, state: NDArray[np.float32], actions: NDArray[np.float32]
    ) -> JointTrajectory:
        # Raises ChunkRejectedError (a RuntimeError); the base rollout loop then stops
        # and cancels the trajectory.
        chunk = guard_chunk(
            state,
            actions,
            groups=self._joint_groups,
            fps=self.config.fps,
            limits=self._guard_limits,
        )
        return super()._trajectory(chunk.start, chunk.actions)
