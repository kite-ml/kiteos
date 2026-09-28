"""Isolated-environment implementation of DualArmPolicyModule.

Extends dimOS's LeRobotPolicyRuntime (dimos_lerobot.runtime), which keeps doing the
rollout loop, checkpoint-range clipping, trajectory execution, stop/cancel and Quest
grip takeover. This subclass only changes the observation (three cameras), blends each
chunk's first steps from the measured pose, and adds start-jump and speed guards.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any

from dimos_lerobot.runtime import _ACTION_FEATURE, _STATE_FEATURE, LeRobotPolicyRuntime, _LoadedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference
import numpy as np
from numpy.typing import NDArray
from reactivex.disposable import Disposable
import torch

from dimos.core.core import rpc
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory
from dimos.utils.logging_config import setup_logger
from kite_rollout.module import CAMERA_FEATURES, DualArmPolicyModule

logger = setup_logger()

Images = dict[str, NDArray[np.uint8]]


class DualArmPolicyRuntime(LeRobotPolicyRuntime, DualArmPolicyModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_images: dict[str, tuple[NDArray[np.uint8], float]] = {}

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
            logger.warning("Ignoring policy image with unexpected shape", feature=feature,
                           shape=image.data.shape, expected=expected)
            return
        with self._lock:
            self._latest_images[feature] = (np.ascontiguousarray(image.data), image.ts)

    def _snapshot_observation(self, now: float) -> tuple[Images, NDArray[np.float32], float]:
        max_age = self.config.max_observation_age_s
        images: Images = {}
        for feature in CAMERA_FEATURES.values():
            if feature not in self._latest_images:
                raise RuntimeError(f"no image received for {feature}")
            data, ts = self._latest_images[feature]
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
            raise ValueError(f"policy is missing input features: {sorted(missing)}")
        if _ACTION_FEATURE not in outputs:
            raise ValueError(f"policy has no {_ACTION_FEATURE!r} output feature")
        if getattr(policy_config, "temporal_ensemble_coeff", None) is not None:
            raise ValueError("policies using temporal ensembling are not supported")
        expected_image = (3, self.config.image_height, self.config.image_width)
        for feature in CAMERA_FEATURES.values():
            if tuple(inputs[feature].shape) != expected_image:
                raise ValueError(f"{feature} shape {tuple(inputs[feature].shape)} != {expected_image}")
        joints = len(self.config.joint_names)
        if tuple(inputs[_STATE_FEATURE].shape)[:1] != (joints,):
            raise ValueError(f"policy state shape {inputs[_STATE_FEATURE].shape} != ({joints},)")
        if tuple(outputs[_ACTION_FEATURE].shape)[:1] != (joints,):
            raise ValueError(f"policy action shape {outputs[_ACTION_FEATURE].shape} != ({joints},)")

    def _predict(self, loaded_policy: _LoadedPolicy, image: Images, state: NDArray[np.float32],
                 *, task: str) -> NDArray[np.float32]:
        observation = {**image, _STATE_FEATURE: state}
        amp = loaded_policy.device.type == "cuda" and loaded_policy.use_amp
        with torch.inference_mode(), torch.autocast(device_type="cuda") if amp else nullcontext():
            prepared = prepare_observation_for_inference(
                observation, loaded_policy.device, task=task, robot_type=self.config.robot_type
            )
            prepared = loaded_policy.preprocessor(prepared)
            action_chunk = loaded_policy.postprocessor(loaded_policy.policy.predict_action_chunk(prepared))
        return np.asarray(action_chunk.to("cpu").numpy(), dtype=np.float32)

    def _trajectory(self, state: NDArray[np.float32], actions: NDArray[np.float32]) -> JointTrajectory:
        names = list(self.config.joint_names)
        arm = [i for i, name in enumerate(names) if name not in self.config.gripper_joints]
        jump = np.abs(actions[0, arm] - state[arm])
        worst = int(np.argmax(jump))
        if jump[worst] > self.config.max_start_jump_rad:
            raise RuntimeError(
                f"policy's first target is {jump[worst]:.2f} rad from the current pose on "
                f"{names[arm[worst]]} (limit {self.config.max_start_jump_rad} rad); refusing to jump. "
                "Move the arms near the training start pose."
            )
        steps = min(self.config.blend_steps, len(actions))
        alpha = (np.arange(1, steps + 1, dtype=np.float32) / steps)[:, None]
        blended = actions.copy()
        blended[:steps] = (1.0 - alpha) * state[None, :] + alpha * actions[:steps]
        path = np.vstack([state[None, arm], blended[:, arm]])
        speed = np.abs(np.diff(path, axis=0)).max(axis=0) * self.config.fps
        worst = int(np.argmax(speed))
        if speed[worst] > self.config.max_joint_speed_rad_s:
            raise RuntimeError(
                f"policy chunk would move {names[arm[worst]]} at {speed[worst]:.1f} rad/s "
                f"(limit {self.config.max_joint_speed_rad_s} rad/s); refusing to execute"
            )
        return super()._trajectory(state, blended)
