# Copyright 2026 Kite ML.
"""Station cameras: webcams, or a synthetic test pattern for runs without camera hardware.

``KiteCameraModule`` is the runtime's ``CameraModule`` with one addition: the device ``mock``
(``--<role>.hardware.camera-index mock``, or ``mock`` in the station file) streams a synthetic
640x480 test pattern (the role name, a moving bar and a frame counter) instead of opening a
camera. It lets a MOCK run go through collection, conversion and rollout on a machine without
cameras (a devbox, CI). The KiteOS CLI refuses ``--real`` with a mock camera: a policy must
never drive real arms from images that are not the scene.
"""

from __future__ import annotations

import colorsys
import threading
import time
import zlib
from typing import Any

import numpy as np
import reactivex as rx
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.hardware.sensors.camera.module import CameraModule
from dimos.hardware.sensors.camera.spec import CameraHardware
from dimos.hardware.sensors.camera.webcam import Webcam, WebcamConfig
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat, sharpness_barrier
from reactivex import create
from reactivex.observable import Observable

MOCK_CAMERA = "mock"


def is_mock_camera(device: object) -> bool:
    return isinstance(device, str) and device.strip().lower() == MOCK_CAMERA


class SyntheticCamera(CameraHardware):
    """A camera that draws a test pattern at the configured size and rate (no device)."""

    config: WebcamConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._count = 0
        label = (self.config.frame_id_prefix or "camera").removesuffix("_image")
        self._label = label
        # A stable colour per role, so the three views are told apart at a glance.
        hue = (zlib.crc32(label.encode()) % 360) / 360.0
        self._base = np.array([int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.35, 0.85)], dtype=np.uint8)

    def _frame_id(self) -> str:
        prefix = self.config.frame_id_prefix
        return f"{prefix}/camera_optical" if prefix else "camera_optical"

    def render(self, t: float) -> np.ndarray:
        """One RGB frame (H, W, 3) uint8 at time ``t``."""
        h, w = int(self.config.height), int(self.config.width)
        rows = np.linspace(0.6, 1.0, h, dtype=np.float32)[:, None, None]
        img = (rows * self._base[None, None, :].astype(np.float32)).astype(np.uint8)
        img = np.broadcast_to(img, (h, w, 3)).copy()
        x = int((t * 120.0) % max(w, 1))
        img[:, max(0, x - 6) : x + 6] = (40, 40, 40)
        try:
            import cv2

            cv2.putText(img, f"KiteOS mock camera: {self._label}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
            cv2.putText(img, f"frame {self._count}  t={t:.2f}", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        except Exception:  # text is decoration only
            pass
        return img

    def image_stream(self) -> Observable[Image]:
        def subscribe(observer: Any, scheduler: Any = None) -> Any:
            self._stop.clear()

            def loop() -> None:
                period = 1.0 / float(self.config.fps) if self.config.fps and self.config.fps > 0 else 1.0 / 30.0
                next_t = time.time()
                while not self._stop.is_set():
                    now = time.time()
                    frame = self.render(now)
                    self._count += 1
                    observer.on_next(
                        Image.from_numpy(frame, format=ImageFormat.RGB, frame_id=self._frame_id(), ts=now)
                    )
                    next_t += period
                    delay = next_t - time.time()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_t = time.time()

            self._thread = threading.Thread(target=loop, name=f"KiteOS-mock-camera-{self._label}", daemon=True)
            self._thread.start()

            def dispose() -> None:
                self.stop()

            return dispose

        return create(subscribe)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    @property
    def camera_info(self) -> CameraInfo:
        return CameraInfo.from_fov(
            60.0, int(self.config.width), int(self.config.height), axis="horizontal", frame_id=self._frame_id()
        )


class KiteCameraModule(CameraModule):
    """``CameraModule`` whose device ``mock`` is a ``SyntheticCamera`` (otherwise a webcam)."""

    @rpc
    def start(self) -> None:
        Module.start(self)
        options = self.config.hardware.model_dump()
        if is_mock_camera(self.config.hardware.camera_index):
            self.hardware = SyntheticCamera(**options)
        else:
            self.hardware = Webcam(**options)
        stream = self.hardware.image_stream()
        if self.config.frequency > 0:
            stream = stream.pipe(sharpness_barrier(self.config.frequency))
        self.register_disposable(stream.subscribe(self.color_image.publish))
        self.register_disposable(rx.interval(1.0).subscribe(lambda _: self.publish_metadata()))
