# Copyright 2026 Kite ML.
"""Quest (WebXR) arm-teleop module whose web server listens on loopback only.

the runtime's ``WebXRTeleopModule`` builds its HTTPS server with ``host="0.0.0.0"``: anyone on the
station's LAN could open ``https://<station>:8443/teleop``, connect before the operator's Quest
and drive both arms. The Quest never needs that: it reaches the server over USB (``adb reverse``
to the KiteOS proxy on 127.0.0.1:8080, which relays to 127.0.0.1:8443). ``KiteArmTeleopModule``
is the runtime's ``ArmTeleopModule`` with the server bound to ``server_host`` (default 127.0.0.1);
``--armteleopmodule.server-host 0.0.0.0`` restores LAN access deliberately.
"""

from __future__ import annotations

from dimos.teleop.webxr.extensions import ArmTeleopModule
from dimos.teleop.webxr.module import WebXRTeleopConfig
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Keeps the runtime's flag prefix (--armteleopmodule.*) and remapping keys.
INSTANCE_NAME = "armteleopmodule"
LOOPBACK = "127.0.0.1"


class KiteArmTeleopConfig(WebXRTeleopConfig):
    """``WebXRTeleopConfig`` plus the interface the teleop web server binds."""

    server_host: str = LOOPBACK


class KiteArmTeleopModule(ArmTeleopModule):
    """``ArmTeleopModule`` serving the WebXR page on ``server_host`` (loopback by default)."""

    config: KiteArmTeleopConfig

    def _start_server(self) -> None:
        server = getattr(self, "_web_server", None)
        if server is not None:
            server.host = self.config.server_host
            if self.config.server_host != LOOPBACK:
                logger.warning(
                    f"KiteOS WebXR teleop server bound to {self.config.server_host}: anyone who can "
                    "reach this port can drive the arms"
                )
        super()._start_server()
