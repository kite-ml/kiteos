# Copyright 2026 Kite ML.
"""KiteOS one-arm OpenYAM coordinator: the runtime's one-arm Quest coordinator, with an explicit CAN port.

The runtime's one-arm OpenYAM setups build their hardware when their module is imported, from the
global ``--can-port`` option, which has to come before ``run`` on its command line. ``kiteos run``
passes arguments after the setup. So this coordinator builds the same hardware component (the
runtime's ``openyam_hardware``) in ``_setup_from_config`` from ``--controlcoordinator.can-port``
instead: mock arms unless a CAN port is given. ``kiteos run`` refuses a CAN port without ``--real``.

The KiteOS fence does not cover the one-arm OpenYAM yet. Like the other coordinator modules, this
one imports no teleop blueprint (no WebXR server), because a policy's reference to it is unpickled
in the runtime's isolated LeRobot environment.
"""

from __future__ import annotations

from dataclasses import replace

from dimos.control.components import HardwareComponent
from dimos.control.coordinator import ControlCoordinatorConfig
from dimos.control.teleop_coordinator import TeleopControlCoordinator
from dimos.hardware.spec import JointLimits
from dimos.hardware.whole_body.damiao.config import DamiaoRuntimeConfig
from dimos.robot.manipulators.openyam.config import OPENYAM_DOF, OPENYAM_JOINTS, openyam_hardware

from kiteos.robots import MOCK_ADAPTER

OPENYAM_REAL_ADAPTER = "openyam_damiao"
OPENYAM_BUS_DEVICE = "openyam"  # the runtime's bus device name for the arm


def openyam_arm_hardware(can_port: str | None) -> HardwareComponent:
    """The runtime's OpenYAM hardware component: mock arms unless ``can_port`` is given.

    Joints, gains and the hardware id are the runtime's; only the adapter is chosen here, as the
    runtime's ``openyam_hardware`` does for simulation (mock) and for a ``--can-port`` (real).
    """
    base = openyam_hardware()
    if not can_port:
        return replace(
            base,
            adapter_type=MOCK_ADAPTER,
            adapter_kwargs={},
            limits=JointLimits(
                position_lower=[*([None] * OPENYAM_DOF), 0.0],
                position_upper=[*([None] * OPENYAM_DOF), 1.0],
                velocity_max=[None] * len(OPENYAM_JOINTS),
            ),
        )
    return replace(
        base,
        adapter_type=OPENYAM_REAL_ADAPTER,
        adapter_kwargs={
            "runtime_config": DamiaoRuntimeConfig(bus_devices={OPENYAM_BUS_DEVICE: can_port}, gravity_comp=True)
        },
        limits=None,
    )


class KiteOpenYamCoordinatorConfig(ControlCoordinatorConfig):
    """The arm's CAN port (a USB serial on macOS, a SocketCAN interface on Linux); none for mock arms."""

    can_port: str | None = None


class KiteOpenYamCoordinator(TeleopControlCoordinator):
    """The runtime's one-arm Quest coordinator with its hardware built from ``can_port``."""

    config: KiteOpenYamCoordinatorConfig

    def _setup_from_config(self) -> None:
        self.config.hardware = [openyam_arm_hardware(self.config.can_port)]
        super()._setup_from_config()
