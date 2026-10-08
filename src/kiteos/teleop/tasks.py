# Copyright 2026 Kite ML.
"""KiteOS control-task types and their registration with the runtime's task registry.

A coordinator builds its tasks from ``TaskConfig.type`` through the runtime's control-task
registry. KiteOS adds its own types there by import path, without editing the runtime. The
registry only stores the path, so registering imports nothing heavy. That matters because a policy's
reference to a KiteOS coordinator is unpickled in an environment without the IK stack.

* ``kiteos_teleop_ik``: the runtime's Quest teleop IK task (``teleop_ik``) with the carry lock
  (``kiteos.teleop.carry``); the same input streams, plus the ``task_invoke`` commands
  ``set_carry_lock`` and ``carry_status``.
"""

from __future__ import annotations

from typing import Any, Literal

KITEOS_TELEOP_IK = "kiteos_teleop_ik"
CARRY_COMMANDS = ("set_carry_lock", "carry_status")
_FACTORIES = {KITEOS_TELEOP_IK: "kiteos.teleop.carry:create_task"}

# Carry-lock defaults (the station file's `carry:` section overrides them).
CarryYaw = Literal["follow", "lock"]
CARRY_MAX_SPEED_M_S = 0.25
CARRY_MAX_ACCEL_M_S2 = 1.0
CARRY_MAX_YAW_RATE_DEG_S = 60.0
CARRY_YAW: CarryYaw = "follow"
CARRY_MAX_TILT_ERROR_DEG = 3.0


def register_kiteos_tasks(registry: Any = None) -> None:
    """Register the KiteOS task types with the runtime's control-task registry (idempotent)."""
    # Inline, as in the runtime: importing the registry runs task-manifest discovery.
    from dimos.control.tasks.registry import control_task_registry
    from dimos.control.tasks.teleop_ik_task._registry import TASK_CONSUMES

    target = control_task_registry if registry is None else registry
    for task_type, factory in _FACTORIES.items():
        target.register_path(task_type, factory)
    target.register_bindings(
        KITEOS_TELEOP_IK,
        consumes=TASK_CONSUMES["teleop_ik"],
        exposes=CARRY_COMMANDS,
        source=__name__,
    )
