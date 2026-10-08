# Copyright 2026 Kite ML.
"""KiteOS teleop extensions: control tasks the KiteOS setups use instead of the runtime's.

``kiteos.teleop.tasks`` names them and registers them with the runtime's control-task registry
(by import path, so a coordinator module can register them without importing the IK stack);
``kiteos.teleop.carry`` is the Quest teleop task with the per-arm carry lock.
"""
