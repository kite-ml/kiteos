# Copyright 2026 Kite ML.
"""Helper scripts that run inside the runtime's isolated LeRobot environment.

They are executed as files (``python <path> ...`` via ``isolated_python_run_command``), never
imported by KiteOS, and must not import ``kiteos``: that environment has LeRobot, torch and
the runtime, but KiteOS only through an inherited PYTHONPATH.
"""
