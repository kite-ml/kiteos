"""Collect demos on one OpenYAM arm with the Quest (right controller), wrist + top camera.

  cd ~/teleop_stack/kiteos-runtime
  PYTHONPATH=~/teleop_stack/kiteos uv run --no-sync python -m kite_collect.run [--real]

Each run records into the next free recordings/<prefix>-NNN (default pick-plate-NNN).
Without --real the arm is simulated (mock adapter, no CAN). Episode controls, in another
terminal:  ~/teleop_stack/kiteos-runtime/.venv/bin/python ~/teleop_stack/tools/collect.py --prefix pick-plate
Run from a terminal with camera access (the Claude Terminal panel), in the foreground.
"""

import argparse
import os
import signal
import subprocess
from pathlib import Path

RECORDINGS = Path.home() / "teleop_stack" / "kiteos-runtime" / "recordings"
PID_FILE = RECORDINGS / ".kite_collect.pid"  # lets collect.py stop this stack on pause
CAN_PORT = "006C003E4845500F20303750"  # canable2 in the camera hub


def next_session(prefix: str) -> Path:
    n = 1
    while (RECORDINGS / f"{prefix}-{n:03d}").exists():
        n += 1
    return RECORDINGS / f"{prefix}-{n:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true", help="drive the real arm (default: simulated)")
    parser.add_argument("--task", default="pick up the black 384-well plate")
    parser.add_argument("--prefix", default="pick-plate", help="session folder prefix")
    parser.add_argument("--can-port", default=CAN_PORT)
    # AVFoundation order on this Mac: 0 = Innomaker (top view), 1 = USB 4K fisheye (wrist).
    parser.add_argument("--wrist", type=int, default=1, help="wrist camera index")
    parser.add_argument("--top", type=int, default=0, help="top camera index")
    args = parser.parse_args()
    if args.wrist == args.top:
        parser.error("--wrist and --top must be different cameras")

    if args.real:
        print("REAL ARM: the motors will be powered.")
        if input("Is someone supporting the arm and is the workspace clear? [y/N] ").strip().lower() != "y":
            print("Not starting.")
            return

    # The OpenYAM teleop blueprint chooses its adapter and CAN port at import time, as
    # `dimos run --can-port ...` does; the environment variables reach worker processes too.
    from dimos.core.global_config import global_config

    settings = {"can_port": args.can_port, "simulation": ""} if args.real else {"can_port": None, "simulation": "mujoco"}
    global_config.update(**settings)
    os.environ["SIMULATION"] = settings["simulation"]
    if args.real:
        os.environ["CAN_PORT"] = args.can_port
    else:
        os.environ.pop("CAN_PORT", None)

    from dimos.core.coordination.module_coordinator import ModuleCoordinator
    from kite_collect.blueprint import build_openyam_2cam_collection

    recording = next_session(args.prefix)
    blueprint = build_openyam_2cam_collection(
        recording=recording, task=args.task, wrist_index=args.wrist, top_index=args.top
    )
    print(f"{'REAL' if args.real else 'SIMULATED'} arm · recording to {recording} · task: {args.task}")
    # A sleeping Mac cuts USB power (cameras + CAN adapter vanish mid-run). Lid close still sleeps.
    subprocess.Popen(["caffeinate", "-dis", "-w", str(os.getpid())])
    # Treat SIGTERM like Ctrl-C so the coordinator shuts down and the recorder finalizes.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n")
    try:
        ModuleCoordinator.build(blueprint).loop()
    finally:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
