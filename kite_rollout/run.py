"""Run a trained LeRobot policy on the Dual OpenYAM (simulated arms unless CAN ports are given).

  cd ~/teleop_stack/dimos
  PYTHONPATH=~/teleop_stack/kiteos uv run --no-sync python -m kite_rollout.run \
      --checkpoint ~/teleop_stack/policies/act-close-box-step30000 [--real] [--speed 0.5]

Then, in another terminal (PYTHONPATH is needed there too, to find the policy module):
  PYTHONPATH=~/teleop_stack/kiteos uv run --no-sync dimos imitation rollout
Space starts/stops
the policy. On the Quest, A toggles it and holding both grips takes over (stops it).
Run from a terminal with camera access (the Claude Terminal panel), in the foreground.
"""

import argparse
import signal

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from kite_rollout.blueprint import build_dual_openyam_rollout

LEFT_CAN = "006C003E4845500F20303750"
RIGHT_CAN = "003900374845500F20303750"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="close the box")
    parser.add_argument("--real", action="store_true", help="drive the real arms (default: simulated)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 0.5 = half speed")
    # Checkpoints trained in the cloud say "cuda"; on a Mac run on the Apple GPU.
    parser.add_argument("--device", default="mps", help="torch device (default: mps)")
    parser.add_argument("--max-start-jump", type=float, default=0.2,
                        help="refuse chunks whose first target is further than this (rad) from the arms")
    args = parser.parse_args()
    if not 0.1 <= args.speed <= 1.0:
        parser.error("--speed must be between 0.1 and 1.0")
    if args.real and args.max_start_jump > 0.3:
        parser.error("--max-start-jump above 0.3 rad is only allowed with simulated arms")

    if args.real:
        print("REAL ARMS: the motors will be powered.")
        if input("Is someone supporting both arms and is the workspace clear? [y/N] ").strip().lower() != "y":
            print("Not starting.")
            return
    blueprint = build_dual_openyam_rollout(
        args.checkpoint,
        task=args.task,
        left_can_port=LEFT_CAN if args.real else None,
        right_can_port=RIGHT_CAN if args.real else None,
        speed=args.speed,
        device=args.device,
        max_start_jump_rad=args.max_start_jump,
    )
    # Treat SIGTERM like Ctrl-C so the coordinator shuts down cleanly.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    ModuleCoordinator.build(blueprint).loop()


if __name__ == "__main__":
    main()
