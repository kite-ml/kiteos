# Copyright 2026 Kite ML.
"""Drive a MOCK KiteOS OpenArm stack without a Quest: the scripted side of the OpenArm workflow.

Start a stack on the OpenArm station first (``kiteos --station config/station.openarm.yaml
teleop``, ``collect start`` or ``rollout start``), then run this with the runtime's Python (the
one ``kiteos`` runs in):

    python tools/openarm_workflow.py status       # fence mode, holds and each arm's clearance
    python tools/openarm_workflow.py ready        # both arms to a ready pose (a fenced trajectory)
    python tools/openarm_workflow.py through      # a trajectory through the glass: the fence rejects it
    python tools/openarm_workflow.py stream       # teleop-like joint commands toward the glass: held
    python tools/openarm_workflow.py home         # both arms back to zero
    python tools/openarm_workflow.py collect 6    # six scripted picks, one episode each (collect stack)
    python tools/openarm_workflow.py rollout 12   # from the ready pose, run the policy 12 s (rollout stack)

It joins the stack's bus from the KiteOS run-state file and refuses anything but a mock KiteOS
OpenArm run: every action moves arms. The poses assume the example fence
(``config/fence.openarm.yaml``: glass at x = 0.60 m).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from kiteos.cli._common import attach_to_run
from kiteos.robots import OPENARM

run = attach_to_run("openarm workflow")
if run is None or run.real or not run.blueprint.startswith("kiteos.openarm-"):
    sys.exit("openarm_workflow drives only a mock KiteOS OpenArm stack (config/station.openarm.yaml)")

from dimos.core.transport_factory import make_transport  # noqa: E402
from dimos.imitation.collection.episode_monitor import EpisodeControlSpec  # noqa: E402
from dimos.imitation.policy.lerobot.module import RolloutControlSpec  # noqa: E402
from dimos.msgs.sensor_msgs.JointState import JointState  # noqa: E402
from dimos.msgs.trajectory_msgs.JointTrajectory import JointTrajectory  # noqa: E402
from dimos.msgs.trajectory_msgs.TrajectoryPoint import TrajectoryPoint  # noqa: E402
from dimos.porcelain.dimos import Dimos  # noqa: E402

LEFT = list(OPENARM.arm_joints["left"])
ARMS = OPENARM.arm_joint_names  # 14: left arm, right arm
ALL = OPENARM.joint_names  # 16: the arms, then the left and right grippers
READY = [-0.5, 0, 0, 1.2, 0, 0, 0] + [0.5, 0, 0, 1.2, 0, 0, 0]
ZERO = [0.0] * 14
THROUGH_LEFT = [-1.2, 0, 0, 0.3, 0, 0, 0]  # the left hand ends up past the glass
REACH_LEFT = [-1.57, 0, 0, 0, 0, 0, 0]  # the left arm straight out toward the glass


def positions(cc, names: list[str]) -> list[float]:  # type: ignore[no-untyped-def]
    named = cc.get_joint_positions()
    return [float(named[n]) for n in names]


def trajectory(names: list[str], points: list[tuple[float, list[float]]]) -> JointTrajectory:
    return JointTrajectory(
        joint_names=names,
        points=[TrajectoryPoint(time_from_start=t, positions=[float(v) for v in q]) for t, q in points],
    )


def accepted(result) -> bool:  # type: ignore[no-untyped-def]
    return result.status.name == "ACCEPTED"


def wait_near(cc, goal: list[float], timeout: float) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if np.max(np.abs(np.array(positions(cc, ARMS)) - goal)) < 0.02:
            return True
        time.sleep(0.1)
    return False


def summary(cc) -> None:  # type: ignore[no-untyped-def]
    status = cc.fence_status()
    hardware = status["hardware"]["openarm"]
    cancels = status["trajectory_cancels"]
    print(f"fence {hardware['mode']} · holds {hardware['holds']} · trajectory cancels {cancels}")
    for side, arm in cc.fence_check(cc.get_joint_positions())["arms"].items():
        clearance = arm.get("min_clearance_m")
        at = "" if clearance is None else f" {clearance * 100:+.1f} cm ({arm.get('sphere')})"
        print(f"  {side}: {'clear' if arm['ok'] else 'VIOLATION'}{at}")


def move(cc, name: str, goal: list[float]) -> None:  # type: ignore[no-untyped-def]
    result = cc.execute_trajectory(trajectory(ARMS, [(0.0, positions(cc, ARMS)), (3.0, goal)]))
    print(f"{name}: {result.status.name} {result.message}".rstrip())
    if accepted(result):
        print("  reached" if wait_near(cc, goal, 10.0) else "  did not reach the pose")
    summary(cc)


def stream(cc) -> None:  # type: ignore[no-untyped-def]
    """Joint commands like the Quest teleop sends (0.01 rad steps at 50 Hz), toward the glass."""
    commands = make_transport("/joint_command", JointState)
    q, goal = np.array(positions(cc, LEFT)), np.array(REACH_LEFT)
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        q = q + np.clip(goal - q, -0.01, 0.01)
        commands.broadcast(None, JointState(name=LEFT, position=[float(v) for v in q]))
        time.sleep(0.02)
    time.sleep(0.5)
    print(f"streamed toward {REACH_LEFT}; the left arm stopped at {np.round(positions(cc, LEFT), 2).tolist()}")
    summary(cc)


def collect(driver, cc, count: int, seed: int) -> None:  # type: ignore[no-untyped-def]
    """One episode per scripted pick; a pick the fence rejects is discarded, not saved."""
    episodes = driver.find_module_by_spec(EpisodeControlSpec)
    rng = np.random.default_rng(seed)
    for i in range(count):
        dx, dz = rng.uniform(-0.25, 0.25, size=2)
        reach = [-0.5 + dx, 0.2, 0, 1.2 + dz, 0, 0.3, 0] + [0.5 - dx, -0.2, 0, 1.2 + dz, 0, -0.3, 0]
        grip = 0.2 + 0.1 * float(rng.uniform())
        print(f"episode {i}: {episodes.command('start').state}")
        pick = trajectory(
            ALL,
            [
                (0.0, positions(cc, ALL)),
                (0.8, READY + [grip, grip]),  # open the fingers on the way
                (3.0, reach + [grip, grip]),  # approach
                (4.0, reach + [0.0, 0.0]),  # close
                (6.5, READY + [0.0, 0.0]),  # retreat
            ],
        )
        result = cc.execute_trajectory(pick)
        if not accepted(result):
            print(f"  pick {result.status.name}: {result.message}")
            print(f"  discarded ({episodes.command('discard').episodes_discarded} so far)")
            continue
        time.sleep(6.6)  # the whole pick; its end state must be in the episode
        wait_near(cc, READY, 5.0)
        time.sleep(0.5)
        print(f"  saved ({episodes.command('save').episodes_saved} so far)")
        time.sleep(1.0)


def rollout(driver, cc, seconds: float) -> None:  # type: ignore[no-untyped-def]
    """Run the policy for ``seconds`` from the ready pose the scripted demonstrations start at
    (the rollout guard refuses a first target far from the arms); the recorded setup records it
    as one episode, discarded when the policy sent nothing."""
    move(cc, "ready", READY)
    policy = driver.find_module_by_spec(RolloutControlSpec)
    ready = policy.preflight_rollout()
    if not (ready["policy_ready"] and ready["observations_ready"]):
        sys.exit(f"the policy is not ready: {ready['last_error']}")
    episodes = driver.find_module_by_spec(EpisodeControlSpec) if run.blueprint.endswith("-recorded") else None
    if episodes is not None:
        print(f"episode: {episodes.command('start').state}")
    status = policy.start_rollout()
    print(f"rollout {'started' if status['active'] else 'did not start'} on {status['device']}")
    start = np.array(positions(cc, ARMS))
    deadline = time.monotonic() + seconds
    while status["active"] and time.monotonic() < deadline:
        time.sleep(1.0)
        status = policy.rollout_status()
        moved = float(np.max(np.abs(np.array(positions(cc, ARMS)) - start)))
        print(f"  chunks {status['chunks_accepted']} · moved up to {moved:.3f} rad")
    # The stop waits 2 s for the policy thread; one still inside an inference reports active a
    # little longer (it sends no further chunk).
    status = policy.stop_rollout()
    deadline = time.monotonic() + 5.0
    while status["active"] and time.monotonic() < deadline:
        time.sleep(0.2)
        status = policy.rollout_status()
    state = "still active" if status["active"] else "stopped"
    print(f"{state} · chunks {status['chunks_accepted']} · error {status['last_error']}")
    if episodes is not None and status["chunks_accepted"]:
        print(f"episode saved ({episodes.command('save').episodes_saved} so far)")
    elif episodes is not None:
        discarded = episodes.command("discard").episodes_discarded
        print(f"episode discarded, the policy sent nothing ({discarded} so far)")
    summary(cc)


def main(action: str, args: list[str]) -> None:
    driver = Dimos.connect(timeout=10.0)
    try:
        cc = driver.get_module("ControlCoordinator")
        if action == "status":
            summary(cc)
        elif action == "ready":
            move(cc, "ready", READY)
        elif action == "home":
            move(cc, "home", ZERO)
        elif action == "through":
            now = positions(cc, ARMS)
            result = cc.execute_trajectory(trajectory(ARMS, [(0.0, now), (3.0, THROUGH_LEFT + now[7:])]))
            print(f"through the glass: {result.status.name} {result.message}".rstrip())
        elif action == "stream":
            stream(cc)
        elif action == "collect":
            collect(driver, cc, int(args[0]) if args else 6, int(args[1]) if len(args) > 1 else 3)
        elif action == "rollout":
            rollout(driver, cc, float(args[0]) if args else 12.0)
        else:
            sys.exit(f"unknown action {action!r}; see the header of {__file__}")
    finally:
        driver.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status", sys.argv[2:])
