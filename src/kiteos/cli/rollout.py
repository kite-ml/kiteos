# Copyright 2026 Kite ML.
"""``kiteos rollout start|controls``: run a trained policy on the arms.

``start`` runs the rollout blueprint (policy + coordinator + cameras + Quest takeover) in the
foreground; ``--record`` uses the recorded variant (collection recorder + episode monitor, Quest
B/Y) and records into ``<recordings>/rollout-<prefix>-NNN``. ``controls`` attaches the policy
start/stop controls from a second terminal. On the Quest, A toggles the policy and holding the
grips takes over (stops it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from kiteos import brand
from kiteos.cli._common import (
    StationOption,
    attach_to_run,
    console,
    die,
    runtime_argv,
    kiteos_env,
    load_station,
    note,
)
from kiteos.cli.collect import next_free_session
from kiteos.cli.launch import MAX_REAL_START_JUMP_RAD, collection_flags, launch, resolve_device, rollout_flags
from kiteos.cli.upload import MOCK_PREFIX
from kiteos.config import platform_key

ROLLOUT_PREFIX = "rollout-"

app = typer.Typer(help="Run a trained policy on the arms.", no_args_is_help=True)


@app.command("start")
def start(
    checkpoint: Annotated[
        Optional[Path], typer.Option("--checkpoint", help="Policy checkpoint dir (default: station policy).")
    ] = None,
    real: Annotated[bool, typer.Option("--real", help="Drive the real arms (default: simulated).")] = False,
    speed: Annotated[
        float, typer.Option("--speed", min=0.1, max=1.0, help="Playback speed, e.g. 0.5 = half speed.")
    ] = 1.0,
    device: Annotated[
        Optional[str], typer.Option("--device", help="auto | cuda | mps | cpu (default: station policy.device).")
    ] = None,
    max_start_jump: Annotated[
        Optional[float],
        typer.Option(
            "--max-start-jump",
            min=0.01,
            max=0.5,
            help=f"Refuse chunks starting further than this (rad) from the arms (default: station "
            f"policy.max_start_jump); <= {MAX_REAL_START_JUMP_RAD} with --real.",
        ),
    ] = None,
    max_joint_speed: Annotated[
        Optional[float],
        typer.Option(
            "--max-joint-speed",
            min=0.1,
            max=3.0,
            help="Refuse chunks moving an arm joint faster than this (rad/s; default: station policy.max_joint_speed).",
        ),
    ] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task instruction (default: station task).")] = None,
    record: Annotated[
        bool, typer.Option("--record", help="Record the rollout in the collection format (Quest B/Y episodes).")
    ] = False,
    session: Annotated[
        Optional[str], typer.Option("--session", help="With --record: session prefix (default: rollout-<station prefix>).")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the runtime command and exit.")] = False,
    allow_concurrent: Annotated[
        bool,
        typer.Option("--allow-concurrent", help="MOCK only: start even if another robot stack runs on this host."),
    ] = False,
    station_ports: Annotated[
        bool, typer.Option("--station-ports", help="MOCK only: use the station's WebXR/viser ports.")
    ] = False,
    station: StationOption = None,
) -> None:
    """Run the policy in the foreground (attach controls with `kiteos rollout controls`)."""
    config = load_station(station)
    if session is not None and not record:
        die("--session needs --record", code=2)
    if max_start_jump is None:
        max_start_jump = config.policy.max_start_jump
    if max_joint_speed is None:
        max_joint_speed = config.policy.max_joint_speed
    ckpt = checkpoint.expanduser().resolve() if checkpoint else config.checkpoint_path()
    if ckpt is None:
        die("no checkpoint: pass --checkpoint or set policy.checkpoint in the station file", code=2)
    if not ckpt.is_dir() and not dry_run:
        die(f"checkpoint not found: {ckpt}", code=2)
    if real and max_start_jump is not None and max_start_jump > MAX_REAL_START_JUMP_RAD:
        die(f"--max-start-jump above {MAX_REAL_START_JUMP_RAD} rad is only allowed with simulated arms", code=2)
    try:
        resolved = resolve_device(device or config.policy.device)
    except ValueError as exc:
        die(str(exc), code=2)
    if platform_key() == "darwin":
        note("macOS: run this from a terminal with camera permission; it stays in the foreground")
    console.print(f"Policy [kite]{ckpt.name}[/kite] on {resolved} at {speed:g}x speed")
    note("attach the policy controls from another terminal: kiteos rollout controls")
    task_text = task or config.task
    flags = rollout_flags(
        ckpt,
        task=task_text,
        device=resolved,
        speed=speed,
        max_start_jump_rad=max_start_jump,
        max_joint_speed_rad_s=max_joint_speed,
    )
    run_info = {}
    if record:
        prefix = session or f"{ROLLOUT_PREFIX}{config.prefix}"
        if not real and not prefix.startswith(MOCK_PREFIX):
            prefix = f"{MOCK_PREFIX}{prefix}"
        recording = next_free_session(config.recordings_path(), prefix)
        console.print(f"Recording the rollout to [kite]{recording}[/kite]")
        flags += collection_flags(recording, task_text)
        run_info = {"recording": recording, "session_prefix": prefix}
    launch(
        config,
        "rollout_recorded" if record else "rollout",
        real=real,
        extra_flags=flags,
        dry_run=dry_run,
        allow_concurrent=allow_concurrent,
        station_ports_for_mock=station_ports,
        run_info=run_info,
    )


@app.command("controls")
def controls(station: StationOption = None) -> None:
    """Attach policy start/stop controls to the running rollout (Space toggles, Q detaches)."""
    config = load_station(station)
    attach_to_run("rollout controls")
    try:
        from dimos.imitation.tui import RolloutApp, RolloutSession
        from dimos.porcelain.dimos import Dimos
    except ImportError:
        # Not in the runtime venv: fall back to the runtime's own controls (unbranded).
        import os

        argv = runtime_argv("imitation", "rollout")
        os.chdir(config.runtime_root_path())
        os.execvpe(argv[0], argv, kiteos_env())

    class KiteRolloutApp(RolloutApp):
        CSS = (
            RolloutApp.CSS
            + f"\n#dashboard {{ border: double {brand.PRIMARY}; }}\n#title {{ color: {brand.PRIMARY}; }}\n"
        )

    driver = None
    try:
        driver = Dimos.connect()
        KiteRolloutApp(RolloutSession(driver), title=brand.title("rollout", config.task)).run()
    except Exception as exc:
        die(f"rollout controls failed: {exc}\nIs `kiteos rollout start` running?", code=1)
    finally:
        if driver is not None:
            driver.stop()
