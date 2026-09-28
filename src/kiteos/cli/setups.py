# Copyright 2026 Kite ML.
"""``kiteos list``, ``kiteos run`` and ``kiteos status``: every embodiment the runtime supports.

KiteOS runs on the whole base runtime, so every setup it ships (arms, quadrupeds, humanoids,
drones, sensor rigs, simulators) is available here next to the fenced KiteOS setups:

* ``kiteos list`` shows them grouped by embodiment, with the ways KiteOS can run each one without
  robot hardware (``kiteos.embodiments.HARDWARE_FREE``);
* ``kiteos run <setup>`` starts one through ``kiteos-runtime`` with the KiteOS gates:
  - hardware-free runs (no flag, ``--sim`` or ``--replay``, only where the table says the setup
    moves no real robot in that mode) get a private bus, the runtime's user config file is
    ignored, robot addresses and ``<MODULE>__<FIELD>`` settings are removed from the
    environment, and a runtime ``.env`` or an argument that sets them is refused;
  - ``--real`` shows that the KiteOS fence does not cover the robot and needs the station name
    typed; the fenced dual OpenYAM only runs for real through ``kiteos teleop|collect|rollout``;
  - either way: one setup per run, no other stack on this host, a run-state file for
    ``kiteos stop`` and ``kiteos status``;
* ``kiteos status`` shows the robot stacks running on this host.

``kiteos-runtime`` is the runtime's own command line without these gates, for advanced use.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Annotated, Literal, Optional

import typer
from rich.markup import escape
from rich.panel import Panel

from kiteos import embodiments
from kiteos.cli import launch, runs, stop
from kiteos.cli._common import StationOption, confirm_typed, console, die, kiteos_env, load_station, note, runtime_argv
from kiteos.config import StationConfig
from kiteos.embodiments import Setup

Mode = Literal["default", "sim", "replay", "real"]

ENTRY_POINT_GROUP = "dimos.blueprints"
# GlobalConfig fields (the runtime reads them from plain, case-insensitive environment names) and
# environment variables that point a stack at a robot, plus the robot network interface of the G1.
HARDWARE_ENV_KEYS = frozenset(
    {
        "robot_ip",
        "robot_ips",
        "xarm7_ip",
        "xarm6_ip",
        "lite6_ip",
        "can_port",
        "device_path",
        "robot_interface",
        "zenoh_connect",
        "relay_url",
        "local_relay",
    }
)
# `<MODULE>__<FIELD>` (and `G__`, `TRANSPORTS__`): the runtime turns these into setup configuration.
MODULE_ENV_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*?__[A-Za-z0-9]")
# Arguments a hardware-free run must not take: a robot address, a bus, or the runtime's mode.
HARDWARE_FLAG = re.compile(
    r"(?i)^--(?:[a-z0-9_]+\.)*"
    r"(robot[-_]ips?|xarm[67][-_]ip|lite6[-_]ip|(?:left[-_]|right[-_])?can[-_]ports?|device[-_]path|"
    r"robot[-_]interface|zenoh[-_](?:connect|scout[-_]addr|interface|mode)|relay[-_]url|local[-_]relay|"
    r"address|ip|connection[-_]string|bus[-_]devices?|simulation|replay|no[-_]replay)(?:=|$)"
)
# The bus and per-stream transports (a ROS transport could reach a robot's topics) stay KiteOS's choice.
TRANSPORT_FLAG = re.compile(r"(?i)^--transports?(\.|=|$)")
CONFIG_FLAG = re.compile(r"^(-c|--config)(=|$)")
KITEOS_COMMANDS = {
    "teleop": "kiteos teleop",
    "collection": "kiteos collect start",
    "rollout": "kiteos rollout start",
    "rollout_recorded": "kiteos rollout start --record",
}

# Replaced in tests: nothing in the unit tests may look at this host's processes.
find_stacks: Callable[[], list[stop.RunningStack]] = stop.find_stacks
stack_processes: Callable[[], list[tuple[int, str]]] = runs.stack_processes
process_table: Callable[[], dict[int, tuple[int, str]]] = stop._process_table


class SetupRunError(ValueError):
    """The setup cannot be started the way it was asked for."""


# --------------------------------------------------------------------------- the catalog


def plugin_setups() -> dict[str, str]:
    """Setups other installed packages register with the runtime (``<package>.<name>``), KiteOS's excluded."""
    found: dict[str, str] = {}
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        package = entry.dist.name if entry.dist is not None else None
        if package and package != launch.DIST_NAME:
            found[f"{package}.{entry.name}"] = entry.value
    return found


def load_registry(station: StationConfig) -> dict[str, str]:
    try:
        return embodiments.read_registry(station.runtime_root_path())
    except embodiments.RegistryError as exc:
        die(f"{exc}\nCheck runtime_root in the station file; `kiteos doctor` checks the runtime checkout.", code=2)


def resolve_setup(name: str, registry: Mapping[str, str], plugins: Mapping[str, str] | None = None) -> Setup:
    """The setup to run, or ``SetupRunError`` (KiteOS setups have their own commands)."""
    kiteos_names = {spec.name for spec in launch.BLUEPRINTS.values()}
    if name.startswith(f"{launch.DIST_NAME}.") or name in kiteos_names:
        raise SetupRunError(
            f"{name} is a KiteOS setup: start it with `kiteos teleop`, `kiteos collect start` or "
            "`kiteos rollout start` (they add the station file and the fence)"
        )
    if name in registry:
        return Setup(name, registry[name], embodiments.classify(name, registry[name]))
    if plugins and name in plugins:
        return Setup(name, plugins[name], embodiments.PLUGINS)
    close = difflib.get_close_matches(name, [*registry, *(plugins or {})], n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise SetupRunError(f"unknown setup {name!r}.{hint} `kiteos list` shows them all.")


# --------------------------------------------------------------------------- planning a run


def _not_checked(setup: Setup, how: str) -> str:
    checked = setup.hardware_free
    options = checked.summary() if checked is not None else ""
    why = embodiments.ALWAYS_REAL_NOTES.get(setup.name)
    return (
        f"KiteOS has not checked that {setup.name} moves no real robot {how}"
        + (f" (it {why} even with --simulation)" if why else "")
        + f".\nWithout hardware it runs: {options or 'not at all (--real only)'}. "
        "Use --real to drive the robot (typed confirmation)."
    )


def mode_runtime_options(setup: Setup, mode: Mode, engine: str | None) -> tuple[list[str], str]:
    """(runtime global options, what the run drives) for ``mode``; ``SetupRunError`` if not allowed."""
    checked = setup.hardware_free
    if engine is not None and mode != "sim":
        raise SetupRunError("--engine chooses the simulator for --sim")
    if mode == "real":
        if setup.embodiment.fenced:
            raise SetupRunError(
                f"{setup.name} drives the dual OpenYAM without the KiteOS fence. Drive the real arms with "
                "`kiteos teleop`, `kiteos collect start` or `kiteos rollout start`: they enforce the fence "
                "(the keep-out zones such as the glass)."
            )
        return [], "real"
    if mode == "sim":
        if checked is None or not checked.sim:
            raise SetupRunError(_not_checked(setup, "in a simulator"))
        chosen = engine or checked.sim[0]
        if chosen not in checked.sim:
            raise SetupRunError(f"{setup.name} runs in {', '.join(checked.sim)}, not {chosen}")
        return ["--simulation", chosen], f"{chosen} simulator"
    if mode == "replay":
        if checked is None or not checked.replay:
            raise SetupRunError(_not_checked(setup, "with --replay"))
        return ["--replay"], "recorded data"
    if checked is None or checked.default is None:
        raise SetupRunError(_not_checked(setup, "without --sim or --replay"))
    return list(checked.default), checked.default_label


def refused_arguments(extra: Sequence[str], *, real: bool) -> list[str]:
    """Extra arguments the run must not take (see the module doc)."""
    bad = []
    if extra and not extra[0].startswith("-"):
        bad.append(f"{extra[0]} (a second setup: one setup per run)")
    if not real:
        bad += [arg for arg in extra if HARDWARE_FLAG.match(arg) or TRANSPORT_FLAG.match(arg) or CONFIG_FLAG.match(arg)]
    return bad


def hardware_env_keys(environ: Iterable[str]) -> list[str]:
    """Environment names that could point a hardware-free run at a robot (see ``HARDWARE_ENV_KEYS``)."""
    return sorted(key for key in environ if key.lower() in HARDWARE_ENV_KEYS or MODULE_ENV_KEY.match(key))


def hardware_dotenv_settings(station: StationConfig) -> list[str]:
    """Robot addresses and setup settings in the runtime's ``.env`` files (it reads them itself)."""
    found = []
    for path in launch.runtime_dotenv_files(station.runtime_root_path()):
        found += [f"{key} in {path}" for key in hardware_env_keys(launch.dotenv_keys(path))]
    return found


def plan_setup_run(
    station: StationConfig,
    setup: Setup,
    *,
    mode: Mode,
    engine: str | None = None,
    extra: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
    private_bus: Mapping[str, str] | None = None,
    runtime_config: Path | None = None,
    repo: Path | None = None,
) -> launch.LaunchPlan:
    """The ``kiteos-runtime [options] run <setup> ...`` command for this run (no filesystem, no host)."""
    real = mode == "real"
    options, drives = mode_runtime_options(setup, mode, engine)
    bad = refused_arguments(extra, real=real)
    if bad:
        raise SetupRunError(
            "refused arguments: "
            + ", ".join(bad)
            + ("" if real else f".\nA run on {drives} takes no robot address, bus or runtime mode from its "
               "arguments (KiteOS sets them); use --real to drive the robot.")
        )
    if not real and not private_bus:
        raise SetupRunError("a hardware-free run needs a private bus")
    env = kiteos_env(environ, repo)
    notes: list[str] = []
    if real:
        notes.append(f"the KiteOS fence does not cover the {setup.embodiment.title}")
    else:
        removed = hardware_env_keys(env)
        for key in removed:
            env.pop(key)
        env.update(private_bus or {})
        notes.append(f"{drives}: KiteOS checked that {setup.name} moves no real robot this way")
        notes.append("private bus: this run cannot reach (or be reached by) another stack")
        if removed:
            notes.append(f"not passed to the run: {', '.join(removed)}")
    argv = runtime_argv(*options, "run", setup.name)
    if not real and runtime_config is not None:
        argv += [launch.RUNTIME_CONFIG_FLAG, str(runtime_config)]
    argv += list(extra)
    return launch.LaunchPlan(
        argv=argv,
        cwd=station.runtime_root_path(),
        env=env,
        real=real,
        blueprint=launch.BlueprintSpec(setup.name, False, drives, package=None),
        fence_path=None,
        notes=notes,
        private_bus=not real,
    )


# --------------------------------------------------------------------------- commands


def list_setups(
    embodiment: Annotated[
        Optional[str],
        typer.Option("--embodiment", "-e", help="Only one embodiment (key or name, e.g. xarm, go2, g1)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    station: StationOption = None,
) -> None:
    """List every setup KiteOS can run, by embodiment, and how each runs without robot hardware."""
    config = load_station(station)
    wanted = None
    if embodiment is not None:
        wanted = embodiments.find_embodiment(embodiment)
        if wanted is None:
            keys = ", ".join(embodiments.BY_KEY)
            die(f"unknown embodiment {embodiment!r}; one of: {keys}", code=2)
    setups = [
        s for s in embodiments.catalog(load_registry(config), plugin_setups()) if wanted is None or s.embodiment == wanted
    ]
    kiteos_setups = [] if wanted not in (None, embodiments.BY_KEY["dual-openyam"]) else list(launch.BLUEPRINTS.items())
    if as_json:
        data = {
            "runtime_root": str(config.runtime_root_path()),
            "kiteos": [
                {
                    "setup": spec.run_name,
                    "command": KITEOS_COMMANDS[key],
                    "description": spec.description,
                    "embodiment": "dual-openyam",
                    "fenced": True,
                }
                for key, spec in kiteos_setups
            ],
            "setups": [
                {
                    "setup": s.name,
                    "embodiment": s.embodiment.key,
                    "embodiment_title": s.embodiment.title,
                    "module": s.module,
                    "without_hardware": None
                    if s.hardware_free is None
                    else {
                        "default": s.hardware_free.default_label if s.hardware_free.default is not None else None,
                        "sim": list(s.hardware_free.sim),
                        "replay": s.hardware_free.replay,
                    },
                    "note": embodiments.ALWAYS_REAL_NOTES.get(s.name),
                }
                for s in setups
            ],
        }
        typer.echo(json.dumps(data, indent=2))
        return
    if kiteos_setups:
        console.print("[kite]KiteOS setups[/kite]: dual OpenYAM with the KiteOS fence (keep-out zones enforced)")
        width = max(len(KITEOS_COMMANDS[key]) for key, _ in kiteos_setups)
        for key, spec in kiteos_setups:
            console.print(f"  {KITEOS_COMMANDS[key]:<{width}}  {spec.run_name} · {spec.description}", soft_wrap=True)
        console.print()
    groups: dict[str, list[Setup]] = {}
    for setup in setups:
        groups.setdefault(setup.embodiment.key, []).append(setup)
    console.print(
        f"[kite]Runtime setups[/kite]: {len(setups)} in {len(groups)} groups. Start one with `kiteos run <setup>` "
        "(add --sim, --replay or --real); each line says how it runs without robot hardware.",
        soft_wrap=True,
    )
    for key, members in groups.items():
        group = embodiments.BY_KEY[key]
        extra = " · real arms only through the KiteOS setups" if group.fenced else ""
        console.print(f"\n[accent]{escape(group.title)}[/accent] [muted]({len(members)}){extra}[/muted]")
        width = max(len(s.name) for s in members)
        for setup in members:
            why = embodiments.ALWAYS_REAL_NOTES.get(setup.name)
            detail = setup.summary() + (f" ({why})" if why else "")
            style = "muted" if setup.hardware_free is None else "ok"
            console.print(f"  {setup.name:<{width}}  [{style}]{escape(detail)}[/{style}]", soft_wrap=True)


def show_real_warning(setup: Setup, station: StationConfig) -> None:
    console.print(
        Panel.fit(
            f"[fail]REAL HARDWARE[/fail]: {escape(setup.name)} ({escape(setup.embodiment.title)}).\n"
            "The KiteOS fence does not cover this robot (it protects the dual OpenYAM): only the\n"
            "robot's own limits keep it off the glass, people and fixtures.\n"
            "Stopping the stack disables the motors: arms can drop and legged robots can fall.\n"
            "Robot settings come from your arguments, the environment and the runtime's config.\n"
            f"Station: {escape(station.describe_sources())}",
            border_style="fail",
        )
    )


def confirm_real_setup(station: StationConfig) -> bool:
    console.print(
        "Someone must be at the robot's e-stop and the workspace must be clear.\n"
        f"Type the station name [kite]{station.station}[/kite] to confirm."
    )
    return confirm_typed(station.station, "Station name")


def run_setup(
    ctx: typer.Context,
    setup: Annotated[str, typer.Argument(help="Setup name (see `kiteos list`).", show_default=False)],
    sim: Annotated[bool, typer.Option("--sim", help="Run in a simulator (see --engine).")] = False,
    engine: Annotated[
        Optional[str], typer.Option("--engine", help="Simulator for --sim: mujoco or dimsim (default: the first listed).")
    ] = None,
    replay: Annotated[bool, typer.Option("--replay", help="Run on recorded data instead of the robot.")] = False,
    real: Annotated[
        bool, typer.Option("--real", help="Drive the real robot (not fenced by KiteOS; typed confirmation).")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the command, start nothing.")] = False,
    allow_concurrent: Annotated[
        bool,
        typer.Option("--allow-concurrent", help="Start a hardware-free run while another stack runs (private bus)."),
    ] = False,
    station: StationOption = None,
) -> None:
    """Run any setup of the runtime: every embodiment (see `kiteos list`).

    Without a flag the setup runs on mock hardware where KiteOS has checked it can; --sim and
    --replay run it in a simulator or on recorded data; --real drives the robot. Arguments after
    the setup (or after --) go to the runtime.
    """
    config = load_station(station)
    chosen = [flag for flag, on in (("sim", sim), ("replay", replay), ("real", real)) if on]
    if len(chosen) > 1:
        die("choose one of --sim, --replay and --real", code=2)
    mode: Mode = chosen[0] if chosen else "default"  # type: ignore[assignment]
    if real and allow_concurrent:
        die("--allow-concurrent is only for hardware-free runs; a real run never shares the host", code=2)
    try:
        target = resolve_setup(setup, load_registry(config), plugin_setups())
    except SetupRunError as exc:
        die(escape(str(exc)), code=2)
    injected = launch.config_injection(config, os.environ)
    if injected:
        die(
            "the runtime would pick up settings from outside this command:\n  - "
            + "\n  - ".join(escape(i) for i in injected)
            + "\nRemove them (CONTROLCOORDINATOR__* / G__* / TRANSPORTS__*) and pass settings as arguments.",
            code=2,
        )
    if not real:
        in_dotenv = hardware_dotenv_settings(config)
        if in_dotenv:
            die(
                "the runtime reads robot addresses or setup settings from a .env file, and a hardware-free run "
                "must not get them:\n  - " + "\n  - ".join(escape(i) for i in in_dotenv) + "\nRemove them from the "
                "file (pass them as arguments to --real runs instead).",
                code=2,
            )
    try:
        plan = plan_setup_run(
            config,
            target,
            mode=mode,
            engine=engine,
            extra=list(ctx.args),
            environ=os.environ,
            private_bus=None if real else runs.random_private_bus(r.bus for r in runs.alive_run_states()),
            runtime_config=None if real else launch.runtime_empty_config(),
        )
    except SetupRunError as exc:
        die(escape(str(exc)), code=2)
    if not plan.cwd.is_dir() and not dry_run:
        die(f"runtime checkout not found at {plan.cwd} (station: runtime_root)", code=2)
    launch.refuse_other_stacks(config, plan, allow_concurrent=allow_concurrent, dry_run=dry_run)
    if real:
        show_real_warning(target, config)
        if not dry_run and not confirm_real_setup(config):
            die("not confirmed; not starting", code=1)
    for message in plan.notes:
        console.print(f"[warn]note:[/warn] {escape(message)}")
    drives = "[fail]REAL HARDWARE[/fail]" if real else f"[ok]{escape(plan.blueprint.description)}[/ok]"
    console.print(f"[kite]{escape(target.name)}[/kite] ({escape(target.embodiment.title)}) on {config.station} · {drives}")
    console.print(f"[muted]{escape(plan.shell())}[/muted]", soft_wrap=True)
    if dry_run:
        note("dry run: not starting the runtime")
        return
    launch.record_run(
        runs.RunState(
            pid=os.getpid(),  # execvpe keeps the pid: this is the runtime process
            real=real,
            blueprint=target.name,
            station=config.station,
            bus=runs.bus_of(plan.env),
            private_bus=plan.private_bus,
            started_at=runs.now_iso(),
            argv=list(plan.argv),
            hardware=None if real else plan.blueprint.description,
        )
    )
    launch.execute(plan)


def _descends_from(pid: int, ancestors: set[int], table: Mapping[int, tuple[int, str]]) -> bool:
    seen = 0
    while pid in table and seen < 64:
        if pid in ancestors:
            return True
        pid = table[pid][0]
        seen += 1
    return pid in ancestors


def status(station: StationOption = None) -> None:
    """Show the robot stacks running on this host (KiteOS runs, the runtime's registry, others)."""
    load_station(station)
    stacks = find_stacks()
    table = process_table()
    claimed = {s.pid for s in stacks} | {pid for s in stacks for pid in s.related}
    others = [(pid, cmd) for pid, cmd in stack_processes() if not _descends_from(pid, claimed, table)]
    if not stacks and not others:
        console.print("[ok]No robot stack is running on this host.[/ok]")
        return
    for stack in stacks:
        console.print(f"  {escape(stack.label())}")
    if others:
        console.print("Robot stack processes that KiteOS did not start and the runtime has not registered:")
        for pid, cmd in others[:12]:
            console.print(f"  PID {pid}: {escape(cmd[:160])}")
    note("Stop one with `kiteos stop` (`--pid` when several run).")
