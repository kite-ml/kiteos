# Copyright 2026 Kite ML.
"""Run helpers: turn the station config into a ``kiteos-runtime run`` command and start it safely.

``build_runtime_command`` is pure (no filesystem, no hardware) and unit-tested. ``launch`` adds
the operator-facing safety gates:

* nothing inherited can change what the stack drives: CONTROLCOORDINATOR__/G__/TRANSPORTS__
  settings in the environment or in the runtime checkout's ``.env`` are refused, the runtime's user config
  file is replaced by an empty one (``--config``), and the fence requirement and the expected arm
  type (``--controlcoordinator.expect-mock``) are always passed as flags, which beat every other
  source;
* another robot stack on this host (registry, KiteOS run, process, WebXR/viser listener, or for
  ``--real`` traffic on the target bus) is refused (see ``kiteos.cli.runs``);
* without ``--real``: CAN ports are never passed and the coordinator refuses any real adapter, the
  run gets a private bus (own LCM port + zenoh scouting group), free WebXR/viser ports and no
  rerun viewer, so it cannot talk to a real stack;
* with ``--real``: the fence file must exist and pass the real-hardware checks (enforce mode, a
  zone for each arm unless ``allow_no_zones``, plausible speed), the enabled zones are shown, and
  the operator must type the station name to confirm someone is at the e-stop.

A run-state file (pid, REAL/MOCK, bus, ports, recording) is written right before ``exec``, so the
client commands can join the right bus and target the right run.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kiteos.cli import runs
from kiteos.cli._common import confirm_typed, console, die, kiteos_env, note, runtime_argv, warn
from kiteos.config import CAMERA_ROLES, SIDES, StationConfig, is_mock_camera, is_placeholder, platform_key
from kiteos.robots import DUAL_OPENYAM, ROBOTS, RobotProfile, get_robot

if TYPE_CHECKING:
    from kiteos.safety.fence import FenceConfig

DIST_NAME = "kiteos"
COORDINATOR_FLAG = "controlcoordinator"
POLICY_INSTANCE = "policy"
RECORDER_INSTANCE = "recorder"
EPISODES_INSTANCE = "episodes"
VISER_FLAG = "--manipulationmodule.visualization.host"
VIEWER_FENCE_FLAG = "--manipulationmodule.fence-path"  # the fence zones drawn in the viewer
VISER_PORT_FLAG = "--manipulationmodule.visualization.port"
WEBXR_PORT_FLAG = "--armteleopmodule.server-port"
VIEWER_FLAG = "--viewer"
RUNTIME_CONFIG_FLAG = "--config"
DEFAULT_VISER_HOST = "127.0.0.1"
POLICY_FPS = 30.0
MAX_REAL_START_JUMP_RAD = 0.3
# NativeModule sends SIGKILL to the recorder this long after SIGTERM (the runtime default 2 s). Kept
# below the runtime's 120 s module-stop RPC timeout so the stop RPC does not give up first.
RECORDER_SHUTDOWN_TIMEOUT_S = 100.0
# Environment keys the runtime turns into module / global / transport config: they could add CAN ports
# to a mock run, switch the fence requirement off, or move the run onto another bus.
INJECTION_KEY = re.compile(r"(?i)^(controlcoordinator|g|transports)__")

Device = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class BlueprintSpec:
    """A setup to run: a KiteOS blueprint (registered under the ``dimos.blueprints`` entry-point
    group, run as ``kiteos.<name>``) or, with ``package=None``, one of the runtime's own setups."""

    name: str
    has_cameras: bool
    description: str
    package: str | None = DIST_NAME

    @property
    def run_name(self) -> str:
        return f"{self.package}.{self.name}" if self.package else self.name


# Setup kind -> (has cameras, description); the setup name comes from the station's robot.
SETUP_KINDS: dict[str, tuple[bool, str]] = {
    "teleop": (False, "Quest teleop of both arms"),
    "collection": (True, "teleop + 3 cameras + episode recorder"),
    "rollout": (True, "policy rollout with Quest takeover"),
    "rollout_recorded": (True, "policy rollout with Quest takeover, recorded"),
}


def setup_spec(robot: str | RobotProfile, kind: str) -> BlueprintSpec:
    """The KiteOS setup of ``kind`` for ``robot`` (``kiteos.<robot>-<kind>``)."""
    profile = get_robot(robot)
    try:
        has_cameras, description = SETUP_KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown setup kind {kind!r}; one of {', '.join(SETUP_KINDS)}") from None
    return BlueprintSpec(profile.setup_name(kind), has_cameras, description)


def all_setup_specs() -> list[tuple[RobotProfile, str, BlueprintSpec]]:
    """(robot, kind, setup) for every KiteOS setup of every robot."""
    return [(robot, kind, setup_spec(robot, kind)) for robot in ROBOTS.values() for kind in SETUP_KINDS]


# The dual OpenYAM setups by kind (the default robot).
BLUEPRINTS: dict[str, BlueprintSpec] = {kind: setup_spec(DUAL_OPENYAM, kind) for kind in SETUP_KINDS}


class LaunchError(ValueError):
    """The station cannot produce a safe, complete launch command."""


@dataclass(frozen=True)
class LaunchPlan:
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    real: bool
    blueprint: BlueprintSpec
    fence_path: Path | None
    notes: list[str] = field(default_factory=list)
    private_bus: bool = False
    webxr_port: int | None = None
    viser_port: int | None = None

    def shell(self) -> str:
        """Copy-pasteable command line (PYTHONPATH and a private bus shown, since the runtime needs them)."""
        exports = [f"PYTHONPATH={shlex.quote(self.env.get('PYTHONPATH', ''))}"]
        if self.private_bus:
            exports += [f"{k}={shlex.quote(self.env[k])}" for k in ("LCM_DEFAULT_URL", "ZENOH_SCOUT_ADDR")]
        return f"cd {shlex.quote(str(self.cwd))} && {' '.join(exports)} " + shlex.join(self.argv)


def camera_flag(role: str) -> str:
    return f"--{role.replace('_', '-')}.hardware.camera-index"


def build_runtime_command(
    station: StationConfig,
    blueprint: str | BlueprintSpec,
    *,
    real: bool,
    fence_path: Path | None,
    cameras: bool = True,
    extra_flags: Sequence[str] = (),
    repo: Path | None = None,
    environ: Mapping[str, str] | None = None,
    private_bus: Mapping[str, str] | None = None,
    ports: tuple[int, int] | None = None,
    runtime_config: Path | None = None,
) -> LaunchPlan:
    """Build the ``uv run --no-sync kiteos-runtime run kiteos.<blueprint> ...`` command for this host.

    ``private_bus`` (mock runs) puts the child on its own LCM URL / zenoh scouting group;
    ``ports`` = (WebXR, viser) moves the servers off the station's fixed ports and disables the
    rerun viewer. Raises ``LaunchError`` when a real run lacks a fence file or a CAN port, when a
    camera selector for this host is still a template placeholder, or when a private bus is
    requested for a real run.
    """
    spec = setup_spec(station.profile, blueprint) if isinstance(blueprint, str) else blueprint
    notes: list[str] = []
    argv = runtime_argv("run", spec.run_name)
    if runtime_config is not None:
        argv += [RUNTIME_CONFIG_FLAG, str(runtime_config)]

    if real and fence_path is None:
        raise LaunchError("a fence file is required to drive the real arms")
    if real and private_bus:
        raise LaunchError("a real run uses the station's bus, never a private one")
    if fence_path is not None:
        resolved_fence = str(Path(fence_path).expanduser().resolve())
        argv += [f"--{COORDINATOR_FLAG}.fence-path", resolved_fence, VIEWER_FENCE_FLAG, resolved_fence]
    # CLI flags beat the environment and config files: pin the safety-relevant coordinator config.
    argv += [
        f"--{COORDINATOR_FLAG}.require-fence-on-hardware",
        "true",
        f"--{COORDINATOR_FLAG}.expect-mock",
        "false" if real else "true",
    ]

    if real:
        for side in SIDES:
            port = station.can_port(side)
            if not port or is_placeholder(port):
                raise LaunchError(
                    f"no CAN port configured for the {side} arm on {platform_key()} "
                    f"(station file: can.{side}.{platform_key()})"
                )
            argv += [f"--{COORDINATOR_FLAG}.{side}-can-port", port]
    else:
        notes.append("arms are SIMULATED (mock hardware); pass --real to drive the real arms")

    if spec.has_cameras and cameras:
        for role in CAMERA_ROLES:
            selector = station.camera(role)
            if selector is None or is_placeholder(selector):
                raise LaunchError(
                    f"camera {role} is not configured for {platform_key()} "
                    f"(station file: cameras.{role}.{platform_key()}); find it with `kiteos cameras list`"
                )
            if real and is_mock_camera(selector):
                raise LaunchError(
                    f"camera {role} is the synthetic 'mock' camera: a real run needs the station's "
                    f"real cameras (station file: cameras.{role}.{platform_key()})"
                )
            argv += [camera_flag(role), str(selector)]
        mocked = [role for role in CAMERA_ROLES if is_mock_camera(station.camera(role))]
        if mocked:
            notes.append(f"cameras {', '.join(mocked)} are synthetic test patterns (mock)")

    if station.viser_host != DEFAULT_VISER_HOST:
        argv += [VISER_FLAG, station.viser_host]
        notes.append(f"viser viewer bound to {station.viser_host}: anyone who can reach it can see the robot")

    webxr_port = viser_port = None
    if ports is not None:
        webxr_port, viser_port = ports
        argv += [WEBXR_PORT_FLAG, str(webxr_port), VISER_PORT_FLAG, str(viser_port), VIEWER_FLAG, "none"]

    argv += list(extra_flags)
    env = kiteos_env(environ, repo)
    if private_bus:
        env.update(private_bus)
        notes.append("private bus: this run cannot reach (or be reached by) another stack")
    return LaunchPlan(
        argv=argv,
        cwd=station.runtime_root_path(),
        env=env,
        real=real,
        blueprint=spec,
        fence_path=Path(fence_path).expanduser().resolve() if fence_path is not None else None,
        notes=notes,
        private_bus=bool(private_bus),
        webxr_port=webxr_port,
        viser_port=viser_port,
    )


def collection_flags(recording: Path, task: str, fmt: str = "mcap") -> list[str]:
    return [
        f"--{RECORDER_INSTANCE}.recording",
        str(recording),
        f"--{RECORDER_INSTANCE}.format",
        fmt,
        f"--{RECORDER_INSTANCE}.shutdown-timeout",
        f"{RECORDER_SHUTDOWN_TIMEOUT_S:g}",
        f"--{EPISODES_INSTANCE}.task",
        task,
    ]


def rollout_flags(
    checkpoint: Path,
    *,
    task: str,
    device: str,
    speed: float = 1.0,
    max_start_jump_rad: float | None = None,
    max_joint_speed_rad_s: float | None = None,
) -> list[str]:
    flags = [
        f"--{POLICY_INSTANCE}.policy-path",
        str(checkpoint),
        f"--{POLICY_INSTANCE}.task",
        task,
        f"--{POLICY_INSTANCE}.device",
        device,
        f"--{POLICY_INSTANCE}.fps",
        f"{POLICY_FPS * speed:g}",
    ]
    if max_start_jump_rad is not None:
        flags += [f"--{POLICY_INSTANCE}.max-start-jump-rad", f"{max_start_jump_rad:g}"]
    if max_joint_speed_rad_s is not None:
        flags += [f"--{POLICY_INSTANCE}.max-joint-speed-rad-s", f"{max_joint_speed_rad_s:g}"]
    return flags


def resolve_device(
    requested: str,
    *,
    host_platform: str | None = None,
    machine: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[str], bool] = os.path.exists,
) -> Device:
    """``auto`` -> cuda when an NVIDIA driver is present, mps on Apple silicon, else cpu."""
    if requested in ("cuda", "mps", "cpu"):
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError(f"device must be auto, cuda, mps or cpu, got {requested!r}")
    host_platform = host_platform or sys.platform
    machine = machine or platform.machine()
    if host_platform == "darwin":
        return "mps" if machine == "arm64" else "cpu"
    if which("nvidia-smi") or exists("/proc/driver/nvidia/version"):
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- safety gates


def load_fence_for_launch(station: StationConfig, *, real: bool) -> tuple[Path | None, FenceConfig | None]:
    """Resolve and validate the fence file. Exits on anything unsafe for a real run."""
    from kiteos.safety.fence import FenceConfig, FenceConfigError, real_hardware_problems

    path = station.fence_path()
    if not path.is_file():
        if real:
            die(
                f"fence file not found: {path}\n"
                "The real arms never run without a fence. Create it from config/fence.example.yaml "
                "and check it with `kiteos fence validate`.",
                code=2,
            )
        warn(f"fence file not found ({path}); the mock run gets joint and speed limits only")
        return None, None
    try:
        fence = FenceConfig.load(path)
    except (FenceConfigError, OSError) as exc:
        die(f"fence file is invalid, not starting:\n{exc}", code=2)
    if fence.robot != station.robot:
        die(
            f"fence file {path} is for the {fence.profile.title} (robot: {fence.robot}), but this station "
            f"runs the {station.profile.title} (robot: {station.robot}). A fence protects only the robot it "
            "was written for; point `fence:` in the station file at this robot's fence.",
            code=2,
        )
    if real:
        problems = real_hardware_problems(fence)
        if problems:
            die(f"{path} cannot drive the real arms:\n  - " + "\n  - ".join(problems), code=2)
    if fence.station and fence.station != station.station:
        message = f"fence file {path} is for station {fence.station!r}, this station is {station.station!r}"
        if real:
            die(
                f"{message}.\nA fence measured elsewhere does not protect this site; check its zones, "
                "then set `station:` in the fence file.",
                code=2,
            )
        warn(message)
    return path, fence


def dotenv_keys(path: Path) -> list[str]:
    """Keys assigned in a dotenv file (``KEY=value`` / ``export KEY=value`` lines)."""
    if not path.is_file():
        return []
    keys = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=", line)
        if match:
            keys.append(match.group(1))
    return keys


def runtime_dotenv_files(root: Path) -> list[Path]:
    """The dotenv files the runtime reads: the first ``.env`` above its command-line package (loaded
    into the environment at start) and ``<root>/.env`` (read by its settings)."""
    found: list[Path] = []
    start = root / "dimos" / "cli"
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            found.append(candidate)
            break
    own = root / ".env"
    if own.is_file() and own not in found:
        found.append(own)
    return found


def config_injection(station: StationConfig, environ: Mapping[str, str]) -> list[str]:
    """Settings outside the station file that would change what the runtime drives (see module doc)."""
    found = [f"environment variable {key}" for key in environ if INJECTION_KEY.match(key)]
    for dotenv in runtime_dotenv_files(station.runtime_root_path()):
        found += [f"{key} in {dotenv}" for key in dotenv_keys(dotenv) if INJECTION_KEY.match(key)]
    return found


def runtime_empty_config() -> Path:
    """An empty runtime config file, passed with ``--config`` so the user's runtime config file is ignored."""
    path = runs.kiteos_runs_dir().parent / "runtime-config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_text().strip() != "{}":
        path.write_text("{}\n")
    return path


def station_ports(station: StationConfig) -> tuple[int, int]:
    """(WebXR, viser) ports the station's stacks use."""
    match = re.search(r":(\d+)(/|$)", station.quest.upstream)
    return (int(match.group(1)) if match else runs.DEFAULT_WEBXR_PORT, runs.DEFAULT_VISER_PORT)


# Replaced in tests: nothing in the unit tests may look at (or refuse because of) this host.
host_stacks: Callable[..., list[runs.Evidence]] = runs.other_stacks
bus_probe: Callable[..., runs.BusProbe] = runs.probe_bus
record_run: Callable[[runs.RunState], Any] = runs.write_run_state


def refuse_other_stacks(station: StationConfig, plan: LaunchPlan, *, allow_concurrent: bool, dry_run: bool) -> None:
    """Refuse to start while another robot stack runs on this host (see ``kiteos.cli.runs``)."""
    evidence = list(host_stacks(ports=station_ports(station)))
    if plan.real and not evidence:
        probe = bus_probe(plan.env, cwd=plan.cwd if plan.cwd.is_dir() else None)
        if probe.busy:
            evidence.append(
                runs.Evidence("bus", f"{probe.samples} runtime messages on the target bus: {', '.join(probe.topics[:4])}")
            )
        elif probe.error:
            warn(f"could not listen on the bus before starting ({escape(probe.error.splitlines()[-1][:160])})")
    if not evidence:
        return
    listing = "\n".join(f"  - {e.source}: {escape(e.detail)}" for e in evidence)
    if dry_run:
        warn(f"another robot stack appears to be running on this host:\n{listing}")
        return
    kind = "MOCK" if plan.blueprint.package else "hardware-free"
    if not plan.real and allow_concurrent:
        ports = "its own ports" if plan.webxr_port is not None else "the setup's usual ports (they may clash)"
        warn(
            f"another robot stack appears to be running on this host:\n{listing}\n"
            f"Continuing (--allow-concurrent): this {kind} run uses a private bus and {ports}."
            + (" It opens the station cameras too." if plan.blueprint.has_cameras else "")
        )
        return
    hint = (
        f"Stop it first (`kiteos stop`), or pass --allow-concurrent to run this {kind} stack on its "
        "private bus anyway."
        if not plan.real
        else "Stop it first (`kiteos stop`, or in its own terminal). Two stacks on one bus share "
        "topics and every RPC runs on both."
    )
    die(f"another robot stack is already running on this host:\n{listing}\n{hint}", code=2)


def show_fence(fence: FenceConfig, path: Path) -> int:
    """Print the enabled zones; returns how many are enabled."""
    enabled = [z for z in fence.zones if z.enabled]
    table = Table(title=f"Fence {path}", title_justify="left", show_edge=False)
    for column in ("zone", "kind", "frame", "applies to"):
        table.add_column(column)
    for zone in enabled:
        table.add_row(zone.name, zone.kind, zone.frame, zone.applies_to)
    console.print(
        f"fence mode [kite]{fence.mode}[/kite], margin {fence.margin_m * 100:.0f} cm, "
        f"joint speed limit {fence.max_joint_speed_rad_s:g} rad/s"
    )
    if enabled:
        console.print(table)
    disabled = [z.name for z in fence.zones if not z.enabled]
    if disabled:
        note(f"disabled zones: {', '.join(disabled)}")
    return len(enabled)


def show_real_warning(fence: FenceConfig, fence_path: Path, station: StationConfig | None = None) -> None:
    """The REAL ARMS banner, the enabled zones, and loud warnings for unprotected arms."""
    sources = f"\nStation: {station.describe_sources()}\nFence: {fence_path}" if station is not None else ""
    console.print(
        Panel.fit(
            f"[fail]REAL ARMS[/fail]: both {fence.profile.title} arms' motors will be powered.\n"
            "Stopping KiteOS disables the motors: the arms may DROP." + sources,
            border_style="fail",
        )
    )
    if show_fence(fence, fence_path) == 0:
        console.print(
            Panel.fit(
                "[fail]NO KEEP-OUT ZONES ARE ENABLED.[/fail]\n"
                "Only joint limits and the speed limit are enforced.\n"
                "Nothing keeps the arms off the glass, people or fixtures.\n"
                "Teach a zone with `kiteos fence teach`.",
                border_style="fail",
                title="WARNING",
            )
        )
    else:
        for side in fence.unprotected_arms():
            console.print(
                Panel.fit(
                    f"[fail]THE {side.upper()} ARM HAS NO KEEP-OUT ZONE.[/fail]\n"
                    f"No enabled zone applies to the {side} arm (allow_no_zones: true).\n"
                    f"Teach it with `kiteos fence teach --arms {side}`.",
                    border_style="fail",
                    title="WARNING",
                )
            )
    wrist_links = fence.profile.wrist_links
    wrist = [
        side for side in SIDES if not any(s.side in ("both", side) and s.link in wrist_links for s in fence.extra_spheres)
    ]
    if wrist:
        warn(
            f"no extra sphere models the wrist camera on the {' and '.join(wrist)} gripper; the camera "
            "body is NOT fenced (see docs/FENCE.md, extra_spheres)"
        )


def confirm_real_run(station: StationConfig) -> bool:
    """Require the operator to type the station name."""
    console.print(
        "Someone must be at the e-stop and supporting both arms, and the workspace must be clear.\n"
        f"Type the station name [kite]{station.station}[/kite] to confirm."
    )
    return confirm_typed(station.station, "Station name")


def execute(plan: LaunchPlan) -> None:
    """Replace this process with the runtime in the foreground (Ctrl-C goes straight to the runtime)."""
    os.chdir(plan.cwd)
    os.execvpe(plan.argv[0], plan.argv, plan.env)


def launch(
    station: StationConfig,
    blueprint: str,
    *,
    real: bool,
    cameras: bool = True,
    extra_flags: Sequence[str] = (),
    dry_run: bool = False,
    allow_concurrent: bool = False,
    station_ports_for_mock: bool = False,
    run_info: Mapping[str, Any] | None = None,
) -> LaunchPlan:
    """Run the safety gates, print the plan, then exec the runtime (unless ``dry_run``)."""
    if real and allow_concurrent:
        die("--allow-concurrent is only for MOCK runs; a real run never shares the host", code=2)
    injected = config_injection(station, os.environ)
    if injected:
        die(
            "the runtime would pick up settings that bypass the station file:\n  - "
            + "\n  - ".join(injected)
            + "\nRemove them (CONTROLCOORDINATOR__* / G__* / TRANSPORTS__*): they can power real arms "
            "in a mock run or switch the fence requirement off.",
            code=2,
        )
    fence_path, fence = load_fence_for_launch(station, real=real)
    private_bus = None
    ports = None
    if not real:
        private_bus = runs.random_private_bus(r.bus for r in runs.alive_run_states())
        if not station_ports_for_mock:
            ports = (runs.free_tcp_port(), runs.free_tcp_port())
    try:
        plan = build_runtime_command(
            station,
            blueprint,
            real=real,
            fence_path=fence_path,
            cameras=cameras,
            extra_flags=extra_flags,
            private_bus=private_bus,
            ports=ports,
            runtime_config=runtime_empty_config(),
        )
    except LaunchError as exc:
        die(str(exc), code=2)

    if not plan.cwd.is_dir() and not dry_run:
        die(f"runtime checkout not found at {plan.cwd} (station: runtime_root)", code=2)

    refuse_other_stacks(station, plan, allow_concurrent=allow_concurrent, dry_run=dry_run)

    if real:
        if fence is None or fence_path is None:  # load_fence_for_launch exits first; never run unfenced
            die("no fence loaded; not starting the real arms", code=2)
        show_real_warning(fence, fence_path, station)
        if not dry_run and not confirm_real_run(station):
            die("not confirmed; not starting", code=1)
    elif fence is not None and fence_path is not None:
        show_fence(fence, fence_path)

    for message in plan.notes:
        console.print(f"[warn]note:[/warn] {message}")
    mode = "[fail]REAL ARMS[/fail]" if real else "[ok]MOCK arms[/ok]"
    console.print(
        f"[kite]{plan.blueprint.run_name}[/kite] ({station.profile.title}) on {station.station} · {mode}"
    )
    if plan.webxr_port is not None:
        console.print(
            f"MOCK run on its own ports: WebXR https://127.0.0.1:{plan.webxr_port}/teleop · viser "
            f"http://127.0.0.1:{plan.viser_port} (for the Quest: `kiteos quest proxy` picks this run up)"
        )
    console.print(f"[muted]{plan.shell()}[/muted]", soft_wrap=True)
    if dry_run:
        note("dry run: not starting the runtime")
        return plan
    info = dict(run_info or {})
    record_run(
        runs.RunState(
            pid=os.getpid(),  # execvpe keeps the pid: this is the runtime process
            real=real,
            blueprint=plan.blueprint.run_name,
            station=station.station,
            bus=runs.bus_of(plan.env),
            private_bus=plan.private_bus,
            webxr_port=plan.webxr_port if plan.webxr_port is not None else station_ports(station)[0],
            viser_port=plan.viser_port if plan.viser_port is not None else station_ports(station)[1],
            recording=str(info["recording"]) if info.get("recording") else None,
            session_prefix=info.get("session_prefix"),
            started_at=runs.now_iso(),
            argv=list(plan.argv),
        )
    )
    execute(plan)
    return plan  # pragma: no cover - execvpe does not return
