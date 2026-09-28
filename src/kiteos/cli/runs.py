# Copyright 2026 Kite ML.
"""Which robot stacks run on this host, and which bus a KiteOS run uses.

The runtime picks its pub/sub + RPC bus from ``LCM_DEFAULT_URL`` and ``ZENOH_SCOUT_ADDR``; two stacks
on the same host and bus share topics, and a zenoh RPC runs on EVERY stack that serves the name. So:

* a mock run (no ``--real``) gets a private bus (its own LCM port and zenoh scouting group) and
  free ports for WebXR and viser, so it can never talk to a real stack; so does every
  hardware-free ``kiteos run``;
* ``kiteos teleop|collect start|rollout start|run`` refuse to start while another stack runs on
  this host (the runtime's run registry, KiteOS run-state files, a process-table scan for runtime
  and kite_rollout stacks that are not registered, listeners on the WebXR / viser ports and, for
  ``--real``, a short passive listen on the target bus that never publishes);
* every KiteOS launch writes a run-state file (pid, real/mock, bus, ports, recording) that the
  client commands (``collect controls``, ``rollout controls``, ``fence check|teach``, ``stop``,
  ``quest proxy``) read to join the right bus and target the right run.

Nothing here opens hardware. The probes are read-only: files, ``ps``, ``lsof`` (or a TCP connect to
127.0.0.1 when ``lsof`` is missing) and a zenoh/LCM subscriber that never publishes.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR_ENV = "KITEOS_RUNS_DIR"
BUS_ENV_KEYS: tuple[str, ...] = ("LCM_DEFAULT_URL", "ZENOH_SCOUT_ADDR", "DIMOS_TRANSPORT")
DEFAULT_LCM_PORT = 7667
DEFAULT_ZENOH_SCOUT_PORT = 7446
DEFAULT_WEBXR_PORT = 8443
DEFAULT_VISER_PORT = 8095
DEFAULT_QUEST_PORT = 8080
# Ports a mock run must never take (the station's real stack and the Quest proxy use them).
RESERVED_PORTS = frozenset({DEFAULT_WEBXR_PORT, DEFAULT_VISER_PORT, DEFAULT_QUEST_PORT})
_LCM_GROUP = "239.255.76.67"
_ZENOH_GROUP = "224.0.0.224"

# Command lines of processes that are (part of) a running robot stack. Client TUIs such as
# `kiteos-runtime imitation rollout` are deliberately not listed: they are not stacks.
STACK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # KiteOS launches through its own `kiteos-runtime` entry point; stacks started with the
    # runtime's own command (`dimos run`) are other stacks on this host and must be found too.
    # Global options may come before `run` (`kiteos-runtime --simulation mujoco run unitree-go2`).
    re.compile(r"(^|[\s/])(kiteos-runtime|dimos)(\s+-\S*(\s+[^\s-]\S*)?)*\s+run(\s|$)"),
    re.compile(r"\bkite_rollout\.run\b"),
    re.compile(r"\bdimos\.experimental\.isolated_python\.bootstrap\b"),
    re.compile(r"multiprocessing\.forkserver.*(dimos|kiteos-runtime)"),
)

PROBE_SCRIPT = r"""
import json, sys, time
duration = float(sys.argv[1])
seen = {"samples": 0, "topics": set()}
def note(key):
    seen["samples"] += 1
    if len(seen["topics"]) < 50:
        seen["topics"].add(str(key))
from dimos.core.global_config import global_config
if global_config.transport == "zenoh":
    from dimos.protocol.service.zenohservice import ZenohConfig, ZenohSessionPool
    pool = ZenohSessionPool()
    session = pool.acquire(ZenohConfig())
    sub = session.declare_subscriber("dimos/**", lambda sample: note(sample.key_expr))
    time.sleep(duration)
    sub.undeclare()
    pool.close_all()
else:
    import lcm
    from dimos.protocol.service import lcmservice
    lc = lcm.LCM(lcmservice._DEFAULT_LCM_URL)
    lc.subscribe(".*", lambda channel, data: note(channel))
    end = time.monotonic() + duration
    while time.monotonic() < end:
        lc.handle_timeout(100)
print(json.dumps({"transport": global_config.transport, "samples": seen["samples"],
                  "topics": sorted(seen["topics"])}))
"""


# --------------------------------------------------------------------------- run-state files


@dataclass
class RunState:
    """What one KiteOS launch started (written right before ``exec``; pid survives exec)."""

    pid: int
    real: bool
    blueprint: str
    station: str
    bus: dict[str, str | None] = field(default_factory=dict)  # BUS_ENV_KEYS as the stack saw them
    private_bus: bool = False
    webxr_port: int | None = None
    viser_port: int | None = None
    recording: str | None = None
    session_prefix: str | None = None
    started_at: str = ""
    argv: list[str] = field(default_factory=list)
    hardware: str | None = None  # what a `kiteos run` drives ("mock hardware", "mujoco simulator", ...)

    @property
    def mode(self) -> str:
        if self.real:
            return "REAL"
        return self.hardware.upper() if self.hardware else "MOCK"

    def describe(self) -> str:
        bus = "private bus" if self.private_bus else "default bus"
        extra = f", recording {Path(self.recording).name}" if self.recording else ""
        return f"PID {self.pid} {self.mode} {self.blueprint} ({bus}{extra})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunState:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")


def kiteos_runs_dir() -> Path:
    override = os.environ.get(RUNS_DIR_ENV)
    return Path(override).expanduser() if override else state_home() / "kiteos" / "runs"


def runtime_registry_dir() -> Path:
    return state_home() / "dimos" / "runs"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def process_command(pid: int) -> str | None:
    """Command line of ``pid`` (``ps``), or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["ps", "-o", "command=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    text = out.stdout.strip()
    return text or None


def write_run_state(state: RunState, runs_dir: Path | None = None) -> Path:
    """Atomically write ``<runs_dir>/<pid>.json``."""
    directory = runs_dir or kiteos_runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{state.pid}.json"
    fd, tmp = tempfile.mkstemp(prefix=f".{state.pid}.", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(state), fh, indent=2)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def alive_run_states(
    runs_dir: Path | None = None,
    *,
    alive: Callable[[int], bool] = pid_alive,
    command: Callable[[int], str | None] = process_command,
) -> list[RunState]:
    """KiteOS runs whose process is still ``kiteos-runtime run``; stale files (the pid ended or was reused)
    are removed."""
    directory = runs_dir or kiteos_runs_dir()
    if not directory.is_dir():
        return []
    states: list[RunState] = []
    for path in sorted(directory.glob("*.json")):
        try:
            state = RunState.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            continue
        cmd = command(state.pid) if alive(state.pid) else None
        if cmd is None or not STACK_PATTERNS[0].search(cmd):
            path.unlink(missing_ok=True)  # our own state file for a run that has ended
            continue
        states.append(state)
    return states


def runtime_registry_runs(
    registry_dir: Path | None = None, *, alive: Callable[[int], bool] = pid_alive
) -> list[dict[str, Any]]:
    """Alive entries of the runtime's run registry (read-only: never deletes the runtime's files)."""
    directory = registry_dir or runtime_registry_dir()
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            pid = int(data["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if alive(pid):
            runs.append(data)
    return runs


def registry_run_is_mock(entry: Mapping[str, Any]) -> bool | None:
    """True/False when a registry entry's argv proves mock/real arms, None when it cannot tell."""
    argv = [str(a) for a in entry.get("original_argv") or entry.get("cli_args") or []]
    text = " ".join(argv)
    if "can-port" in text or "can_port" in text:
        return False
    if re.search(r"expect[-_]mock(\s+|=)true", text):
        return True
    return None


# --------------------------------------------------------------------------- host probes


def stack_processes(
    ps_output: str | None = None, *, exclude: Iterable[int] = ()
) -> list[tuple[int, str]]:
    """``(pid, command)`` of processes that look like a running robot stack."""
    if ps_output is None:
        try:
            ps_output = subprocess.run(
                ["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
    skip = set(exclude) | {os.getpid()}
    found = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in skip:
            continue
        if any(pattern.search(cmd) for pattern in STACK_PATTERNS):
            found.append((pid, cmd))
    return found


def port_listening(port: int, *, lsof: str | None = None) -> bool:
    """Whether something listens on TCP ``port`` (lsof; else a connect to 127.0.0.1, never a bind)."""
    tool = lsof if lsof is not None else _which("lsof")
    if tool:
        try:
            out = subprocess.run(
                [tool, "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if out.stdout.strip():
                return True
            if out.returncode in (0, 1):
                return False
        except (OSError, subprocess.SubprocessError):
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


@dataclass(frozen=True)
class BusProbe:
    samples: int
    topics: list[str]
    error: str | None = None

    @property
    def busy(self) -> bool:
        return self.samples > 0


def probe_bus(
    env: Mapping[str, str],
    *,
    python: str | None = None,
    duration_s: float = 1.5,
    timeout_s: float = 60.0,
    cwd: Path | None = None,
) -> BusProbe:
    """Listen passively on the bus ``env`` selects and count runtime messages. Never publishes."""
    if os.environ.get("KITEOS_TEST_PRIVATE_BUS") == "1" and is_default_bus(bus_of(env)):
        # The test suite runs next to real rigs: it never listens on the default bus.
        return BusProbe(0, [], error="refusing to probe the default bus from the test suite")
    try:
        out = subprocess.run(
            [python or sys.executable, "-c", PROBE_SCRIPT, f"{duration_s:g}"],
            env=dict(env),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return BusProbe(0, [], error=f"{type(exc).__name__}: {exc}")
    for line in reversed(out.stdout.strip().splitlines()):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        return BusProbe(int(data.get("samples", 0)), list(data.get("topics", [])))
    return BusProbe(0, [], error=(out.stderr or out.stdout).strip()[-300:] or "no probe output")


@dataclass(frozen=True)
class Evidence:
    source: str
    detail: str


def other_stacks(
    *,
    ports: Sequence[int] = (DEFAULT_WEBXR_PORT, DEFAULT_VISER_PORT),
    registry: Callable[[], list[dict[str, Any]]] = runtime_registry_runs,
    kiteos_runs: Callable[[], list[RunState]] = alive_run_states,
    processes: Callable[[], list[tuple[int, str]]] = stack_processes,
    listening: Callable[[int], bool] = port_listening,
) -> list[Evidence]:
    """Everything on this host that suggests a robot stack is already running."""
    evidence: list[Evidence] = []
    for entry in registry():
        evidence.append(
            Evidence("runtime registry", f"PID {entry.get('pid')} {entry.get('blueprint', '?')} (run {entry.get('run_id', '?')})")
        )
    for state in kiteos_runs():
        evidence.append(Evidence("KiteOS run", state.describe()))
    for pid, cmd in processes():
        evidence.append(Evidence("process", f"PID {pid}: {cmd[:140]}"))
    for port in dict.fromkeys(ports):
        if listening(port):
            evidence.append(Evidence("port", f"something listens on TCP {port}"))
    return evidence


# --------------------------------------------------------------------------- private bus


def random_private_bus(taken: Iterable[Mapping[str, str | None]] = ()) -> dict[str, str]:
    """A private LCM port + zenoh scouting group, not the defaults and not used by another run."""
    used = {str(v) for bus in taken for v in bus.values() if v}
    while True:
        lcm_port = 20000 + secrets.randbelow(9000)
        scout_port = 30000 + secrets.randbelow(9000)
        if lcm_port == DEFAULT_LCM_PORT or scout_port == DEFAULT_ZENOH_SCOUT_PORT:
            continue
        bus = {
            # ttl=0 keeps LCM multicast on this host.
            "LCM_DEFAULT_URL": f"udpm://{_LCM_GROUP}:{lcm_port}?ttl=0",
            "ZENOH_SCOUT_ADDR": f"{_ZENOH_GROUP}:{scout_port}",
        }
        if not used & set(bus.values()):
            return bus


def free_tcp_port() -> int:
    """An ephemeral TCP port that is free right now (never one of the station's fixed ports)."""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in RESERVED_PORTS:
            return port


def bus_of(env: Mapping[str, str]) -> dict[str, str | None]:
    return {key: env.get(key) for key in BUS_ENV_KEYS}


def is_default_bus(bus: Mapping[str, str | None]) -> bool:
    lcm = bus.get("LCM_DEFAULT_URL") or ""
    scout = bus.get("ZENOH_SCOUT_ADDR") or ""
    return (not lcm or f":{DEFAULT_LCM_PORT}" in lcm) and (not scout or scout.endswith(f":{DEFAULT_ZENOH_SCOUT_PORT}"))


def apply_bus(bus: Mapping[str, str | None], environ: dict[str, str] | None = None) -> None:
    """Make ``environ`` (default ``os.environ``) select exactly ``bus`` (set or remove each key)."""
    target = os.environ if environ is None else environ
    for key in BUS_ENV_KEYS:
        value = bus.get(key)
        if value:
            target[key] = str(value)
        else:
            target.pop(key, None)


def runtime_already_imported() -> bool:
    return any(name in sys.modules for name in ("dimos.core.global_config", "dimos.protocol.service.lcmservice"))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

