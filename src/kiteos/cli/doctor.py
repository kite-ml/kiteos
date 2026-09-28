# Copyright 2026 Kite ML.
"""``kiteos doctor``: is this host ready to run KiteOS?

Every check prints PASS/WARN/FAIL with remediation text. FAIL means a KiteOS run would not
work (or not be safe); the command then exits 1. Hardware is only *listed* (``kiteos-runtime hardware can
list``, ``adb devices``, device nodes / system_profiler); no camera or CAN bus is opened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.markup import escape

from kiteos import __version__
from kiteos.cli._common import (
    STATE,
    CmdResult,
    Level,
    StationOption,
    console,
    runtime_argv,
    kiteos_env,
    run_capture,
    status_line,
)
from kiteos.config import CAMERA_ROLES, SIDES, StationConfig, StationError, is_placeholder, platform_key, repo_root

EXPECTED_BLUEPRINTS = (
    "kiteos.dual-openyam-teleop",
    "kiteos.dual-openyam-collection",
    "kiteos.dual-openyam-rollout",
    "kiteos.dual-openyam-rollout-recorded",
)
# system_profiler names of cameras that are never station cameras (built-in, iPhone Continuity).
NON_STATION_CAMERA = re.compile(r"(?i)facetime|macbook|built-?in|iphone|ipad|continuity|desk view")
RECORDER_BINARY = Path("dimos/experimental/memory/rust/result/bin/dimos-memory-recorder")
RUNTIME_URDF = Path("data/dual_openyam_abc_box_v2/dual_openyam.urdf")
PIN_FILE = Path("deploy/runtime/PIN")
VENDORED_URDF = Path(__file__).resolve().parents[1] / "data" / "dual_openyam.urdf"
# FK self-test vectors from CONTRACTS.md (left arm, default mounts, world frame).
FK_VECTORS: tuple[tuple[tuple[float, ...], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.2574, 0.31, 0.1735)),
    ((0.0, 1.047, 1.047, 0.0, 0.0, 0.0), (0.3894, 0.31, 0.4021)),
    ((0.5, 1.2, 0.8, -0.3, 0.4, 1.0), (0.3486, 0.4195, 0.1635)),
)
FK_TOLERANCE_M = 1e-3


@dataclass(frozen=True)
class Check:
    level: Level
    name: str
    detail: str = ""
    fix: str | None = None


class Probe:
    """What the doctor observes about the host. Tests substitute a fake."""

    def __init__(self, station: StationConfig | None) -> None:
        self.station = station
        self.platform = platform_key()

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def is_executable(self, path: Path) -> bool:
        return path.is_file() and os.access(path, os.X_OK)

    def glob(self, pattern: str) -> list[str]:
        import glob

        return sorted(glob.glob(pattern))

    def run(self, argv: Sequence[str], cwd: Path | None = None, timeout: float = 60.0) -> CmdResult:
        return run_capture(argv, cwd=cwd, timeout=timeout)

    def runtime(self, *args: str, timeout: float = 120.0) -> CmdResult:
        assert self.station is not None
        root = self.station.runtime_root_path()
        if not root.is_dir():
            return CmdResult(2, "", f"runtime checkout not found at {root}")
        return run_capture(runtime_argv(*args), cwd=root, timeout=timeout, env=kiteos_env())

    def hf_token(self) -> str | None:
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception:
            token = os.environ.get("HF_TOKEN")
            if token:
                return token
            home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            path = home / "token"
            return path.read_text().strip() if path.is_file() else None

    def imported_runtime_root(self) -> Path | None:
        spec = importlib.util.find_spec("dimos")
        if spec is None or not spec.origin:
            return None
        return Path(spec.origin).resolve().parent.parent

    def python_version(self) -> tuple[int, int]:
        return sys.version_info[:2]

    def interpreter(self) -> Path:
        return Path(sys.executable)

    def running_stacks(self) -> list[str]:
        """Descriptions of robot stacks running on this host (registry, KiteOS runs, processes)."""
        from kiteos.cli.runs import other_stacks

        return [f"{e.source}: {e.detail}" for e in other_stacks(ports=())]


# --------------------------------------------------------------------------- checks


def check_python(probe: Probe) -> list[Check]:
    major, minor = probe.python_version()
    if (major, minor) < (3, 10):
        return [Check("FAIL", "python", f"{major}.{minor}", "KiteOS needs Python >= 3.10 (the runtime uses 3.12)")]
    level: Level = "PASS" if (major, minor) == (3, 12) else "WARN"
    return [
        Check(
            level,
            "python",
            f"{major}.{minor} ({probe.interpreter()})",
            None if level == "PASS" else "the runtime is tested on 3.12",
        )
    ]


def check_station_values(station: StationConfig, probe: Probe) -> list[Check]:
    checks = []
    for side in SIDES:
        port = station.can_port(side)
        if not port or is_placeholder(port):
            checks.append(
                Check(
                    "WARN", f"station CAN {side}", f"not set for {probe.platform}", f"set can.{side}.{probe.platform}"
                )
            )
    for role in CAMERA_ROLES:
        selector = station.camera(role)
        if selector is None or is_placeholder(selector):
            checks.append(
                Check(
                    "FAIL",
                    f"station camera {role}",
                    f"not set for {probe.platform} ({selector}); `kiteos collect` and `kiteos rollout` refuse to start",
                    "find the device with `kiteos cameras list` and set it in the station file "
                    "(teleop alone works with --no-cameras)",
                )
            )
        elif probe.platform == "linux" and isinstance(selector, int):
            checks.append(
                Check(
                    "WARN",
                    f"station camera {role}",
                    f"integer index {selector} on Linux is not stable (two /dev/video nodes per camera)",
                    "use /dev/v4l/by-id or by-path (see `kiteos cameras list`)",
                )
            )
    if not checks:
        checks.append(Check("PASS", "station devices", f"CAN and cameras set for {probe.platform}"))
    return checks


def check_fence(station: StationConfig) -> list[Check]:
    from kiteos.cli.fence import validate_fence
    from kiteos.safety.fence import FenceConfig, FenceConfigError

    path = station.fence_path()
    if not path.is_file():
        return [
            Check(
                "FAIL",
                "fence",
                f"missing: {path}",
                "copy config/fence.example.yaml, then `kiteos fence validate` (required for --real)",
            )
        ]
    try:
        config = FenceConfig.load(path)
    except (FenceConfigError, OSError) as exc:
        return [Check("FAIL", "fence", f"invalid: {path}", str(exc))]
    return [
        Check(f.level, f"fence {f.name}", f.detail, f.fix)  # type: ignore[arg-type]
        for f in validate_fence(config, station_name=station.station)
    ]


def check_urdf(station: StationConfig, probe: Probe) -> list[Check]:
    vendored = VENDORED_URDF
    if not probe.exists(vendored):
        return [Check("FAIL", "URDF", f"missing {vendored}", "reinstall KiteOS (the URDF ships with the package)")]
    upstream = station.runtime_root_path() / RUNTIME_URDF
    if probe.exists(upstream):
        same = vendored.read_bytes() == upstream.read_bytes()
        if not same:
            return [
                Check(
                    "WARN",
                    "URDF",
                    "vendored copy differs from the runtime's model",
                    f"compare {vendored} with {upstream}; the fence uses the vendored copy",
                )
            ]
        return [Check("PASS", "URDF", "vendored copy matches the runtime's model")]
    return [Check("PASS", "URDF", f"{vendored.name} (runtime model not downloaded yet)")]


def check_fk() -> list[Check]:
    import numpy as np

    from kiteos.safety.kinematics import DualArmKinematics

    try:
        kinematics = DualArmKinematics()
        worst = 0.0
        for q, expected in FK_VECTORS:
            tip = kinematics.fingertip("left", np.array(q))
            worst = max(worst, float(np.max(np.abs(tip - np.array(expected)))))
            right = kinematics.fingertip("right", np.array(q))
            worst = max(worst, float(np.max(np.abs(right - (np.array(expected) - [0.0, 0.62, 0.0])))))
    except Exception as exc:
        return [
            Check(
                "FAIL", "FK self-test", f"{type(exc).__name__}: {exc}", "the fence cannot be trusted; reinstall KiteOS"
            )
        ]
    if worst > FK_TOLERANCE_M:
        return [
            Check(
                "FAIL",
                "FK self-test",
                f"fingertip off by {worst * 1000:.2f} mm",
                "the kinematic model does not match the reference; do not run with --real",
            )
        ]
    return [Check("PASS", "FK self-test", f"{len(FK_VECTORS)} poses per arm within {worst * 1000:.2f} mm")]


def read_pin(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "commit":
            return value.split("#")[0].strip() or None
    return None


def check_runtime(station: StationConfig, probe: Probe) -> list[Check]:
    checks: list[Check] = []
    root = station.runtime_root_path()
    if not probe.which("uv"):
        checks.append(Check("FAIL", "uv", "not on PATH", "install uv >= 0.9.25 (https://docs.astral.sh/uv/)"))
    else:
        checks.append(Check("PASS", "uv", probe.which("uv") or ""))
    if not probe.exists(root):
        checks.append(
            Check(
                "FAIL",
                "runtime checkout",
                f"missing: {root}",
                "re-run deploy/install.sh (it sets up the pinned runtime; see docs/RUNBOOK.md)",
            )
        )
        return checks
    pinned = read_pin(repo_root() / PIN_FILE)
    head = probe.run(["git", "-C", str(root), "rev-parse", "HEAD"], timeout=30.0)
    if not head.ok:
        checks.append(
            Check(
                "FAIL",
                "runtime commit",
                head.output or "not a git checkout",
                "the runtime must be a git clone (it resolves data and the LeRobot env through .git)",
            )
        )
    elif pinned and head.stdout.strip() != pinned:
        checks.append(
            Check(
                "FAIL",
                "runtime commit",
                f"{head.stdout.strip()[:10]} != pinned {pinned[:10]}",
                f"git -C {root} checkout {pinned[:10]}  (KiteOS depends on runtime internals at that commit)",
            )
        )
    else:
        checks.append(Check("PASS", "runtime commit", head.stdout.strip()[:10] + (" (pinned)" if pinned else "")))
        dirty = probe.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], timeout=30.0)
        if dirty.ok and dirty.stdout.strip():
            checks.append(
                Check(
                    "WARN",
                    "the runtime tree",
                    "has local modifications",
                    "KiteOS never edits the runtime; `git -C <runtime checkout> diff` to review",
                )
            )
    imported = probe.imported_runtime_root()
    if imported is None:
        checks.append(
            Check(
                "FAIL",
                "runtime import",
                "the runtime package is not importable",
                f"run kiteos from {root}/.venv (uv pip install --no-deps -e <kiteos>)",
            )
        )
    elif imported.resolve() != root.resolve():
        checks.append(Check("WARN", "runtime import", f"imports the runtime from {imported}", f"station runtime_root is {root}"))
    else:
        checks.append(Check("PASS", "runtime import", str(imported)))

    listing = probe.runtime("list")
    if not listing.ok:
        checks.append(
            Check(
                "FAIL",
                "setups",
                listing.output[-300:] or "`kiteos-runtime list` failed",
                f"cd {root} && uv run --no-sync kiteos-runtime list",
            )
        )
    else:
        names = {line.strip() for line in listing.stdout.splitlines()}
        missing = [bp for bp in EXPECTED_BLUEPRINTS if bp not in names]
        if missing:
            checks.append(
                Check(
                    "FAIL",
                    "setups",
                    f"the runtime does not see {', '.join(missing)}",
                    f"uv pip install --python {root}/.venv/bin/python --no-deps -e {repo_root()}  "
                    "(never a plain `uv sync` in the runtime checkout afterwards)",
                )
            )
        else:
            checks.append(Check("PASS", "setups", ", ".join(EXPECTED_BLUEPRINTS)))

    recorder = root / RECORDER_BINARY
    if probe.is_executable(recorder):
        checks.append(Check("PASS", "recorder", str(RECORDER_BINARY)))
    else:
        checks.append(
            Check(
                "FAIL",
                "recorder",
                f"missing {recorder}",
                f"cd {root} && cargo build --release --locked -p dimos-memory-recorder && "
                f"mkdir -p {recorder.parent} && cp target/release/dimos-memory-recorder {recorder.parent}/",
            )
        )
    return checks


def check_adb(station: StationConfig, probe: Probe) -> list[Check]:
    if not station.quest.adb_enabled:
        return [Check("INFO", "adb", "disabled in the station (quest.adb)")]
    configured = station.quest.adb if isinstance(station.quest.adb, str) and station.quest.adb != "auto" else None
    adb = (
        configured
        or probe.which("adb")
        or (
            "/opt/homebrew/bin/adb"
            if probe.platform == "darwin" and probe.exists(Path("/opt/homebrew/bin/adb"))
            else None
        )
    )
    if not adb:
        install = "brew install android-platform-tools" if probe.platform == "darwin" else "sudo apt install adb"
        return [Check("WARN", "adb", "not found (needed for the USB-tethered Quest)", install)]
    devices = probe.run([adb, "devices"], timeout=15.0)
    if not devices.ok:
        return [Check("WARN", "adb", devices.output or "adb devices failed")]
    rows = [line.split() for line in devices.stdout.splitlines()[1:] if line.strip()]
    if not rows:
        return [Check("WARN", "Quest", "no headset connected over USB", "plug in the Quest and allow USB debugging")]
    states = ", ".join(f"{r[0]} ({r[1]})" for r in rows if len(r) >= 2)
    if any(len(r) >= 2 and r[1] == "unauthorized" for r in rows):
        return [Check("WARN", "Quest", states, "accept 'Allow USB debugging' inside the headset")]
    return [Check("PASS", "Quest", states)]


def check_can(station: StationConfig, probe: Probe) -> list[Check]:
    from kiteos.cli.can import parse_gs_usb_list, parse_ip_brief

    running = probe.running_stacks()
    if running:
        # Enumerating the adapters opens their USB descriptors; never while a stack drives them.
        return [
            Check(
                "WARN",
                "CAN",
                f"not checked: a robot stack is running ({escape(running[0][:120])})",
                "re-run `kiteos doctor` after `kiteos stop` to check the adapters",
            )
        ]
    listing = probe.runtime("hardware", "can", "list")
    if not listing.ok:
        return [Check("WARN", "CAN", listing.output[-300:] or "listing failed", "needed only with --real")]
    checks = []
    if probe.platform == "darwin":
        serials = parse_gs_usb_list(listing.stdout)
        for side in SIDES:
            port = station.can_port(side)
            if port and port in serials:
                checks.append(Check("PASS", f"CAN {side}", port))
            else:
                checks.append(
                    Check(
                        "WARN",
                        f"CAN {side}",
                        f"{port} not connected (seen: {', '.join(serials) or 'none'})",
                        "plug in the adapter; needed only with --real",
                    )
                )
    else:
        interfaces = parse_ip_brief(listing.stdout)
        for side in SIDES:
            port = station.can_port(side) or ""
            state = interfaces.get(port)
            if state is None:
                checks.append(
                    Check(
                        "WARN",
                        f"CAN {side}",
                        f"{port} missing",
                        "install deploy/udev rules and replug; needed only with --real",
                    )
                )
            elif state.upper() not in ("UP", "UNKNOWN"):
                checks.append(Check("WARN", f"CAN {side}", f"{port} is {state}", "kiteos can setup"))
            else:
                checks.append(Check("PASS", f"CAN {side}", f"{port} {state}"))
    return checks


def check_cameras(station: StationConfig, probe: Probe) -> list[Check]:
    if probe.platform == "darwin":
        result = probe.run(["system_profiler", "SPCameraDataType", "-json"], timeout=30.0)
        try:
            devices = json.loads(result.stdout).get("SPCameraDataType", []) if result.ok else []
        except (ValueError, AttributeError):
            devices = []
        names = [str(d.get("_name", "")) for d in devices if isinstance(d, dict)]
        station_cams = [n for n in names if not NON_STATION_CAMERA.search(n)]
        other = [n for n in names if NON_STATION_CAMERA.search(n)]
        level: Level = "PASS" if len(station_cams) >= len(CAMERA_ROLES) else "WARN"
        detail = f"{len(station_cams)} USB camera(s)" + (f" (+ not counted: {', '.join(other)})" if other else "")
        return [
            Check(
                level,
                "cameras",
                detail,
                None
                if level == "PASS"
                else "connect the three station cameras; the built-in and iPhone (Continuity) cameras do not count",
            ),
            Check(
                "INFO",
                "camera indices",
                "OpenCV's index order can differ from this list; do the cover test (`kiteos cameras check`)",
            ),
        ]
    checks = []
    nodes = probe.glob("/dev/video*")
    for role in CAMERA_ROLES:
        selector = station.camera(role)
        if isinstance(selector, str) and not is_placeholder(selector):
            if probe.exists(Path(selector)):
                checks.append(Check("PASS", f"camera {role}", selector))
            else:
                checks.append(
                    Check(
                        "WARN",
                        f"camera {role}",
                        f"{selector} missing",
                        "`kiteos cameras list` shows the by-id/by-path names",
                    )
                )
    checks.append(Check("INFO", "video nodes", f"{len(nodes)} /dev/video* nodes"))
    return checks


def check_data(station: StationConfig, probe: Probe) -> list[Check]:
    checks = []
    token = probe.hf_token()
    if token:
        checks.append(Check("PASS", "Hugging Face", "token found"))
    else:
        checks.append(
            Check(
                "WARN",
                "Hugging Face",
                "not logged in (needed for `kiteos upload`)",
                f"{station.runtime_root_path()}/.venv/bin/hf auth login",
            )
        )
    checkpoint = station.checkpoint_path()
    if checkpoint is None:
        checks.append(Check("INFO", "policy", "no checkpoint configured"))
    elif probe.exists(checkpoint):
        checks.append(Check("PASS", "policy", str(checkpoint)))
    else:
        checks.append(Check("WARN", "policy", f"missing {checkpoint}", "copy the checkpoint or set policy.checkpoint"))
    return checks


def run_doctor(
    station_path: Path | None, probe_factory: Callable[[StationConfig | None], Probe] | None = None
) -> list[Check]:
    """Run every check; the station is loaded first because the others depend on it."""
    checks: list[Check] = [Check("INFO", "KiteOS", f"{__version__} from {repo_root()}")]
    try:
        station: StationConfig | None = StationConfig.load(station_path)
    except StationError as exc:
        station = None
        checks.append(Check("FAIL", "station", str(exc).splitlines()[0], "fix the station file; other checks skipped"))
    probe = (probe_factory or Probe)(station)
    checks += check_python(probe)
    if station is None:
        return checks
    checks.append(Check("PASS", "station", f"{station.station} · {station.task} · {station.source}"))
    checks += check_station_values(station, probe)
    checks += check_fence(station)
    checks += check_urdf(station, probe)
    checks += check_fk()
    checks += check_runtime(station, probe)
    checks += check_can(station, probe)
    checks += check_cameras(station, probe)
    checks += check_adb(station, probe)
    checks += check_data(station, probe)
    return checks


def doctor(station: StationOption = None) -> None:
    """Check the host, the runtime, the station and fence files, and the attached devices."""
    checks = run_doctor(station or STATE.station)
    for item in checks:
        status_line(item.level, item.name, item.detail, item.fix)
    failures = [c for c in checks if c.level == "FAIL"]
    warnings = [c for c in checks if c.level == "WARN"]
    console.print(
        f"\n{len(checks)} checks: [fail]{len(failures)} FAIL[/fail], [warn]{len(warnings)} WARN[/warn]"
        + ("" if failures else " · [ok]ready[/ok]")
    )
    if failures:
        raise typer.Exit(1)
