# Copyright 2026 Kite ML.
"""Every embodiment of the runtime through KiteOS: `kiteos list`, `kiteos run`, `kiteos status`.

Nothing here starts the runtime or looks at this host: the registry is a fake file in a tmp
runtime root, `execute`/`record_run` are captured, and the process probes are replaced. The
`runtime` tests read the pinned runtime's registry file (never import it) to keep the checked
table in step with the pin.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from kiteos import embodiments
from kiteos.cli import app, launch, runs, setups, stop
from kiteos.cli._common import STATE
from kiteos.config import StationConfig
from kiteos.embodiments import HARDWARE_FREE, Setup, classify

STATION_NAME = "kite-test"
runner = CliRunner()
REAL_PLUGIN_SETUPS = setups.plugin_setups  # the autouse fixture replaces it

REGISTRY = {
    "coordinator-dual-openyam": "dimos.robot.manipulators.dual_openyam.blueprints.basic:coordinator_dual_openyam",
    "teleop-webxr-dual-openyam": "dimos.robot.manipulators.dual_openyam.blueprints.teleop:teleop_webxr_dual_openyam",
    "teleop-webxr-openyam": "dimos.robot.manipulators.openyam.blueprints.teleop:teleop_webxr_openyam",
    "teleop-webxr-xarm7": "dimos.teleop.webxr.blueprints:teleop_webxr_xarm7",
    "coordinator-trajectory-xarm6": "dimos.robot.manipulators.xarm.blueprints.teleop:coordinator_trajectory_xarm6",
    "teleop-webxr-dual": "dimos.teleop.webxr.blueprints:teleop_webxr_dual",
    "coordinator-piper-xarm": "dimos.robot.manipulators.common.mixed:coordinator_piper_xarm",
    "coordinator-mock": "dimos.robot.manipulators.common.mock:coordinator_mock",
    "unitree-go2-basic": "dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic:unitree_go2_basic",
    "unitree-go2-keyboard-teleop": "dimos.robot.unitree.go2.blueprints.basic.unitree_go2_keyboard_teleop:x",
    "unitree-g1-coordinator": "dimos.robot.unitree.g1.blueprints.basic.unitree_g1_coordinator:x",
    "unitree-g1-sim": "dimos.robot.unitree.g1.blueprints.perceptive.unitree_g1_sim:unitree_g1_sim",
    "drone-basic": "dimos.robot.drone.blueprints.basic.drone_basic:drone_basic",
    "demo-camera": "dimos.hardware.sensors.camera.module:demo_camera",
    "teleop-webxr-rerun": "dimos.teleop.webxr.blueprints:teleop_webxr_rerun",
}


def write_registry(root: Path, table: dict[str, str] | None = None) -> Path:
    path = root / embodiments.REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f'    "{k}": "{v}",' for k, v in (table or REGISTRY).items())
    path.write_text(f"# generated\n\nall_blueprints = {{\n{body}\n}}\n\nall_modules = {{}}\n")
    return path


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITEOS_STATION", raising=False)
    monkeypatch.delenv("KITEOS_ROOT", raising=False)
    monkeypatch.setenv("KITEOS_NO_BANNER", "1")
    STATE.station = None

    def no_host_commands(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"test tried to run a host command: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "run", no_host_commands)
    monkeypatch.setattr(os, "execvpe", no_host_commands)
    monkeypatch.setattr(setups, "plugin_setups", lambda: {})
    monkeypatch.setattr(runs, "alive_run_states", lambda *a, **k: [])
    for key in ("ROBOT_IP", "CAN_PORT", "XARM7_IP", "GO2CONNECTION__IP"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    root = tmp_path / "kiteos-runtime"
    root.mkdir()
    write_registry(root)
    station = {
        "station": STATION_NAME,
        "task": "close the box",
        "session_prefix": "close-box",
        "can": {"left": {"darwin": "L", "linux": "can_left"}, "right": {"darwin": "R", "linux": "can_right"}},
        "cameras": {"left_wrist": 0, "right_wrist": 2, "top": 1},
        "fence": "config/fence.yaml",
        "recordings_dir": str(tmp_path / "recordings"),
        "datasets_dir": str(tmp_path / "datasets"),
        "hub": {"repo_id": "kiteml/test-data", "private": True},
        "policy": {"checkpoint": str(tmp_path / "ckpt"), "device": "auto"},
        "runtime_root": str(root),
    }
    (config_dir / "station.yaml").write_text(yaml.safe_dump(station))
    return config_dir / "station.yaml"


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[launch.LaunchPlan]:
    calls: list[launch.LaunchPlan] = []
    monkeypatch.setattr(launch, "execute", calls.append)
    return calls


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[runs.RunState]:
    states: list[runs.RunState] = []
    monkeypatch.setattr(launch, "record_run", states.append)
    return states


def invoke(site: Path, *args: str, input: str | None = None) -> Any:
    return runner.invoke(app, ["--station", str(site), *args], input=input)


def setup(name: str) -> Setup:
    return Setup(name, REGISTRY[name], classify(name, REGISTRY[name]))


def station(site: Path) -> StationConfig:
    return StationConfig.load(site)


PRIVATE = {"LCM_DEFAULT_URL": "udpm://239.255.76.67:21234?ttl=0", "ZENOH_SCOUT_ADDR": "224.0.0.224:31234"}


# =========================================================================== registry and embodiments


def test_read_registry_parses_without_importing(tmp_path: Path) -> None:
    write_registry(tmp_path, {"a": "dimos.x:a", "b": "dimos.y:b"})
    assert embodiments.read_registry(tmp_path) == {"a": "dimos.x:a", "b": "dimos.y:b"}


@pytest.mark.parametrize(
    "text",
    [None, "all_modules = {}\n", "all_blueprints = dict(a='x')\n", "all_blueprints = {'a': 1}\n", "all_blueprints = {\n"],
)
def test_read_registry_errors(tmp_path: Path, text: str | None) -> None:
    if text is not None:
        path = tmp_path / embodiments.REGISTRY_FILE
        path.parent.mkdir(parents=True)
        path.write_text(text)
    with pytest.raises(embodiments.RegistryError):
        embodiments.read_registry(tmp_path)


@pytest.mark.parametrize(
    ("name", "target", "key"),
    [
        ("coordinator-dual-openyam", REGISTRY["coordinator-dual-openyam"], "dual-openyam"),
        ("teleop-webxr-openyam", REGISTRY["teleop-webxr-openyam"], "openyam"),
        ("teleop-webxr-xarm7", REGISTRY["teleop-webxr-xarm7"], "xarm"),
        ("teleop-webxr-dual", REGISTRY["teleop-webxr-dual"], "xarm-piper"),
        ("coordinator-mock", REGISTRY["coordinator-mock"], "mock"),
        ("learning-collect-webxr-piper", "dimos.imitation.collection.blueprint:x", "piper"),
        ("teleop-hosted-go2-multicam", "dimos.teleop.hosted.blueprints.cloudflare:x", "go2"),
        ("xarm6-worldbelief", "dimos.experimental.world_belief.xarm6_blueprint:x", "xarm"),
        ("unitree-g1-sim", REGISTRY["unitree-g1-sim"], "g1"),
        ("mid360-realsense-record", "dimos.robot.assembly.mid360_realsense_30:x", "sensors"),
        ("demo-error-on-name-conflicts", "dimos.robot.unitree.demo_error_on_name_conflicts:x", "agents"),
        ("teleop-webxr-rerun", REGISTRY["teleop-webxr-rerun"], "teleop"),
        ("acme.arm-teleop", "acme.blueprints:arm", "plugins"),
        ("something-new", "dimos.brand_new.module:x", "other"),
    ],
)
def test_classify(name: str, target: str, key: str) -> None:
    assert classify(name, target).key == key


def test_the_robots_kiteos_fences_are_fenced() -> None:
    from kiteos.robots import ROBOTS

    assert sorted(e.key for e in embodiments.BY_KEY.values() if e.fenced) == sorted(ROBOTS)


def test_catalog_orders_by_embodiment_then_name() -> None:
    names = [s.name for s in embodiments.catalog(REGISTRY, {"acme.arm": "acme:arm"})]
    assert names.index("coordinator-dual-openyam") < names.index("teleop-webxr-openyam") < names.index("demo-camera")
    assert names[-1] == "acme.arm"


def test_find_embodiment() -> None:
    assert embodiments.find_embodiment("GO2").key == "go2"
    assert embodiments.find_embodiment("humanoid").key == "g1"
    assert embodiments.find_embodiment("arm") is None  # ambiguous
    assert embodiments.find_embodiment("nope") is None


def test_always_real_setups_are_never_hardware_free() -> None:
    assert not set(embodiments.ALWAYS_REAL_NOTES) & set(HARDWARE_FREE)
    for name in ("coordinator-cartesian-ik-piper", "coordinator-trajectory-xarm6", "unitree-go2-keyboard-teleop"):
        assert name not in HARDWARE_FREE
    # --simulation does not stop these from opening hardware: never offered as --sim.
    for name in ("drone-basic", "drone-agentic", "keyboard-teleop-xarm7", "coordinator-dual-openyam"):
        assert not HARDWARE_FREE[name].sim


def _pinned_runtime_root() -> Path | None:
    spec = importlib.util.find_spec("dimos")
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).resolve().parents[1]
    return root if (root / embodiments.REGISTRY_FILE).is_file() else None


@pytest.mark.runtime
def test_checked_table_matches_the_pinned_runtime() -> None:
    root = _pinned_runtime_root()
    if root is None:
        pytest.skip("the runtime checkout is not importable here")
    registry = embodiments.read_registry(root)
    kiteos_run = set(embodiments.KITEOS_RUN_SETUPS)  # KiteOS's own, run like the runtime's
    assert not set(HARDWARE_FREE) - set(registry) - kiteos_run, "checked setups missing from the runtime (pin moved?)"
    assert kiteos_run <= set(HARDWARE_FREE)
    assert not set(embodiments.ALWAYS_REAL_NOTES) - set(registry)
    grouped = embodiments.catalog(registry)
    assert not [s.name for s in grouped if s.embodiment is embodiments.OTHER], "classify the new setups"
    fenced = {s.name for s in grouped if s.embodiment.fenced}
    assert fenced == {n for n, t in registry.items() if ".dual_openyam." in t or ".openarm." in t}


# =========================================================================== planning a run


def test_modes() -> None:
    assert setups.mode_runtime_options(setup("teleop-webxr-openyam"), "default", None) == (
        ["--simulation", "mujoco"],
        "mock arms",
    )
    assert setups.mode_runtime_options(setup("teleop-webxr-xarm7"), "default", None) == ([], "mock hardware")
    assert setups.mode_runtime_options(setup("teleop-webxr-xarm7"), "sim", None) == (
        ["--simulation", "mujoco"],
        "mujoco simulator",
    )
    assert setups.mode_runtime_options(setup("unitree-go2-basic"), "sim", "dimsim")[0] == ["--simulation", "dimsim"]
    assert setups.mode_runtime_options(setup("unitree-go2-basic"), "replay", None) == (["--replay"], "recorded data")
    assert setups.mode_runtime_options(setup("unitree-g1-sim"), "default", None)[0] == ["--simulation", "mujoco"]
    assert setups.mode_runtime_options(setup("unitree-g1-sim"), "sim", None)[0] == ["--simulation", "mujoco"]
    assert setup("unitree-g1-sim").summary() == "MuJoCo (simulation only)"
    assert setup("teleop-webxr-xarm7").summary() == "mock hardware · --sim mujoco"
    assert setup("coordinator-trajectory-xarm6").summary() == "--real only"
    assert setups.mode_runtime_options(setup("coordinator-piper-xarm"), "real", None) == ([], "real")


@pytest.mark.parametrize(
    ("name", "mode", "engine", "message"),
    [
        ("coordinator-dual-openyam", "real", None, "without the KiteOS fence"),
        ("teleop-webxr-dual-openyam", "real", None, "kiteos teleop"),
        ("coordinator-piper-xarm", "sim", None, "opens a real Piper on can0"),
        ("coordinator-piper-xarm", "default", None, "--real only"),
        ("coordinator-trajectory-xarm6", "sim", None, "has not checked"),
        ("unitree-g1-coordinator", "sim", None, "DDS"),
        ("unitree-go2-basic", "default", None, "--sim mujoco|dimsim · --replay"),
        ("unitree-go2-basic", "sim", "gazebo", "not gazebo"),
        ("teleop-webxr-xarm7", "replay", None, "with --replay"),
        ("teleop-webxr-xarm7", "default", "mujoco", "--engine"),
        ("drone-basic", "sim", None, "--replay"),
    ],
)
def test_refused_modes(name: str, mode: setups.Mode, engine: str | None, message: str) -> None:
    with pytest.raises(setups.SetupRunError, match=None) as info:
        setups.mode_runtime_options(setup(name), mode, engine)
    assert message in str(info.value)


def test_refused_arguments() -> None:
    assert setups.refused_arguments(["--viewer", "none"], real=False) == []
    assert setups.refused_arguments(["coordinator-dual-openyam", "--x"], real=True)
    bad = setups.refused_arguments(
        [
            "--robot-ip", "10.0.0.2", "--can_port=can0", "--controlcoordinator.left-can-port", "x",
            "--go2connection.ip", "1.2.3.4", "--simulation", "--zenoh-connect", "tcp/1.2.3.4:7447",
            "--config", "c.json", "--mavlinkconnection.connection-string", "udp:1:2",
            "--transports.cmd_vel", "ros", "--transport=lcm",
        ],
        real=False,
    )
    assert bad == [
        "--robot-ip", "--can_port=can0", "--controlcoordinator.left-can-port", "--go2connection.ip",
        "--simulation", "--zenoh-connect", "--config", "--mavlinkconnection.connection-string",
        "--transports.cmd_vel", "--transport=lcm",
    ]
    assert setups.refused_arguments(["--robot-ip", "10.0.0.2"], real=True) == []


def test_hardware_env_keys() -> None:
    env = {
        "ROBOT_IP": "1", "robot_ip": "1", "CAN_PORT": "can0", "GO2CONNECTION__IP": "1", "G__ROBOT_IP": "1",
        "ROBOT_INTERFACE": "en0", "PATH": "/bin", "__CF_USER_TEXT_ENCODING": "x", "HOME": "/h", "RELAY_URL": "u",
    }
    assert setups.hardware_env_keys(env) == [
        "CAN_PORT", "GO2CONNECTION__IP", "G__ROBOT_IP", "RELAY_URL", "ROBOT_INTERFACE", "ROBOT_IP", "robot_ip",
    ]


def test_plan_sim_run_is_isolated(site: Path, tmp_path: Path) -> None:
    config = tmp_path / "runtime-config.json"
    plan = setups.plan_setup_run(
        station(site),
        setup("teleop-webxr-xarm7"),
        mode="sim",
        extra=["--viewer", "none"],
        environ={"PATH": "/bin", "ROBOT_IP": "10.0.0.2", "XARM7_IP": "10.0.0.3", "GO2CONNECTION__IP": "x"},
        private_bus=PRIVATE,
        runtime_config=config,
    )
    assert plan.argv == [
        "uv", "run", "--no-sync", "kiteos-runtime", "--simulation", "mujoco", "run", "teleop-webxr-xarm7",
        "--config", str(config), "--viewer", "none",
    ]
    assert not plan.real and plan.private_bus
    assert "ROBOT_IP" not in plan.env and "XARM7_IP" not in plan.env and "GO2CONNECTION__IP" not in plan.env
    assert plan.env["LCM_DEFAULT_URL"] == PRIVATE["LCM_DEFAULT_URL"] and plan.env["PATH"] == "/bin"
    assert plan.blueprint.run_name == "teleop-webxr-xarm7" and plan.blueprint.description == "mujoco simulator"
    assert any("GO2CONNECTION__IP, ROBOT_IP, XARM7_IP" in n for n in plan.notes)
    assert plan.cwd == station(site).runtime_root_path()


def test_plan_real_run_keeps_the_operators_settings(site: Path) -> None:
    plan = setups.plan_setup_run(
        station(site),
        setup("unitree-go2-basic"),
        mode="real",
        extra=["--robot-ip", "10.0.0.2"],
        environ={"PATH": "/bin", "ROBOT_IP": "10.0.0.2"},
    )
    assert plan.argv == ["uv", "run", "--no-sync", "kiteos-runtime", "run", "unitree-go2-basic", "--robot-ip", "10.0.0.2"]
    assert plan.real and not plan.private_bus and plan.env["ROBOT_IP"] == "10.0.0.2"
    assert "LCM_DEFAULT_URL" not in plan.env


def test_plan_hardware_free_run_needs_a_private_bus(site: Path) -> None:
    with pytest.raises(setups.SetupRunError, match="private bus"):
        setups.plan_setup_run(station(site), setup("demo-camera"), mode="default", environ={})


def test_resolve_setup() -> None:
    assert setups.resolve_setup("teleop-webxr-xarm7", REGISTRY).embodiment.key == "xarm"
    assert setups.resolve_setup("acme.arm", REGISTRY, {"acme.arm": "acme:arm"}).embodiment.key == "plugins"
    for name in ("kiteos.dual-openyam-teleop", "dual-openyam-collection"):
        with pytest.raises(setups.SetupRunError, match="KiteOS setup"):
            setups.resolve_setup(name, REGISTRY)
    with pytest.raises(setups.SetupRunError, match="Did you mean teleop-webxr-xarm7"):
        setups.resolve_setup("teleop-webxr-xarm", REGISTRY)


def test_runtime_dotenv_files(tmp_path: Path) -> None:
    root = tmp_path / "rt"
    (root / "dimos" / "cli").mkdir(parents=True)
    assert launch.runtime_dotenv_files(root) == []
    (root / ".env").write_text("A=1\n")
    assert launch.runtime_dotenv_files(root) == [root / ".env"]
    (root / "dimos" / ".env").write_text("B=1\n")  # found first walking up from dimos/cli
    assert launch.runtime_dotenv_files(root) == [root / "dimos" / ".env", root / ".env"]


def test_plugin_setups_exclude_kiteos(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dist:
        def __init__(self, name: str) -> None:
            self.name = name

    class Entry:
        def __init__(self, dist: str, name: str) -> None:
            self.dist, self.name, self.value = Dist(dist), name, f"{dist}.mod:{name}"

    entries = [Entry("kiteos", "dual-openyam-teleop"), Entry("acme-robots", "arm")]
    monkeypatch.setattr(setups.metadata, "entry_points", lambda group: entries if group == "dimos.blueprints" else [])
    assert REAL_PLUGIN_SETUPS() == {"acme-robots.arm": "acme-robots.mod:arm"}


# =========================================================================== kiteos list


def test_list_groups_every_setup(site: Path) -> None:
    result = invoke(site, "list")
    assert result.exit_code == 0, result.output
    out = result.output
    for name in REGISTRY:
        assert name in out
    assert "kiteos.dual-openyam-teleop" in out and "kiteos collect start" in out
    assert "Unitree Go2 quadruped" in out and "--sim mujoco|dimsim · --replay" in out
    assert "opens a real Piper on can0" in out and "--real only" in out


def test_list_json_and_filter(site: Path) -> None:
    result = invoke(site, "list", "--json", "--embodiment", "go2")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["kiteos"] == []
    assert [s["setup"] for s in data["setups"]] == ["unitree-go2-basic", "unitree-go2-keyboard-teleop"]
    assert data["setups"][0]["without_hardware"] == {"default": None, "sim": ["mujoco", "dimsim"], "replay": True}
    assert data["setups"][1]["without_hardware"] is None and "DDS" in data["setups"][1]["note"]
    dual = json.loads(invoke(site, "list", "--json", "-e", "dual-openyam").output)
    assert [k["setup"] for k in dual["kiteos"]][0] == "kiteos.dual-openyam-teleop"
    assert all(k["fenced"] for k in dual["kiteos"])


def test_list_errors(site: Path, tmp_path: Path) -> None:
    assert invoke(site, "list", "-e", "zeppelin").exit_code == 2
    (tmp_path / "kiteos-runtime" / embodiments.REGISTRY_FILE).unlink()
    result = invoke(site, "list")
    assert result.exit_code == 2 and "runtime_root" in result.output


# =========================================================================== kiteos run


def test_run_sim_dry_run_prints_the_command(site: Path, executed: list[launch.LaunchPlan]) -> None:
    result = invoke(site, "run", "teleop-webxr-xarm7", "--sim", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "kiteos-runtime --simulation mujoco run teleop-webxr-xarm7" in " ".join(result.output.split())
    assert "private bus" in result.output and executed == []


def test_run_sim_executes_on_a_private_bus(
    site: Path, executed: list[launch.LaunchPlan], recorded: list[runs.RunState], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROBOT_IP", "10.0.0.2")
    result = invoke(site, "run", "unitree-go2-basic", "--sim", "--engine", "dimsim", "--", "--viewer", "none")
    assert result.exit_code == 0, result.output
    (plan,) = executed
    assert plan.argv[3:8] == ["kiteos-runtime", "--simulation", "dimsim", "run", "unitree-go2-basic"]
    assert plan.argv[-2:] == ["--viewer", "none"] and "--config" in plan.argv
    assert "ROBOT_IP" not in plan.env and plan.private_bus and ":7667" not in plan.env["LCM_DEFAULT_URL"]
    (state,) = recorded
    assert not state.real and state.private_bus and state.hardware == "dimsim simulator"
    assert state.blueprint == "unitree-go2-basic" and state.mode == "DIMSIM SIMULATOR"
    assert runs.STACK_PATTERNS[0].search(" ".join(state.argv))


def test_run_default_mode_uses_the_checked_mock(site: Path, executed: list[launch.LaunchPlan]) -> None:
    assert invoke(site, "run", "teleop-webxr-openyam").exit_code == 0
    assert executed[0].argv[4:8] == ["--simulation", "mujoco", "run", "teleop-webxr-openyam"]
    assert invoke(site, "run", "demo-camera").exit_code == 0
    assert executed[1].argv[4:6] == ["run", "demo-camera"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["coordinator-piper-xarm", "--sim"], "opens a real Piper on can0"),
        (["coordinator-piper-xarm"], "--real only"),
        (["coordinator-dual-openyam", "--real"], "KiteOS fence"),
        (["kiteos.dual-openyam-teleop"], "KiteOS setup"),
        (["teleop-webxr-xarm7", "--sim", "--real"], "choose one of"),
        (["teleop-webxr-xarm7", "--sim", "--", "--robot-ip", "10.0.0.2"], "refused arguments: --robot-ip"),
        (["teleop-webxr-xarm7", "--", "--config", "x.json"], "refused arguments: --config"),
        (["unitree-go2-basic", "--real", "--", "coordinator-dual-openyam"], "one setup per run"),
        (["unitree-go2-basic", "--real", "--allow-concurrent"], "only for hardware-free runs"),
        (["teleop-webxr-xarm", "--sim"], "Did you mean"),
    ],
)
def test_run_refusals(site: Path, executed: list[launch.LaunchPlan], args: list[str], message: str) -> None:
    result = invoke(site, "run", *args)
    assert result.exit_code == 2, result.output
    assert message in " ".join(result.output.split())
    assert executed == []


def test_run_refuses_robot_settings_in_the_runtime_dotenv(site: Path, tmp_path: Path, executed: list) -> None:
    dotenv = tmp_path / "kiteos-runtime" / ".env"
    dotenv.write_text("export ROBOT_IP=10.0.0.2\nOPENAI_API_KEY=x\n")
    result = invoke(site, "run", "unitree-go2-basic", "--sim")
    assert result.exit_code == 2 and "ROBOT_IP in" in result.output and executed == []
    assert "OPENAI_API_KEY" not in result.output
    dotenv.write_text("CONTROLCOORDINATOR__LEFT_CAN_PORT=can0\n")  # refused for every mode
    assert invoke(site, "run", "unitree-go2-basic", "--real", input=f"{STATION_NAME}\n").exit_code == 2
    dotenv.write_text("ROBOT_IP=10.0.0.2\n")  # a real run takes the operator's settings
    assert invoke(site, "run", "unitree-go2-basic", "--real", input=f"{STATION_NAME}\n").exit_code == 0
    assert executed[0].real


def test_run_real_needs_the_typed_station_name(
    site: Path, executed: list[launch.LaunchPlan], recorded: list[runs.RunState]
) -> None:
    result = invoke(site, "run", "unitree-go2-basic", "--real", input="nope\n")
    assert result.exit_code == 1 and "REAL HARDWARE" in result.output and "does not cover" in result.output
    assert executed == [] and recorded == []
    result = invoke(site, "run", "coordinator-piper-xarm", "--real", "--", "--can-port", "can0", input=f"{STATION_NAME}\n")
    assert result.exit_code == 0, result.output
    (plan,) = executed
    assert plan.real and "--config" not in plan.argv and plan.argv[-2:] == ["--can-port", "can0"]
    assert recorded[0].real and recorded[0].hardware is None and recorded[0].mode == "REAL"


def test_run_refuses_while_another_stack_runs(
    site: Path, executed: list[launch.LaunchPlan], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "host_stacks", lambda **_: [runs.Evidence("process", "PID 7: dimos run x")])
    result = invoke(site, "run", "demo-camera")
    assert result.exit_code == 2 and "hardware-free stack" in result.output and executed == []
    result = invoke(site, "run", "demo-camera", "--allow-concurrent")
    assert result.exit_code == 0 and "setup's usual ports" in " ".join(result.output.split())
    assert invoke(site, "run", "unitree-go2-basic", "--real", input=f"{STATION_NAME}\n").exit_code == 2


# =========================================================================== kiteos status / stop labels


def test_status_nothing_running(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setups, "find_stacks", lambda: [])
    monkeypatch.setattr(setups, "stack_processes", lambda: [])
    monkeypatch.setattr(setups, "process_table", lambda: {})
    result = invoke(site, "status")
    assert result.exit_code == 0 and "No robot stack is running" in result.output


def test_status_lists_stacks_and_unclaimed_processes(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kiteos_run = stop.RunningStack(
        pid=101, blueprint="unitree-go2-basic", mock=True, source="KiteOS run", related=(100,), hardware="mujoco simulator"
    )
    monkeypatch.setattr(setups, "find_stacks", lambda: [kiteos_run])
    monkeypatch.setattr(
        setups,
        "stack_processes",
        lambda: [(100, "uv run --no-sync kiteos-runtime --simulation mujoco run unitree-go2-basic"),
                 (102, "python -c from multiprocessing.forkserver import main; kiteos-runtime"),
                 (300, "/v/bin/dimos run coordinator-xarm7")],
    )
    monkeypatch.setattr(setups, "process_table", lambda: {100: (1, "uv"), 101: (100, "rt"), 102: (101, "fs"), 300: (1, "d")})
    out = invoke(site, "status").output
    assert "PID 101 unitree-go2-basic · mujoco simulator · KiteOS run" in out
    assert "PID 300" in out and "PID 102" not in out and "PID 100:" not in out


def test_stop_labels_hardware_free_runs() -> None:
    stack = stop.RunningStack(pid=1, blueprint="x", mock=True, hardware="recorded data")
    assert stack.mode == "recorded data"
    assert stop.RunningStack(pid=1, mock=True).mode == "MOCK arms"
    assert stop.RunningStack(pid=1, mock=False, hardware="ignored").mode == "REAL ARMS"


def test_find_stacks_carries_the_run_hardware() -> None:
    state = runs.RunState(pid=100, real=False, blueprint="drone-basic", station="s", hardware="recorded data")
    found = stop.find_stacks(kiteos_runs=lambda: [state], registry=lambda: [], processes=lambda: {})
    assert found[0].hardware == "recorded data" and found[0].mock is True


def test_run_state_round_trips_hardware(tmp_path: Path) -> None:
    state = runs.RunState(pid=5, real=False, blueprint="x", station="s", hardware="mujoco simulator")
    path = runs.write_run_state(state, tmp_path)
    loaded = runs.RunState.from_dict(json.loads(path.read_text()))
    assert loaded.hardware == "mujoco simulator" and loaded.mode == "MUJOCO SIMULATOR"
    legacy = runs.RunState.from_dict({"pid": 1, "real": False, "blueprint": "b", "station": "s"})
    assert legacy.hardware is None and legacy.mode == "MOCK"


@pytest.mark.parametrize(
    ("command", "stack"),
    [
        ("/x/.venv/bin/python /x/.venv/bin/kiteos-runtime run kiteos.dual-openyam-teleop --config y", True),
        ("uv run --no-sync kiteos-runtime --simulation mujoco run unitree-go2 --config /c", True),
        ("/v/bin/dimos --replay run unitree-go2", True),
        ("dimos --simulation mujoco --scene-package office run unitree-g1-groot-wbc", True),
        ("dimos imitation rollout", False),
        ("uv run --no-sync kiteos-runtime list", False),
        ("dimos log run-2", False),
        ("/Users/x/teleop_stack/dimos/.venv/bin/python /Users/x/.local/bin/kiteos run unitree-go2 --sim", False),
    ],
)
def test_stack_pattern_allows_global_options_before_run(command: str, stack: bool) -> None:
    assert bool(runs.STACK_PATTERNS[0].search(command)) is stack
