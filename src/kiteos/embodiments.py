# Copyright 2026 Kite ML.
"""Robot embodiments of the base runtime, and which of its runs KiteOS has checked stay off hardware.

KiteOS runs on the whole base runtime: every setup the runtime ships (arms, quadrupeds,
humanoids, drones, sensor rigs, simulators) can be listed with ``kiteos list`` and started with
``kiteos run``. The setups are read from the station's runtime checkout
(``dimos/robot/all_blueprints.py``, parsed, never imported) and grouped here by embodiment.

``HARDWARE_FREE`` records, per setup, the runs in which nothing can move a real robot: only mock
or simulated actuators and no connection to a robot, provided no robot address reaches the stack
(``kiteos run`` strips them; cameras and other sensors may still open). Every entry was checked
against the pinned runtime source (deploy/runtime/PIN). A setup or mode that is not listed may
drive real hardware whatever the flags say: ``coordinator-piper-xarm`` opens a real Piper on
``can0`` and ``unitree-g1-coordinator`` joins the robot's DDS network, both even with
``--simulation``. ``kiteos run`` therefore treats anything unlisted as a real run. Re-check the
table whenever the runtime pin moves.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REGISTRY_FILE = Path("dimos") / "robot" / "all_blueprints.py"
REGISTRY_TABLE = "all_blueprints"


class RegistryError(RuntimeError):
    """The runtime checkout's setup registry is missing or unreadable."""


@dataclass(frozen=True)
class Embodiment:
    """One kind of robot (or a family of non-robot setups) in the runtime."""

    key: str
    title: str
    modules: tuple[str, ...] = ()  # registry module prefixes, without the leading "dimos."
    words: tuple[str, ...] = ()  # setup-name words that place a setup from a shared module here
    setups: tuple[str, ...] = ()  # setups placed here by name
    robot: bool = True  # False for the sensor, agent and teleop-tool groups
    fenced: bool = False  # KiteOS fences this robot: drive it with the KiteOS setups


EMBODIMENTS: tuple[Embodiment, ...] = (
    Embodiment("dual-openyam", "Dual OpenYAM (bimanual arms)", modules=("robot.manipulators.dual_openyam",), fenced=True),
    Embodiment("openyam", "OpenYAM arm", modules=("robot.manipulators.openyam",), words=("openyam",)),
    Embodiment(
        "xarm-piper",
        "xArm6 + Piper (mixed pair)",
        setups=("coordinator-piper-xarm", "coordinator-teleop-dual", "teleop-webxr-dual"),
    ),
    Embodiment(
        "xarm",
        "UFactory xArm 6/7 and Lite6",
        modules=("robot.manipulators.xarm", "experimental.world_belief.xarm6_blueprint"),
        words=("xarm", "xarm6", "xarm7", "lite6"),
    ),
    Embodiment("piper", "AgileX Piper arm", modules=("robot.manipulators.piper",), words=("piper",)),
    Embodiment(
        "openarm", "OpenArm (bimanual arms)", modules=("robot.manipulators.openarm",), words=("openarm",), fenced=True
    ),
    Embodiment("a1z", "Galaxea A1Z arm", modules=("robot.manipulators.a1z",), words=("a1z",)),
    Embodiment("a750", "A-750 arm", modules=("robot.manipulators.a750",), words=("a750",)),
    Embodiment(
        "mock",
        "Mock arms and bases",
        setups=(
            "coordinator-basic",
            "coordinator-mock",
            "coordinator-dual-mock",
            "coordinator-mock-twist-base",
            "coordinator-mobile-manip-mock",
        ),
    ),
    Embodiment("flowbase", "FlowBase mobile base", setups=("coordinator-flowbase", "coordinator-flowbase-keyboard-teleop")),
    Embodiment("go2", "Unitree Go2 quadruped", modules=("robot.unitree.go2",), words=("go2",)),
    Embodiment("g1", "Unitree G1 humanoid", modules=("robot.unitree.g1",), words=("g1",)),
    Embodiment("r1pro", "Galaxea R1 Pro (mobile manipulator)", modules=("robot.galaxea.r1pro",), words=("r1pro",)),
    Embodiment("m20", "DeepRobotics M20 quadruped", modules=("robot.deeprobotics",), words=("m20",)),
    Embodiment("spot", "Boston Dynamics Spot", modules=("experimental.robot.bosdyn",), words=("spot",)),
    Embodiment("drone", "Drone (MAVLink)", modules=("robot.drone",), words=("drone",)),
    Embodiment("alfred", "Alfred (DIY mobile robot)", modules=("robot.diy.alfred",), words=("alfred",)),
    Embodiment("habitat", "Habitat simulator (virtual robot)", modules=("simulation.habitat",)),
    Embodiment(
        "sensors",
        "Sensors, mapping and perception (no actuators)",
        modules=("hardware.sensors", "mapping", "perception", "memory", "robot.assembly"),
        robot=False,
    ),
    Embodiment(
        "agents",
        "Agents and software demos (no actuators)",
        modules=("agents", "core.demos", "robot.unitree.demo_error_on_name_conflicts"),
        robot=False,
    ),
    Embodiment("teleop", "Teleop tools", modules=("teleop",), robot=False),
)
PLUGINS = Embodiment("plugins", "Other installed setups (plugins)", robot=False)
OTHER = Embodiment("other", "Other", robot=False)
BY_KEY: dict[str, Embodiment] = {e.key: e for e in (*EMBODIMENTS, PLUGINS, OTHER)}


@dataclass(frozen=True)
class HardwareFree:
    """The runs of one setup that KiteOS checked stay off robot hardware (see the module doc)."""

    default: tuple[str, ...] | None = None  # runtime options for a run without --sim/--replay; None: unchecked
    default_label: str = "mock hardware"  # what that run drives
    sim: tuple[str, ...] = ()  # simulators for --sim; the first is the default
    replay: bool = False  # --replay (recorded data instead of the robot)
    sim_only: bool = False  # a simulation setup: it runs the simulator with or without --sim

    def summary(self) -> str:
        parts = []
        if self.sim_only:
            parts.append(f"{self.default_label} (simulation only)")
        elif self.default is not None:
            parts.append(self.default_label)
        if self.sim and not self.sim_only:
            parts.append("--sim " + "|".join(self.sim))
        if self.replay:
            parts.append("--replay")
        return " · ".join(parts)


REAL_ONLY_SUMMARY = "--real only"

# The runtime selects mock arms for OpenYAM only in its simulation mode (there is no MuJoCo model).
_OPENYAM = HardwareFree(default=("--simulation", "mujoco"), default_label="mock arms")
_MOCK = HardwareFree(default=(), default_label="mock hardware")
_MOCK_OR_MUJOCO = HardwareFree(default=(), default_label="mock hardware", sim=("mujoco",))
_MUJOCO = HardwareFree(sim=("mujoco",))
_SIM_ONLY = HardwareFree(default=("--simulation", "mujoco"), default_label="MuJoCo", sim=("mujoco",), sim_only=True)
_GO2 = HardwareFree(sim=("mujoco", "dimsim"), replay=True)
_REPLAY = HardwareFree(replay=True)
_HABITAT = HardwareFree(default=(), default_label="Habitat simulator")
_NO_ACTUATORS = HardwareFree(default=(), default_label="no actuators")


def _each(names: str, support: HardwareFree) -> dict[str, HardwareFree]:
    return dict.fromkeys(names.split(), support)


# Checked against the runtime source pinned in deploy/runtime/PIN (see the module doc for the rule).
HARDWARE_FREE: dict[str, HardwareFree] = {
    # Mock coordinators: every adapter is mock (or there is no hardware at all).
    **_each(
        "coordinator-basic coordinator-mock coordinator-dual-mock coordinator-mock-twist-base "
        "coordinator-mobile-manip-mock coordinator-cartesian-ik-mock dual-xarm6-planner-coordinator",
        _MOCK,
    ),
    # Dual OpenYAM and OpenArm: mock unless both CAN ports are configured.
    **_each(
        "coordinator-dual-openyam dual-openyam-planner-coordinator dual-openyam-quest-collection "
        "teleop-webxr-dual-openyam coordinator-openarm openarm-planner-coordinator teleop-webxr-openarm",
        _MOCK,
    ),
    # OpenYAM: mock arms in the runtime's simulation mode; without it the real arm on the default bus.
    **_each(
        "coordinator-openyam keyboard-teleop-openyam keyboard-teleop-openyam-planner openyam-planner-coordinator "
        "teleop-webxr-openyam openyam-quest-collection openyam-teach-collection openyam-lerobot-rollout "
        "openyam-lerobot-quest-rollout",
        _OPENYAM,
    ),
    # A1Z (real only with a CAN port and not simulating), A-750 (mock without a device path).
    **_each(
        "a1z-planner-coordinator coordinator-a1z coordinator-teleop-a1z keyboard-teleop-a1z teleop-webxr-a1z "
        "keyboard-teleop-a750",
        _MOCK,
    ),
    # xArm and Piper with mock_without_address: mock without an address, MuJoCo with --simulation.
    **_each(
        "coordinator-teleop-xarm7 coordinator-teleop-xarm6 teleop-webxr-xarm7 teleop-webxr-hand-xarm7 "
        "teleop-webxr-xarm7-video teleop-webxr-xarm6 learning-collect-webxr-xarm7 coordinator-piper "
        "coordinator-teleop-piper teleop-webxr-piper learning-collect-webxr-piper",
        _MOCK_OR_MUJOCO,
    ),
    # Mock without an address; --simulation would pick a MuJoCo adapter with no simulator running.
    **_each(
        "keyboard-teleop-xarm6 keyboard-teleop-xarm7 xarm7-planner-coordinator xarm7-planner-coordinator-agent "
        "keyboard-teleop-lite6 lite6-planner-coordinator keyboard-teleop-piper",
        _MOCK,
    ),
    # Real hardware unless --simulation, which switches to a MuJoCo simulator (G1: MuJoCo only).
    **_each(
        "coordinator-xarm7 coordinator-xarm6 xarm-grasp xarm-grasp-agent xarm-grasp-graspgenx "
        "xarm-grasp-graspgenx-agent unitree-g1-groot-wbc unitree-g1-teleop",
        _MUJOCO,
    ),
    # Simulation-only setups (they still need the runtime's simulation mode).
    **_each(
        "xarm-perception-sim xarm-perception-sim-agent unitree-g1-sim unitree-g1-basic-sim unitree-g1-agentic-sim",
        _SIM_ONLY,
    ),
    # Go2 setups built on the Go2 connection alone: MuJoCo or DimSim with --simulation, a dataset with --replay.
    **_each("unitree-go2-basic unitree-go2 teleop-webxr-go2", _GO2),
    # Drone: recorded MAVLink with --replay; otherwise it listens for a real drone on UDP 14550.
    **_each("drone-basic drone-agentic", _REPLAY),
    **_each("habitat-nav habitat-raycaster habitat-teleop habitat-voxel", _HABITAT),
    # No actuator and no robot connection in the setup (cameras and lidars may open).
    **_each(
        "demo-camera real-sense-camera-vis mid360 mid360-fastlio mid360-fastlio-ray-trace mid360-fastlio-voxels "
        "mid360-fastlio-voxels-native mid360-pointlio mid360-pointlio-voxels pointlio-rust pointlio-rust-replay "
        "demo-mid360-pcap-replay demo-virtual-mid360-fastlio demo-virtual-mid360-pointlio mid360-realsense-record "
        "mid360-realsense-record-with-pcap relocalize-mid360 demo-dim-slam-realsense demo-osm desk-marker-tf "
        "demo-object-scene-registration replay demo-agent demo-agent-camera demo-capabilities "
        "demo-google-maps-skill demo-gps-nav demo-skill demo-mcp-stress-test demo-error-on-name-conflicts "
        "teleop-webxr-rerun demo-pico-body-tracking",
        _NO_ACTUATORS,
    ),
}

# Setups that always open real hardware or join a robot's network, even with --simulation: listed so
# the README and `kiteos list` can say why they are --real only (the rule is the table above).
ALWAYS_REAL_NOTES: dict[str, str] = {
    "coordinator-piper-xarm": "opens a real Piper on can0 and a real xArm6",
    "coordinator-teleop-dual": "opens a real Piper on can0 and a real xArm6",
    "teleop-webxr-dual": "opens a real Piper on can0 and a real xArm6",
    "coordinator-cartesian-ik-piper": "opens a real Piper on can0",
    "unitree-g1-coordinator": "joins the G1's DDS network",
    "unitree-go2-keyboard-teleop": "drives the Go2 over DDS",
}


@dataclass(frozen=True)
class Setup:
    """One runnable setup: its registry name, where it lives, and how KiteOS may run it."""

    name: str
    target: str  # "module:attribute" in the runtime ("" for plugins)
    embodiment: Embodiment

    @property
    def module(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def hardware_free(self) -> HardwareFree | None:
        return HARDWARE_FREE.get(self.name)

    def summary(self) -> str:
        checked = self.hardware_free
        return checked.summary() if checked is not None else REAL_ONLY_SUMMARY


def _relative_module(target: str) -> str:
    module = target.split(":", 1)[0]
    return module[len("dimos.") :] if module.startswith("dimos.") else module


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def classify(name: str, target: str) -> Embodiment:
    """The embodiment of one registry entry: by name, then module, then name words, then family."""
    if "." in name:
        return PLUGINS
    for embodiment in EMBODIMENTS:
        if name in embodiment.setups:
            return embodiment
    module = _relative_module(target)
    for embodiment in EMBODIMENTS:
        if embodiment.robot and _under(module, embodiment.modules):
            return embodiment
    words = set(name.split("-"))
    for embodiment in EMBODIMENTS:
        if embodiment.robot and words & set(embodiment.words):
            return embodiment
    for embodiment in EMBODIMENTS:
        if not embodiment.robot and _under(module, embodiment.modules):
            return embodiment
    return OTHER


def read_registry(runtime_root: Path) -> dict[str, str]:
    """``{setup: "module:attribute"}`` from the runtime checkout's generated registry."""
    path = runtime_root / REGISTRY_FILE
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError, ValueError) as exc:
        raise RegistryError(f"cannot read the runtime's setup registry {path}: {exc}") from exc
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if not any(isinstance(t, ast.Name) and t.id == REGISTRY_TABLE for t in targets) or node.value is None:
            continue
        try:
            table = ast.literal_eval(node.value)
        except ValueError as exc:
            raise RegistryError(f"{path}: {REGISTRY_TABLE} is not a literal table") from exc
        if isinstance(table, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in table.items()):
            return dict(table)
        raise RegistryError(f"{path}: {REGISTRY_TABLE} is not a table of names")
    raise RegistryError(f"{path} has no {REGISTRY_TABLE} table")


def catalog(registry: Mapping[str, str], plugins: Mapping[str, str] | None = None) -> list[Setup]:
    """Every setup, classified, in embodiment order then by name."""
    entries = [Setup(name, target, classify(name, target)) for name, target in registry.items()]
    entries += [Setup(name, target, PLUGINS) for name, target in (plugins or {}).items()]
    order = {key: index for index, key in enumerate(BY_KEY)}
    return sorted(entries, key=lambda s: (order[s.embodiment.key], s.name))


def find_embodiment(query: str) -> Embodiment | None:
    """An embodiment by key, or by a unique case-insensitive match on its title."""
    wanted = query.strip().lower()
    if wanted in BY_KEY:
        return BY_KEY[wanted]
    matches = [e for e in BY_KEY.values() if wanted and wanted in e.title.lower()]
    return matches[0] if len(matches) == 1 else None
