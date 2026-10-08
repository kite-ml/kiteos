# Copyright 2026 Kite ML.
"""The thin KiteOS one-arm OpenYAM setups: the runtime's one-arm Quest setups, nearly unchanged.

They must stay the runtime's composition except for the loopback Quest server, the top camera in
the recording and the CAN port as a coordinator option, and ``kiteos run`` must start them with
the gates of the runtime's own OpenYAM setups (no CAN port without --real).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.runtime
pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.robot.manipulators.openyam.blueprints.teleop import (  # noqa: E402
    OPENYAM_QUEST_KINEMATICS,
    OPENYAM_QUEST_MODEL,
    openyam_quest_tasks,
)
from dimos.robot.manipulators.openyam.collection import OPENYAM_QUEST_COLLECTION  # noqa: E402
from dimos.robot.manipulators.openyam.config import OPENYAM_JOINTS, openyam_hardware  # noqa: E402

from kiteos import embodiments  # noqa: E402
from kiteos.blueprints import openyam  # noqa: E402
from kiteos.blueprints.cameras import KiteCameraModule  # noqa: E402
from kiteos.blueprints.webxr import KiteArmTeleopModule  # noqa: E402
from kiteos.cli import setups  # noqa: E402
from kiteos.config import StationConfig  # noqa: E402
from kiteos.robots import MOCK_ADAPTER  # noqa: E402
from kiteos.safety.openyam_coordinator import (  # noqa: E402
    KiteOpenYamCoordinator,
    KiteOpenYamCoordinatorConfig,
    openyam_arm_hardware,
)

REPO = Path(__file__).resolve().parents[1]
CAN_PORT = "006C003E4845500F20303750"


def _atoms(blueprint: object) -> dict[str, object]:
    return {atom.name: atom for atom in blueprint.active_blueprints}  # type: ignore[attr-defined]


def test_teleop_is_the_runtime_one_arm_teleop_with_the_loopback_quest_server() -> None:
    atoms = _atoms(openyam.openyam_teleop)
    assert set(atoms) == {"armteleopmodule", "ControlCoordinator", "manipulationmodule"}
    assert atoms["armteleopmodule"].module is KiteArmTeleopModule  # binds 127.0.0.1
    coordinator = atoms["ControlCoordinator"]
    assert coordinator.module is KiteOpenYamCoordinator
    ours, theirs = coordinator.kwargs["tasks"], openyam_quest_tasks()
    assert [(t.name, t.type, list(t.joint_names), t.priority) for t in ours] == [
        (t.name, t.type, list(t.joint_names), t.priority) for t in theirs
    ]
    manipulation = atoms["manipulationmodule"].kwargs
    assert manipulation["model"] is OPENYAM_QUEST_MODEL and manipulation["kinematics"] is OPENYAM_QUEST_KINEMATICS
    remapping = dict(openyam.openyam_teleop.remapping_map)
    assert remapping[("armteleopmodule", "right_controller_output")] == "right_cartesian_command"


def test_collection_records_the_runtime_profile_plus_the_top_camera() -> None:
    atoms = _atoms(openyam.openyam_collection)
    assert {"wrist", "top", "recorder", "episodes"} <= set(atoms)
    for role in ("wrist", "top"):
        assert atoms[role].module is KiteCameraModule
    profile = openyam.OPENYAM_COLLECTION
    assert profile.robot_type == "openyam"
    assert set(profile.observations) == {*OPENYAM_QUEST_COLLECTION.observations, "observation.images.top"}
    assert profile.observations["observation.images.top"].stream == "top_image"
    assert profile.actions == OPENYAM_QUEST_COLLECTION.actions and profile.sync == OPENYAM_QUEST_COLLECTION.sync
    remapping = dict(openyam.openyam_collection.remapping_map)
    assert remapping[("wrist", "color_image")] == "wrist_image" and remapping[("top", "color_image")] == "top_image"


def test_the_arm_is_mock_unless_a_can_port_is_given() -> None:
    runtime = openyam_hardware()
    mock = openyam_arm_hardware(None)
    assert mock.adapter_type == MOCK_ADAPTER and mock.adapter_kwargs == {}
    assert mock.limits is not None and mock.limits.position_upper[-1] == 1.0
    real = openyam_arm_hardware(CAN_PORT)
    assert real.adapter_type == "openyam_damiao" and real.limits is None
    config = real.adapter_kwargs["runtime_config"]
    assert config.bus_devices == {"openyam": CAN_PORT} and config.gravity_comp
    for built in (mock, real):  # the runtime's joints, gains and id
        assert list(built.joints) == list(OPENYAM_JOINTS) == list(runtime.joints)
        assert built.wb_config == runtime.wb_config and built.hardware_id == runtime.hardware_id
    assert KiteOpenYamCoordinatorConfig().can_port is None


def test_the_one_arm_coordinator_module_imports_no_webxr_server() -> None:
    code = (
        "import sys, kiteos.safety.openyam_coordinator\n"
        "print([m for m in ('fastapi', 'dimos.teleop.webxr.module', 'dimos.teleop.webxr.extensions') if m in sys.modules])"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


def _station(tmp_path: Path) -> StationConfig:
    root = tmp_path / "kiteos-runtime"
    root.mkdir()
    return StationConfig.model_validate(
        {
            "station": "lab",
            "task": "t",
            "can": {"left": {"darwin": "a", "linux": "can_left"}, "right": {"darwin": "b", "linux": "can_right"}},
            "cameras": {r: {"darwin": i, "linux": f"/dev/video{i}"} for i, r in enumerate(("left_wrist", "right_wrist", "top"))},
            "hub": {"repo_id": "org/repo"},
            "runtime_root": str(root),
            "recordings_dir": str(tmp_path / "recordings"),
            "datasets_dir": str(tmp_path / "datasets"),
        }
    )


def test_kiteos_run_starts_them_with_the_runtime_openyam_gates(tmp_path: Path) -> None:
    station = _station(tmp_path)
    setup = setups.resolve_setup("kiteos.openyam-collection", {})
    assert setup.embodiment.key == "openyam" and not setup.embodiment.fenced
    assert {s.name for s in embodiments.catalog({})} >= set(embodiments.KITEOS_RUN_SETUPS)
    with pytest.raises(setups.SetupRunError, match="can-port"):  # no CAN port on a mock run
        setups.plan_setup_run(
            station, setup, mode="default", extra=["--controlcoordinator.can-port", CAN_PORT], environ={}, private_bus={"X": "1"}
        )
    mock = setups.plan_setup_run(station, setup, mode="default", environ={}, private_bus={"LCM_DEFAULT_URL": "udpm://x"})
    assert mock.argv[-2:] == ["run", "kiteos.openyam-collection"] and not mock.real
    real = setups.plan_setup_run(station, setup, mode="real", extra=["--controlcoordinator.can-port", CAN_PORT], environ={})
    assert real.real and real.argv[-2:] == ["--controlcoordinator.can-port", CAN_PORT]
    assert any("fence does not cover" in note for note in real.notes)
    with pytest.raises(setups.SetupRunError, match="KiteOS setup"):  # the fenced ones keep their commands
        setups.resolve_setup("kiteos.dual-openyam-teleop", {})
