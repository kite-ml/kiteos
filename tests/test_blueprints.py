# Copyright 2026 Kite ML.
"""KiteOS blueprint and rollout-module composition tests.

These need the KiteOS runtime virtualenv. Nothing is launched: blueprints are inspected
and parsed with the runtime's own CLI configuration parser, with model downloads
forbidden.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import pickle
import sys
import tomllib
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.runtime

pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.core.coordination.blueprint_config.errors import BlueprintConfigError  # noqa: E402
from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser  # noqa: E402
from dimos.core.coordination.blueprints import Blueprint, BlueprintAtom  # noqa: E402
from dimos.robot.external_blueprints import (  # noqa: E402
    ENTRY_POINT_GROUP,
    LOCAL_BLUEPRINT_NAME_PATTERN,
    canonicalize_distribution_namespace,
    is_valid_external_local_blueprint_name,
)
from dimos.robot.manipulators.dual_openyam.joints import (  # noqa: E402
    DUAL_OPENYAM_GRIPPER_JOINTS,
    DUAL_OPENYAM_JOINTS,
)
from dimos.utils.data import LfsPath  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from kiteos.rollout.module import (  # noqa: E402
    CAMERA_FEATURES,
    DualArmPolicyConfig,
    DualArmPolicyModule,
    kiteos_import_root,
    with_kiteos_pythonpath,
)

REPO = Path(__file__).resolve().parents[1]
CAMERA_ROLES = ("left_wrist", "right_wrist", "top")
EXPECTED_ENTRY_POINTS = {
    "dual-openyam-teleop": "kiteos.blueprints.teleop:dual_openyam_teleop",
    "dual-openyam-collection": "kiteos.blueprints.collection:dual_openyam_collection",
    "dual-openyam-rollout": "kiteos.blueprints.rollout:dual_openyam_rollout",
    "dual-openyam-rollout-recorded": "kiteos.blueprints.rollout:dual_openyam_rollout_recorded",
    "openarm-teleop": "kiteos.blueprints.openarm:openarm_teleop",
    "openarm-collection": "kiteos.blueprints.openarm:openarm_collection",
    "openarm-rollout": "kiteos.blueprints.openarm:openarm_rollout",
    "openarm-rollout-recorded": "kiteos.blueprints.openarm:openarm_rollout_recorded",
}
TELEOP_INSTANCES = {"armteleopmodule", "ControlCoordinator", "manipulationmodule"}
COLLECTION_INSTANCES = TELEOP_INSTANCES | set(CAMERA_ROLES) | {"recorder", "episodes"}
ROLLOUT_INSTANCES = TELEOP_INSTANCES | set(CAMERA_ROLES) | {"policy"}


def _blueprint_module(name: str) -> ModuleType:
    """Import ``kiteos.blueprints.<name>``, skipping while the fenced coordinator is missing."""
    try:
        return importlib.import_module(f"kiteos.blueprints.{name}")
    except ModuleNotFoundError as error:
        if error.name == "kiteos.safety.coordinator":
            pytest.skip(
                "kiteos.safety.coordinator is not written yet (built concurrently); "
                "blueprint tests need KiteDualOpenYamCoordinator"
            )
        raise


def _coordinator_class() -> type:
    _blueprint_module("teleop")
    from kiteos.safety.coordinator import KiteDualOpenYamCoordinator

    return KiteDualOpenYamCoordinator


def _atoms(blueprint: Blueprint) -> dict[str, BlueprintAtom]:
    return {atom.name: atom for atom in blueprint.active_blueprints}


def _task_names(atom: BlueprintAtom) -> list[str]:
    return [task.name for task in atom.kwargs["tasks"]]


def _parse(blueprint: Blueprint, *args: str) -> Any:
    return BlueprintConfigParser(blueprint).parse(list(args), environ={})


def _comparable(value: Any) -> Any:
    """Parsed kwargs with lazy LFS model paths replaced by their names.

    Comparing an ``LfsPath`` resolves (downloads) it, which these tests forbid.
    """
    if isinstance(value, LfsPath):
        return ("lfs", object.__getattribute__(value, "_lfs_filename"))
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("blueprint configuration must not download models")

    monkeypatch.setattr("dimos.utils.data.get_data", forbidden)


@pytest.fixture(scope="module")
def teleop() -> ModuleType:
    return _blueprint_module("teleop")


@pytest.fixture(scope="module")
def collection() -> ModuleType:
    return _blueprint_module("collection")


@pytest.fixture(scope="module")
def rollout() -> ModuleType:
    return _blueprint_module("rollout")


# --- entry points -----------------------------------------------------------


def _pyproject_entry_points() -> dict[str, str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["name"] == "kiteos"
    return dict(data["project"]["entry-points"][ENTRY_POINT_GROUP])


def test_pyproject_entry_points_follow_the_runtime_naming_rule():
    entry_points = _pyproject_entry_points()
    assert entry_points == EXPECTED_ENTRY_POINTS
    assert canonicalize_distribution_namespace("kiteos") == "kiteos"
    for name in entry_points:
        assert LOCAL_BLUEPRINT_NAME_PATTERN.fullmatch(name), name
        assert is_valid_external_local_blueprint_name(name), name


def test_entry_point_targets_are_blueprint_objects():
    _coordinator_class()
    for target in _pyproject_entry_points().values():
        module_name, attr = target.split(":")
        value = getattr(importlib.import_module(module_name), attr)
        # The runtime rejects factory functions; the target must be a Blueprint object.
        assert isinstance(value, Blueprint), target


def test_host_blueprints_never_import_the_isolated_runtime(rollout):
    # kiteos.rollout.runtime needs torch/LeRobot and must only load in the isolated env.
    assert "kiteos.rollout.runtime" not in sys.modules


# --- teleop -----------------------------------------------------------------


def test_teleop_uses_the_fenced_coordinator(teleop):
    coordinator_cls = _coordinator_class()
    atoms = _atoms(teleop.dual_openyam_teleop)
    assert set(atoms) == TELEOP_INSTANCES
    assert atoms["ControlCoordinator"].module is coordinator_cls
    assert _task_names(atoms["ControlCoordinator"]) == [
        "teleop_dual_openyam",
        "left_arm_gripper",
        "right_arm_gripper",
        "joint_trajectory",
    ]


def test_teleop_matches_the_runtime_blueprint_except_the_coordinator(teleop):
    from dimos.robot.manipulators.dual_openyam.blueprints.teleop import (
        teleop_webxr_dual_openyam,
    )

    ours, theirs = teleop.dual_openyam_teleop, teleop_webxr_dual_openyam
    assert dict(ours.remapping_map) == dict(theirs.remapping_map)
    assert [atom.name for atom in ours.active_blueprints] == [
        atom.name for atom in theirs.active_blueprints
    ]
    ours_parsed, theirs_parsed = _parse(ours), _parse(theirs)
    for name in TELEOP_INSTANCES:
        assert _comparable(ours_parsed.module_kwargs(name)) == _comparable(
            theirs_parsed.module_kwargs(name)
        ), name


def test_teleop_viser_is_local_by_default_and_can_be_opened(teleop):
    parsed = _parse(teleop.dual_openyam_teleop)
    assert parsed.module_kwargs("manipulationmodule")["visualization"]["host"] == "127.0.0.1"
    opened = _parse(
        teleop.dual_openyam_teleop, "--manipulationmodule.visualization.host", "0.0.0.0"
    )
    assert opened.module_kwargs("manipulationmodule")["visualization"]["host"] == "0.0.0.0"


def _fence_file(tmp_path: Path) -> str:
    path = tmp_path / "fence.yaml"
    path.write_text("version: 1\nzones: []\n")
    return str(path)


def test_teleop_coordinator_flags(teleop, tmp_path):
    fence = _fence_file(tmp_path)
    parsed = _parse(
        teleop.dual_openyam_teleop,
        "--controlcoordinator.left-can-port",
        "can_left",
        "--controlcoordinator.right-can-port",
        "can_right",
        "--controlcoordinator.fence-path",
        fence,
    )
    config = parsed.module_kwargs("ControlCoordinator")
    assert config["left_can_port"] == "can_left"
    assert config["right_can_port"] == "can_right"
    assert config["fence_path"] == fence


# --- collection ---------------------------------------------------------------

COLLECTION_ARGS = (
    "--recorder.recording",
    "/tmp/kiteos-test-recording",
    "--episodes.task",
    "close the box",
)


def test_collection_composition(collection):
    coordinator_cls = _coordinator_class()
    from dimos.hardware.sensors.camera.module import CameraModule
    from dimos.imitation.collection.episode_monitor import EpisodeMonitorModule

    from kiteos.blueprints.cameras import KiteCameraModule

    atoms = _atoms(collection.dual_openyam_collection)
    assert set(atoms) == COLLECTION_INSTANCES
    assert atoms["ControlCoordinator"].module is coordinator_cls
    assert atoms["episodes"].module is EpisodeMonitorModule
    for role in CAMERA_ROLES:
        camera = atoms[role]
        # The runtime's camera module, plus the synthetic device "mock" for runs without cameras.
        assert camera.module is KiteCameraModule and issubclass(KiteCameraModule, CameraModule)
        hardware = camera.kwargs["hardware"]
        assert (hardware.width, hardware.height, hardware.fps) == (640, 480, 30)
        assert hardware.frame_id_prefix == f"{role}_image"
        assert camera.kwargs["frame_id"] == f"{role}_camera_link"
    recorder_inputs = {
        port.name
        for port in atoms["recorder"].streams
        if port.direction == "in" and port.name != "tf"
    }
    assert recorder_inputs == {
        "left_wrist_image",
        "right_wrist_image",
        "top_image",
        "coordinator_joint_state",
        "applied_joint_position_command",
        "status",
    }


def test_collection_mirrors_the_runtime_blueprint(collection):
    from dimos.robot.manipulators.dual_openyam.blueprints.learning_collection import (
        dual_openyam_quest_collection,
    )

    ours, theirs = collection.dual_openyam_collection, dual_openyam_quest_collection
    assert dict(ours.remapping_map) == dict(theirs.remapping_map)
    assert set(_atoms(ours)) == set(_atoms(theirs))
    ours_parsed, theirs_parsed = _parse(ours, *COLLECTION_ARGS), _parse(theirs, *COLLECTION_ARGS)
    for name in COLLECTION_INSTANCES:
        ours_kwargs = _comparable(ours_parsed.module_kwargs(name))
        theirs_kwargs = _comparable(theirs_parsed.module_kwargs(name))
        if name == "manipulationmodule":
            # KiteOS keeps viser local; the runtime's collection opens it to the LAN.
            assert ours_kwargs["visualization"]["host"] == "127.0.0.1"
            assert theirs_kwargs["visualization"]["host"] == "0.0.0.0"
            ours_kwargs["visualization"]["host"] = theirs_kwargs["visualization"]["host"]
        if name == "armteleopmodule":
            # KiteOS binds the WebXR server to loopback (the Quest comes in over USB).
            assert ours_kwargs.pop("server_host", "127.0.0.1") == "127.0.0.1"
            ours_kwargs.pop("instance_name", None)
            theirs_kwargs.pop("instance_name", None)
        if name in collection.CAMERA_ROLES:
            # KiteOS defaults to the lab rig's mapping (the runtime enumerates the roles 0, 1, 2).
            assert ours_kwargs["hardware"]["camera_index"] == collection.DEFAULT_CAMERA_INDICES[name]
            ours_kwargs["hardware"]["camera_index"] = theirs_kwargs["hardware"]["camera_index"]
        assert ours_kwargs == theirs_kwargs, name


def test_collection_takes_station_devices_by_flag(collection, tmp_path):
    fence = _fence_file(tmp_path)
    parsed = _parse(
        collection.dual_openyam_collection,
        *COLLECTION_ARGS,
        "--controlcoordinator.left-can-port",
        "006C003E4845500F20303750",
        "--controlcoordinator.right-can-port",
        "003900374845500F20303750",
        "--controlcoordinator.fence-path",
        fence,
        "--left-wrist.hardware.camera-index",
        "0",
        "--right-wrist.hardware.camera-index",
        "/dev/v4l/by-path/right-video-index0",
        "--top.hardware.camera-index",
        "1",
    )
    assert parsed.module_kwargs("left_wrist")["hardware"]["camera_index"] == 0
    assert (
        parsed.module_kwargs("right_wrist")["hardware"]["camera_index"]
        == "/dev/v4l/by-path/right-video-index0"
    )
    assert parsed.module_kwargs("top")["hardware"]["camera_index"] == 1
    assert parsed.module_kwargs("ControlCoordinator")["fence_path"] == fence
    assert parsed.module_kwargs("episodes")["task"] == "close the box"


def test_collection_requires_task_and_recording(collection):
    with pytest.raises(BlueprintConfigError):
        _parse(collection.dual_openyam_collection)


# --- rollout ------------------------------------------------------------------

ROLLOUT_ARGS = (
    "--policy.policy-path",
    "/nonexistent/act-checkpoint",
    "--policy.task",
    "close the box",
)


def test_rollout_composition(rollout):
    coordinator_cls = _coordinator_class()
    from dimos.teleop.webxr.extensions import ArmTeleopModule

    from kiteos.blueprints.webxr import KiteArmTeleopModule

    atoms = _atoms(rollout.dual_openyam_rollout)
    assert set(atoms) == ROLLOUT_INSTANCES
    assert atoms["ControlCoordinator"].module is coordinator_cls
    assert atoms["armteleopmodule"].module is KiteArmTeleopModule
    assert issubclass(KiteArmTeleopModule, ArmTeleopModule)
    assert atoms["policy"].module is DualArmPolicyModule

    tasks = {task.name: task for task in atoms["ControlCoordinator"].kwargs["tasks"]}
    # No Quest gripper tasks: the policy owns the grippers through the trajectory task.
    assert set(tasks) == {"teleop_dual_openyam", "joint_trajectory"}
    assert tasks["teleop_dual_openyam"].priority == 20
    trajectory = tasks["joint_trajectory"]
    assert trajectory.joint_names == list(DUAL_OPENYAM_JOINTS)
    assert trajectory.priority == 10
    assert trajectory.params["start_position_tolerance"] == 0.05


def test_rollout_streams_reach_the_policy(rollout):
    blueprint = rollout.dual_openyam_rollout
    atoms = _atoms(blueprint)
    policy_inputs = {port.name for port in atoms["policy"].streams if port.direction == "in"}
    assert set(CAMERA_FEATURES) <= policy_inputs
    for role in CAMERA_ROLES:
        assert blueprint.remapping_map[(role, "color_image")] == f"{role}_image"
        assert f"{role}_image" in policy_inputs
    assert {"coordinator_joint_state", "teleop_buttons", "button_pressed"} <= policy_inputs
    assert blueprint.remapping_map[("armteleopmodule", "left_controller_output")] == (
        "left_cartesian_command"
    )
    assert blueprint.remapping_map[("armteleopmodule", "right_controller_output")] == (
        "right_cartesian_command"
    )


def test_rollout_policy_path_is_a_run_time_flag(rollout):
    blueprint = rollout.dual_openyam_rollout
    policy = _atoms(blueprint)["policy"].kwargs
    assert {"policy_path", "task", "device"}.isdisjoint(policy)
    assert policy["joint_names"] == list(DUAL_OPENYAM_JOINTS)
    assert policy["gripper_joints"] == list(DUAL_OPENYAM_GRIPPER_JOINTS)

    with pytest.raises(BlueprintConfigError, match="policy_path"):
        _parse(blueprint, "--policy.task", "close the box")

    parsed = _parse(
        blueprint,
        *ROLLOUT_ARGS,
        "--policy.device",
        "mps",
        "--policy.fps",
        "15",
        "--policy.max-start-jump-rad",
        "0.3",
    )
    config = parsed.module_kwargs("policy")
    assert config["policy_path"] == "/nonexistent/act-checkpoint"
    assert config["task"] == "close the box"
    assert config["device"] == "mps"
    assert config["fps"] == 15.0
    assert config["max_start_jump_rad"] == 0.3
    assert config["robot_type"] == "dual_openyam"
    assert (config["image_width"], config["image_height"]) == (640, 480)


def test_rollout_guard_flags_are_bounded(rollout):
    with pytest.raises(BlueprintConfigError, match="max_joint_speed_rad_s"):
        _parse(rollout.dual_openyam_rollout, *ROLLOUT_ARGS, "--policy.max-joint-speed-rad-s", "5")


def test_rollout_cameras_come_from_flags_not_hard_coded(rollout, collection):
    blueprint = rollout.dual_openyam_rollout
    atoms = _atoms(blueprint)
    defaults = {role: atoms[role].kwargs["hardware"].camera_index for role in CAMERA_ROLES}
    assert defaults == collection.DEFAULT_CAMERA_INDICES
    parsed = _parse(
        blueprint,
        *ROLLOUT_ARGS,
        "--left-wrist.hardware.camera-index",
        "0",
        "--right-wrist.hardware.camera-index",
        "2",
        "--top.hardware.camera-index",
        "1",
    )
    assert [parsed.module_kwargs(role)["hardware"]["camera_index"] for role in CAMERA_ROLES] == [
        0,
        2,
        1,
    ]


def test_rollout_viser_is_local(rollout):
    parsed = _parse(rollout.dual_openyam_rollout, *ROLLOUT_ARGS)
    assert parsed.module_kwargs("manipulationmodule")["visualization"]["host"] == "127.0.0.1"


def test_recorded_rollout_adds_recorder_and_episodes(rollout):
    atoms = _atoms(rollout.dual_openyam_rollout_recorded)
    assert set(atoms) == ROLLOUT_INSTANCES | {"recorder", "episodes"}
    parsed = _parse(rollout.dual_openyam_rollout_recorded, *ROLLOUT_ARGS, *COLLECTION_ARGS)
    assert parsed.module_kwargs("episodes")["task"] == "close the box"


def test_rollout_builder_pins_values(rollout):
    blueprint = rollout.build_kiteos_dual_openyam_rollout(
        policy_path="/ckpt", task="t", device="cpu", fps=15.0, camera_indices={"top": "/dev/video4"}
    )
    atoms = _atoms(blueprint)
    assert atoms["policy"].kwargs["policy_path"] == "/ckpt"
    assert atoms["policy"].kwargs["fps"] == 15.0
    assert atoms["top"].kwargs["hardware"].camera_index == "/dev/video4"
    with pytest.raises(ValueError, match="unknown camera roles"):
        rollout.build_kiteos_dual_openyam_rollout(camera_indices={"wrist": 0})


def test_policy_config_crosses_the_process_boundary(rollout):
    # The host pickles config.model_dump() to the isolated runtime, which re-validates it.
    parsed = _parse(rollout.dual_openyam_rollout, *ROLLOUT_ARGS)
    dumped = DualArmPolicyConfig(**parsed.module_kwargs("policy")).model_dump()
    restored = DualArmPolicyConfig(**pickle.loads(pickle.dumps(dumped)))
    assert restored.gripper_joints == list(DUAL_OPENYAM_GRIPPER_JOINTS)
    assert restored.max_joint_speed_rad_s == 3.0


# --- rollout policy module ------------------------------------------------------


def _policy_config(**overrides: Any) -> DualArmPolicyConfig:
    values: dict[str, Any] = {
        "policy_path": "/nonexistent/ckpt",
        "task": "close the box",
        "joint_names": list(DUAL_OPENYAM_JOINTS),
        "gripper_joints": list(DUAL_OPENYAM_GRIPPER_JOINTS),
    }
    values.update(overrides)
    return DualArmPolicyConfig(**values)


def test_policy_module_points_at_the_kiteos_runtime():
    assert DualArmPolicyModule.implementation == "kiteos.rollout.runtime:DualArmPolicyRuntime"
    assert DualArmPolicyModule.project_dir == "native/python/lerobot"


def test_policy_guard_defaults():
    config = _policy_config()
    assert config.max_joint_speed_rad_s == 3.0
    assert config.max_start_jump_rad == 0.3
    assert config.max_step_delta_rad == 0.1
    assert config.max_gripper_start_jump == 1.0
    assert config.max_gripper_speed == 60.0
    assert config.blend_steps == 10


@pytest.mark.parametrize(
    "overrides",
    [
        {"gripper_joints": ["not_a_joint"]},
        {"gripper_joints": ["left_arm/gripper", "left_arm/gripper"]},
        {"joint_names": ["left_arm/gripper"], "gripper_joints": ["left_arm/gripper"]},
        {"max_joint_speed_rad_s": 0.0},
        {"max_joint_speed_rad_s": 3.5},
        {"max_start_jump_rad": 0.6},
        {"max_step_delta_rad": 0.5},
        {"max_gripper_start_jump": 1.5},
        {"max_gripper_speed": -1.0},
        {"blend_steps": 0},
    ],
)
def test_policy_config_rejects_unsafe_or_inconsistent_values(overrides):
    with pytest.raises(ValidationError):
        _policy_config(**overrides)


def test_policy_config_accepts_tighter_guards():
    config = _policy_config(max_joint_speed_rad_s=1.0, max_gripper_speed=1.0, blend_steps=20)
    assert config.max_joint_speed_rad_s == 1.0


def test_kiteos_import_root_is_this_source_tree():
    root = kiteos_import_root()
    assert root is not None
    assert (Path(root) / "kiteos" / "__init__.py").is_file()
    assert Path(root).name not in {"site-packages", "dist-packages"}


def test_runtime_pythonpath_prepends_the_kiteos_root_once():
    sep = os.pathsep
    assert with_kiteos_pythonpath({}, "/r")["PYTHONPATH"] == "/r"
    env = {"PYTHONPATH": f"/a{sep}/r{sep}/b", "OTHER": "1"}
    result = with_kiteos_pythonpath(env, "/r")
    assert result["PYTHONPATH"] == f"/r{sep}/a{sep}/b"
    assert result["OTHER"] == "1"
    assert env["PYTHONPATH"] == f"/a{sep}/r{sep}/b"


def test_runtime_pythonpath_skips_site_packages(monkeypatch):
    import kiteos
    import kiteos.rollout.module as module

    fake = Path("/venv/lib/python3.12/site-packages/kiteos/__init__.py")
    monkeypatch.setattr(kiteos, "__file__", str(fake))
    assert module.kiteos_import_root() is None
    assert module.with_kiteos_pythonpath({"PYTHONPATH": "/a"}) == {"PYTHONPATH": "/a"}


def test_policy_module_exports_kiteos_to_the_isolated_runtime(monkeypatch):
    # Built without __init__ so no RPC transport is started.
    policy = object.__new__(DualArmPolicyModule)
    policy.__dict__["config"] = _policy_config(extra_env={"KITEOS_TEST": "1"})
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    env = policy._runtime_env()
    assert env["PYTHONPATH"].split(os.pathsep) == [kiteos_import_root(), "/elsewhere"]
    assert env["KITEOS_TEST"] == "1"
    assert "UV_PROJECT_ENVIRONMENT" in env


def test_isolated_runtime_satisfies_the_contract():
    """Only runs where dimos_lerobot, LeRobot and torch are importable (the isolated env)."""
    pytest.importorskip("dimos_lerobot.runtime", reason="needs the runtime's isolated LeRobot env")
    from dimos.experimental.isolated_python.bootstrap import load_class, validate_runtime

    runtime = load_class(DualArmPolicyModule.implementation)
    validate_runtime(DualArmPolicyModule, runtime)



def test_webxr_server_binds_loopback_by_default():
    import threading

    from kiteos.blueprints.webxr import KiteArmTeleopConfig, KiteArmTeleopModule

    assert KiteArmTeleopConfig().server_host == "127.0.0.1"

    class FakeServer:
        host = "0.0.0.0"  # what the runtime's start() constructs
        ran = threading.Event()

        def run(self, **_: object) -> None:
            self.bound = self.host
            self.ran.set()

    module = object.__new__(KiteArmTeleopModule)
    module.config = KiteArmTeleopConfig()
    module._web_server = FakeServer()
    module._web_server_thread = None
    module._start_server()
    assert module._web_server.ran.wait(5.0)
    assert module._web_server.bound == "127.0.0.1"


def test_rollout_trajectory_task_caps_execution_explicitly(rollout):
    atoms = _atoms(rollout.dual_openyam_rollout)
    tasks = {task.name: task for task in atoms["ControlCoordinator"].kwargs["tasks"]}
    limits = tasks["joint_trajectory"].params["velocity_limits"]
    assert set(limits) == set(DUAL_OPENYAM_JOINTS) and set(limits.values()) == {1.0}


def test_recorder_shutdown_timeout_flag_reaches_the_recorder(collection, tmp_path):
    from kiteos.cli.launch import RECORDER_SHUTDOWN_TIMEOUT_S, collection_flags

    flags = collection_flags(tmp_path / "rec", "close the box")
    parsed = _parse(collection.dual_openyam_collection, *flags)
    assert parsed.module_kwargs("recorder")["shutdown_timeout"] == RECORDER_SHUTDOWN_TIMEOUT_S
    assert RECORDER_SHUTDOWN_TIMEOUT_S < 120.0  # below the runtime's module stop RPC timeout


def test_cli_safety_flags_beat_environment_and_config_file(teleop, tmp_path):
    """The CLI always passes require-fence and expect-mock: env/config-file values cannot win."""
    import json

    from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser

    config = tmp_path / "config"
    config.write_text(json.dumps({"controlcoordinator": {"left_can_port": "L", "right_can_port": "R"}}))
    parsed = BlueprintConfigParser(teleop.dual_openyam_teleop).parse(
        ["--controlcoordinator.require-fence-on-hardware", "true", "--controlcoordinator.expect-mock", "true"],
        config_path=config,
        environ={"CONTROLCOORDINATOR__REQUIRE_FENCE_ON_HARDWARE": "false", "CONTROLCOORDINATOR__EXPECT_MOCK": "false"},
    )
    cc = parsed.module_kwargs("ControlCoordinator")
    assert cc["require_fence_on_hardware"] is True and cc["expect_mock"] is True
    # With expect_mock the coordinator refuses the real adapter these CAN ports would select
    # (tests/test_coordinator.py::test_expect_mock_refuses_real_can_ports_from_any_source).
    assert cc["left_can_port"] == "L"
