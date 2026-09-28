# Copyright 2026 Kite ML.
"""The OpenArm as a KiteOS robot: model, fence, adapter, coordinator, setups and CLI.

The fence model is checked against pinocchio (FK of every link) and against the OpenArm meshes
(every vertex inside the covering spheres), when the runtime and its cached
``enactic/openarm_description`` checkout are available. Everything runs with mock hardware; a
"real" adapter is a stub, as in tests/test_coordinator.py.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from kiteos.cli.runs import RunState
from kiteos.robots import DUAL_OPENYAM, OPENARM, get_robot, robot_for_setup
from kiteos.safety.fence import FenceChecker, FenceConfig
from kiteos.safety.kinematics import DualArmKinematics, default_urdf_path, parse_urdf_joints, read_urdf_text

REPO = Path(__file__).resolve().parents[1]
ZERO = np.zeros(7)
# Shoulder pitch -90 degrees: the arm points straight forward (+x), about 0.57 m out.
REACH = {"left": np.array([-1.57, 0, 0, 0, 0, 0, 0.0]), "right": np.array([1.57, 0, 0, 0, 0, 0, 0.0])}
# Shoulder pitch about -57 degrees: the fingers are 8 cm through the 0.45 m glass plane.
THROUGH = {"left": np.array([-1.0, 0, 0, 0, 0, 0, 0.0]), "right": np.array([1.0, 0, 0, 0, 0, 0, 0.0])}
GLASS = {"kind": "half_space", "name": "glass", "frame": "world", "point": [0.45, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]}


def openarm_fence(**kw: Any) -> FenceConfig:
    data = {"robot": "openarm", "margin_m": 0.03, "zones": [GLASS], **kw}
    return FenceConfig.model_validate(data)


# =========================================================================== profile and model


def test_profile_joint_names_follow_the_runtime_order() -> None:
    assert OPENARM.dof == 7
    assert OPENARM.arm_joints["left"] == tuple(f"openarm_left_joint{i}" for i in range(1, 8))
    assert OPENARM.joint_names[-2:] == ["left_arm/gripper", "right_arm/gripper"]
    assert len(OPENARM.joint_names) == 16
    assert get_robot("openarm") is OPENARM and get_robot(None) is DUAL_OPENYAM
    assert robot_for_setup("kiteos.openarm-collection") is OPENARM
    assert robot_for_setup("dual-openyam-teleop") is DUAL_OPENYAM
    assert robot_for_setup("unitree-go2") is None
    with pytest.raises(ValueError, match="unknown robot"):
        get_robot("zeppelin")


@pytest.mark.runtime
def test_profile_matches_the_runtime_openarm_config() -> None:
    config = pytest.importorskip("dimos.robot.manipulators.openarm.config")
    assert list(OPENARM.joint_names) == list(config.OPENARM_JOINTS)
    assert OPENARM.hold_kp == config._ARM_KP and OPENARM.hold_kd == config._ARM_KD
    assert OPENARM.dof == config.OPENARM_DOF


def test_vendored_urdf_parses_into_both_seven_joint_chains() -> None:
    kinematics = DualArmKinematics(robot=OPENARM)
    for side in ("left", "right"):
        chain = kinematics.chain(side)
        assert chain.joint_names == list(OPENARM.arm_joints[side])
        assert chain.dof == 7
        for link in OPENARM.required_links:
            assert link in chain.link_index
    # URDF mounts: the shoulders, 0.698 m above the bottom of the body.
    np.testing.assert_allclose(kinematics.mounts["left"][:3, 3], (0.0, 0.031, 0.698))
    np.testing.assert_allclose(kinematics.mounts["right"][:3, 3], (0.0, -0.031, 0.698))


def test_vendored_urdf_has_portable_mesh_paths_and_provenance() -> None:
    text = Path(default_urdf_path(OPENARM)).read_text()
    assert "1fba2cbc05001f05b4514120b70130b4ac06f409" in text and "Apache" in text
    assert "/Users/" not in text and "package://openarm_description/" in text


def _runtime_openarm_xml() -> str | None:
    """The runtime's expanded OpenArm model, only if its description is already cached."""
    try:
        from dimos.robot.manipulators.openarm.config import OPENARM_BIMANUAL_MODEL, OPENARM_DESCRIPTION_SOURCE
    except ImportError:
        return None
    cache = Path.home() / ".cache" / "dimos" / "robot_assets" / "sources"
    if not any(cache.glob("*/openarm_description")):
        return None  # never download from the test suite
    try:
        OPENARM_DESCRIPTION_SOURCE.checkout_path()
        return OPENARM_BIMANUAL_MODEL.load().xml
    except Exception:
        return None


@pytest.mark.runtime
def test_vendored_urdf_matches_the_runtime_model() -> None:
    xml = _runtime_openarm_xml()
    if xml is None:
        pytest.skip("the runtime's OpenArm description is not cached here")
    ours = parse_urdf_joints(read_urdf_text(robot=OPENARM))
    theirs = parse_urdf_joints(xml)
    assert set(ours) == set(theirs)
    for name, joint in theirs.items():
        mine = ours[name]
        assert (mine.type, mine.parent, mine.child) == (joint.type, joint.parent, joint.child), name
        np.testing.assert_allclose(mine.xyz, joint.xyz, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(mine.rpy, joint.rpy, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(mine.axis, joint.axis, atol=1e-12, err_msg=name)
        assert (mine.lower, mine.upper) == (joint.lower, joint.upper), name


@pytest.mark.runtime
def test_forward_kinematics_matches_pinocchio() -> None:
    pin = pytest.importorskip("pinocchio")
    xml = read_urdf_text(robot=OPENARM)
    model = pin.buildModelFromXML(xml)
    data = model.createData()
    kinematics = DualArmKinematics(robot=OPENARM)
    rng = np.random.default_rng(7)
    for side in ("left", "right"):
        chain = kinematics.chain(side)
        limits = chain.joint_limits
        for _ in range(25):
            q = rng.uniform(limits[:, 0], limits[:, 1])
            full = pin.neutral(model)
            for i, name in enumerate(OPENARM.arm_joints[side]):
                full[model.joints[model.getJointId(name)].idx_q] = q[i]
            pin.forwardKinematics(model, data, full)
            pin.updateFramePlacements(model, data)
            ours = kinematics.link_poses(side, q)
            for link in chain.link_names:
                frame = data.oMf[model.getFrameId(f"openarm_{side}_{link}")]
                np.testing.assert_allclose(ours[link][:3, 3], frame.translation, atol=1e-9, err_msg=link)
                np.testing.assert_allclose(ours[link][:3, :3], frame.rotation, atol=1e-9, err_msg=link)


def test_doctor_fk_self_test_passes_for_openarm() -> None:
    from kiteos.cli import doctor

    (check,) = doctor.check_fk(OPENARM)
    assert check.level == "PASS", check


def test_sphere_table_is_complete_and_sane() -> None:
    for side in ("left", "right"):
        table = OPENARM.link_spheres(side)
        assert set(table) == {"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "ee_base_link", "ee_link1", "ee_link2"}
        for link, spheres in table.items():
            assert spheres, link
            for center, radius in spheres:
                assert 0.005 < radius < 0.2, (link, radius)
                assert float(np.linalg.norm(center)) < 0.3, (link, center)
    assert {link for link, _ in OPENARM.finger_spheres} == {"ee_link1", "ee_link2"}
    assert 0.0 < OPENARM.gripper_stroke_m < 0.1


def _mesh_points(xml: str, link_name: str) -> np.ndarray | None:
    """Every vertex of a link's collision (and visual, when loadable) meshes, in the link frame."""
    trimesh = pytest.importorskip("trimesh")
    from kiteos.safety.kinematics import pose_matrix

    root = ET.fromstring(xml)
    link = next(e for e in root.findall("link") if e.get("name") == link_name)
    points = []
    for element in [*link.findall("collision"), *link.findall("visual")]:
        mesh = element.find("geometry/mesh")
        if mesh is None:
            continue
        path = Path(mesh.get("filename", ""))
        if not path.is_file():
            continue
        try:
            loaded = trimesh.load(path, force="mesh")
        except Exception:
            continue  # .dae needs pycollada
        scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
        origin = element.find("origin")
        T = pose_matrix(
            [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()],
            [float(v) for v in (origin.get("rpy") if origin is not None else "0 0 0").split()],
        )
        v = np.asarray(loaded.vertices, dtype=float) * scale
        points.append(v @ T[:3, :3].T + T[:3, 3])
    return np.vstack(points) if points else None


@pytest.mark.runtime
def test_spheres_cover_every_mesh_vertex() -> None:
    xml = _runtime_openarm_xml()
    if xml is None:
        pytest.skip("the runtime's OpenArm description is not cached here")
    for side in ("left", "right"):
        for link, spheres in OPENARM.link_spheres(side).items():
            points = _mesh_points(xml, f"openarm_{side}_{link}")
            assert points is not None, link
            centers = np.array([c for c, _ in spheres])
            radii = np.array([r for _, r in spheres])
            excess = np.min(np.linalg.norm(points[:, None, :] - centers[None], axis=2) - radii[None], axis=1)
            assert float(np.max(excess)) <= 1e-6, (side, link, float(np.max(excess)))


# =========================================================================== fence config and checker


def test_fence_config_defaults_come_from_the_robot() -> None:
    config = FenceConfig.model_validate({"robot": "openarm"})
    assert config.mounts["left"].xyz == (0.0, 0.031, 0.698)
    assert config.gripper_stroke_m == pytest.approx(OPENARM.gripper_stroke_m)
    dual = FenceConfig()
    assert dual.robot == "dual-openyam" and dual.mounts["left"].xyz == (0.0, 0.31, 0.0)
    assert dual.gripper_stroke_m == 0.05


def test_openarm_zones_default_to_the_world_frame_and_reject_other_robots_frames() -> None:
    zone = {"kind": "half_space", "name": "g", "point": [0.45, 0, 0], "normal": [1, 0, 0]}
    config = FenceConfig.model_validate({"robot": "openarm", "zones": [zone]})
    assert config.zones[0].frame == "world"
    for frame in ("openarm_body_link0", "left_base", "right_base"):
        FenceConfig.model_validate({"robot": "openarm", "zones": [{**zone, "frame": frame}]})
    with pytest.raises(ValueError, match="not a frame of the OpenArm"):
        FenceConfig.model_validate({"robot": "openarm", "zones": [{**zone, "frame": "dual_openyam_base"}]})
    with pytest.raises(ValueError, match="not a frame of the Dual OpenYAM"):
        FenceConfig.model_validate({"zones": [{**zone, "frame": "openarm_body_link0"}]})


def test_extra_sphere_links_are_checked_against_the_robot() -> None:
    cam = {"name": "cam", "side": "both", "center": [0.05, 0, -0.03], "radius": 0.03}
    FenceConfig.model_validate({"robot": "openarm", "extra_spheres": [{**cam, "link": "ee_base_link"}]})
    with pytest.raises(ValueError, match="is not a OpenArm"):
        FenceConfig.model_validate({"robot": "openarm", "extra_spheres": [{**cam, "link": "gripper"}]})


def test_shipped_openarm_fence_and_station_are_valid() -> None:
    from kiteos.config import StationConfig

    fence = FenceConfig.load(REPO / "config" / "fence.openarm.yaml")
    assert fence.robot == "openarm" and fence.unprotected_arms() == []
    station = StationConfig.load(REPO / "config" / "station.openarm.yaml")
    assert station.robot == "openarm" and station.profile is OPENARM
    assert station.fence_path() == (REPO / "config" / "fence.openarm.yaml").resolve()
    assert station.mock_cameras() == ["left_wrist", "right_wrist", "top"]


def test_fence_checker_holds_a_reach_through_the_glass() -> None:
    checker = FenceChecker(openarm_fence())
    assert checker.dof == 7 and checker.robot is OPENARM
    for side in ("left", "right"):
        assert checker.check_arm(side, ZERO, 1.0).ok
        reach = checker.check_arm(side, REACH[side], 0.0)
        assert not reach.ok and reach.zone == "glass" and reach.min_clearance_m < 0.03
        assert not checker.check_arm_motion(side, ZERO, REACH[side], 0.0).ok
        # Lowering the arm out of the glass is an escape; going deeper is not.
        assert not checker.check_arm(side, THROUGH[side], 0.0).ok
        assert checker.is_escape_motion(side, THROUGH[side], ZERO, 0.0)
        assert not checker.is_escape_motion(side, THROUGH[side], REACH[side], 0.0)
        # From the horizontal reach, swinging straight down first pushes the wrist deeper into
        # the zone, so the strict escape rule refuses that path.
        assert not checker.is_escape_motion(side, REACH[side], ZERO, 0.0)
    report = checker.check(np.concatenate([ZERO, REACH["right"], [0.0, 0.0]]))
    assert report.arms["left"].ok and not report.arms["right"].ok and not report.ok
    bad = checker.check(np.zeros(14 + 3))
    assert not bad.ok and "expected 14 or 16" in (bad.arms["left"].detail or "")
    assert not checker.check_arm("left", np.zeros(6)).ok  # six values are a dual OpenYAM pose


def test_gripper_opening_grows_the_finger_spheres() -> None:
    checker = FenceChecker(openarm_fence())
    closed = checker.spheres("left", ZERO, 0.0)
    wide = checker.spheres("left", ZERO, 1.0)
    grew = wide.radii - closed.radii
    fingers = [i for i, label in enumerate(closed.labels) if ".ee_link" in label]
    assert fingers and np.allclose(grew[fingers], OPENARM.gripper_stroke_m)
    others = [i for i in range(len(grew)) if i not in fingers]
    assert np.allclose(grew[others], 0.0)


def test_fingertip_is_beyond_the_grasp_frame() -> None:
    kinematics = DualArmKinematics(robot=OPENARM)
    for side in ("left", "right"):
        tip = kinematics.fingertip(side, ZERO)
        grasp = kinematics.link_poses(side, ZERO)["grasp_frame"][:3, 3]
        # At zero the hand points down: the finger end is below the grasp frame, within 4 cm.
        assert tip[2] < grasp[2] and float(np.linalg.norm(tip - grasp)) < 0.04


# =========================================================================== adapter


class FakeInner:
    """16-joint whole-body adapter: motors follow commands."""

    def __init__(self) -> None:
        from dimos.hardware.whole_body.spec import MotorState

        self._MotorState = MotorState
        self.q = [0.0] * 16
        self.writes: list[list[Any]] = []

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def has_motor_states(self) -> bool:
        return True

    def read_imu(self) -> Any:
        return None

    def get_limits(self) -> Any:
        return None

    def read_motor_states(self) -> list[Any]:
        return [self._MotorState(q=v, dq=0.0, tau=0.0) for v in self.q]

    def write_motor_commands(self, commands: list[Any]) -> bool:
        self.writes.append(list(commands))
        self.q = [float(c.q) for c in commands]
        return True


@pytest.mark.runtime
def test_adapter_fences_all_sixteen_joints() -> None:
    pytest.importorskip("dimos")
    from dimos.hardware.whole_body.spec import MotorCommand

    from kiteos.safety.adapter import FencedWholeBodyAdapter

    inner = FakeInner()
    clock = iter(np.arange(0.0, 1000.0, 0.01))
    adapter = FencedWholeBodyAdapter(
        inner, joint_names=OPENARM.joint_names, checker=FenceChecker(openarm_fence()), clock=lambda: float(next(clock))
    )
    assert adapter.robot is OPENARM
    adapter.read_motor_states()
    assert adapter.activate()

    def command(left: np.ndarray, right: np.ndarray, grippers: tuple[float, float] = (0.0, 0.0)) -> list[MotorCommand]:
        q = [*left, *right, *grippers]
        return [MotorCommand(q=float(v), dq=0.0, kp=0.0, kd=0.0, tau=0.0) for v in q]

    # Step toward the reach pose until the fence holds the left arm short of the glass.
    target = REACH["left"]
    for _ in range(400):
        adapter.read_motor_states()
        adapter.write_motor_commands(command(target, ZERO))
    left_q = np.array(inner.q[:7])
    checker = FenceChecker(openarm_fence())
    held = checker.check_arm("left", left_q, 0.0)
    assert held.ok and held.min_clearance_m < 0.2  # stopped clear of the margin, near the glass
    assert adapter.status()["arms"]["left"]["holding"]
    # Held joints keep at least the OpenArm hold gains.
    last = inner.writes[-1]
    assert [c.kp for c in last[:7]] == list(OPENARM.hold_kp)
    # The right arm, commanded to stay at zero, is untouched by the left arm's hold (zero clamped
    # to the joint limits minus the margin: joint 4's lower limit is exactly 0).
    assert np.allclose(inner.q[7:14], checker.clamp_to_limits("right", ZERO))
    # A dual OpenYAM fence cannot be swapped in.
    with pytest.raises(ValueError, match="this adapter drives the OpenArm"):
        adapter.reload(FenceConfig())


# =========================================================================== coordinator and setups


@pytest.mark.runtime
def test_openarm_coordinator_fences_the_mock_arms(tmp_path: Path) -> None:
    pytest.importorskip("dimos")
    os.environ.setdefault("LCM_DEFAULT_URL", "udpm://239.255.76.67:7918?ttl=0")
    os.environ.setdefault("ZENOH_SCOUT_ADDR", "224.0.0.224:17918")
    from dimos.robot.manipulators.openarm.config import openarm_hardware

    from kiteos.safety.adapter import FencedWholeBodyAdapter
    from kiteos.safety.openarm_coordinator import KiteOpenArmCoordinator

    fence = tmp_path / "fence.yaml"
    openarm_fence().dump(fence)
    coordinator = KiteOpenArmCoordinator(publish_joint_state=False, fence_path=str(fence), expect_mock=True)
    try:
        component = openarm_hardware()
        assert component.adapter_type == "mock_whole_body"
        adapter = coordinator._create_whole_body_adapter(component)
        assert isinstance(adapter, FencedWholeBodyAdapter) and adapter.robot is OPENARM
        status = coordinator.fence_status()
        assert status["robot"] == "openarm"
        # A dual OpenYAM fence file is refused before any adapter exists.
        dual = tmp_path / "dual.yaml"
        FenceConfig().dump(dual)
        other = KiteOpenArmCoordinator(publish_joint_state=False, fence_path=str(dual), expect_mock=True)
        try:
            with pytest.raises(RuntimeError, match="is for the Dual OpenYAM"):
                other._create_whole_body_adapter(openarm_hardware())
        finally:
            other.stop()
        # expect_mock refuses the real adapter.
        real = openarm_hardware(left_can_port="can_left", right_can_port="can_right")
        with pytest.raises(RuntimeError, match="MOCK arms"):
            coordinator._create_whole_body_adapter(real)
    finally:
        coordinator.stop()


@pytest.mark.runtime
def test_openarm_setups_compose_on_the_fenced_coordinator() -> None:
    pytest.importorskip("dimos")
    from dimos.robot.manipulators.openarm.config import OPENARM_GRIPPER_JOINTS, OPENARM_JOINTS

    from kiteos.blueprints import openarm
    from kiteos.blueprints.cameras import KiteCameraModule
    from kiteos.rollout.module import DualArmPolicyModule
    from kiteos.safety.openarm_coordinator import KiteOpenArmCoordinator

    def atoms(bp: Any) -> dict[str, Any]:
        return {atom.name: atom for atom in bp.blueprints}

    teleop = atoms(openarm.openarm_teleop)
    assert set(teleop) == {"armteleopmodule", "ControlCoordinator", "manipulationmodule"}
    assert teleop["ControlCoordinator"].module is KiteOpenArmCoordinator
    tasks = {t.name: t for t in teleop["ControlCoordinator"].kwargs["tasks"]}
    assert set(tasks) == {"teleop_openarm", "left_arm_gripper", "right_arm_gripper", "joint_trajectory"}
    assert tasks["teleop_openarm"].params["bindings"][1]["target_frame"] == "openarm_right_grasp_frame"
    assert teleop["manipulationmodule"].module is openarm.KiteOpenArmManipulationModule
    assert teleop["manipulationmodule"].kwargs["visualization"].host == "127.0.0.1"

    collection = atoms(openarm.openarm_collection)
    assert {"left_wrist", "right_wrist", "top", "recorder", "episodes"} <= set(collection)
    assert collection["top"].module is KiteCameraModule
    assert openarm.OPENARM_COLLECTION.robot_type == "openarm"
    assert openarm.OPENARM_COLLECTION.observations["observation.state"].names == list(OPENARM_JOINTS)
    assert openarm.OPENARM_COLLECTION.actions["action"].stream == "applied_joint_position_command"

    rollout = atoms(openarm.openarm_rollout)
    policy = rollout["policy"]
    assert policy.module is DualArmPolicyModule
    assert policy.kwargs["joint_names"] == list(OPENARM_JOINTS)
    assert policy.kwargs["gripper_joints"] == list(OPENARM_GRIPPER_JOINTS)
    assert policy.kwargs["robot_type"] == "openarm"
    assert "recorder" not in rollout and "recorder" in atoms(openarm.openarm_rollout_recorded)


@pytest.mark.runtime
def test_openarm_setups_parse_with_the_station_flags() -> None:
    """The flags `kiteos collect start` passes for the OpenArm station parse on the setup."""
    pytest.importorskip("dimos")
    from dimos.core.coordination.blueprint_config.parser import BlueprintConfigParser

    from kiteos.blueprints import openarm

    parsed = BlueprintConfigParser(openarm.openarm_collection).parse(
        [
            "--controlcoordinator.fence-path", "/tmp/fence.yaml",
            "--controlcoordinator.require-fence-on-hardware", "true",
            "--controlcoordinator.expect-mock", "true",
            "--left-wrist.hardware.camera-index", "mock",
            "--right-wrist.hardware.camera-index", "mock",
            "--top.hardware.camera-index", "mock",
            "--recorder.recording", "/tmp/kiteos-test-openarm",
            "--episodes.task", "pick up the cube",
            "--armteleopmodule.server-port", "18443",
            "--manipulationmodule.visualization.port", "18095",
        ],
        environ={},
    )
    configs = parsed.module_configs
    assert configs["ControlCoordinator"]["expect_mock"] is True
    assert configs["left_wrist"]["hardware"]["camera_index"] == "mock"
    assert configs["manipulationmodule"]["visualization"]["port"] == 18095


# =========================================================================== the policy environment


WEBXR_MODULES = ("fastapi", "dimos.teleop.webxr.extensions", "dimos.teleop.webxr.module")


@pytest.mark.runtime
def test_coordinator_modules_never_import_the_webxr_server() -> None:
    """A policy's coordinator reference is unpickled in the isolated LeRobot env, which has no
    fastapi; importing a teleop blueprint there left every rollout without its coordinator."""
    pytest.importorskip("dimos")
    import sys

    code = (
        "import sys, kiteos.safety.coordinator, kiteos.safety.openarm_coordinator\n"
        f"print([m for m in {WEBXR_MODULES!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout


@pytest.mark.runtime
def test_openarm_task_name_matches_the_runtime() -> None:
    teleop = pytest.importorskip("dimos.robot.manipulators.openarm.blueprints.teleop")
    from kiteos.safety.openarm_coordinator import OPENARM_TELEOP_TASK_NAME

    assert OPENARM_TELEOP_TASK_NAME == teleop.OPENARM_WEBXR_TASK_NAME


@pytest.mark.runtime
def test_coordinators_import_in_the_policy_environment() -> None:
    """The real check, in the runtime's isolated LeRobot environment (when it is built)."""
    import shutil

    pytest.importorskip("dimos")
    from dimos.experimental.isolated_python.module import isolated_python_environment, isolated_python_run_command
    from dimos.imitation.dataprep.lerobot import lerobot_project

    project = lerobot_project()
    env = isolated_python_environment(project)
    if not Path(env["UV_PROJECT_ENVIRONMENT"]).is_dir() or shutil.which("uv") is None:
        pytest.skip("the isolated LeRobot environment is not built here")
    env["PYTHONPATH"] = str(REPO / "src")
    command = isolated_python_run_command(
        project, "python", "-c", "import kiteos.safety.coordinator, kiteos.safety.openarm_coordinator; print('ok')"
    )
    out = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), (out.stdout + out.stderr)[-3000:]


# =========================================================================== fence zones in the viewer


@pytest.mark.runtime
def test_viewer_draws_the_enabled_zones_as_boxes() -> None:
    pytest.importorskip("dimos")
    from kiteos.blueprints.manipulation import REACH_M, SLAB_DEPTH_M, zone_boxes

    fixture = {
        "kind": "box",
        "name": "fixture",
        "frame": "left_base",
        "center": [0.3, 0.0, 0.0],
        "half_extents": [0.05, 0.1, 0.2],
    }
    cell = {"kind": "keep_in_box", "name": "cell", "frame": "world", "center": [0, 0, 0.5], "half_extents": [1, 1, 1]}
    fence = openarm_fence(zones=[GLASS, fixture, cell, {**GLASS, "name": "off", "enabled": False}])
    boxes = {name: (center, quat, size) for name, center, quat, size in zone_boxes(fence)}
    assert set(boxes) == {"glass", "fixture"}  # keep-in boxes and disabled zones are not drawn
    center, quat, size = boxes["glass"]
    # Centred on the arms (between the shoulders, 0.698 m up), on the plane, depth along +x.
    np.testing.assert_allclose(center, [0.45 + SLAB_DEPTH_M / 2.0, 0.0, 0.698], atol=1e-12)
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    assert size[0] == SLAB_DEPTH_M and size[1] == size[2] == pytest.approx(2 * (0.031 + REACH_M))
    center, quat, size = boxes["fixture"]
    np.testing.assert_allclose(center, [0.3, 0.031, 0.698], atol=1e-12)  # in the left arm's base frame
    np.testing.assert_allclose(size, [0.1, 0.2, 0.4])


@pytest.mark.runtime
def test_viewer_slab_for_a_tilted_plane_is_on_the_forbidden_side() -> None:
    pytest.importorskip("dimos")
    from kiteos.blueprints.manipulation import SLAB_DEPTH_M, zone_boxes
    from kiteos.safety.kinematics import pose_matrix

    tilted = {**GLASS, "normal": [1.0, 0.0, -1.0]}
    ((_, center, (x, y, z, w), _),) = zone_boxes(openarm_fence(zones=[tilted]))
    n = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)
    # The slab centre is SLAB_DEPTH_M / 2 into the forbidden side, and its x axis is the normal.
    assert float(n @ center - n @ np.array([0.45, 0.0, 0.0])) == pytest.approx(SLAB_DEPTH_M / 2.0)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(R[:, 0], n, atol=1e-9)
    assert pose_matrix((0, 0, 0)).shape == (4, 4)


# =========================================================================== synthetic camera


@pytest.mark.runtime
def test_synthetic_camera_streams_test_pattern_frames() -> None:
    pytest.importorskip("dimos")
    from dimos.hardware.sensors.camera.webcam import WebcamConfig

    from kiteos.blueprints.cameras import SyntheticCamera, is_mock_camera

    assert is_mock_camera("mock") and is_mock_camera(" Mock ") and not is_mock_camera(0)
    camera = SyntheticCamera(**WebcamConfig(camera_index="mock", width=64, height=48, fps=200, frame_id_prefix="top_image").model_dump())
    frames: list[Any] = []
    subscription = camera.image_stream().subscribe(frames.append)
    try:
        deadline = 2.0
        import time

        start = time.monotonic()
        while len(frames) < 5 and time.monotonic() - start < deadline:
            time.sleep(0.01)
    finally:
        subscription.dispose()
    assert len(frames) >= 5
    image = frames[0]
    assert image.data.shape == (48, 64, 3) and image.data.dtype == np.uint8
    assert image.frame_id == "top_image/camera_optical"
    assert camera.camera_info.width == 64 and camera.camera_info.height == 48
    assert not np.array_equal(frames[0].data, frames[-1].data)  # the bar moves


# =========================================================================== CLI


runner = CliRunner()


@pytest.fixture
def openarm_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from kiteos.cli._common import STATE

    monkeypatch.delenv("KITEOS_STATION", raising=False)
    monkeypatch.setenv("KITEOS_NO_BANNER", "1")
    STATE.station = None

    def no_host_commands(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"test tried to run a host command: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "run", no_host_commands)
    monkeypatch.setattr(os, "execvpe", no_host_commands)
    config = tmp_path / "config"
    config.mkdir()
    (tmp_path / "kiteos-runtime").mkdir()
    station = yaml.safe_load((REPO / "config" / "station.openarm.yaml").read_text())
    station.update(
        runtime_root=str(tmp_path / "kiteos-runtime"),
        recordings_dir=str(tmp_path / "recordings"),
        datasets_dir=str(tmp_path / "datasets"),
        fence="config/fence.yaml",
        can={"left": {"darwin": "L-SERIAL", "linux": "can_left"}, "right": {"darwin": "R-SERIAL", "linux": "can_right"}},
    )
    (config / "station.yaml").write_text(yaml.safe_dump(station))
    openarm_fence(station=station["station"]).dump(config / "fence.yaml")
    return config / "station.yaml"


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    from kiteos.cli import launch

    calls: list[Any] = []
    monkeypatch.setattr(launch, "execute", calls.append)
    return calls


def invoke(site: Path, *args: str, input: str | None = None) -> Any:
    from kiteos.cli import app

    return runner.invoke(app, ["--station", str(site), *args], input=input)


def test_teleop_runs_the_openarm_setup(openarm_site: Path, executed: list[Any]) -> None:
    result = invoke(openarm_site, "teleop")
    assert result.exit_code == 0, result.output
    (plan,) = executed
    assert plan.argv[3:6] == ["kiteos-runtime", "run", "kiteos.openarm-teleop"]
    assert "--controlcoordinator.expect-mock" in plan.argv and "true" in plan.argv
    assert not any("can-port" in a for a in plan.argv)
    fence = plan.argv[plan.argv.index("--controlcoordinator.fence-path") + 1]
    assert plan.argv[plan.argv.index("--manipulationmodule.fence-path") + 1] == fence  # zones in the viewer
    assert "OpenArm" in result.output


def test_collect_passes_mock_cameras_and_refuses_them_for_real(openarm_site: Path, executed: list[Any]) -> None:
    result = invoke(openarm_site, "collect", "start")
    assert result.exit_code == 0, result.output
    argv = executed[0].argv
    assert "kiteos.openarm-collection" in argv
    index = argv.index("--top.hardware.camera-index")
    assert argv[index + 1] == "mock"
    assert "synthetic" in " ".join(result.output.split())
    real = invoke(openarm_site, "collect", "start", "--real", input="openarm-demo\n")
    assert real.exit_code == 2 and "synthetic 'mock' camera" in " ".join(real.output.split())


def test_real_teleop_passes_the_can_ports(openarm_site: Path, executed: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    result = invoke(openarm_site, "teleop", "--real", input="openarm-demo\n")
    assert result.exit_code == 0, result.output
    argv = executed[0].argv
    assert argv[argv.index("--controlcoordinator.left-can-port") + 1] == "can_left"
    assert argv[argv.index("--controlcoordinator.expect-mock") + 1] == "false"
    assert "REAL ARMS" in result.output and "OpenArm" in result.output


def test_a_fence_for_another_robot_is_refused(openarm_site: Path, executed: list[Any]) -> None:
    FenceConfig(station="openarm-demo").dump(openarm_site.parent / "fence.yaml")  # a dual OpenYAM fence
    result = invoke(openarm_site, "teleop")
    assert result.exit_code == 2 and "is for the Dual OpenYAM" in " ".join(result.output.split())
    assert executed == []


def test_fence_check_offline_takes_sixteen_values(openarm_site: Path) -> None:
    clear = ",".join(["0"] * 16)
    result = invoke(openarm_site, "fence", "check", "--offline", "--positions", clear)
    assert result.exit_code == 0, result.output
    reach = ",".join(str(v) for v in [*REACH["left"], *ZERO, 0.0, 0.0])
    result = invoke(openarm_site, "fence", "check", "--offline", "--positions", reach)
    assert result.exit_code == 1 and "VIOLATION" in result.output
    result = invoke(openarm_site, "fence", "check", "--offline", "--positions", ",".join(["0"] * 12))
    assert result.exit_code == 2 and "14/16 values" in " ".join(result.output.split())


def test_fence_validate_uses_the_openarm_reference_poses(openarm_site: Path) -> None:
    result = invoke(openarm_site, "fence", "validate")
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "ee_base_link/link6" in out  # the wrist-camera warning names the OpenArm wrist links


def test_list_shows_openarm_kiteos_setups(openarm_site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiteos import embodiments
    from kiteos.cli import setups

    registry = openarm_site.parent.parent / "kiteos-runtime" / embodiments.REGISTRY_FILE
    registry.parent.mkdir(parents=True)
    registry.write_text(
        'all_blueprints = {\n    "teleop-webxr-openarm": "dimos.robot.manipulators.openarm.blueprints.teleop:t",\n}\n'
    )
    monkeypatch.setattr(setups, "plugin_setups", lambda: {})
    result = invoke(openarm_site, "list", "--json", "-e", "openarm")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["station_robot"] == "openarm"
    assert [k["setup"] for k in data["kiteos"]] == [
        "kiteos.openarm-teleop",
        "kiteos.openarm-collection",
        "kiteos.openarm-rollout",
        "kiteos.openarm-rollout-recorded",
    ]
    assert data["setups"][0]["setup"] == "teleop-webxr-openarm"
    refused = invoke(openarm_site, "run", "teleop-webxr-openarm", "--real")
    assert refused.exit_code == 2 and "without the KiteOS fence" in " ".join(refused.output.split())


def test_upload_no_push_converts_mock_sessions_locally(openarm_site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kiteos.cli import upload

    session = openarm_site.parent.parent / "recordings" / "mock-openarm-demo-001"
    session.mkdir(parents=True)
    (session / "recording.mcap").write_bytes(b"x")
    converted: list[tuple[Path, Path]] = []

    class Env:
        pass

    monkeypatch.setattr(upload, "IsolatedEnv", Env)

    def prepare(env: Any, recording: Path, datasets: Path) -> bool:
        converted.append((recording, datasets))
        return True

    monkeypatch.setattr(upload, "prepare_session", prepare)
    result = invoke(openarm_site, "upload", "--session", "mock-openarm-demo", "--no-push")
    assert result.exit_code == 0, result.output
    assert converted and converted[0][0].name == "mock-openarm-demo-001"
    assert "not pushed" in " ".join(result.output.split())
    refused = invoke(openarm_site, "upload", "--session", "mock-openarm-demo")
    assert refused.exit_code == 2 and "MOCK arms" in refused.output


# =========================================================================== the scripted driver

WORKFLOW_TOOL = REPO / "tools" / "openarm_workflow.py"


def _load_workflow_tool(monkeypatch: pytest.MonkeyPatch, run: Any) -> dict[str, Any]:
    """Execute the driver's module body (not ``main``) as if ``attach_to_run`` found ``run``."""
    import runpy

    from kiteos.cli import _common

    monkeypatch.setattr(_common, "attach_to_run", lambda purpose: run)
    return runpy.run_path(str(WORKFLOW_TOOL), run_name="openarm_workflow")


@pytest.mark.parametrize(
    "run",
    [
        None,  # no KiteOS run on this host
        RunState(pid=1, real=True, blueprint="kiteos.openarm-teleop", station="s"),
        RunState(pid=1, real=False, blueprint="kiteos.dual-openyam-collection", station="s"),
        RunState(pid=1, real=False, blueprint="coordinator-openarm", station="s", hardware="mock hardware"),
    ],
    ids=["no-run", "real", "other-robot", "runtime-setup"],
)
def test_workflow_driver_refuses_all_but_a_mock_kiteos_openarm_run(monkeypatch: pytest.MonkeyPatch, run: Any) -> None:
    with pytest.raises(SystemExit, match="only a mock KiteOS OpenArm stack"):
        _load_workflow_tool(monkeypatch, run)


@pytest.mark.runtime
def test_workflow_driver_poses_behave_as_documented_with_the_shipped_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_workflow_tool(monkeypatch, RunState(pid=1, real=False, blueprint="kiteos.openarm-teleop", station="s"))
    checker = FenceChecker(FenceConfig.load(REPO / "config" / "fence.openarm.yaml"))
    closed = [0.0, 0.0]  # the grippers as the mock reports them (unknown openings count as open)
    ready, right_ready = tool["READY"], tool["READY"][7:]
    assert tool["ARMS"] == OPENARM.arm_joint_names and tool["ALL"] == OPENARM.joint_names
    assert checker.check(ready + closed).ok and checker.check(tool["ZERO"] + closed).ok
    assert not checker.check(tool["THROUGH_LEFT"] + right_ready + closed).arms["left"].ok  # past the glass
    assert not checker.check(tool["REACH_LEFT"] + right_ready + closed).arms["left"].ok  # where `stream` heads
