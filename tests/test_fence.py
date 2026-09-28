# Copyright 2026 Kite ML.
"""FenceConfig and FenceChecker (kiteos.safety.fence)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from kiteos.safety.fence import (
    MAX_MOTION_SAMPLES,
    ArmReport,
    ExtraSphere,
    FenceChecker,
    FenceConfig,
    FenceConfigError,
    FenceReport,
    Mount,
)
from kiteos.safety.kinematics import ARM_JOINTS, DUAL_OPENYAM_JOINTS, GRIPPER_JOINTS
from kiteos.safety.zones import BoxZone, HalfSpaceZone, KeepInBox

HOME = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])
FORWARD_REACH = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0])  # fingertip at x ~ 0.63 m, gripper level
FAR_FORWARD = np.array([0.0, 3.0, 2.75, 0.5, 0.0, 0.0])  # fingertip at x ~ 0.78 m, z ~ 0.21 m

GLASS = HalfSpaceZone(name="glass_window", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0))


def glass_config(**overrides) -> FenceConfig:
    return FenceConfig(margin_m=0.03, zones=[GLASS], **overrides)


def q14(
    left: np.ndarray, right: np.ndarray, left_open: float = 0.0, right_open: float = 0.0
) -> np.ndarray:
    return np.r_[left, right, left_open, right_open]


# ------------------------------------------------------------------------------ config


def test_defaults_match_contract() -> None:
    cfg = FenceConfig()
    assert cfg.version == 1 and cfg.mode == "enforce"
    assert cfg.margin_m == 0.05 and cfg.gripper_stroke_m == 0.05
    assert cfg.mounts["left"].xyz == (0.0, 0.31, 0.0) and cfg.mounts["right"].xyz == (
        0.0,
        -0.31,
        0.0,
    )
    assert cfg.mounts["left"].rpy == (0.0, 0.0, 0.0)
    assert cfg.zones == [] and cfg.extra_spheres == []
    assert cfg.max_joint_speed_rad_s == 2.5
    assert cfg.joint_limit_margin_rad == 0.01
    assert cfg.interpolation_step_rad == 0.01
    assert cfg.refuse_start_inside is True


def test_margin_floor_enforced() -> None:
    with pytest.raises(ValidationError):
        FenceConfig(margin_m=0.019)
    assert FenceConfig(margin_m=0.02).margin_m == 0.02
    cfg = FenceConfig()
    with pytest.raises(ValidationError):
        cfg.margin_m = 0.0  # assignment is validated too


def full_config() -> FenceConfig:
    return FenceConfig(
        station="kite-lab-dual-openyam",
        mode="enforce",
        margin_m=0.04,
        gripper_stroke_m=0.06,
        mounts={
            "left": Mount(xyz=(0.0, 0.305, 0.0), rpy=(0.0, 0.0, 0.01)),
            "right": Mount(xyz=(0.0, -0.312, 0.0)),
        },
        extra_spheres=[
            ExtraSphere(
                name="wrist_camera", link="gripper", center=(0.0, 0.06, -0.03), radius=0.035
            )
        ],
        zones=[
            GLASS,
            BoxZone(
                name="monitor",
                frame="left_base",
                center=(0.3, 0.2, 0.3),
                half_extents=(0.05, 0.2, 0.15),
                rpy=(0, 0, 0.2),
            ),
            KeepInBox(
                name="cell",
                center=(0.3, 0.0, 0.4),
                half_extents=(0.6, 0.9, 0.6),
                applies_to="right",
                enabled=False,
            ),
        ],
        max_joint_speed_rad_s=2.0,
        interpolation_step_rad=0.005,
        refuse_start_inside=False,
    )


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = full_config()
    path = tmp_path / "sub" / "fence.yaml"
    cfg.dump(path)
    text = path.read_text()
    assert text.startswith("# KiteOS fence configuration")
    assert "kind: half_space" in text
    loaded = FenceConfig.load(path)
    assert loaded == cfg
    assert isinstance(loaded.zones[1], BoxZone) and isinstance(loaded.zones[2], KeepInBox)
    assert not list(tmp_path.glob("sub/.fence.yaml.*"))  # temp file cleaned up


def test_load_refuses_an_empty_or_comment_only_file(tmp_path: Path) -> None:
    """A truncated fence.yaml must never load as a valid fence without zones."""
    path = tmp_path / "fence.yaml"
    for text in ("", "# zones were here\n", "\n\n"):
        path.write_text(text)
        with pytest.raises(FenceConfigError, match="empty"):
            FenceConfig.load(path)
    path.write_text("version: 1\n")  # an explicit (if minimal) mapping is still a fence
    assert FenceConfig.load(path) == FenceConfig()


def test_load_refuses_duplicate_keys(tmp_path: Path) -> None:
    """PyYAML keeps the last duplicate: an uncommented example block above `zones: []` would
    silently drop the zones, and a second `enabled:` would flip a zone."""
    path = tmp_path / "fence.yaml"
    path.write_text(
        "zones:\n  - {kind: half_space, name: glass, point: [0.6, 0, 0], normal: [1, 0, 0]}\n"
        "zones: []\n"
    )
    with pytest.raises(FenceConfigError, match="duplicate key 'zones'"):
        FenceConfig.load(path)
    path.write_text(
        "zones:\n  - kind: half_space\n    name: glass\n    enabled: true\n"
        "    point: [0.6, 0, 0]\n    normal: [1, 0, 0]\n    enabled: false\n"
    )
    with pytest.raises(FenceConfigError, match="duplicate key 'enabled'"):
        FenceConfig.load(path)


def test_plausibility_bounds() -> None:
    with pytest.raises(ValidationError, match="millimetres"):
        ExtraSphere(name="cam", link="gripper", center=(0, -55, -20), radius=0.035)
    with pytest.raises(ValidationError):
        ExtraSphere(name="cam", link="gripper", center=(0, -0.055, -0.02), radius=35.0)
    with pytest.raises(ValidationError, match="millimetres"):
        Mount(xyz=(0.0, 310.0, 0.0))
    with pytest.raises(ValidationError):
        FenceConfig(interpolation_step_rad=1.0)
    with pytest.raises(ValidationError):
        FenceConfig(max_feedforward_tau_nm=-1.0)
    assert FenceConfig().max_feedforward_tau_nm == 0.0 and FenceConfig().allow_no_zones is False


def test_unprotected_arms() -> None:
    left = HalfSpaceZone(name="g_left", frame="left_base", applies_to="left", point=(0.45, 0, 0), normal=(1, 0, 0))
    assert FenceConfig().unprotected_arms() == ["left", "right"]
    assert FenceConfig(zones=[left]).unprotected_arms() == ["right"]
    assert FenceConfig(zones=[GLASS]).unprotected_arms() == []
    off = GLASS.model_copy(update={"enabled": False})
    assert FenceConfig(zones=[off]).unprotected_arms() == ["left", "right"]


def test_escape_path_accepts_only_straight_retreats() -> None:
    checker = FenceChecker(glass_config())
    inside = np.array([0.0, 1.2, 1.047, 0.0, 0.0, 0.0])
    assert not checker.check_arm("left", inside, 0.0).ok
    out = checker.check_escape_path("left", [(inside, HOME, 0.0)])
    assert out.ok and "escape" in (out.detail or "")
    deeper = checker.check_escape_path("left", [(inside, FORWARD_REACH, 0.0)])
    assert not deeper.ok and "does not move straight out" in (deeper.detail or "")
    partial = checker.check_escape_path("left", [(inside, [0.0, 1.19, 1.047, 0.0, 0.0, 0.0], 0.0)])
    assert not partial.ok and "does not end clear" in (partial.detail or "")
    # Opening the gripper on the way out makes the finger pairs lose clearance: refused.
    widen = checker.check_escape_path("left", [(inside, [0.0, 1.15, 1.047, 0, 0, 0], 0.0), ([0.0, 1.15, 1.047, 0, 0, 0], HOME, 1.0)])
    assert not widen.ok
    assert checker.is_escape_motion("left", inside, [0.0, 1.19, 1.047, 0.0, 0.0, 0.0], 0.0)
    assert not checker.is_escape_motion("left", inside, [0.0, 1.21, 1.047, 0.0, 0.0, 0.0], 0.0)
    assert not checker.is_escape_motion("left", HOME, [0.0, 1.0, 1.047, 0.0, 0.0, 0.0], 0.0)
    assert not checker.check_escape_path("left", [(inside, [math.nan] * 6, 0.0)]).ok


def test_load_reports_readable_errors(tmp_path: Path) -> None:
    path = tmp_path / "fence.yaml"
    path.write_text(
        """
margin_m: 0.01
zones:
  - {kind: half_space, name: glass, point: [0.45, 0, 0], normal: [0, 0, 0]}
  - {kind: box, name: shelf, center: [0.3, 0, 0.2], half_extents: [0.1, 0.0, 0.1]}
  - {name: nameless_kind, point: [0, 0, 0], normal: [1, 0, 0]}
  - {kind: cylinder, name: pipe}
extra_spheres:
  - {name: cam, link: left_gripper, center: [0, 0, 0], radius: 0.03}
"""
    )
    with pytest.raises(FenceConfigError) as info:
        FenceConfig.load(path)
    msg = str(info.value)
    assert str(path) in msg
    assert "margin_m" in msg and "2 cm" in msg
    assert "zones[0].normal: normal must be a non-zero vector" in msg
    assert "zones[1].half_extents: half_extents must all be > 0" in msg
    assert "zones[2]: missing 'kind'" in msg
    assert "zones[3]: unknown kind 'cylinder'" in msg
    assert "extra_spheres[0].link: unknown link 'left_gripper'" in msg


def test_load_rejects_bad_yaml_and_shapes(tmp_path: Path) -> None:
    path = tmp_path / "fence.yaml"
    path.write_text("zones: [unclosed")
    with pytest.raises(FenceConfigError, match="invalid YAML"):
        FenceConfig.load(path)
    path.write_text("- just\n- a list\n")
    with pytest.raises(FenceConfigError, match="mapping"):
        FenceConfig.load(path)
    path.write_text("margin: 0.05\n")  # typo'd key
    with pytest.raises(FenceConfigError, match="margin"):
        FenceConfig.load(path)
    assert isinstance(FenceConfigError("x"), ValueError)


def test_docs_example_is_valid(tmp_path: Path) -> None:
    doc = Path(__file__).resolve().parents[1] / "docs" / "FENCE.md"
    if not doc.is_file():
        pytest.skip("docs/FENCE.md not present")
    block = doc.read_text().split("```yaml\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "fence.yaml"
    path.write_text(block)
    config = FenceConfig.load(path)
    checker = FenceChecker(config)
    # the documented behaviour: home is allowed with the grippers closed AND open ...
    assert checker.check(q14(HOME, HOME, 0.0, 0.0)).ok
    assert checker.check(q14(HOME, HOME, 1.0, 1.0)).ok
    assert checker.check_arm("left", HOME, 1.0).min_clearance_m == pytest.approx(0.091, abs=1e-3)
    assert not checker.check_arm("left", FORWARD_REACH, 0.0).ok
    # ... and `kiteos fence validate` passes for it (the review found the old example failing).
    from kiteos.cli.fence import validate_fence

    assert [f for f in validate_fence(config) if f.level == "FAIL"] == []


def test_config_cross_field_validation() -> None:
    with pytest.raises(ValidationError, match="both 'left' and 'right'"):
        FenceConfig(mounts={"left": Mount(xyz=(0, 0.31, 0))})
    with pytest.raises(ValidationError, match="duplicate zone names"):
        FenceConfig(zones=[GLASS, GLASS])
    with pytest.raises(ValidationError):
        FenceConfig(mode="off")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        FenceConfig(interpolation_step_rad=0.0)
    with pytest.raises(ValidationError):
        ExtraSphere(name="cam", link="gripper", center=(0, 0, 0), radius=-0.01)


# ------------------------------------------------------------------------------ checker basics


def test_no_zones_is_ok_with_infinite_clearance() -> None:
    checker = FenceChecker(FenceConfig())
    report = checker.check_arm("left", HOME)
    assert report.ok and report.min_clearance_m == math.inf and report.zone is None
    assert report.to_dict()["min_clearance_m"] is None


def test_margin_semantics() -> None:
    base = FenceChecker(FenceConfig(margin_m=0.03, zones=[GLASS]))
    raw = base.check_arm("left", HOME, 0.0).min_clearance_m
    # move the glass so that the raw clearance sits just above / just below the margin
    for delta, ok in ((0.001, True), (-0.001, False)):
        x = 0.45 - raw + 0.03 + delta
        zone = HalfSpaceZone(name="g", point=(x, 0, 0), normal=(1, 0, 0))
        report = FenceChecker(FenceConfig(margin_m=0.03, zones=[zone])).check_arm("left", HOME, 0.0)
        assert report.ok is ok
        assert report.min_clearance_m == pytest.approx(0.03 + delta)


def test_report_names_closest_zone_and_sphere() -> None:
    checker = FenceChecker(glass_config())
    report = checker.check_arm("left", HOME, 0.0)
    assert report.zone == "glass_window"
    assert report.sphere == "left.gripper[2]"  # the finger sphere is the closest to the glass
    d = report.to_dict()
    assert d["side"] == "left" and d["ok"] is True and d["zone"] == "glass_window"


def test_applies_to_and_enabled_in_checker() -> None:
    left_only = GLASS.model_copy(update={"applies_to": "left", "point": (0.3, 0.0, 0.0)})
    disabled = BoxZone(name="off", center=(0.3, 0.0, 0.3), half_extents=(2, 2, 2), enabled=False)
    checker = FenceChecker(FenceConfig(margin_m=0.03, zones=[left_only, disabled]))
    assert checker.zone_names("left") == ["glass_window"] and checker.zone_names("right") == []
    report = checker.check(q14(HOME, HOME))
    assert not report.arms["left"].ok
    assert report.arms["right"].ok and report.arms["right"].zone is None
    assert not report.ok


def test_mounts_move_the_arms() -> None:
    near = FenceChecker(glass_config()).check_arm("left", HOME, 0.0)
    mounts = {"left": Mount(xyz=(0.05, 0.31, 0.0)), "right": Mount(xyz=(0.0, -0.31, 0.0))}
    moved = FenceChecker(glass_config(mounts=mounts)).check_arm("left", HOME, 0.0)
    assert moved.min_clearance_m == pytest.approx(near.min_clearance_m - 0.05)


def test_arm_frame_zone_follows_its_arm() -> None:
    per_arm = HalfSpaceZone(
        name="g", frame="left_base", point=(0.45, 0.0, 0.0), normal=(1.0, 0.0, 0.0)
    )
    mounts = {"left": Mount(xyz=(0.05, 0.31, 0.0)), "right": Mount(xyz=(0.0, -0.31, 0.0))}
    a = FenceChecker(FenceConfig(margin_m=0.03, zones=[per_arm])).check_arm("left", HOME, 0.0)
    b = FenceChecker(FenceConfig(margin_m=0.03, zones=[per_arm], mounts=mounts)).check_arm(
        "left", HOME, 0.0
    )
    assert a.min_clearance_m == pytest.approx(b.min_clearance_m)


def test_gripper_opening_and_extra_spheres_matter() -> None:
    checker = FenceChecker(glass_config())
    closed = checker.check_arm("left", HOME, 0.0).min_clearance_m
    half = checker.check_arm("left", HOME, 0.5).min_clearance_m
    unknown = checker.check_arm("left", HOME, None).min_clearance_m
    assert half == pytest.approx(closed - 0.025)
    assert unknown == pytest.approx(closed - 0.05)  # unknown opening = fully open
    cam = ExtraSphere(name="camera", link="gripper", center=(0.0, 0.0, -0.16), radius=0.03)
    with_cam = FenceChecker(glass_config(extra_spheres=[cam])).check_arm("left", HOME, 0.0)
    assert with_cam.sphere == "left.camera" and with_cam.min_clearance_m < closed


def test_spheres_and_fingertip_helpers() -> None:
    checker = FenceChecker(
        glass_config(
            extra_spheres=[
                ExtraSphere(name="camera", link="gripper", center=(0, 0.05, 0), radius=0.03)
            ]
        )
    )
    spheres = checker.spheres("right", HOME, 1.0)
    assert len(spheres) == 15 and spheres.labels[-1] == "right.camera"
    np.testing.assert_allclose(checker.fingertip("left", HOME), (0.3894, 0.31, 0.4021), atol=1e-3)


# ------------------------------------------------------------------------------ check() inputs


def test_check_accepts_14_12_and_names() -> None:
    checker = FenceChecker(glass_config())
    by_vector = checker.check(q14(HOME, FORWARD_REACH))
    assert isinstance(by_vector, FenceReport)
    assert by_vector.arms["left"].ok and not by_vector.arms["right"].ok and not by_vector.ok

    by_12 = checker.check(np.r_[HOME, HOME], gripper_openings={"left": 0.0, "right": 0.0})
    assert by_12.ok

    names = dict(zip(DUAL_OPENYAM_JOINTS, q14(HOME, HOME).tolist(), strict=True))
    assert checker.check(names).ok
    d = checker.check(names).to_dict()
    assert set(d) == {"ok", "arms"} and set(d["arms"]) == {"left", "right"}


def test_check_gripper_openings() -> None:
    checker = FenceChecker(glass_config())
    closed = checker.check(q14(HOME, HOME, 0.0, 0.0))
    opened = checker.check(q14(HOME, HOME, 1.0, 0.0))
    assert closed.ok and not opened.arms["left"].ok and opened.arms["right"].ok
    # explicit openings override the vector; gripper joint names work as keys too
    override = checker.check(q14(HOME, HOME, 1.0, 1.0), {"left": 0.0, GRIPPER_JOINTS["right"]: 0.0})
    assert override.ok
    # 12 values and no openings: unknown opening counts as fully open
    assert not checker.check(np.r_[HOME, HOME]).ok


def test_check_missing_and_malformed_inputs_fail_closed() -> None:
    checker = FenceChecker(glass_config())
    partial = {name: 0.0 for name in ARM_JOINTS["left"]}
    report = checker.check(partial)
    assert report.arms["left"].ok and report.arms["left"].zone == "glass_window"
    assert not report.arms["right"].ok and "missing joints" in (report.arms["right"].detail or "")
    for bad in ([0.0] * 13, None, "left_joint1", [[1, 2], [3, 4]], {"left_joint1": "x"}):
        r = checker.check(bad)  # type: ignore[arg-type]
        assert not r.ok
        assert all(not a.ok for a in r.arms.values())


@pytest.mark.parametrize(
    "bad", [[0.0] * 5, [0.0] * 7, [np.nan] * 6, [0, 0, np.inf, 0, 0, 0], None, "abc"]
)
def test_check_arm_never_raises(bad) -> None:
    checker = FenceChecker(glass_config())
    report = checker.check_arm("left", bad)
    assert isinstance(report, ArmReport) and not report.ok and report.min_clearance_m == -math.inf
    report = checker.check_arm_motion("left", HOME, bad)
    assert not report.ok


def test_unknown_side_fails_closed() -> None:
    checker = FenceChecker(glass_config())
    assert not checker.check_arm("middle", HOME).ok
    assert not checker.check_arm_motion("middle", HOME, HOME).ok
    with pytest.raises(ValueError):
        checker.clamp_to_limits("middle", HOME)


def test_internal_errors_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = FenceChecker(glass_config())

    def boom(*_a, **_k):
        raise RuntimeError("simulated")

    monkeypatch.setattr(checker.kinematics.chains["left"], "link_poses_batch", boom)
    report = checker.check_arm("left", HOME, 0.0)
    assert not report.ok and "simulated" in (report.detail or "")
    assert not checker.check_arm_motion("left", HOME, HOME + 0.02).ok
    assert not checker.check(q14(HOME, HOME)).ok
    assert checker.check(q14(HOME, HOME)).arms["right"].ok


def test_nan_clearance_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    checker = FenceChecker(glass_config())
    zone = checker._arms["left"].zones[0]

    class NanZone:
        name = "nan"

        def clearance(self, centers, radii):
            out = zone.clearance(centers, radii)
            out[0, 3] = np.nan
            return out

    arm = checker._arms["left"]
    monkeypatch.setitem(
        checker._arms,
        "left",
        type(arm)(arm.side, arm.base_T, arm.spheres, arm.link_idx, (NanZone(),)),
    )
    report = checker.check_arm("left", HOME, 0.0)
    assert not report.ok and "non-finite" in (report.detail or "")


# ------------------------------------------------------------------------------ motion


def test_motion_samples_include_endpoints_and_respect_step() -> None:
    checker = FenceChecker(glass_config())
    for delta in (0.0, 0.003, 0.01, 0.03, 0.05, 0.0999, 0.7, 2.3):
        target = HOME + np.array([delta, -delta / 2, delta / 3, 0.0, delta, -delta])
        samples = checker.motion_samples(HOME, target)
        assert samples is not None
        np.testing.assert_array_equal(samples[0], HOME)
        np.testing.assert_array_equal(samples[-1], target)
        assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.01


def test_motion_flags_violating_endpoint() -> None:
    checker = FenceChecker(glass_config())
    assert checker.check_arm_motion("left", HOME, HOME, 0.0).ok
    report = checker.check_arm_motion("left", HOME, FORWARD_REACH, 0.0)
    assert not report.ok and report.zone == "glass_window"
    assert f"sample {len(checker.motion_samples(HOME, FORWARD_REACH)) - 1}" in (report.detail or "")


def test_huge_motion_fails_closed() -> None:
    checker = FenceChecker(FenceConfig(interpolation_step_rad=1e-4, zones=[GLASS]))
    report = checker.check_arm_motion("left", HOME, HOME + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    assert MAX_MOTION_SAMPLES < (1.0 / 1e-4) + 1
    assert not report.ok and "too large" in (report.detail or "")


def test_clamp_to_limits() -> None:
    checker = FenceChecker(FenceConfig(joint_limit_margin_rad=0.02))
    lim = checker.joint_limits["left"]
    assert lim.shape == (6, 2)
    q = np.array([-10.0, 10.0, 1.0, -10.0, 10.0, 0.0])
    got = checker.clamp_to_limits("left", q)
    np.testing.assert_allclose(got[[0, 3]], lim[[0, 3], 0] + 0.02)
    np.testing.assert_allclose(got[[1, 4]], lim[[1, 4], 1] - 0.02)
    np.testing.assert_allclose(got[[2, 5]], q[[2, 5]])
    np.testing.assert_allclose(checker.clamp_to_limits("right", HOME), HOME)
    with pytest.raises(ValueError):
        checker.clamp_to_limits("left", [0.0] * 5)


def test_check_arm_motion_benchmark() -> None:
    zones = [
        GLASS,
        BoxZone(
            name="monitor",
            center=(0.2, 0.0, 0.6),
            half_extents=(0.1, 0.2, 0.1),
            rpy=(0.1, 0.2, 0.3),
        ),
        KeepInBox(name="cell", center=(0.2, 0.0, 0.4), half_extents=(0.8, 1.0, 0.8)),
        HalfSpaceZone(name="table", point=(0.0, 0.0, -0.01), normal=(0.0, 0.0, -1.0)),
    ]
    checker = FenceChecker(FenceConfig(margin_m=0.03, zones=zones))
    target = HOME + 0.05
    for _ in range(20):
        checker.check_arm_motion("left", HOME, target, 0.3)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        checker.check_arm_motion("left", HOME, target, 0.3)
        timings.append(time.perf_counter() - t0)
    median_ms = float(np.median(timings)) * 1e3
    assert median_ms < 5.0, f"check_arm_motion took {median_ms:.3f} ms"


# ------------------------------------------------------------------------------ scenario


def test_glass_window_scenario() -> None:
    """Customer glass window 45 cm in front of the arm bases (normal +x)."""
    checker = FenceChecker(glass_config())

    # Home is safe on both arms with the grippers closed.
    home = checker.check(q14(HOME, HOME, 0.0, 0.0))
    assert home.ok, home.to_dict()
    assert home.arms["left"].min_clearance_m > checker.margin_m
    # The finger inflation is isotropic: a wide-open gripper at home already reaches the glass
    # margin, which is why the stroke must be measured (docs/FENCE.md).
    assert not checker.check_arm("left", HOME, 1.0).ok

    # Reaching forward past the glass is flagged, naming the glass.
    assert checker.fingertip("left", FORWARD_REACH)[0] > 0.45
    reach = checker.check(q14(FORWARD_REACH, HOME))
    assert not reach.ok and not reach.arms["left"].ok and reach.arms["right"].ok
    assert reach.arms["left"].zone == "glass_window"
    assert reach.arms["left"].min_clearance_m < 0.0  # actually through the glass

    # The swept check catches the motion toward it.
    assert not checker.check_arm_motion("left", HOME, FORWARD_REACH, 0.0).ok


def test_swept_check_catches_mid_motion_crossing() -> None:
    """Both endpoints clear a keep-out box, but the straight joint motion passes through it."""
    obstacle = BoxZone(
        name="shelf_bracket", center=(0.61, 0.31, 0.41), half_extents=(0.03, 0.03, 0.03)
    )
    checker = FenceChecker(FenceConfig(margin_m=0.03, zones=[obstacle]))

    start = checker.check_arm("left", HOME, 0.0)
    end = checker.check_arm("left", FAR_FORWARD, 0.0)
    assert start.ok and end.ok, (start, end)
    assert min(start.min_clearance_m, end.min_clearance_m) > 0.08  # well clear, not borderline
    assert checker.fingertip("left", FAR_FORWARD)[0] > 0.7

    motion = checker.check_arm_motion("left", HOME, FAR_FORWARD, 0.0)
    assert not motion.ok
    assert motion.zone == "shelf_bracket" and motion.min_clearance_m < 0.0
    n = len(checker.motion_samples(HOME, FAR_FORWARD)) - 1
    sample = int((motion.detail or "").split("sample ")[1].split(" ")[0])
    assert 0 < sample < n  # the violation is strictly inside the motion
    # the reverse motion is flagged too, and the right arm (far from the box) is not
    assert not checker.check_arm_motion("left", FAR_FORWARD, HOME, 0.0).ok
    assert checker.check_arm_motion("right", HOME, FAR_FORWARD, 0.0).ok
