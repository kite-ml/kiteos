# Copyright 2026 Kite ML.
"""Station config (kiteos.config): loading, per-platform resolution, paths and validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from kiteos import brand
from kiteos.config import (
    CAMERA_ROLES,
    PlatformValues,
    StationConfig,
    StationError,
    is_placeholder,
    platform_key,
    repo_root,
    slugify,
)

REPO = Path(__file__).resolve().parents[1]
LEFT_SERIAL = "006C003E4845500F20303750"
RIGHT_SERIAL = "003900374845500F20303750"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITEOS_STATION", raising=False)
    monkeypatch.delenv("KITEOS_ROOT", raising=False)


def minimal(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "station": "test-rig",
        "task": "Close the Box!",
        "can": {"left": {"darwin": "AAA", "linux": "can_left"}, "right": {"darwin": "BBB", "linux": "can_right"}},
        "cameras": {
            "left_wrist": {"darwin": 0, "linux": "/dev/v4l/by-id/left"},
            "right_wrist": 2,
            "top": {"darwin": 1, "linux": "/dev/v4l/by-id/top"},
        },
        "fence": "fence.yaml",
        "hub": {"repo_id": "kiteml/test"},
    }
    data.update(overrides)
    return data


def write_station(directory: Path, data: dict[str, object], name: str = "station.yaml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(yaml.safe_dump(data))
    return path


# --------------------------------------------------------------------------- the rig's files


def test_repo_station_on_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    station = StationConfig.load(REPO / "config" / "station.yaml")
    assert platform_key() == "darwin"
    assert station.station == "kite-lab-dual-openyam"
    assert station.task == "close the box"
    assert station.can_port("left") == LEFT_SERIAL
    assert station.can_port("right") == RIGHT_SERIAL
    assert [station.camera(r) for r in CAMERA_ROLES] == [0, 2, 1]
    assert station.hub.repo_id == "kiteml/dual-openyam-close-box"
    assert station.hub.private is True
    assert station.prefix == "close-box"
    assert station.policy.device == "auto"
    assert station.checkpoint_path() == Path.home() / "teleop_stack" / "policies" / "act-close-box-step30000"


def test_repo_station_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    station = StationConfig.load(REPO / "config" / "station.yaml")
    assert station.can_port("left") == "can_left"
    assert station.can_port("right") == "can_right"
    for role in CAMERA_ROLES:
        selector = station.camera(role)
        assert isinstance(selector, str) and selector.startswith("/dev/v4l/")
        assert is_placeholder(selector), "linux cameras stay placeholders until discovered on the devbox"


def test_other_platforms_use_linux_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert platform_key() == "linux"
    station = StationConfig.load(REPO / "config" / "station.yaml")
    assert station.can_port("left") == "can_left"


def test_default_path_is_repo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    assert repo_root() == REPO
    station = StationConfig.load()
    assert station.source == (REPO / "config" / "station.yaml").resolve()


def test_repo_fence_path_is_absolute_and_exists() -> None:
    station = StationConfig.load(REPO / "config" / "station.yaml")
    fence = station.fence_path()
    assert fence.is_absolute()
    assert fence == (REPO / "config" / "fence.yaml").resolve()
    assert fence.is_file()


def test_example_station_is_valid_and_marks_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    station = StationConfig.load(REPO / "config" / "station.example.yaml")
    assert is_placeholder(station.can_port("left"))
    assert station.hub.private is True


def test_home_paths_are_expanded() -> None:
    station = StationConfig.load(REPO / "config" / "station.yaml")
    home = Path.home()
    assert station.recordings_path() == home / "teleop_stack" / "dimos" / "recordings"
    assert station.datasets_path() == home / "teleop_stack" / "dimos" / "datasets"
    assert station.dimos_root_path() == home / "teleop_stack" / "dimos"


# --------------------------------------------------------------------------- lookup


def test_env_var_selects_station(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_station(tmp_path / "site", minimal(station="from-env"))
    monkeypatch.setenv("KITEOS_STATION", str(path))
    assert StationConfig.load().station == "from-env"


def test_explicit_path_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = write_station(tmp_path / "a", minimal(station="from-env"))
    explicit = write_station(tmp_path / "b", minimal(station="explicit"))
    monkeypatch.setenv("KITEOS_STATION", str(env_path))
    assert StationConfig.load(explicit).station == "explicit"


def test_local_override_merged_on_default_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITEOS_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    config_dir = tmp_path / "config"
    write_station(config_dir, minimal())
    (config_dir / "station.local.yaml").write_text(
        yaml.safe_dump({"cameras": {"top": {"linux": "/dev/v4l/by-path/usb-top"}}, "hub": {"private": False}})
    )
    station = StationConfig.load()
    assert station.camera("top") == "/dev/v4l/by-path/usb-top"
    assert station.camera("left_wrist") == "/dev/v4l/by-id/left"  # untouched keys survive the merge
    assert station.hub.repo_id == "kiteml/test"
    assert station.hub.private is False
    # Naming the same station file explicitly (or via $KITEOS_STATION) keeps the local override:
    # site values such as the customer fence must never be dropped silently.
    explicit = StationConfig.load(config_dir / "station.yaml")
    assert explicit.camera("top") == "/dev/v4l/by-path/usb-top"
    assert explicit.local_source == (config_dir / "station.local.yaml").resolve()
    assert "station.local.yaml" in explicit.describe_sources()
    monkeypatch.setenv("KITEOS_STATION", str(config_dir / "station.yaml"))
    assert StationConfig.load().camera("top") == "/dev/v4l/by-path/usb-top"


def test_data_dirs_must_be_separate(tmp_path: Path) -> None:
    for recordings, datasets in (("~/data", "~/data"), ("~/data", "~/data/datasets"), ("~/d/rec", "~/d")):
        with pytest.raises(StationError, match="separate"):
            StationConfig.load(write_station(tmp_path / recordings.replace("/", "_").replace("~", "h"), minimal(recordings_dir=recordings, datasets_dir=datasets)))


# --------------------------------------------------------------------------- resolution


def test_relative_paths_resolve_against_station_file(tmp_path: Path) -> None:
    path = write_station(tmp_path / "site", minimal(recordings_dir="rec", fence="fence.yaml"))
    station = StationConfig.load(path)
    assert station.fence_path() == (tmp_path / "site" / "fence.yaml").resolve()
    assert station.recordings_path() == (tmp_path / "site" / "rec").resolve()


def test_config_prefixed_fence_resolves_from_repo_layout(tmp_path: Path) -> None:
    config_dir = tmp_path / "repo" / "config"
    path = write_station(config_dir, minimal(fence="config/fence.yaml"))
    (config_dir / "fence.yaml").write_text("version: 1\n")
    assert StationConfig.load(path).fence_path() == (config_dir / "fence.yaml").resolve()


def test_absolute_fence_path(tmp_path: Path) -> None:
    fence = tmp_path / "elsewhere" / "fence.yaml"
    path = write_station(tmp_path / "site", minimal(fence=str(fence)))
    assert StationConfig.load(path).fence_path() == fence


def test_scalar_value_applies_to_both_platforms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    station = StationConfig.load(write_station(tmp_path, minimal()))
    for plat in ("darwin", "linux"):
        monkeypatch.setattr(sys, "platform", plat)
        assert station.camera("right_wrist") == 2
    assert PlatformValues.model_validate("x").for_platform("linux") == "x"


def test_value_coercion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    data = minimal(
        can={"left": {"darwin": 12345, "linux": "can_left"}, "right": {"darwin": "B", "linux": "can_right"}},
        cameras={"left_wrist": "3", "right_wrist": "~/cams/right", "top": 1},
    )
    station = StationConfig.load(write_station(tmp_path, data))
    assert station.can_port("left") == "12345"
    assert station.camera("left_wrist") == 3
    assert station.camera("right_wrist") == str(Path.home() / "cams" / "right")


def test_missing_platform_value_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    data = minimal(can={"left": {"darwin": "A"}, "right": {"darwin": "B", "linux": ""}})
    station = StationConfig.load(write_station(tmp_path, data))
    assert station.can_port("left") is None
    assert station.can_port("right") is None


def test_unknown_side_or_role_rejected(tmp_path: Path) -> None:
    station = StationConfig.load(write_station(tmp_path, minimal()))
    with pytest.raises(ValueError):
        station.can_port("middle")
    with pytest.raises(ValueError):
        station.camera("front")


def test_prefix_defaults_to_task_slug(tmp_path: Path) -> None:
    station = StationConfig.load(write_station(tmp_path, minimal()))
    assert station.prefix == "close-the-box"
    assert slugify("  ") == "session"
    explicit = StationConfig.load(write_station(tmp_path / "b", minimal(session_prefix="close-box")))
    assert explicit.prefix == "close-box"


def test_quest_adb_off(tmp_path: Path) -> None:
    path = tmp_path / "station.yaml"
    text = yaml.safe_dump(minimal()) + "quest:\n  adb: off\n"
    path.write_text(text)
    station = StationConfig.load(path)
    assert station.quest.adb is False  # YAML 1.1 reads `off` as false
    assert not station.quest.adb_enabled
    assert StationConfig.load(write_station(tmp_path / "b", minimal())).quest.adb_enabled


# --------------------------------------------------------------------------- errors


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StationError, match="not found"):
        StationConfig.load(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cameras": {"left_wrist": 0, "right_wrist": 2}}, "missing camera"),
        ({"can": {"left": "a"}}, "missing CAN"),
        ({"station": "two words"}, "single word"),
        ({"hub": {"repo_id": "no-slash"}}, "repo_id"),
        ({"session_prefix": "Close Box"}, "session_prefix"),
        ({"surprise": 1}, "surprise"),
        ({"policy": {"device": "tpu"}}, "device"),
    ],
)
def test_invalid_station(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    path = write_station(tmp_path, minimal(**overrides))
    with pytest.raises(StationError, match=message):
        StationConfig.load(path)


def test_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "station.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(StationError, match="mapping"):
        StationConfig.load(path)


def test_brand() -> None:
    assert brand.BRAND_NAME == "KiteOS"
    assert brand.title("close the box") == "KiteOS · close the box"
    assert "KiteOS 0.1.0" in brand.banner("0.1.0")
