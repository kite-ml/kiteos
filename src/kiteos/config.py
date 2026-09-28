# Copyright 2026 Kite ML.
"""Station configuration: everything site-specific about one KiteOS rig.

A station file (YAML) names the rig, its task, the CAN adapters and cameras per
host OS, the fence file, data directories, the Hub repository and the policy.
Lookup order for :meth:`StationConfig.load` without an explicit path:

1. ``$KITEOS_STATION``
2. ``<repo>/config/station.yaml``

Whichever file is used (also an explicit ``--station`` path), its sibling
``<name>.local.yaml`` (``config/station.local.yaml`` for the default, git-ignored)
is deep-merged on top when it exists, so site values such as the customer fence
are never silently dropped by naming the same station file explicitly.

Per-platform values (``can``, ``cameras``) are mappings ``{darwin: ..., linux: ...}``
or a single scalar used on both. They resolve by ``sys.platform`` at call time:
``"darwin"`` uses the darwin value, every other platform the linux value.

Relative paths resolve against the station file's directory, falling back to its
parent, so both ``fence: fence.yaml`` and ``fence: config/fence.yaml`` work for a
station file at ``<repo>/config/station.yaml``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from kiteos.yamlio import load_yaml_unique

Side = Literal["left", "right"]
CameraRole = Literal["left_wrist", "right_wrist", "top"]
PlatformKey = Literal["darwin", "linux"]

SIDES: tuple[Side, ...] = ("left", "right")
CAMERA_ROLES: tuple[CameraRole, ...] = ("left_wrist", "right_wrist", "top")
STATION_ENV = "KITEOS_STATION"
ROOT_ENV = "KITEOS_ROOT"
PLACEHOLDER_MARK = "REPLACE"


class StationError(RuntimeError):
    """The station file is missing, unreadable or invalid."""


def repo_root() -> Path:
    """The KiteOS checkout (``$KITEOS_ROOT`` or the source tree of this module)."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def platform_key() -> PlatformKey:
    """``"darwin"`` on macOS, ``"linux"`` for every other platform."""
    return "darwin" if sys.platform == "darwin" else "linux"


def is_placeholder(value: object) -> bool:
    """True for unfilled template values such as ``/dev/v4l/by-path/REPLACE``."""
    return isinstance(value, str) and PLACEHOLDER_MARK in value


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "session"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlatformValues(_Model):
    """One setting that differs between macOS and Linux hosts."""

    darwin: int | str | None = None
    linux: int | str | None = None

    @model_validator(mode="before")
    @classmethod
    def _scalar_for_both(cls, value: Any) -> Any:
        if value is None or isinstance(value, (int, str)):
            return {"darwin": value, "linux": value}
        return value

    def for_platform(self, key: PlatformKey) -> int | str | None:
        return self.darwin if key == "darwin" else self.linux


class HubConfig(_Model):
    repo_id: str = Field(min_length=3, pattern=r"^[^/\s]+/[^/\s]+$")
    private: bool = True


class PolicyConfig(_Model):
    checkpoint: str | None = None
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    # Rollout chunk guards (kiteos.rollout.guards): refuse a chunk whose first arm target is
    # further than this from the measured pose (validated on the rig: 0.3 rad) ...
    max_start_jump: float = Field(0.3, gt=0, le=0.5)
    # ... or that moves an arm joint faster than this at execution rate (legacy kite_rollout: 3.0).
    max_joint_speed: float = Field(3.0, gt=0, le=3.0)


class QuestConfig(_Model):
    port: int = Field(8080, ge=1, le=65535)
    upstream: str = "https://127.0.0.1:8443"
    # "auto" finds adb on PATH; "off"/false disables adb reverse; anything else is an adb path.
    adb: str | bool = "auto"

    @property
    def adb_enabled(self) -> bool:
        return not (self.adb is False or str(self.adb).lower() in {"off", "no", "none", "false"})


class StationConfig(_Model):
    """Validated contents of a station YAML file."""

    station: str = Field(min_length=1)
    task: str = Field(min_length=1)
    session_prefix: str | None = None
    can: dict[Side, PlatformValues]
    cameras: dict[CameraRole, PlatformValues]
    fence: str = "config/fence.yaml"
    recordings_dir: str = "~/teleop_stack/dimos/recordings"
    datasets_dir: str = "~/teleop_stack/dimos/datasets"
    hub: HubConfig
    policy: PolicyConfig = PolicyConfig()
    dimos_root: str = "~/teleop_stack/dimos"
    viser_host: str = "127.0.0.1"
    quest: QuestConfig = QuestConfig()

    _source: Path | None = PrivateAttr(default=None)
    _local: Path | None = PrivateAttr(default=None)

    @field_validator("station")
    @classmethod
    def _station_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("station must be a single word of letters, digits, '.', '_' or '-'")
        return value

    @field_validator("session_prefix")
    @classmethod
    def _prefix(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", value):
            raise ValueError("session_prefix must be lowercase words joined by '-', e.g. close-box")
        return value

    @field_validator("can")
    @classmethod
    def _can_both_sides(cls, value: dict[str, PlatformValues]) -> dict[str, PlatformValues]:
        missing = [side for side in SIDES if side not in value]
        if missing:
            raise ValueError(f"missing CAN entries for {missing}")
        for entry in value.values():
            # CAN selectors are serials or interface names; YAML may read an all-digit serial as int.
            for key in ("darwin", "linux"):
                raw = getattr(entry, key)
                if isinstance(raw, int):
                    setattr(entry, key, str(raw))
        return value

    @model_validator(mode="after")
    def _separate_data_dirs(self) -> StationConfig:
        rec = Path(self.recordings_dir).expanduser()
        data = Path(self.datasets_dir).expanduser()
        if rec == data or rec in data.parents or data in rec.parents:
            raise ValueError(
                "recordings_dir and datasets_dir must be separate directories (neither inside the "
                "other): converting a session replaces its dataset folder"
            )
        return self

    @field_validator("cameras")
    @classmethod
    def _cameras_all_roles(cls, value: dict[str, PlatformValues]) -> dict[str, PlatformValues]:
        missing = [role for role in CAMERA_ROLES if role not in value]
        if missing:
            raise ValueError(f"missing camera entries for {missing}")
        for entry in value.values():
            for key in ("darwin", "linux"):
                raw = getattr(entry, key)
                if isinstance(raw, str) and raw.strip().isdigit():
                    setattr(entry, key, int(raw))
        return value

    # ------------------------------------------------------------------ loading

    @classmethod
    def default_path(cls) -> Path:
        env = os.environ.get(STATION_ENV)
        if env:
            return Path(env).expanduser()
        return repo_root() / "config" / "station.yaml"

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> StationConfig:
        """Load and validate a station file (see module docstring for the lookup order)."""
        source = Path(path).expanduser() if path is not None else cls.default_path()
        source = source.resolve()
        data = _read_yaml(source)
        local = local_override_path(source)
        used_local = None
        if local.is_file():
            data = _deep_merge(data, _read_yaml(local))
            used_local = local
        try:
            config = cls.model_validate(data)
        except ValidationError as exc:
            where = f"{source}" + (f" + {used_local.name}" if used_local else "")
            raise StationError(f"invalid station file {where}:\n{exc}") from exc
        config._source = source
        config._local = used_local
        return config

    @property
    def local_source(self) -> Path | None:
        """The ``*.local.yaml`` override merged on top of :attr:`source`, if any."""
        return self._local

    def describe_sources(self) -> str:
        """``station.yaml + station.local.yaml`` style description for operator panels."""
        if self._source is None:
            return "(built in code)"
        return str(self._source) + (f" + {self._local.name}" if self._local else "")

    @property
    def source(self) -> Path | None:
        """The station file this config was loaded from (None if built in code)."""
        return self._source

    # ---------------------------------------------------------------- resolving

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        """Absolute path for a station setting: ``~`` expanded, relative to the station file."""
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        base = self._source.parent if self._source is not None else Path.cwd()
        candidates = [base / path, base.parent / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    def can_port(self, side: str) -> str | None:
        """CAN selector for this host: USB serial on macOS, SocketCAN interface on Linux."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        value = self.can[side].for_platform(platform_key())  # type: ignore[index]
        return None if value in (None, "") else str(value)

    def camera(self, role: str) -> int | str | None:
        """OpenCV camera selector for this host: an index, or a device path (``~`` expanded)."""
        if role not in CAMERA_ROLES:
            raise ValueError(f"camera role must be one of {CAMERA_ROLES}, got {role!r}")
        value = self.cameras[role].for_platform(platform_key())  # type: ignore[index]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            if value.startswith("~"):
                return str(Path(value).expanduser())
        return value

    def fence_path(self) -> Path:
        return self.resolve_path(self.fence)

    def recordings_path(self) -> Path:
        return self.resolve_path(self.recordings_dir)

    def datasets_path(self) -> Path:
        return self.resolve_path(self.datasets_dir)

    def dimos_root_path(self) -> Path:
        return self.resolve_path(self.dimos_root)

    def checkpoint_path(self) -> Path | None:
        return None if not self.policy.checkpoint else self.resolve_path(self.policy.checkpoint)

    @property
    def prefix(self) -> str:
        """Recording session prefix: ``session_prefix`` or a slug of the task."""
        return self.session_prefix or slugify(self.task)


def local_override_path(source: Path) -> Path:
    """``<dir>/<stem>.local<suffix>`` next to a station file (``station.local.yaml``)."""
    return source.with_name(f"{source.stem}.local{source.suffix or '.yaml'}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StationError(
            f"station file not found: {path}\n"
            f"Copy config/station.example.yaml to config/station.yaml, or set {STATION_ENV}."
        )
    try:
        data = load_yaml_unique(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise StationError(f"cannot read station file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise StationError(f"station file {path} must contain a YAML mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
