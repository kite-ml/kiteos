# Copyright 2026 Kite ML.
"""Checks for deploy/install.sh and its udev/limits files.

The installer is only ever run with --help or --dry-run here, against a
throwaway HOME and --stack-dir, and every test asserts that nothing was
created or changed. No packages, network, sudo or hardware are involved.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy"
INSTALL = DEPLOY / "install.sh"
BUNDLE = DEPLOY / "runtime" / "kiteos-runtime.bundle"
CAN_RULES = DEPLOY / "udev" / "80-kiteos-can.rules"
QUEST_RULES = DEPLOY / "udev" / "70-kiteos-quest.rules"
MEMLOCK = DEPLOY / "limits" / "99-kiteos-memlock.conf"
STATION_FILES = (REPO / "config" / "station.yaml", REPO / "config" / "station.example.yaml")

pytestmark = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="the installer targets macOS and Linux"
)


def _pin(key: str) -> str:
    for line in (DEPLOY / "runtime" / "PIN").read_text().splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].split("#", 1)[0].strip()
    raise KeyError(key)


PIN_COMMIT = _pin("commit")
PIN_BRANCH = _pin("branch")


def _bash() -> str:
    # /bin/bash on macOS is bash 3.2, the oldest shell the installer must support.
    if Path("/bin/bash").exists():
        return "/bin/bash"
    found = shutil.which("bash")
    assert found, "bash is required"
    return found


def _isolated_env(home: Path) -> dict[str, str]:
    """Environment whose HOME is a temp dir and that cannot reach user tool state."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("XDG_", "UV_", "CARGO", "RUSTUP", "KITEOS_", "VIRTUAL_ENV", "HOMEBREW_"))
    }
    # rustup's cargo proxy creates $HOME/.rustup even for `cargo --version`.
    path = os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep)
    env["PATH"] = os.pathsep.join(p for p in path if p and ".cargo" not in p and ".rustup" not in p)
    env["HOME"] = str(home)
    env["NO_COLOR"] = "1"
    return env


def _run(
    args: list[str], env: dict[str, str] | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), str(INSTALL), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _station_state() -> dict[str, tuple[bool, int | None]]:
    return {
        str(p): (p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in STATION_FILES
    }


def _flat(output: str) -> str:
    """Collapse whitespace so assertions do not depend on line wrapping."""
    return re.sub(r"\s+", " ", output)


def _assert_no_executed_commands(output: str) -> None:
    for line in output.splitlines():
        if line.lstrip().startswith("$ "):
            pytest.fail(f"dry run executed a command: {line!r}")


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    home = tmp_path / "home"
    home.mkdir()
    stack = tmp_path / "stack"
    return home, stack, _isolated_env(home)


# --- static checks --------------------------------------------------------


@pytest.mark.parametrize(
    "script", [INSTALL, *sorted((DEPLOY / "lib").glob("*.sh"))], ids=lambda p: p.name
)
def test_scripts_parse(script: Path) -> None:
    result = subprocess.run([_bash(), "-n", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_install_script_is_executable_bash() -> None:
    assert os.access(INSTALL, os.X_OK), "deploy/install.sh must be executable"
    lines = INSTALL.read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert "# Copyright 2026 Kite ML." in lines[:3]
    assert "set -euo pipefail" in lines


def test_help_lists_every_option() -> None:
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Usage: deploy/install.sh" in out
    for flag in (
        "--stack-dir",
        "--runtime-dir",
        "--extras",
        "--skip-system",
        "--with-udev",
        "--no-recorder",
        "--no-prewarm",
        "--dry-run",
        "--help",
    ):
        assert flag in out, flag
    assert "kiteos doctor" in out
    assert "KiteOS" in out


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["--bogus"], 2, "unknown option: --bogus"),
        (["--stack-dir"], 2, "--stack-dir needs a value"),
        (["--dry-run", "--extras", "manipulation;rm"], 1, "invalid extra name"),
        (["--dry-run", "--stack-dir="], 1, "--stack-dir must not be empty"),
    ],
)
def test_rejects_bad_arguments(
    args: list[str], code: int, message: str, sandbox: tuple[Path, Path, dict[str, str]]
) -> None:
    home, _, env = sandbox
    result = _run(args, env=env)
    assert result.returncode == code
    assert message in result.stderr
    assert _tree(home) == []


# --- dry runs -------------------------------------------------------------


def test_dry_run_fresh_machine_prints_all_steps_without_side_effects(
    tmp_path: Path, sandbox: tuple[Path, Path, dict[str, str]]
) -> None:
    home, stack, env = sandbox
    before = _station_state()

    result = _run(["--dry-run", "--stack-dir", str(stack)], env=env, cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    flat = _flat(out)
    dimos = stack / "kiteos-runtime"

    positions = [out.find(f"[{n}/10]") for n in range(1, 11)]
    assert all(p >= 0 for p in positions), positions
    assert positions == sorted(positions), "steps must print in order"

    staging = f"{dimos}.kiteos-staging"
    expected = [
        f"git clone https://github.com/dimensionalOS/dimos.git {staging}",
        f"mv {staging} {dimos}",
        f"git bundle verify {BUNDLE}",
        f"git fetch {BUNDLE} {PIN_BRANCH}:{PIN_BRANCH}",
        f"git checkout -B {PIN_BRANCH} {PIN_COMMIT}",
        "GIT_LFS_SKIP_SMUDGE=1",
        "sync --frozen --inexact --python 3.12 --extra manipulation --extra learning",
        "build --release --locked -p dimos-memory-recorder",
        f"{dimos}/dimos/experimental/memory/rust/result/bin/dimos-memory-recorder",
        f"pip install --python {dimos}/.venv/bin/python --no-deps -e {REPO}",
        "git lfs pull --include data/.lfs/dual_openyam_abc_box_v2.tar.gz --exclude=",
        "isolated_python_run_command",
        "kiteos doctor",
        "uninstalls the editable KiteOS package",
        "dry run complete, nothing was changed",
    ]
    for text in expected:
        assert text in flat, f"missing from dry-run output: {text!r}"
    if sys.platform == "darwin":
        assert "addressed by USB serial" in flat
    else:
        assert "pass --with-udev" in flat
    assert "can_left" in flat and "can_right" in flat

    _assert_no_executed_commands(out)
    assert not stack.exists()
    assert _tree(home) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["home"]
    assert _station_state() == before


def test_dry_run_honours_skip_flags(sandbox: tuple[Path, Path, dict[str, str]]) -> None:
    home, stack, env = sandbox
    result = _run(
        [
            "--dry-run",
            "--stack-dir",
            str(stack),
            "--skip-system",
            "--no-recorder",
            "--no-prewarm",
            "--with-udev",
            "--extras",
            "manipulation",
        ],
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    flat = _flat(result.stdout)
    assert "skip --skip-system" in flat
    assert "skip --no-recorder" in flat
    assert "cargo build" not in flat and "sh.rustup.rs" not in flat
    assert "isolated_python_run_command" not in flat
    assert "--extra manipulation" in flat and "--extra learning" not in flat
    if sys.platform == "darwin":
        assert "--with-udev only applies to Linux" in flat
    else:
        assert "udevadm control --reload-rules" in flat
        assert "/etc/udev/rules.d/80-kiteos-can.rules" in flat
    _assert_no_executed_commands(result.stdout)
    assert not stack.exists()
    assert _tree(home) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_dry_run_leaves_existing_checkout_alone(sandbox: tuple[Path, Path, dict[str, str]]) -> None:
    home, stack, env = sandbox
    dimos = stack / "dimos"
    dimos.mkdir(parents=True)
    git = ["git", "-C", str(dimos), "-c", "user.name=KiteOS test", "-c", "user.email=test@kiteos.invalid"]
    subprocess.run([*git, "-c", "init.defaultBranch=main", "init", "-q"], env=env, check=True)
    (dimos / "README.md").write_text("not a runtime checkout\n")
    subprocess.run([*git, "add", "README.md"], env=env, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], env=env, check=True)
    (dimos / "README.md").write_text("local edit\n")
    head = subprocess.run(
        ["git", "-C", str(dimos), "rev-parse", "HEAD"], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    stack_before = _tree(stack)
    home_before = _tree(home)

    result = _run(["--dry-run", "--stack-dir", str(stack)], env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    flat = _flat(result.stdout)
    assert f"not the KiteOS pin {PIN_COMMIT}; leaving it as is" in flat
    assert "local modifications" in flat
    assert "previous folder name" in flat  # <stack>/dimos from before the rename is reused, not recloned
    assert f"fetch {BUNDLE} {PIN_BRANCH}:{PIN_BRANCH}" in flat  # printed as advice only
    assert "git clone" not in flat
    _assert_no_executed_commands(result.stdout)
    assert _tree(stack) == stack_before
    assert _tree(home) == home_before
    assert (dimos / "README.md").read_text() == "local edit\n"
    after = subprocess.run(
        ["git", "-C", str(dimos), "rev-parse", "HEAD"], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert after == head


def test_dry_run_refuses_non_git_runtime_dir(sandbox: tuple[Path, Path, dict[str, str]]) -> None:
    home, stack, env = sandbox
    runtime = stack / "kiteos-runtime"
    runtime.mkdir(parents=True)
    (runtime / "keep.txt").write_text("user data\n")

    result = _run(["--dry-run", "--stack-dir", str(stack)], env=env)

    assert result.returncode == 1
    assert "is not a git checkout" in result.stderr
    assert _tree(stack) == ["kiteos-runtime", "kiteos-runtime/keep.txt"]
    assert (runtime / "keep.txt").read_text() == "user data\n"
    assert _tree(home) == []


def test_dry_run_refuses_foreign_staging_dir(sandbox: tuple[Path, Path, dict[str, str]]) -> None:
    """Only a clone the installer left behind may be removed, never other data."""
    home, stack, env = sandbox
    staging = stack / "kiteos-runtime.kiteos-staging"
    staging.mkdir(parents=True)
    (staging / "notes.txt").write_text("keep me\n")

    result = _run(["--dry-run", "--stack-dir", str(stack)], env=env)

    assert result.returncode == 1
    assert "is not a KiteOS staging clone" in result.stderr
    assert (staging / "notes.txt").read_text() == "keep me\n"
    assert _tree(home) == []


# --- udev / limits --------------------------------------------------------


def _rule_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _can_rule_serials() -> dict[str, str]:
    names: dict[str, str] = {}
    for line in _rule_lines(CAN_RULES):
        serial = re.search(r'ATTRS\{serial\}=="([^"]+)"', line)
        name = re.search(r'NAME="([^"]+)"', line)
        assert serial and name, line
        names[name.group(1)] = serial.group(1)
    return names


def test_can_udev_rules_name_both_adapters() -> None:
    assert CAN_RULES.read_text().startswith("# Copyright 2026 Kite ML.")
    lines = _rule_lines(CAN_RULES)
    assert len(lines) == 2
    for line in lines:
        assert 'SUBSYSTEM=="net"' in line
        assert 'ACTION=="add"' in line
        assert 'ATTRS{idVendor}=="1d50"' in line
        assert 'ATTRS{idProduct}=="606f"' in line
    serials = _can_rule_serials()
    assert set(serials) == {"can_left", "can_right"}
    assert serials["can_left"] != serials["can_right"]
    assert all(len(name) <= 15 for name in serials)  # IFNAMSIZ - 1


@pytest.mark.parametrize("station_file", ["station.yaml", "station.example.yaml"])
def test_can_udev_rules_match_station_config(station_file: str) -> None:
    """Left/right must agree between the udev names and the station's USB serials."""
    yaml = pytest.importorskip("yaml")
    path = REPO / "config" / station_file
    if not path.exists():
        pytest.skip(f"config/{station_file} not present")
    can = (yaml.safe_load(path.read_text()) or {}).get("can")
    if not isinstance(can, dict) or not all(isinstance(can.get(s), dict) for s in ("left", "right")):
        pytest.skip(f"config/{station_file} has no can.left/can.right mapping")
    serials = _can_rule_serials()
    for side in ("left", "right"):
        assert can[side].get("linux") == f"can_{side}", side
        serial = str(can[side].get("darwin"))
        if not serial.startswith("REPLACE"):
            assert serial == serials[f"can_{side}"], side


def test_quest_udev_rule() -> None:
    assert QUEST_RULES.read_text().startswith("# Copyright 2026 Kite ML.")
    (line,) = _rule_lines(QUEST_RULES)
    assert 'SUBSYSTEM=="usb"' in line
    assert 'ATTR{idVendor}=="2833"' in line
    assert 'GROUP="plugdev"' in line
    assert 'TAG+="uaccess"' in line


def test_memlock_limits_template() -> None:
    assert MEMLOCK.read_text().startswith("# Copyright 2026 Kite ML.")
    (line,) = _rule_lines(MEMLOCK)
    domain, kind, item, value = line.split()
    assert (domain, kind, item) == ("KITEOS_USER", "-", "memlock")
    # The runtime's MemlockConfiguratorLinux needs 64 MiB (value is in KiB).
    assert int(value) * 1024 >= 64 * 1024 * 1024
