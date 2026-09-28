#!/usr/bin/env bash
# Copyright 2026 Kite ML.
#
# KiteOS installer for macOS (Apple silicon, Intel) and Linux (Ubuntu/Debian first).
#
# Sets up system prerequisites, the pinned dimOS checkout and its virtualenv, the
# dimOS collection recorder, and installs KiteOS (editable) into the dimOS venv.
# Idempotent: each step checks what is already in place. --dry-run prints every
# step without executing anything. Never touches robot hardware: no CAN traffic,
# no cameras, no arms. Run as your normal user; it calls sudo where needed.
#
# Must run under macOS's bash 3.2 as well as modern bash (see deploy/lib/common.sh).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
# shellcheck source=deploy/lib/common.sh
. "$SCRIPT_DIR/lib/common.sh"

# --- constants ------------------------------------------------------------

TOTAL_STEPS=10
UV_MIN=0.9.25
UV_PIN=0.9.26
RUST_MIN=1.89
PYTHON_VERSION=3.12
DEFAULT_EXTRAS="manipulation learning"
LFS_ASSET=dual_openyam_abc_box_v2
RECORDER_CRATE=dimos-memory-recorder
RECORDER_RESULT_REL=dimos/experimental/memory/rust/result
PIN_FILE="$REPO_DIR/deploy/dimos/PIN"
UDEV_DIR="$REPO_DIR/deploy/udev"
LIMITS_TEMPLATE="$REPO_DIR/deploy/limits/99-kiteos-memlock.conf"
UDEV_RULES="80-kiteos-can.rules 70-kiteos-quest.rules"
LIMITS_DEST=/etc/security/limits.d/99-kiteos-memlock.conf

# Ubuntu/Debian packages: dimOS's own list (scripts/install.sh) plus the recorder
# build (cmake nasm libsqlite3-dev), CAN (iproute2 can-utils), the Quest (adb) and
# diagnostics (v4l-utils usbutils).
APT_PACKAGES="ca-certificates curl git git-lfs g++ build-essential pkg-config portaudio19-dev
libturbojpeg libgl1 libegl1 libglib2.0-0 ffmpeg libsndfile1 openssl cmake nasm libsqlite3-dev
iproute2 can-utils adb v4l-utils usbutils"
BREW_FORMULAE="git git-lfs ffmpeg portaudio libjpeg-turbo pkg-config cmake nasm sqlite"
BREW_CASK=android-platform-tools
LINUX_GROUPS="video plugdev dialout"

# --- arguments ------------------------------------------------------------

usage() {
    cat <<EOF
KiteOS installer

Usage: deploy/install.sh [options]

Installs KiteOS and its pinned dimOS runtime on this machine (macOS or Linux).
Safe to re-run: every step checks what is already in place. Never touches
robot hardware (CAN, cameras, arms).

Options:
  --stack-dir DIR    Workspace root (default: ~/teleop_stack)
  --dimos-dir DIR    dimOS checkout (default: <stack-dir>/dimos). Cloned and
                     pinned if missing; an existing checkout is only verified.
  --extras "A B"     dimOS extras for uv sync (default: "$DEFAULT_EXTRAS")
  --skip-system      Skip system packages (Homebrew/apt) and Linux group changes
  --with-udev        Linux: install the CAN/Quest udev rules and the memlock
                     limit into /etc (sudo)
  --no-recorder      Skip Rust and the dimos-memory-recorder build (collection
                     will not work without the recorder)
  --no-prewarm       Skip pre-building dimOS's isolated LeRobot environment
  --dry-run          Print every step without executing anything
  -h, --help         Show this help

Steps:
   1  system packages (Homebrew on macOS, apt on Ubuntu/Debian)
   2  uv >= $UV_MIN (installs $UV_PIN if missing)
   3  Rust >= $RUST_MIN via rustup (only to build the recorder)
   4  dimOS source at the KiteOS pin (deploy/dimos/PIN)
   5  dimOS virtualenv (uv sync --frozen, Python $PYTHON_VERSION)
   6  collection recorder (cargo build, copied into dimOS)
   7  KiteOS into the dimOS venv (editable) and station config
   8  dual OpenYAM model asset and dimOS's isolated LeRobot env
   9  devices: udev rules, groups, CAN names (Linux)
  10  next steps

After installing, run: kiteos doctor
EOF
}

need_value() {  # need_value OPTION "$#"
    if [ "$2" -lt 2 ]; then
        printf 'KiteOS installer: %s needs a value (see --help)\n' "$1" >&2
        exit 2
    fi
}

STACK_DIR="$HOME/teleop_stack"
DIMOS_DIR=""
DIMOS_DIR_SET=0
EXTRAS=$DEFAULT_EXTRAS
SKIP_SYSTEM=0
WITH_UDEV=0
NO_RECORDER=0
NO_PREWARM=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --stack-dir) need_value "$1" "$#"; STACK_DIR=$2; shift 2 ;;
        --stack-dir=*) STACK_DIR=${1#*=}; shift ;;
        --dimos-dir) need_value "$1" "$#"; DIMOS_DIR=$2; DIMOS_DIR_SET=1; shift 2 ;;
        --dimos-dir=*) DIMOS_DIR=${1#*=}; DIMOS_DIR_SET=1; shift ;;
        --extras) need_value "$1" "$#"; EXTRAS=$2; shift 2 ;;
        --extras=*) EXTRAS=${1#*=}; shift ;;
        --skip-system) SKIP_SYSTEM=1; shift ;;
        --with-udev) WITH_UDEV=1; shift ;;
        --no-recorder) NO_RECORDER=1; shift ;;
        --no-prewarm) NO_PREWARM=1; shift ;;
        --dry-run) KITEOS_DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *)
            printf 'KiteOS installer: unknown option: %s (see --help)\n' "$1" >&2
            exit 2
            ;;
    esac
done

[ -n "$STACK_DIR" ] || die "--stack-dir must not be empty"
[ -n "$DIMOS_DIR" ] || [ "$DIMOS_DIR_SET" = 0 ] || die "--dimos-dir must not be empty"
STACK_DIR=$(expand_path "$STACK_DIR")
if [ -z "$DIMOS_DIR" ]; then
    DIMOS_DIR="$STACK_DIR/dimos"
fi
DIMOS_DIR=$(expand_path "$DIMOS_DIR")

EXTRA_ARGS=()
set -f
for extra in $EXTRAS; do
    case "$extra" in
        *[!a-z0-9-]* | -*) die "invalid extra name: '$extra'" ;;
    esac
    EXTRA_ARGS+=(--extra "$extra")
done
set +f

VENV_PY="$DIMOS_DIR/.venv/bin/python"
RECORDER_RESULT="$DIMOS_DIR/$RECORDER_RESULT_REL"
RECORDER_BIN="$RECORDER_RESULT/bin/$RECORDER_CRATE"
DRY=$KITEOS_DRY_RUN

# --- platform -------------------------------------------------------------

OS_NAME=$(uname -s)
ARCH=$(uname -m)
DISTRO_ID=""
DISTRO_LIKE=""
DISTRO_PRETTY=""
BREW_BIN=""
BREW_PREFIX=""

case "$OS_NAME" in
    Darwin)
        PLATFORM=macos
        DISTRO_PRETTY="macOS $(sw_vers -productVersion 2>/dev/null || echo '?')"
        BREW_BIN=$(command -v brew 2>/dev/null || true)
        for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
            if [ -z "$BREW_BIN" ] && [ -x "$candidate" ]; then
                BREW_BIN=$candidate
            fi
        done
        if [ -n "$BREW_BIN" ]; then
            BREW_PREFIX=$(cd "$(dirname "$BREW_BIN")/.." && pwd)
        fi
        ;;
    Linux)
        PLATFORM=linux
        if [ -r /etc/os-release ]; then
            DISTRO_ID=$(. /etc/os-release && printf '%s' "${ID:-}")
            DISTRO_LIKE=$(. /etc/os-release && printf '%s' "${ID_LIKE:-}")
            DISTRO_PRETTY=$(. /etc/os-release && printf '%s' "${PRETTY_NAME:-Linux}")
        else
            DISTRO_PRETTY=Linux
        fi
        ;;
    *)
        die "unsupported OS '$OS_NAME': KiteOS supports macOS and Linux"
        ;;
esac

is_apt_distro() {
    case " $DISTRO_ID $DISTRO_LIKE " in
        *" ubuntu "* | *" debian "*) return 0 ;;
    esac
    return 1
}

# Tools installed by this script land here; Homebrew goes first on macOS so its
# cmake/sqlite win over e.g. a Miniconda install earlier on PATH.
PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ -n "$BREW_PREFIX" ]; then
    PATH="$BREW_PREFIX/bin:$BREW_PREFIX/sbin:$PATH"
fi
export PATH
# A caller's active venv or project-env override must not redirect the dimOS venv.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT

if [ "$(id -u)" -eq 0 ] && [ "${KITEOS_ALLOW_ROOT:-0}" != 1 ]; then
    if [ "$DRY" = 1 ]; then
        warn "running as root: a real install must run as the station user (it calls sudo itself)"
    else
        die "run the installer as the station user, not root; it calls sudo where needed (override: KITEOS_ALLOW_ROOT=1)"
    fi
fi

[ -r "$PIN_FILE" ] || die "missing $PIN_FILE"
PIN_UPSTREAM=$(pin_value "$PIN_FILE" upstream)
PIN_BRANCH=$(pin_value "$PIN_FILE" branch)
PIN_COMMIT=$(pin_value "$PIN_FILE" commit)
PIN_BASE=$(pin_value "$PIN_FILE" base)
PIN_BUNDLE_REL=$(pin_value "$PIN_FILE" bundle)
[ -n "$PIN_UPSTREAM" ] && [ -n "$PIN_BRANCH" ] && [ -n "$PIN_COMMIT" ] && [ -n "$PIN_BUNDLE_REL" ] ||
    die "$PIN_FILE must define upstream, branch, commit and bundle"
PIN_BUNDLE="$REPO_DIR/$PIN_BUNDLE_REL"
PIN_SHORT=$(printf '%s' "$PIN_COMMIT" | cut -c1-9)

KITEOS_VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$REPO_DIR/src/kiteos/__init__.py" 2>/dev/null | first_line || true)

# --- steps ----------------------------------------------------------------

brew_formula_installed() {
    [ -n "$BREW_PREFIX" ] && [ -e "$BREW_PREFIX/opt/$1" ]
}

step_system_macos() {
    if ! xcode-select -p >/dev/null 2>&1; then
        if [ "$DRY" = 1 ]; then
            warn "Xcode Command Line Tools missing: a real run stops here (run: xcode-select --install)"
        else
            die "Xcode Command Line Tools are required (C/C++ compiler for dimOS). Run: xcode-select --install, then re-run"
        fi
    else
        ok "Xcode Command Line Tools"
    fi
    if [ -z "$BREW_BIN" ]; then
        if [ "$DRY" = 1 ]; then
            warn "Homebrew not found: a real run stops here (install it from https://brew.sh)"
            return 0
        fi
        die "Homebrew is required: install it from https://brew.sh, then re-run"
    fi
    local missing=() f
    for f in $BREW_FORMULAE; do
        if [ "$f" = git ] && have git && git --version >/dev/null 2>&1; then
            continue
        fi
        brew_formula_installed "$f" || missing+=("$f")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        run "$BREW_BIN" install ${missing[@]+"${missing[@]}"}
    else
        ok "Homebrew formulae: $BREW_FORMULAE"
    fi
    if have adb || [ -d "$BREW_PREFIX/Caskroom/$BREW_CASK" ]; then
        ok "adb ($BREW_CASK)"
    else
        run "$BREW_BIN" install --cask "$BREW_CASK"
    fi
}

apt_installed() {
    [ "$(dpkg-query -W -f='${db:Status-Status}' "$1" 2>/dev/null || true)" = installed ]
}

apt_available() {
    apt-cache show "$1" >/dev/null 2>&1
}

step_system_linux() {
    if ! is_apt_distro; then
        warn "no automatic package install for '${DISTRO_ID:-unknown}' Linux; install these (or your distro's equivalents) manually and re-run with --skip-system"
        info "C/C++ toolchain, git, git-lfs, pkg-config, curl, openssl, ca-certificates, cmake, nasm,"
        info "portaudio (dev), libturbojpeg, libGL/libEGL, glib2, ffmpeg, libsndfile, sqlite3 (dev),"
        info "iproute2, can-utils, adb (android-tools), v4l-utils, usbutils"
        info "continuing best-effort"
        return 0
    fi
    have apt-get || die "apt-get not found on $DISTRO_PRETTY"
    local missing=() install=() p
    for p in $APT_PACKAGES; do
        # Debian names the TurboJPEG runtime libturbojpeg0; Ubuntu ships libturbojpeg.
        if [ "$p" = libturbojpeg ] && apt_installed libturbojpeg0; then
            continue
        fi
        apt_installed "$p" || missing+=("$p")
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        ok "apt packages already installed"
        return 0
    fi
    info "missing: ${missing[*]}"
    run_sudo apt-get update
    # Resolve distro-specific names against the freshly updated package lists.
    for p in ${missing[@]+"${missing[@]}"}; do
        if [ "$p" = libturbojpeg ] && ! apt_available libturbojpeg && apt_available libturbojpeg0; then
            p=libturbojpeg0
        fi
        install+=("$p")
    done
    run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y ${install[@]+"${install[@]}"}
}

step_system() {
    step 1 "$TOTAL_STEPS" "System packages"
    if [ "$SKIP_SYSTEM" = 1 ]; then
        skip "--skip-system"
        return 0
    fi
    if [ "$PLATFORM" = macos ]; then
        step_system_macos
    else
        step_system_linux
    fi
    # git-lfs is configured per repository in step 8 (as dimOS itself does), so the
    # user's global git config is left alone.
    if have git-lfs; then
        ok "git-lfs $(git-lfs version 2>/dev/null | awk '{print $1}' | first_line)"
    elif [ "$DRY" != 1 ]; then
        warn "git-lfs not found: the dual OpenYAM model asset cannot be fetched (step 8)"
    fi
}

UV_BIN=""

step_uv() {
    step 2 "$TOTAL_STEPS" "uv (>= $UV_MIN)"
    UV_BIN=$(command -v uv 2>/dev/null || true)
    if [ -n "$UV_BIN" ]; then
        local v
        v=$("$UV_BIN" --version 2>/dev/null | awk '{print $2}' | first_line || true)
        if version_ge "${v:-0}" "$UV_MIN"; then
            ok "uv $v ($UV_BIN)"
            return 0
        fi
        info "uv $v is older than $UV_MIN"
        if ! run "$UV_BIN" self update "$UV_PIN"; then
            die "could not update uv; update it to >= $UV_MIN with the tool that installed it (e.g. brew upgrade uv), then re-run"
        fi
        return 0
    fi
    if [ "$DRY" != 1 ]; then
        have curl || die "curl is required to install uv"
    fi
    run_shell "curl -LsSf https://astral.sh/uv/$UV_PIN/install.sh | sh"
    if [ -n "${XDG_BIN_HOME:-}" ]; then
        PATH="$XDG_BIN_HOME:$PATH"
    fi
    UV_BIN=$(command -v uv 2>/dev/null || true)
    if [ -z "$UV_BIN" ]; then
        [ "$DRY" = 1 ] || die "uv was installed but is not on PATH; open a new shell and re-run"
        UV_BIN=uv
    fi
}

CARGO_CMD=(cargo)

cargo_version() {  # prints the cargo version of "$@" or nothing
    "$@" --version 2>/dev/null | awk '{print $2}' | first_line || true
}

step_rust() {
    step 3 "$TOTAL_STEPS" "Rust toolchain (>= $RUST_MIN, for the recorder)"
    if [ "$NO_RECORDER" = 1 ]; then
        skip "--no-recorder"
        return 0
    fi
    if [ -x "$RECORDER_BIN" ]; then
        skip "recorder already built; Rust not needed"
        return 0
    fi
    local v=""
    if have cargo; then
        v=$(cargo_version cargo)
    fi
    if [ -n "$v" ] && version_ge "$v" "$RUST_MIN"; then
        ok "cargo $v"
        CARGO_CMD=(cargo)
        return 0
    fi
    if [ -n "$v" ]; then
        info "cargo $v is older than $RUST_MIN (distribution Rust is usually too old)"
    fi
    if have rustup; then
        run rustup toolchain install stable --profile minimal
        CARGO_CMD=(cargo +stable)
        return 0
    fi
    if [ "$DRY" != 1 ]; then
        have curl || die "curl is required to install Rust"
    fi
    # --no-modify-path: cargo is only needed for this build; nothing else in KiteOS uses it.
    run_shell "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | RUSTUP_INIT_SKIP_PATH_CHECK=yes sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path"
    CARGO_CMD=("${CARGO_HOME:-$HOME/.cargo}/bin/cargo")
}

git_dimos() {  # read-only git query in the dimOS checkout
    # GIT_OPTIONAL_LOCKS=0: `git status` must not refresh (write) dimOS's index.
    GIT_OPTIONAL_LOCKS=0 git -C "$DIMOS_DIR" "$@"
}

step_dimos_source() {
    step 4 "$TOTAL_STEPS" "dimOS source ($PIN_BRANCH @ $PIN_SHORT)"
    if [ -e "$DIMOS_DIR/.git" ]; then
        local head branch dirty
        head=$(git_dimos rev-parse HEAD 2>/dev/null || true)
        branch=$(git_dimos rev-parse --abbrev-ref HEAD 2>/dev/null || true)
        if [ "$head" = "$PIN_COMMIT" ]; then
            ok "$DIMOS_DIR is at the pin ($branch @ $PIN_SHORT)"
        else
            warn "$DIMOS_DIR is at ${head:-unknown} (${branch:-?}), not the KiteOS pin $PIN_COMMIT; leaving it as is"
            info "to move it to the pin (after saving your work):"
            if git_dimos cat-file -e "$PIN_COMMIT^{commit}" 2>/dev/null; then
                info "  git -C $DIMOS_DIR checkout $PIN_BRANCH   # or: checkout $PIN_COMMIT"
            else
                info "  git -C $DIMOS_DIR fetch $PIN_BUNDLE $PIN_BRANCH:$PIN_BRANCH"
                info "  git -C $DIMOS_DIR checkout $PIN_BRANCH"
            fi
        fi
        dirty=$(git_dimos status --porcelain --untracked-files=no 2>/dev/null || true)
        if [ -n "$dirty" ]; then
            warn "$DIMOS_DIR has local modifications to tracked files; KiteOS expects an unmodified dimOS"
        fi
        if [ "$(git_dimos rev-parse --is-shallow-repository 2>/dev/null || echo false)" = true ]; then
            warn "$DIMOS_DIR is a shallow clone; the KiteOS bundle needs full history (git -C $DIMOS_DIR fetch --unshallow)"
        fi
        return 0
    fi

    if [ -e "$DIMOS_DIR" ] && [ -n "$(ls -A "$DIMOS_DIR" 2>/dev/null || true)" ]; then
        die "$DIMOS_DIR exists but is not a git checkout. dimOS needs its .git (otherwise it clones upstream main at runtime). Move it aside or pass --dimos-dir"
    fi
    [ -r "$PIN_BUNDLE" ] || die "missing $PIN_BUNDLE (the $PIN_BRANCH branch is not on a public remote)"
    have git || [ "$DRY" = 1 ] || die "git is required"

    # Clone into a staging directory and move it into place only once HEAD is verified,
    # so an interrupted run never leaves a half-set-up checkout that later runs would trust.
    local staging="$DIMOS_DIR.kiteos-staging"
    info "cloning upstream dimOS (full history: the bundle builds on upstream ${PIN_BASE:-main})"
    if [ -e "$staging" ]; then
        if [ ! -e "$staging/.git" ] && [ -n "$(ls -A "$staging" 2>/dev/null || true)" ]; then
            die "$staging exists and is not a KiteOS staging clone; move it aside and re-run"
        fi
        info "removing the incomplete clone left by an earlier run: $staging"
        run rm -rf "$staging"
    fi
    run mkdir -p "$(dirname "$DIMOS_DIR")"
    run env GIT_LFS_SKIP_SMUDGE=1 git clone "$PIN_UPSTREAM" "$staging"
    run_in "$staging" git bundle verify "$PIN_BUNDLE"
    run_in "$staging" env GIT_LFS_SKIP_SMUDGE=1 git fetch "$PIN_BUNDLE" "$PIN_BRANCH:$PIN_BRANCH"
    # Fresh clone, nothing to lose: make the branch point exactly at the pin.
    run_in "$staging" env GIT_LFS_SKIP_SMUDGE=1 git checkout -B "$PIN_BRANCH" "$PIN_COMMIT"
    if [ "$DRY" != 1 ]; then
        local head
        head=$(GIT_OPTIONAL_LOCKS=0 git -C "$staging" rev-parse HEAD)
        [ "$head" = "$PIN_COMMIT" ] || die "dimOS HEAD is $head after checkout, expected $PIN_COMMIT"
        ok "dimOS HEAD verified: $PIN_COMMIT"
    else
        info "verify: git -C $staging rev-parse HEAD == $PIN_COMMIT"
    fi
    if [ -d "$DIMOS_DIR" ]; then
        run rmdir "$DIMOS_DIR"  # empty (checked above)
    fi
    run mv "$staging" "$DIMOS_DIR"
}

step_dimos_venv() {
    step 5 "$TOTAL_STEPS" "dimOS virtualenv (uv sync, Python $PYTHON_VERSION)"
    if [ "$PLATFORM" = linux ] && [ "$ARCH" = x86_64 ]; then
        info "Linux x86_64 pulls CUDA torch: expect 10-30 min and ~30-40 GB disk including step 8"
    fi
    if [ "$PLATFORM" = macos ] && [ "$ARCH" = x86_64 ]; then
        warn "Intel Mac: the pinned dimOS lockfile has no macOS x86_64 torch wheels, so uv sync is expected to fail; use an Apple silicon Mac or Linux"
    fi
    info "extras: ${EXTRAS:-(none)}"
    # --inexact keeps packages that are not in the lock (notably the editable KiteOS
    # install from step 7) instead of uninstalling them on re-runs.
    run_in "$DIMOS_DIR" "$UV_BIN" sync --frozen --inexact --python "$PYTHON_VERSION" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

step_recorder() {
    step 6 "$TOTAL_STEPS" "Collection recorder ($RECORDER_CRATE)"
    if [ "$NO_RECORDER" = 1 ]; then
        skip "--no-recorder"
        warn "no recorder: 'kiteos collect' cannot record (dimOS would try 'nix build' at startup)"
        return 0
    fi
    if [ -x "$RECORDER_BIN" ]; then
        ok "already present: $RECORDER_BIN"
        return 0
    fi
    if [ -L "$RECORDER_RESULT" ]; then
        warn "$RECORDER_RESULT is a symlink (Nix output) without the binary; rebuild it with nix or remove it, then re-run"
        return 0
    fi
    local target_dir=${CARGO_TARGET_DIR:-$DIMOS_DIR/target}
    run_in "$DIMOS_DIR" "${CARGO_CMD[@]}" build --release --locked -p "$RECORDER_CRATE"
    run mkdir -p "$RECORDER_RESULT/bin"
    run cp "$target_dir/release/$RECORDER_CRATE" "$RECORDER_RESULT/bin/$RECORDER_CRATE"
    if [ "$DRY" != 1 ]; then
        [ -x "$RECORDER_BIN" ] || die "recorder build finished but $RECORDER_BIN is missing"
        ok "installed $RECORDER_BIN"
    fi
}

KITEOS_LINK="$HOME/.local/bin/kiteos"

step_kiteos() {
    step 7 "$TOTAL_STEPS" "KiteOS ${KITEOS_VERSION:-} into the dimOS venv (editable)"
    run "$UV_BIN" pip install --python "$VENV_PY" --no-deps -e "$REPO_DIR"
    if [ "$DRY" != 1 ]; then
        "$VENV_PY" -c 'import kiteos' || die "KiteOS is not importable from $VENV_PY"
        local names
        names=$("$VENV_PY" -c 'from importlib.metadata import entry_points
print(" ".join(sorted(e.name for e in entry_points(group="dimos.blueprints") if e.value.startswith("kiteos."))))' || true)
        if [ -n "$names" ]; then
            ok "dimOS blueprints registered: $names"
        else
            warn "no kiteos.* blueprints registered in the dimOS venv"
        fi
    fi

    # Make `kiteos` callable from any shell without activating the venv.
    local target="$DIMOS_DIR/.venv/bin/kiteos"
    if [ -L "$KITEOS_LINK" ] && [ "$(readlink "$KITEOS_LINK")" = "$target" ]; then
        ok "$KITEOS_LINK -> $target"
    elif [ -e "$KITEOS_LINK" ] || [ -L "$KITEOS_LINK" ]; then
        warn "$KITEOS_LINK already exists and points elsewhere; not replacing it (use $target)"
    else
        run mkdir -p "$(dirname "$KITEOS_LINK")"
        run ln -s "$target" "$KITEOS_LINK"
    fi

    # Station config: create from the example once, never overwrite.
    local station="$REPO_DIR/config/station.yaml" example="$REPO_DIR/config/station.example.yaml"
    if [ -e "$station" ]; then
        ok "keeping existing station config: $station"
    elif [ -e "$example" ]; then
        run cp "$example" "$station"
        info "edit $station for this site (cameras, CAN, task, hub repo)"
    else
        warn "no $example to start the station config from"
    fi
    if [ -n "${KITEOS_STATION:-}" ]; then
        info "KITEOS_STATION is set: KiteOS will use $KITEOS_STATION"
    fi
    if [ "$DIMOS_DIR" != "$HOME/teleop_stack/dimos" ]; then
        info "non-default dimOS location: set 'dimos_root: $DIMOS_DIR' in the station config"
    fi
}

step_assets() {
    step 8 "$TOTAL_STEPS" "Dual OpenYAM model asset and LeRobot environment"
    local tarball="$DIMOS_DIR/data/.lfs/$LFS_ASSET.tar.gz"
    local extracted="$DIMOS_DIR/data/$LFS_ASSET"
    if [ -f "$extracted/dual_openyam.urdf" ]; then
        ok "model asset present: $extracted"
    else
        if [ "$DRY" != 1 ] && [ ! -e "$tarball" ]; then
            die "$tarball is not in the dimOS checkout (is $DIMOS_DIR at the pin?)"
        fi
        if [ "$DRY" = 1 ] || is_lfs_pointer "$tarball"; then
            run_in "$DIMOS_DIR" git lfs install --local --skip-repo
            # --exclude= overrides lfs.fetchexclude=data/.lfs/* from dimOS's .lfsconfig.
            run_in "$DIMOS_DIR" git lfs pull --include "data/.lfs/$LFS_ASSET.tar.gz" --exclude=
        fi
        if [ "$DRY" != 1 ] && is_lfs_pointer "$tarball"; then
            die "git lfs pull left $tarball as a pointer (is lfs.dimensionalos.com reachable?)"
        fi
        run tar -xzf "$tarball" -C "$DIMOS_DIR/data"
        if [ "$DRY" != 1 ]; then
            [ -f "$extracted/dual_openyam.urdf" ] || die "extracting $tarball did not produce $extracted/dual_openyam.urdf"
            ok "model asset extracted: $extracted"
        fi
    fi

    if [ "$NO_PREWARM" = 1 ]; then
        skip "LeRobot environment pre-warm (--no-prewarm); the first rollout/upload builds it"
        return 0
    fi
    info "building dimOS's isolated LeRobot environment (one-off, several GB; same command dimOS uses for upload/rollout)"
    run_in "$DIMOS_DIR" "$VENV_PY" -c 'import subprocess, sys; from dimos.experimental.isolated_python.module import isolated_python_environment as env, isolated_python_run_command as cmd; from dimos.imitation.dataprep.lerobot import lerobot_project; p = lerobot_project(); sys.exit(subprocess.run(cmd(p, "python", "-c", "import lerobot, torch"), cwd=p, env=env(p)).returncode)'
}

print_can_mapping() {
    local rules="$UDEV_DIR/80-kiteos-can.rules"
    info "CAN adapters (USB serial -> interface):"
    sed -n 's/.*ATTRS{serial}=="\([^"]*\)".*NAME="\([^"]*\)".*/\1 -> \2/p' "$rules" | while IFS= read -r line; do
        info "  $line"
    done
}

install_udev_file() {  # install_udev_file NAME
    local src="$UDEV_DIR/$1" dest="/etc/udev/rules.d/$1"
    if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
        ok "$dest up to date"
    else
        run_sudo install -m 0644 "$src" "$dest"
    fi
}

install_limits() {
    local user rendered
    user=$(id -un)
    if [ -f "$LIMITS_DEST" ] && sed "s/KITEOS_USER/$user/" "$LIMITS_TEMPLATE" | cmp -s - "$LIMITS_DEST"; then
        ok "$LIMITS_DEST up to date"
        return 0
    fi
    if [ "$DRY" = 1 ]; then
        _show "sed 's/KITEOS_USER/$user/' $LIMITS_TEMPLATE > <tmp>"
        run_sudo install -m 0644 "<tmp>" "$LIMITS_DEST"
        return 0
    fi
    rendered=$(mktemp "${TMPDIR:-/tmp}/kiteos-memlock.XXXXXX")
    sed "s/KITEOS_USER/$user/" "$LIMITS_TEMPLATE" >"$rendered"
    run_sudo install -m 0644 "$rendered" "$LIMITS_DEST"
    rm -f "$rendered"
}

step_devices_linux() {
    if [ "$WITH_UDEV" = 1 ]; then
        local r
        for r in $UDEV_RULES; do
            install_udev_file "$r"
        done
        # The Quest rule used to be 81-*: too late for systemd's uaccess (73-seat-late.rules).
        if [ -f /etc/udev/rules.d/81-kiteos-quest.rules ]; then
            run_sudo rm -f /etc/udev/rules.d/81-kiteos-quest.rules
        fi
        install_limits
        run_sudo udevadm control --reload-rules
        info "replug both CAN adapters and the Quest so the rules apply (nothing is triggered from here)"
        info "the memlock limit applies at your next login"
    else
        skip "udev rules and memlock limit (pass --with-udev)"
        info "without them CAN interfaces are can0/can1 in plug order and the arms can swap"
    fi

    if [ "$SKIP_SYSTEM" = 1 ]; then
        skip "group membership (--skip-system)"
    else
        local user add=() g
        user=$(id -un)
        for g in $LINUX_GROUPS; do
            if ! getent group "$g" >/dev/null 2>&1; then
                warn "group '$g' does not exist on this system; skipping it"
                continue
            fi
            case " $(id -nG "$user") " in
                *" $g "*) ;;
                *) add+=("$g") ;;
            esac
        done
        if [ "${#add[@]}" -gt 0 ]; then
            local joined
            joined=$(printf '%s,' "${add[@]}")
            run_sudo usermod -aG "${joined%,}" "$user"
            if [ "$DRY" = 1 ]; then
                info "would add $user to ${joined%,} (takes effect at the next login)"
            else
                warn "added $user to ${joined%,}: log out and back in before using cameras/CAN/Quest"
            fi
        else
            ok "$user is in: $LINUX_GROUPS"
        fi
    fi

    if have modinfo && ! modinfo gs_usb >/dev/null 2>&1; then
        warn "kernel module gs_usb not found; install linux-modules-extra-$(uname -r) (or your distro's equivalent)"
    fi
    local glibc
    glibc=$(ldd --version 2>&1 | first_line | awk '{print $NF}' || true)
    if [ -n "$glibc" ] && ! version_ge "$glibc" 2.34; then
        warn "glibc $glibc < 2.34: dimOS's CAN driver wheels need Ubuntu 22.04+ / Debian 12+"
    fi
    print_can_mapping
    info "after every plug or reboot: kiteos can setup   (sudo ip link, 1 Mbit/s)"
    info "firewall: KiteOS binds the WebXR teleop server to 127.0.0.1; still block 8443 from the LAN as defence in depth (e.g. sudo ufw deny 8443/tcp)"
}

step_devices() {
    step 9 "$TOTAL_STEPS" "Devices (udev, groups, CAN names)"
    if [ "$PLATFORM" = linux ]; then
        step_devices_linux
        return 0
    fi
    if [ "$WITH_UDEV" = 1 ]; then
        skip "--with-udev only applies to Linux"
    fi
    ok "macOS needs no udev rules or groups; CAN adapters are addressed by USB serial"
    print_can_mapping
    info "(on macOS the station config uses the serials: can.left.darwin / can.right.darwin)"
    info "cameras: run KiteOS from a terminal app that has Camera permission (System Settings > Privacy & Security > Camera)"
}

step_next() {
    step 10 "$TOTAL_STEPS" "Next steps"
    local relogin=""
    if [ "$PLATFORM" = linux ]; then
        relogin=" (log out and back in if groups or limits changed)"
    fi
    cat <<EOF
    1. Open a new terminal$relogin.
    2. Hugging Face login (uploads): $DIMOS_DIR/.venv/bin/hf auth login
    3. Review $REPO_DIR/config/station.yaml and the fence (docs/FENCE.md).
    4. Run:  kiteos doctor
       (if 'kiteos' is not on PATH: $DIMOS_DIR/.venv/bin/kiteos doctor)
    Morning checklist: $REPO_DIR/docs/RUNBOOK.md

    Keep the KiteOS install: a plain 'uv sync' in $DIMOS_DIR is exact and
    uninstalls the editable KiteOS package (and its kiteos.* dimOS blueprints).
    Use 'uv run --no-sync' / 'uv sync --inexact'. To restore it, re-run this
    installer or:
      uv pip install --python $VENV_PY --no-deps -e $REPO_DIR
EOF
}

# --- main -----------------------------------------------------------------

printf '%sKiteOS installer%s %s\n' "$_K_BOLD" "$_K_OFF" "${KITEOS_VERSION:-}"
info "platform   $DISTRO_PRETTY ($ARCH)"
info "stack dir  $STACK_DIR"
info "dimOS      $DIMOS_DIR  (pin $PIN_BRANCH @ $PIN_SHORT)"
info "KiteOS     $REPO_DIR"
info "extras     ${EXTRAS:-(none)}"
if [ "$DRY" = 1 ]; then
    info "mode       dry run: commands are printed, nothing is changed"
fi

step_system
step_uv
step_rust
step_dimos_source
step_dimos_venv
step_recorder
step_kiteos
step_assets
step_devices
step_next
print_warnings

if [ "$DRY" = 1 ]; then
    printf '\nKiteOS installer: dry run complete, nothing was changed.\n'
else
    printf '\nKiteOS installer: done.\n'
fi
