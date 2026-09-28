# Copyright 2026 Kite ML.
# shellcheck shell=bash
#
# Shared helpers for deploy/install.sh.
#
# Must stay compatible with bash 3.2 (the /bin/bash that ships with macOS):
# no associative arrays, no ${var,,}, no mapfile, and a possibly-empty array
# is expanded as ${arr[@]+"${arr[@]}"} so `set -u` does not trip on it.
#
# Every state-changing command goes through run/run_in/run_sudo/run_shell so
# that --dry-run (KITEOS_DRY_RUN=1) prints it instead of executing it.
# Read-only probes (command -v, --version, git rev-parse, ...) run directly.

KITEOS_DRY_RUN=${KITEOS_DRY_RUN:-0}
KITEOS_WARNINGS=()

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    _K_BOLD=$'\033[1m'
    _K_DIM=$'\033[2m'
    _K_RED=$'\033[31m'
    _K_GREEN=$'\033[32m'
    _K_YELLOW=$'\033[33m'
    _K_CYAN=$'\033[36m'
    _K_OFF=$'\033[0m'
else
    _K_BOLD=''
    _K_DIM=''
    _K_RED=''
    _K_GREEN=''
    _K_YELLOW=''
    _K_CYAN=''
    _K_OFF=''
fi

# --- output ---------------------------------------------------------------

step() {  # step N TOTAL TITLE
    printf '\n%s==> [%s/%s] %s%s\n' "$_K_BOLD$_K_CYAN" "$1" "$2" "$3" "$_K_OFF"
}

info() {
    printf '    %s\n' "$*"
}

ok() {
    printf '    %sok%s  %s\n' "$_K_GREEN" "$_K_OFF" "$*"
}

skip() {
    printf '    %sskip%s %s\n' "$_K_DIM" "$_K_OFF" "$*"
}

warn() {
    KITEOS_WARNINGS+=("$*")
    printf '    %swarn%s %s\n' "$_K_YELLOW" "$_K_OFF" "$*"
}

die() {
    printf '\n%sKiteOS installer: error:%s %s\n' "$_K_RED$_K_BOLD" "$_K_OFF" "$*" >&2
    exit 1
}

print_warnings() {
    if [ "${#KITEOS_WARNINGS[@]}" -eq 0 ]; then
        return 0
    fi
    printf '\n%sWarnings (%s):%s\n' "$_K_YELLOW$_K_BOLD" "${#KITEOS_WARNINGS[@]}" "$_K_OFF"
    local w
    for w in ${KITEOS_WARNINGS[@]+"${KITEOS_WARNINGS[@]}"}; do
        printf '  - %s\n' "$w"
    done
}

# --- command execution ----------------------------------------------------

quote_arg() {  # quote one word for display, single quotes only when needed
    case "$1" in
        '' | *[!A-Za-z0-9_./:=@%+,-]*)
            printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
            ;;
        *) printf '%s' "$1" ;;
    esac
}

quote_cmd() {  # shell-quoted rendering of argv, for display only
    local out='' arg
    for arg in "$@"; do
        out="$out $(quote_arg "$arg")"
    done
    printf '%s' "${out# }"
}

_show() {
    if [ "$KITEOS_DRY_RUN" = 1 ]; then
        printf '    %s[dry-run]%s $ %s\n' "$_K_DIM" "$_K_OFF" "$1"
    else
        printf '    %s$ %s%s\n' "$_K_DIM" "$1" "$_K_OFF"
    fi
}

run() {  # run CMD [ARGS...]
    _show "$(quote_cmd "$@")"
    if [ "$KITEOS_DRY_RUN" = 1 ]; then
        return 0
    fi
    "$@"
}

run_in() {  # run_in DIR CMD [ARGS...]
    local dir=$1
    shift
    _show "(cd $(quote_arg "$dir") && $(quote_cmd "$@"))"
    if [ "$KITEOS_DRY_RUN" = 1 ]; then
        return 0
    fi
    (cd "$dir" && "$@")
}

run_sudo() {  # run_sudo CMD [ARGS...]; plain run when already root
    if [ "$(id -u)" -eq 0 ]; then
        run "$@"
    else
        if [ "$KITEOS_DRY_RUN" != 1 ] && ! have sudo; then
            die "sudo is required for: $(quote_cmd "$@")"
        fi
        run sudo "$@"
    fi
}

run_shell() {  # run_shell 'pipeline string'; only for fixed, trusted strings
    _show "$1"
    if [ "$KITEOS_DRY_RUN" = 1 ]; then
        return 0
    fi
    bash -c "set -euo pipefail; $1"
}

# --- probes ---------------------------------------------------------------

have() {
    command -v "$1" >/dev/null 2>&1
}

first_line() {  # first line of stdin without SIGPIPE-ing the producer
    sed -n '1p'
}

version_ge() {  # version_ge A B -> success when A >= B (numeric, dot separated)
    local IFS=.
    local -a a b
    a=($1)
    b=($2)
    local i x y
    for i in 0 1 2 3; do
        x=${a[$i]:-0}
        y=${b[$i]:-0}
        x=${x%%[!0-9]*}
        y=${y%%[!0-9]*}
        x=${x:-0}
        y=${y:-0}
        if [ "$((10#$x))" -gt "$((10#$y))" ]; then
            return 0
        fi
        if [ "$((10#$x))" -lt "$((10#$y))" ]; then
            return 1
        fi
    done
    return 0
}

expand_path() {  # expand a leading ~ and make the path absolute (no symlink resolution)
    local p=$1
    case "$p" in
        "~") p=$HOME ;;
        "~/"*) p="$HOME/${p#\~/}" ;;
    esac
    case "$p" in
        /*) ;;
        *) p="$PWD/$p" ;;
    esac
    # Drop trailing slashes (but keep "/").
    while [ "${#p}" -gt 1 ] && [ "${p%/}" != "$p" ]; do
        p=${p%/}
    done
    printf '%s' "$p"
}

pin_value() {  # pin_value FILE KEY -> value of KEY=VALUE, ignoring trailing comments
    sed -n "s/^$2=\\([^[:space:]#]*\\).*/\\1/p" "$1" | first_line
}

is_lfs_pointer() {  # is_lfs_pointer FILE -> success if FILE is a git-lfs pointer stub
    local f=$1 size line=''
    [ -f "$f" ] || return 1
    size=$(wc -c <"$f" | tr -d '[:space:]')
    [ "$size" -lt 1024 ] || return 1
    IFS= read -r line <"$f" || true
    case "$line" in
        "version https://git-lfs.github.com/spec/"*) return 0 ;;
    esac
    return 1
}
