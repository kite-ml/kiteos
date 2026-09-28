# Copyright 2026 Kite ML.
"""``kiteos can setup|status``: the two CAN adapters that drive the arms.

macOS talks to the gs_usb adapters directly (selected by USB serial); nothing needs setting up.
Linux exposes them as SocketCAN interfaces (named can_left/can_right by deploy/udev) that must be
brought up at 1 Mbit/s with sudo after every plug or reboot.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from kiteos.cli._common import StationOption, console, die, runtime_argv, kiteos_env, load_station, note, run_runtime
from kiteos.config import SIDES, StationConfig, platform_key

app = typer.Typer(help="Set up and check the arms' CAN adapters.", no_args_is_help=True)

BITRATE = 1_000_000


@dataclass(frozen=True)
class CanState:
    side: str
    selector: str | None
    present: bool
    detail: str


def parse_gs_usb_list(text: str) -> list[str]:
    """Serials from ``kiteos-runtime hardware can list`` on macOS (``INDEX  SERIAL`` table)."""
    serials = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+\s+(\S+)\s*$", line)
        if match and match.group(1) != "<missing>":
            serials.append(match.group(1))
    return serials


def parse_ip_brief(text: str) -> dict[str, str]:
    """``ip -brief link show type can`` -> {interface: state}."""
    states = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0].split("@")[0]] = parts[1]
    return states


def can_states(station: StationConfig) -> list[CanState]:
    """Which configured adapters this host can see (lists devices only; never opens a bus)."""
    listing = run_runtime(station, "hardware", "can", "list", timeout=60.0)
    states = []
    if platform_key() == "darwin":
        serials = parse_gs_usb_list(listing.stdout) if listing.ok else []
        for side in SIDES:
            port = station.can_port(side)
            present = port is not None and port in serials
            detail = "connected" if present else ("not connected" if listing.ok else listing.output or "listing failed")
            states.append(CanState(side, port, present, detail))
    else:
        interfaces = parse_ip_brief(listing.stdout) if listing.ok else {}
        for side in SIDES:
            port = station.can_port(side)
            state = interfaces.get(port or "")
            if state is None:
                detail = "interface missing (udev rule installed? adapter plugged?)" if listing.ok else listing.output
                states.append(CanState(side, port, False, detail))
            else:
                states.append(CanState(side, port, state.upper() in ("UP", "UNKNOWN"), f"state {state}"))
    return states


@app.command("status")
def status(station: StationOption = None) -> None:
    """Show whether both arms' CAN adapters are present (and up, on Linux)."""
    config = load_station(station)
    table = Table(title=f"CAN adapters ({platform_key()})", title_justify="left", show_edge=False)
    for column in ("arm", "selector", "status"):
        table.add_column(column)
    states = can_states(config)
    for state in states:
        style = "ok" if state.present else "fail"
        table.add_row(state.side, str(state.selector), f"[{style}]{state.detail}[/{style}]")
    console.print(table)
    if not all(s.present for s in states):
        if platform_key() == "linux":
            note("bring the interfaces up with `kiteos can setup` (needs sudo)")
        raise typer.Exit(1)


def interface_ready(details: str, bitrate: int = BITRATE) -> bool:
    """``ip -details link show <if>`` output says UP at ``bitrate``."""
    up = re.search(r"<[^>]*\bUP\b[^>]*>", details) is not None
    rate = re.search(r"\bbitrate\s+(\d+)", details)
    return up and rate is not None and int(rate.group(1)) == bitrate


def running_stacks() -> list[str]:
    """robot stacks on this host (tests replace this)."""
    from kiteos.cli.runs import other_stacks

    return [f"{e.source}: {e.detail}" for e in other_stacks(ports=())]


@app.command("setup")
def setup(
    force: Annotated[
        bool, typer.Option("--force", help="Reconfigure even while a robot stack is running (drops both buses).")
    ] = False,
    station: StationOption = None,
) -> None:
    """Linux: configure both SocketCAN interfaces at 1 Mbit/s (sudo). macOS: nothing to do."""
    config = load_station(station)
    if platform_key() == "darwin":
        console.print("macOS needs no CAN setup: the gs_usb transport sets the bitrate itself.")
        note("check the adapters with `kiteos can status`")
        return
    running = running_stacks()
    if running and not force:
        die(
            "a robot stack is running (" + escape(running[0][:120]) + "): `can setup` takes the CAN links DOWN, "
            "which cuts the motors mid-motion. Stop the stack first (or --force).",
            code=2,
        )
    root = config.runtime_root_path()
    for side in SIDES:
        port = config.can_port(side)
        if not port:
            die(f"no CAN interface configured for the {side} arm (station can.{side}.linux)", code=2)
        details = subprocess.run(
            ["ip", "-details", "link", "show", port], capture_output=True, text=True, check=False
        ).stdout
        if interface_ready(details) and not force:
            console.print(f"[kite]{port}[/kite] ({side} arm) is already up at {BITRATE} bit/s; left alone")
            continue
        console.print(f"Setting up [kite]{port}[/kite] ({side} arm) at {BITRATE} bit/s ...")
        argv = runtime_argv("hardware", "can", "setup", port, "--bitrate", str(BITRATE))
        result = subprocess.run(argv, cwd=root, env=kiteos_env(), check=False)
        if result.returncode != 0:
            die(f"CAN setup failed for {port}", code=1)
    console.print("[ok]both CAN interfaces are up[/ok]")
