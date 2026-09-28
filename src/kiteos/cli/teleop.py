# Copyright 2026 Kite ML.
"""``kiteos teleop``: Quest teleoperation of both arms (no recording)."""

from __future__ import annotations

from typing import Annotated

import typer

from kiteos.cli._common import StationOption, load_station
from kiteos.cli.launch import launch


def teleop(
    real: Annotated[bool, typer.Option("--real", help="Drive the real arms (default: simulated).")] = False,
    no_cameras: Annotated[
        bool,
        typer.Option("--no-cameras", help="Never pass camera selectors to the runtime."),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the runtime command and exit.")] = False,
    allow_concurrent: Annotated[
        bool,
        typer.Option("--allow-concurrent", help="MOCK only: start even if another robot stack runs on this host."),
    ] = False,
    station_ports: Annotated[
        bool, typer.Option("--station-ports", help="MOCK only: use the station's WebXR/viser ports (Quest practice).")
    ] = False,
    station: StationOption = None,
) -> None:
    """Teleoperate both arms from the Quest (run `kiteos quest proxy` alongside)."""
    config = load_station(station)
    launch(
        config,
        "teleop",
        real=real,
        cameras=not no_cameras,
        dry_run=dry_run,
        allow_concurrent=allow_concurrent,
        station_ports_for_mock=station_ports,
    )
