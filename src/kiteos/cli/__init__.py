# Copyright 2026 Kite ML.
"""The ``kiteos`` command line.

Every command reads the station file (``--station PATH``, ``$KITEOS_STATION`` or
``config/station.yaml``) and runs dimOS from the station's checkout with
``uv run --no-sync`` and ``PYTHONPATH=<repo>/src``. Heavy imports (dimOS, textual, OpenCV)
happen inside the commands that need them, so ``kiteos --help`` and the offline commands stay fast.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from kiteos import __version__, brand
from kiteos.cli import cameras, can, collect, doctor, evaluate, fence, quest, rollout, stop, teleop, upload
from kiteos.cli._common import STATE, console, print_banner

NO_BANNER_ENV = "KITEOS_NO_BANNER"

app = typer.Typer(
    name="kiteos",
    help=f"{brand.BRAND_NAME} {__version__}: {brand.TAGLINE}.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)


def _version(value: bool) -> None:
    if value:
        console.print(f"{brand.BRAND_NAME} {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    station: Annotated[
        Optional[Path],
        typer.Option(
            "--station",
            "-s",
            help="Station file for every command (default: $KITEOS_STATION or config/station.yaml).",
            show_default=False,
        ),
    ] = None,
    no_banner: Annotated[bool, typer.Option("--no-banner", help="Do not print the KiteOS banner.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Print the version and exit.")
    ] = False,
) -> None:
    STATE.station = station
    STATE.banner = not (no_banner or os.environ.get(NO_BANNER_ENV))
    if STATE.banner:
        print_banner()


app.command("doctor")(doctor.doctor)
app.command("teleop")(teleop.teleop)
app.add_typer(collect.app, name="collect")
app.add_typer(rollout.app, name="rollout")
app.add_typer(evaluate.app, name="eval")
app.add_typer(quest.app, name="quest")
app.add_typer(cameras.app, name="cameras")
app.add_typer(can.app, name="can")
app.command("upload")(upload.upload)
app.command("stop")(stop.stop)
app.add_typer(fence.app, name="fence")


def main() -> None:
    """Console-script entry point (``kiteos``)."""
    app(prog_name="kiteos")


__all__ = ["app", "main"]
