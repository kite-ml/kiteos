# Copyright 2026 Kite ML.
"""``kiteos quest proxy``: the USB-tethered Quest's way into the teleop server.

A thin wrapper: the proxy itself lives in ``kiteos.quest.proxy``. Settings come from the
station's ``quest`` section; extra arguments after ``--`` are passed through. When exactly one
KiteOS stack runs and it serves WebXR on its own port (a MOCK run), the proxy relays to that port.
"""

from __future__ import annotations

import inspect
import sys
from typing import Annotated, Optional

import typer

from kiteos.cli._common import StationOption, die, load_station, note
from kiteos.config import StationConfig

app = typer.Typer(help="Meta Quest teleop helpers.", no_args_is_help=True)


def running_upstream(station: StationConfig) -> str:
    """The WebXR upstream of the one running KiteOS stack, else the station's."""
    from kiteos.cli.runs import alive_run_states

    stacks = alive_run_states()
    if len(stacks) == 1 and stacks[0].webxr_port:
        upstream = f"https://127.0.0.1:{stacks[0].webxr_port}"
        if upstream != station.quest.upstream:
            note(f"relaying to the running {stacks[0].mode} stack at {upstream}")
        return upstream
    return station.quest.upstream


def proxy_argv(station: StationConfig, extra: list[str] | None = None, upstream: str | None = None) -> list[str]:
    """Arguments for ``kiteos.quest.proxy.main`` from the station's quest section."""
    quest = station.quest
    argv = ["--port", str(quest.port), "--upstream", upstream or quest.upstream]
    if not quest.adb_enabled:
        argv.append("--no-adb")
    elif isinstance(quest.adb, str) and quest.adb.lower() != "auto":
        argv += ["--adb-path", quest.adb]
    return argv + list(extra or [])


@app.command(
    "proxy",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def proxy(
    ctx: typer.Context,
    port: Annotated[Optional[int], typer.Option("--port", help="Local port (default: station quest.port).")] = None,
    station: StationOption = None,
) -> None:
    """Serve the branded teleop page to the Quest over USB (adb reverse) and relay to the runtime."""
    config = load_station(station)
    if port is not None:
        config.quest.port = port
    try:
        from kiteos.quest import proxy as quest_proxy
    except ImportError as exc:
        die(f"the Quest proxy is not available: {exc}", code=2)
    extra = list(ctx.args)
    upstream = None if any(a.startswith("--upstream") for a in extra) else running_upstream(config)
    argv = proxy_argv(config, extra, upstream=upstream)
    main = quest_proxy.main
    if inspect.signature(main).parameters:
        code = main(argv)
    else:
        saved = sys.argv
        sys.argv = ["kiteos quest proxy", *argv]
        try:
            code = main()
        finally:
            sys.argv = saved
    raise typer.Exit(int(code or 0))
