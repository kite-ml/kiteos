# Copyright 2026 Kite ML.
"""``kiteos eval offline``: open-loop test of a checkpoint on recorded episodes (no robot)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from kiteos.cli._common import StationOption, console, die, load_station

app = typer.Typer(help="Evaluate policies without the robot.", no_args_is_help=True)

DEVICES = ("auto", "cuda", "mps", "cpu")


def parse_episodes(text: str) -> list[int]:
    """``"0,2,5-7"`` -> ``[0, 2, 5, 6, 7]``."""
    episodes: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad episode range {part!r}")
            episodes.extend(range(lo, hi + 1))
        else:
            episodes.append(int(part))
    if not episodes or any(e < 0 for e in episodes):
        raise ValueError("episodes must be non-negative indices, e.g. 0,1,2 or 0-4")
    return sorted(set(episodes))


@app.command("offline")
def offline(
    dataset: Annotated[Path, typer.Option("--dataset", help="Prepared LeRobot dataset root.")],
    episodes: Annotated[str, typer.Option("--episodes", help="Episode indices, e.g. 0,1,2 or 0-4.")] = "0",
    checkpoint: Annotated[
        Optional[Path], typer.Option("--checkpoint", help="Checkpoint dir (default: station policy).")
    ] = None,
    out: Annotated[
        Optional[Path], typer.Option("--out", help="Output dir (default: <checkpoint>/eval-<dataset>).")
    ] = None,
    device: Annotated[
        Optional[str], typer.Option("--device", help="auto | cuda | mps | cpu (default: station policy.device).")
    ] = None,
    max_frames: Annotated[
        Optional[int], typer.Option("--max-frames", min=1, help="Frames per episode (large models are slow).")
    ] = None,
    station: StationOption = None,
) -> None:
    """Compare the policy's predicted actions with the recorded ones, frame by frame."""
    config = load_station(station)
    ckpt = checkpoint.expanduser().resolve() if checkpoint else config.checkpoint_path()
    if ckpt is None or not ckpt.is_dir():
        die(f"checkpoint not found: {ckpt or '(none configured)'}", code=2)
    root = dataset.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        die(f"not a LeRobot dataset (no meta/info.json): {root}", code=2)
    try:
        episode_list = parse_episodes(episodes)
    except ValueError as exc:
        die(str(exc), code=2)
    chosen = device or config.policy.device
    if chosen not in DEVICES:
        die(f"device must be one of {', '.join(DEVICES)}", code=2)
    out_dir = (out or ckpt / f"eval-{root.name}").expanduser().resolve()

    from kiteos.cli.upload import IsolatedEnv

    env = IsolatedEnv()
    args = [str(ckpt), str(root), ",".join(map(str, episode_list)), str(out_dir), chosen]
    if max_frames is not None:
        args.append(str(max_frames))
    console.print(f"Evaluating [kite]{ckpt.name}[/kite] on {root.name} episodes {episode_list} -> {out_dir}")
    result = env.run("eval_policy.py", *args, capture=False)
    raise typer.Exit(result.returncode)
