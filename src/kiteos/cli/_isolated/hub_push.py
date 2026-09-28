# Copyright 2026 Kite ML.
"""Merge prepared LeRobot sessions into one dataset and push it to the Hugging Face Hub.

The merged dataset is rebuilt from all sessions on every run. Runs in the runtime's isolated
LeRobot env (see ``kiteos upload``).
Usage: hub_push.py REPO_ID MERGED_ROOT PRIVATE(True|False) SESSION_ROOT...
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def existing_visibility(repo_id: str) -> bool | None:
    """True/False if the dataset repo exists and is private/public, None if it does not exist."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        return bool(HfApi().repo_info(repo_id, repo_type="dataset").private)
    except RepositoryNotFoundError:
        return None


def main(repo_id: str, merged: Path, private: bool, sessions: list[Path]) -> int:
    # push_to_hub only applies `private` when it creates the repo, so an existing public repo
    # would silently receive data meant to stay private.
    try:
        visibility = existing_visibility(repo_id)
    except Exception as exc:  # network/auth problems: do not guess about data visibility
        print(f"[kiteos] cannot check https://huggingface.co/datasets/{repo_id}: {exc}", flush=True)
        return 4
    if private and visibility is False:
        print(
            f"[kiteos] refusing to push: {repo_id} already exists and is PUBLIC, but this station "
            "uploads privately. Make the repo private on the Hub, or pass --public to publish.",
            flush=True,
        )
        return 5
    if merged.exists():
        shutil.rmtree(merged)
    aggregate_datasets(
        repo_ids=[f"local/{s.name}" for s in sessions],
        aggr_repo_id=repo_id,
        roots=sessions,
        aggr_root=merged,
    )
    dataset = LeRobotDataset(repo_id, root=merged)
    print(
        f"[kiteos] merged {len(sessions)} session(s): {dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames",
        flush=True,
    )
    dataset.push_to_hub(private=private)
    visibility = "private" if private else "public"
    print(f"[kiteos] pushed ({visibility}) to https://huggingface.co/datasets/{repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[3] not in ("True", "False"):
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "True", [Path(p) for p in sys.argv[4:]]))
