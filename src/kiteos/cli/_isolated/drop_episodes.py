# Copyright 2026 Kite ML.
"""Remove episodes from a prepared LeRobot session in place.

Used by ``kiteos upload`` when episodes were marked for deletion (U in
``kiteos collect controls``) after the session had already been converted. Indices are the
dataset's own episode indices. Runs in dimOS's isolated LeRobot env.
Usage: drop_episodes.py DATASET_ROOT INDEX...
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

KEEP_FILES = ("kiteos_deleted_episodes.txt", "dimos_meta.json")


def main(root: Path, indices: list[int]) -> int:
    repo_id = f"local/{root.name}"
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    dataset = LeRobotDataset(repo_id, root=root)
    wanted = sorted({i for i in indices if 0 <= i < dataset.meta.total_episodes})
    if not wanted:
        print("[kiteos] no episodes to remove", flush=True)
        return 0
    if len(wanted) >= dataset.meta.total_episodes:
        print("[kiteos] refusing to remove every episode of the session", flush=True)
        return 1
    kept = delete_episodes(dataset, episode_indices=wanted, output_dir=tmp, repo_id=repo_id)
    # delete_episodes writes a fresh dataset; carry over the KiteOS/dimOS side files.
    for name in KEEP_FILES:
        if (root / name).exists() and not (tmp / name).exists():
            shutil.copy2(root / name, tmp / name)
    shutil.rmtree(root)
    tmp.rename(root)
    print(f"[kiteos] removed episodes {wanted}; {kept.meta.total_episodes} left", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), [int(i) for i in sys.argv[2:]]))
