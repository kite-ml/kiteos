"""Remove episodes from a prepared LeRobot session in place (runs in the LeRobot env).

Usage (called by upload.py): drop_episodes.py DATASET_ROOT INDEX...
"""

import shutil
import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1])
indices = sorted({int(i) for i in sys.argv[2:]})
repo_id = f"local/{root.name}"
tmp = root.with_name(root.name + ".tmp")
if tmp.exists():
    shutil.rmtree(tmp)

dataset = LeRobotDataset(repo_id, root=root)
indices = [i for i in indices if i < dataset.meta.total_episodes]
if indices:
    kept = delete_episodes(dataset, episode_indices=indices, output_dir=tmp, repo_id=repo_id)
    shutil.rmtree(root)
    tmp.rename(root)
    print(f"  removed episodes {indices}; {kept.meta.total_episodes} left")
