"""Merge prepared LeRobot sessions into one dataset and push it (runs in the LeRobot env).

Usage (called by upload.py): hub_push.py REPO_ID MERGED_ROOT PRIVATE SESSION_ROOT...
"""

import shutil
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id, merged, private = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "True"
sessions = [Path(p) for p in sys.argv[4:]]

if merged.exists():
    shutil.rmtree(merged)  # rebuilt from all sessions each time
aggregate_datasets(
    repo_ids=[f"local/{s.name}" for s in sessions],
    aggr_repo_id=repo_id,
    roots=sessions,
    aggr_root=merged,
)
dataset = LeRobotDataset(repo_id, root=merged)
print(f"Merged {len(sessions)} session(s): {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
dataset.push_to_hub(private=private)
print(f"Pushed to https://huggingface.co/datasets/{repo_id}")
