"""Rebuild a checkpoint's missing LeRobot pre/post-processors from its dataset's stats.

Use when a checkpoint download lost policy_preprocessor*/policy_postprocessor* files. The
normalizer statistics come from the dataset the policy was trained on, which is how
LeRobot builds them at training time. Runs in the LeRobot env.

Usage: rebuild_processors.py CHECKPOINT_DIR DATASET_ROOT
"""

import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors

ckpt, root = Path(sys.argv[1]), Path(sys.argv[2])
if list(ckpt.glob("policy_preprocessor*.json")) or list(ckpt.glob("policy_postprocessor*.json")):
    sys.exit(f"{ckpt} already has processor files; not overwriting")

config = PreTrainedConfig.from_pretrained(ckpt)
meta = LeRobotDatasetMetadata(f"local/{root.name}", root=root)
pre, post = make_pre_post_processors(config, dataset_stats=meta.stats)
pre.save_pretrained(ckpt)
post.save_pretrained(ckpt)
(ckpt / "REBUILT_PROCESSORS.txt").write_text(
    f"The downloaded checkpoint was truncated after model.safetensors; the pre/post-processors\n"
    f"here were rebuilt by tools/rebuild_processors.py from the stats of {root}\n"
    f"(normalization {config.normalization_mapping}). Replace them with the originals if available.\n"
)
print("rebuilt:", sorted(p.name for p in ckpt.iterdir() if p.name.startswith("policy_")))
