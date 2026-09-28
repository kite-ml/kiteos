"""Offline (open-loop) test of a LeRobot ACT checkpoint on recorded episodes. No robot.

Feeds each recorded frame (3 cameras + state) to the policy exactly as a rollout would,
and compares the predicted action with the action the operator actually commanded.
Runs in the LeRobot env (see upload.py for the launcher pattern).

Usage: eval_policy.py CHECKPOINT_DIR DATASET_ROOT EPISODES(comma list) OUT_DIR [MAX_FRAMES]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ckpt, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[4])
episodes = [int(e) for e in sys.argv[3].split(",")]
max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else None  # per episode; large models are slow
out.mkdir(parents=True, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"

policy_config = PreTrainedConfig.from_pretrained(ckpt)
policy_config.device = device
policy = get_policy_class(policy_config.type).from_pretrained(ckpt, config=policy_config)
policy.config.device = device
policy.to(device).eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=ckpt,
    preprocessor_overrides={"device_processor": {"device": device}},
)
dataset = LeRobotDataset(f"local/{root.name}", root=root, episodes=episodes)
names = dataset.meta.features["action"]["names"]
obs_keys = [k for k in policy.config.input_features]

results, timings = {}, []
current, pred, true = None, [], []


def finish(ep):
    p, t = np.array(pred), np.array(true)
    results[ep] = {"frames": len(t), "pred": p.tolist(), "true": t.tolist()}


for i in range(len(dataset)):
    item = dataset[i]
    ep = int(item["episode_index"])
    if ep != current:
        if current is not None:
            finish(current)
        current, pred, true = ep, [], []
        policy.reset()
        print(f"episode {ep} ...", flush=True)
    if max_frames is not None and len(true) >= max_frames:
        continue
    batch = {k: item[k].unsqueeze(0) for k in obs_keys}
    batch["task"] = [item["task"]]
    start = time.perf_counter()
    with torch.inference_mode():
        action = post(policy.select_action(pre(batch)))
    timings.append(time.perf_counter() - start)
    pred.append(action.squeeze(0).float().cpu().numpy())
    true.append(item["action"].float().numpy())
finish(current)

groups = {
    "left arm (rad)": [i for i, n in enumerate(names) if n.startswith("left_joint")],
    "right arm (rad)": [i for i, n in enumerate(names) if n.startswith("right_joint")],
    "grippers (0-1)": [i for i, n in enumerate(names) if "gripper" in n],
}
summary = {"device": device, "episodes": {}, "timing_ms": {}}
for ep, r in results.items():
    err = np.abs(np.array(r["pred"]) - np.array(r["true"]))
    span = np.ptp(np.array(r["true"]), axis=0)
    summary["episodes"][ep] = {
        "frames": r["frames"],
        **{g: {"mae": round(float(err[:, idx].mean()), 4), "motion_range": round(float(span[idx].mean()), 3)}
           for g, idx in groups.items()},
    }
t = np.array(timings) * 1000
summary["timing_ms"] = {"median": round(float(np.median(t)), 1), "max (new chunk)": round(float(t.max()), 1),
                        "chunk_compute_median": round(float(np.median(t[t > np.percentile(t, 97)])), 1)}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "traces.json").write_text(json.dumps({"names": names, "episodes": results}))
print(json.dumps(summary, indent=2))
