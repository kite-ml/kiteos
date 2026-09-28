# Copyright 2026 Kite ML.
"""Offline (open-loop) test of a LeRobot checkpoint on recorded episodes. No robot.

Feeds each recorded frame (3 cameras + state) to the policy exactly as a rollout would and
compares the predicted action with the action the operator actually commanded. Writes
``summary.json`` (per-episode MAE and motion range per joint group, timing) and
``traces.json`` (predicted and true actions).

Runs in dimOS's isolated LeRobot env (see ``kiteos eval offline``).
Usage: eval_policy.py CHECKPOINT_DIR DATASET_ROOT EPISODES(comma list) OUT_DIR DEVICE [MAX_FRAMES]
DEVICE is auto (cuda > mps > cpu), cuda, mps or cpu.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(ckpt: str, root: Path, episodes: list[int], out: Path, device: str, max_frames: int | None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    print(f"[kiteos] device: {device}", flush=True)

    policy_config = PreTrainedConfig.from_pretrained(ckpt)
    policy_config.device = device
    policy = get_policy_class(policy_config.type).from_pretrained(ckpt, config=policy_config)
    policy.config.device = device
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    dataset = LeRobotDataset(f"local/{root.name}", root=root, episodes=episodes)
    names = dataset.meta.features["action"]["names"]
    obs_keys = list(policy.config.input_features)

    results: dict[int, dict[str, object]] = {}
    timings: list[float] = []
    current: int | None = None
    pred: list[np.ndarray] = []
    true: list[np.ndarray] = []

    def finish(ep: int) -> None:
        results[ep] = {"frames": len(true), "pred": np.array(pred).tolist(), "true": np.array(true).tolist()}

    for i in range(len(dataset)):
        item = dataset[i]
        ep = int(item["episode_index"])
        if ep != current:
            if current is not None:
                finish(current)
            current, pred, true = ep, [], []
            policy.reset()
            print(f"[kiteos] episode {ep} ...", flush=True)
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
    if current is None:
        print("[kiteos] no frames in the selected episodes", flush=True)
        return 1
    finish(current)

    groups = {
        "left arm (rad)": [i for i, n in enumerate(names) if n.startswith("left_joint")],
        "right arm (rad)": [i for i, n in enumerate(names) if n.startswith("right_joint")],
        "grippers (0-1)": [i for i, n in enumerate(names) if "gripper" in n],
    }
    summary: dict[str, object] = {"device": device, "checkpoint": ckpt, "dataset": str(root), "episodes": {}}
    for ep, r in results.items():
        err = np.abs(np.array(r["pred"]) - np.array(r["true"]))
        span = np.ptp(np.array(r["true"]), axis=0)
        summary["episodes"][ep] = {  # type: ignore[index]
            "frames": r["frames"],
            **{
                g: {"mae": round(float(err[:, idx].mean()), 4), "motion_range": round(float(span[idx].mean()), 3)}
                for g, idx in groups.items()
                if idx
            },
        }
    t = np.array(timings) * 1000
    summary["timing_ms"] = {
        "median": round(float(np.median(t)), 1),
        "max (new chunk)": round(float(t.max()), 1),
        "chunk_compute_median": round(float(np.median(t[t >= np.percentile(t, 97)])), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "traces.json").write_text(json.dumps({"names": names, "episodes": results}))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7):
        print(__doc__)
        sys.exit(2)
    sys.exit(
        main(
            sys.argv[1],
            Path(sys.argv[2]),
            [int(e) for e in sys.argv[3].split(",") if e.strip()],
            Path(sys.argv[4]),
            sys.argv[5],
            int(sys.argv[6]) if len(sys.argv) == 7 else None,
        )
    )
