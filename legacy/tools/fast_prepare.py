"""Convert one recording to LeRobot without the slow per-episode quality pre-check.

- Discarded takes are excluded (as in `dimos imitation prepare`).
- Saved episodes listed in the recording's deleted_episodes.txt (U in collect.py) are dropped.
- Quality mode is "fill": camera gaps repeat the last frame and are flagged in
  complementary_info.is_filled (what the published dataset used).
- The pre-check is skipped, so a broken episode fails during writing instead of being
  rejected up front.

Runs in the LeRobot env (see upload.py). Usage: fast_prepare.py RECORDING_DIR OUTPUT_DIR
"""

import sys
from pathlib import Path

from dimos.imitation.collection.recording import RecordingSchema
from dimos.imitation.dataprep import build
from dimos.imitation.dataprep.core import EpisodeQualityReport, OutputConfig
from dimos_lerobot.dataprep import write

recording, output = Path(sys.argv[1]), Path(sys.argv[2])
marks = recording / "deleted_episodes.txt"
deleted = {int(i) for i in marks.read_text().split()} if marks.exists() else set()

config = RecordingSchema.read(recording).dataprep_config(recording, OutputConfig(path=output))
config = config.model_copy(update={"quality": config.quality.model_copy(update={"mode": "fill"})})

_extract = build.extract_episodes


def keep_undeleted(store, cfg):
    episodes = _extract(store, cfg)
    saved = [e for e in episodes if e.success]
    dropped = {saved[i].id for i in deleted if i < len(saved)}
    print(f"saved episodes: {len(saved)}, deleted: {sorted(deleted)}, converting: {len(saved) - len(dropped)}", flush=True)
    return [e for e in episodes if e.id not in dropped]


def assume_valid(store, episode, streams, sync, quality):
    return EpisodeQualityReport(episode_id=episode.id, valid=True, mode=quality.mode)


build.extract_episodes = keep_undeleted
build.inspect_episode_quality = assume_valid
print(build.run_dataprep(config, writer=write), flush=True)
