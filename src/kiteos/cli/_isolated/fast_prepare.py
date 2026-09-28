# Copyright 2026 Kite ML.
"""Convert one KiteOS recording to a LeRobot dataset without the slow quality pre-check.

- Discarded takes are excluded (as in the runtime's ``imitation prepare``).
- Saved episodes listed in the recording's ``deleted_episodes.txt`` (U in
  ``kiteos collect controls``; 0-based saved-episode indices) are dropped. The indices actually
  applied are written to ``<output>/kiteos_deleted_episodes.txt`` so later deletions can be
  applied with drop_episodes.py instead of a full re-conversion.
- Quality mode is "fill": camera gaps repeat the last frame and are flagged in
  ``complementary_info.is_filled``.
- The per-episode pre-check is skipped, so a broken episode fails while writing instead of
  being rejected up front.

Runs in the runtime's isolated LeRobot env (see ``kiteos upload``).
Usage: fast_prepare.py RECORDING_DIR OUTPUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

from dimos.imitation.collection.recording import RecordingSchema
from dimos.imitation.dataprep import build
from dimos.imitation.dataprep.core import EpisodeQualityReport, OutputConfig
from dimos_lerobot.dataprep import write

DELETED_FILE = "deleted_episodes.txt"
APPLIED_FILE = "kiteos_deleted_episodes.txt"


def _read_indices(path: Path) -> set[int]:
    return {int(i) for i in path.read_text().split()} if path.exists() else set()


def main(recording: Path, output: Path) -> int:
    # These two dataprep internals are replaced below; fail loudly if the runtime renamed them.
    for name in ("extract_episodes", "inspect_episode_quality", "run_dataprep"):
        if not hasattr(build, name):
            print(f"[kiteos] the runtime dataprep has no build.{name}; update fast_prepare.py", flush=True)
            return 3

    deleted = _read_indices(recording / DELETED_FILE)
    config = RecordingSchema.read(recording).dataprep_config(recording, OutputConfig(path=output))
    config = config.model_copy(update={"quality": config.quality.model_copy(update={"mode": "fill"})})

    extract = build.extract_episodes
    applied: set[int] = set()

    def keep_undeleted(store, cfg):  # type: ignore[no-untyped-def]
        episodes = extract(store, cfg)
        saved = [e for e in episodes if e.success]
        applied.update(i for i in deleted if 0 <= i < len(saved))
        dropped = {saved[i].id for i in applied}
        print(
            f"[kiteos] saved episodes: {len(saved)}, deleted: {sorted(applied)}, "
            f"converting: {len(saved) - len(dropped)}",
            flush=True,
        )
        return [e for e in episodes if e.id not in dropped]

    def assume_valid(store, episode, streams, sync, quality):  # type: ignore[no-untyped-def]
        return EpisodeQualityReport(episode_id=episode.id, valid=True, mode=quality.mode)

    build.extract_episodes = keep_undeleted
    build.inspect_episode_quality = assume_valid
    print(build.run_dataprep(config, writer=write), flush=True)
    (output / APPLIED_FILE).write_text("".join(f"{i}\n" for i in sorted(applied)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
