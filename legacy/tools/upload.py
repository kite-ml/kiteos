"""Convert finished collection sessions and push them as one dataset to Hugging Face.

1. Every recordings/close-box-NNN that has no datasets/close-box-NNN yet is converted
   with fast_prepare.py (fill mode, no slow quality pre-check; skipped with a note if
   dimOS is still writing it), dropping episodes marked with U in collect.py.
2. All converted sessions are merged into datasets/_hub and pushed to REPO_ID.

Run with the dimOS venv:  ../dimos/.venv/bin/python upload.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

from dimos.experimental.isolated_python.module import isolated_python_environment
from dimos.imitation.dataprep.lerobot import isolated_python_run_command, lerobot_project

REPO_ID = "kiteml/dual-openyam-close-box"
PRIVATE = False
TOOLS = Path(__file__).resolve().parent
DIMOS = TOOLS.parent / "dimos"
RECORDINGS = DIMOS / "recordings"
DATASETS = DIMOS / "datasets"
MERGED = DATASETS / "_hub"


def prepare(recording: Path) -> bool:
    output = DATASETS / recording.name
    if (output / "meta" / "info.json").exists():
        return True
    if output.exists():  # left over from a failed attempt
        shutil.rmtree(output)
    print(f"Converting {recording.name} ...", flush=True)
    # fast_prepare.py: fill mode, no slow quality pre-check, drops deleted_episodes.txt entries.
    project = lerobot_project()
    command = isolated_python_run_command(project, "python", str(TOOLS / "fast_prepare.py"), str(recording), str(output))
    result = subprocess.run(command, cwd=project, env=isolated_python_environment(project),
                            text=True, capture_output=True)
    lines = [l for l in (result.stdout + result.stderr).strip().splitlines() if "[dataprep]" in l or "episodes" in l]
    tail = "\n".join(lines[-4:])
    if result.returncode != 0:
        if "opcode 255" in result.stdout + result.stderr:
            print(f"  skipped: {recording.name} is still being recorded (stop dimOS first)")
        else:
            print("  FAILED:\n" + "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:]))
        shutil.rmtree(output, ignore_errors=True)
        return False
    print(f"  done\n{tail}")
    return True


def main() -> int:
    recordings = sorted(p for p in RECORDINGS.glob("close-box-*") if (p / "recording.mcap").exists())
    if not recordings:
        print("No recordings found.")
        return 1
    DATASETS.mkdir(exist_ok=True)
    ready = [DATASETS / r.name for r in recordings if prepare(r)]
    if not ready:
        print("Nothing converted yet; nothing to upload.")
        return 1
    print(f"Merging {len(ready)} session(s) and pushing to https://huggingface.co/datasets/{REPO_ID}")
    project = lerobot_project()
    command = isolated_python_run_command(
        project, "python", str(TOOLS / "hub_push.py"), REPO_ID, str(MERGED), str(PRIVATE), *map(str, ready)
    )
    return subprocess.run(command, cwd=project, env=isolated_python_environment(project)).returncode


if __name__ == "__main__":
    sys.exit(main())
