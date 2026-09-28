"""Episode controls for a running collection session, plus restart and pause.

Keys: Space start/save   R restart (discard this take, start a new one)
      D discard          U delete the last saved episode (before upload)
      P pause (save this take, power down, upload)
      Q detach (robot and recording keep running)
Quest controllers: B start/save, Y discard (dimOS default).

Pause saves the take in progress, asks you to hold the arms, stops dimOS so the
recording file is finalized, then runs upload.py to convert every finished session
and push the merged dataset to Hugging Face.

Run with the dimOS venv:  ../dimos/.venv/bin/python collect.py
"""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from textual.binding import Binding

from dimos.imitation.tui import CollectionApp, CollectionSession
from dimos.porcelain.dimos import Dimos

TOOLS = Path(__file__).resolve().parent
DIMOS = TOOLS.parent / "dimos"
RECORDINGS = DIMOS / "recordings"
DELETED_FILE = "deleted_episodes.txt"  # 0-based saved-episode indices, applied by upload.py


class SessionApp(CollectionApp):
    BINDINGS = [
        Binding("space", "toggle_recording", "Start / save"),
        Binding("r", "restart", "Restart take"),
        Binding("d", "discard", "Discard"),
        Binding("u", "delete_last", "Delete last saved"),
        Binding("p", "pause", "Pause + upload"),
        Binding("q", "quit", "Detach"),
        Binding("ctrl+c", "quit", "Detach", show=False),
    ]

    def action_restart(self) -> None:
        if self._status.state == "recording":
            self._episode_command("discard")
        self._episode_command("start")
        self._message = "Take restarted. Recording."
        self._refresh()

    def action_delete_last(self) -> None:
        """Mark the newest saved episode of this session for removal before upload."""
        if self._status.state == "recording":
            self._message = "A take is recording: press D to discard it instead."
            self._refresh()
            return
        session = sorted(RECORDINGS.glob("close-box-*"))[-1]
        marks = session / DELETED_FILE
        deleted = {int(x) for x in marks.read_text().split()} if marks.exists() else set()
        remaining = [i for i in range(self._status.episodes_saved) if i not in deleted]
        if not remaining:
            self._message = "No saved episodes left to delete in this session."
        else:
            with marks.open("a") as f:
                f.write(f"{remaining[-1]}\n")
            self._message = (
                f"Saved episode #{remaining[-1] + 1} of {session.name} deleted "
                f"(removed before upload; {len(remaining) - 1} kept). Press U again to delete the one before."
            )
        self._refresh()

    def action_pause(self) -> None:
        if self._status.state == "recording":
            self._episode_command("save")
        self.exit("pause")


def dimos_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["uv", "run", "--no-sync", "dimos", *args], cwd=DIMOS, text=True, capture_output=True)


def power_down() -> bool:
    print("\nPausing: the take in progress was saved.")
    print("Stopping dimOS turns the motors off, and the arms may drop.")
    if input("Type 'stop' once someone is holding both arms: ").strip().lower() != "stop":
        print("Not stopping. The robot is still running; run collect.py again to continue.")
        return False
    time.sleep(1.0)  # let the save marker reach the recorder
    match = re.search(r"PID:\s*(\d+)", dimos_cli("status").stdout)
    if match is None:
        print("No running dimOS found; nothing to stop.")
        return True
    pid = int(match.group(1))
    # Graceful SIGTERM only. `dimos stop` escalates to SIGKILL after 5 s, which could
    # interrupt the recorder while it finalizes a multi-GB file.
    os.kill(pid, signal.SIGTERM)
    print("Stopping dimOS and finalizing the recording", end="", flush=True)
    for _ in range(300):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print("\ndimOS stopped; the recording is finalized.")
            return True
        print(".", end="", flush=True)
        time.sleep(1.0)
    print("\ndimOS is still shutting down after 5 min; check the collection tab, then run upload.py.")
    return False


def main() -> int:
    driver = Dimos.connect()
    result = SessionApp(CollectionSession(driver), title="Close the box").run()
    if result != "pause":
        return 0
    if not power_down():
        return 1
    return subprocess.run([sys.executable, str(TOOLS / "upload.py")]).returncode


if __name__ == "__main__":
    sys.exit(main())
