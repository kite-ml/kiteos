#!/bin/bash
# Start a collection session on the real arms, recording into the next free
# recordings/close-box-NNN folder (dimOS never reuses a recording folder).
# Runs in the foreground: `dimos run --daemon` crashes on macOS (pyarrow's libcurl
# init after fork). Run it from the Claude Terminal panel or another terminal that
# has camera access.
set -euo pipefail
cd "$(dirname "$0")/../dimos"

n=1
while [ -e "recordings/close-box-$(printf %03d "$n")" ]; do n=$((n + 1)); done
rec="recordings/close-box-$(printf %03d "$n")"

echo "This powers both arms' motors."
read -r -p "Is someone supporting both arms and is the workspace clear? [y/N] " ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || { echo "Not starting."; exit 1; }

echo "Recording to $rec"
exec uv run --no-sync dimos run dual-openyam-quest-collection \
  --recorder.recording "$rec" \
  --recorder.format mcap \
  --episodes.task "close the box" \
  --controlcoordinator.left-can-port 006C003E4845500F20303750 \
  --controlcoordinator.right-can-port 003900374845500F20303750 \
  --left-wrist.hardware.camera-index 0 \
  --right-wrist.hardware.camera-index 2 \
  --top.hardware.camera-index 1
