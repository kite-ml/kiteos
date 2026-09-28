#!/bin/bash
# Attach policy start/stop controls to a running kite_rollout (simulated or real arms).
# The dual-arm policy module lives in ~/teleop_stack/kiteos, so dimOS needs it on PYTHONPATH.
cd "$(dirname "$0")/../dimos" || exit 1
PYTHONPATH="$(cd ../kiteos && pwd)${PYTHONPATH:+:$PYTHONPATH}" exec uv run --no-sync dimos imitation rollout
