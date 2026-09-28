# Copyright 2026 Kite ML.
"""Suite-wide test setup: keep every dimOS session the tests open on a private bus.

dimOS picks its pub/sub + RPC bus from ``LCM_DEFAULT_URL`` (read when ``lcmservice`` is imported)
and ``ZENOH_SCOUT_ADDR`` (read when ``dimos.core.global_config`` is imported). A robot stack
running on the same machine uses the defaults, so a test that serves a ``ControlCoordinator`` or
publishes ``/joint_command`` on the default bus could reach real arms. This conftest is imported
before any test module, so the values below are in place before the first dimOS import; they are
per-process, so concurrent test runs do not share a bus either.

Set ``KITEOS_TEST_EXTERNAL_BUS=1`` to keep the caller's bus settings (the mock e2e test then
refuses to run, see tests/e2e).
"""

from __future__ import annotations

import os
import tempfile

import pytest

_EXTERNAL = os.environ.get("KITEOS_TEST_EXTERNAL_BUS", "").strip().lower() in {"1", "true", "yes"}

if not _EXTERNAL:
    _BUCKET = os.getpid() % 5000
    # ttl=0 keeps LCM multicast on this host; the port (not the default 7667) keeps it private.
    os.environ["LCM_DEFAULT_URL"] = f"udpm://239.255.76.67:{21000 + _BUCKET}?ttl=0"
    # A private zenoh scouting group (the default is 224.0.0.224:7446).
    os.environ["ZENOH_SCOUT_ADDR"] = f"224.0.0.224:{31000 + _BUCKET}"
    os.environ["KITEOS_TEST_PRIVATE_BUS"] = "1"

# KiteOS run-state files of the tests never mix with (or reveal) the operator's real runs.
os.environ["KITEOS_RUNS_DIR"] = tempfile.mkdtemp(prefix="kiteos-test-runs-")


@pytest.fixture(autouse=True)
def _no_host_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never scan this host for stacks or listen on a bus (a real rig may be running).

    Tests that exercise the probes call ``kiteos.cli.runs`` directly with fakes.
    """
    from kiteos.cli import launch

    monkeypatch.setattr(launch, "host_stacks", lambda **_: [])
    monkeypatch.setattr(launch, "record_run", lambda state: None)
    monkeypatch.setattr(
        launch, "bus_probe", lambda *_a, **_k: __import__("kiteos.cli.runs", fromlist=["BusProbe"]).BusProbe(0, [])
    )
