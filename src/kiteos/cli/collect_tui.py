# Copyright 2026 Kite ML.
"""KiteOS episode controls: the runtime's CollectionApp with KiteOS branding and extra keys.

Keys: Space start/save   R restart (discard this take, start a new one)
      D discard          U delete the last saved episode (before upload)
      P pause (save this take, power down, upload)
      Q detach (robot and recording keep running)
Quest controllers: B start/save, Y discard (the runtime default).

Imports the runtime and textual; loaded lazily by ``kiteos collect controls``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dimos.imitation.tui import CollectionApp, CollectionSession
from textual.binding import Binding

from kiteos import brand
from kiteos.cli.collect import delete_last_saved

PAUSE_RESULT = "pause"


class KiteCollectionApp(CollectionApp):
    """Collection controls with restart, delete-last and pause-and-upload."""

    CSS = (
        CollectionApp.CSS
        + f"""
    #dashboard {{ border: double {brand.PRIMARY}; }}
    #title {{ color: {brand.PRIMARY}; }}
    #state {{ border: round {brand.SUCCESS}; color: {brand.SUCCESS}; }}
    #state.recording, #state.disconnected {{ border: round {brand.ERROR}; color: {brand.ERROR}; }}
    #message {{ color: {brand.WARNING}; }}
    """
    )
    BINDINGS = [
        Binding("space", "toggle_recording", "Start / save"),
        Binding("r", "restart", "Restart take"),
        Binding("d", "discard", "Discard"),
        Binding("u", "delete_last", "Delete last saved"),
        Binding("p", "pause", "Pause + upload"),
        Binding("q", "quit", "Detach"),
        Binding("ctrl+c", "quit", "Detach", show=False),
    ]

    def __init__(
        self,
        session: CollectionSession,
        *,
        title: str,
        session_dir: Callable[[], Path | None],
    ) -> None:
        super().__init__(session, title=title)
        self._session_dir = session_dir

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
        session_dir = self._session_dir()
        if session_dir is None:
            self._message = "No recording session found for this station."
            self._refresh()
            return
        deleted, remaining = delete_last_saved(session_dir, self._status.episodes_saved)
        if deleted is None:
            self._message = "No saved episodes left to delete in this session."
        else:
            self._message = (
                f"Saved episode #{deleted + 1} of {session_dir.name} deleted "
                f"(removed before upload; {remaining} kept). Press U again to delete the one before."
            )
        self._refresh()

    def action_pause(self) -> None:
        if self._status.state == "recording":
            self._episode_command("save")
        self.exit(PAUSE_RESULT)  # type: ignore[arg-type]
