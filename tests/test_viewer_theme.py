# Copyright 2026 Kite ML.
"""The KiteOS look of the runtime's viser viewer (kiteos.blueprints.viewer_theme)."""

from __future__ import annotations

import base64
from unittest import mock

import pytest

pytestmark = pytest.mark.runtime
pytest.importorskip("dimos", reason="needs the KiteOS runtime virtualenv")

from dimos.manipulation.visualization.viser import visualizer  # noqa: E402

from kiteos.blueprints import viewer_theme  # noqa: E402


def test_the_theme_puts_the_kiteos_name_logo_and_colour_in_the_viewer() -> None:
    server = mock.Mock()
    assert viewer_theme.apply_kiteos_theme(server)
    kwargs = server.gui.configure_theme.call_args.kwargs
    assert kwargs["brand_color"] == viewer_theme.KITEOS_BRAND_RGB and kwargs["dark_mode"]
    titlebar = kwargs["titlebar_content"]
    assert titlebar["buttons"][0]["text"] == "KiteOS Manipulation"  # viser's titlebar types are TypedDicts
    logo = base64.b64decode(titlebar["image"]["image_url_dark"].split(",", 1)[1])
    assert b"<title>KiteOS</title>" in logo
    server.gui.set_panel_label.assert_called_once_with("KiteOS")


def test_a_failing_theme_never_stops_the_viewer() -> None:
    server = mock.Mock()
    server.gui.configure_theme.side_effect = TypeError("old viser")
    assert viewer_theme.apply_kiteos_theme(server) is False


def test_the_runtime_viewer_starts_with_the_kiteos_theme(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hook the runtime's visualizer calls when its viewer starts (pinned runtime).
    assert callable(visualizer.apply_dimos_theme)
    monkeypatch.setattr(visualizer, "apply_dimos_theme", visualizer.apply_dimos_theme)  # restored after
    assert viewer_theme.install_kiteos_viewer_theme()
    assert visualizer.apply_dimos_theme is viewer_theme.apply_kiteos_theme
