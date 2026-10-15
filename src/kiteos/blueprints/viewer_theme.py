# Copyright 2026 Kite ML.
"""The KiteOS look for the runtime's viser manipulation viewer: KiteOS logo, name and colours.

The runtime themes its viewer in one place: its visualizer calls ``apply_dimos_theme(server)`` when
the viewer starts (``dimos/manipulation/visualization/viser/visualizer.py``). Without editing the
runtime, ``install_kiteos_viewer_theme()`` points that name at ``apply_kiteos_theme`` in the viewer's
own process; ``KiteManipulationModule`` calls it before the viewer starts. If the runtime ever
renames the hook, the viewer keeps the runtime's theme and a warning says so
(tests/test_viewer_theme.py checks the hook against the pinned runtime).
"""

from __future__ import annotations

import base64
from importlib import resources
from typing import Any

from dimos.utils.logging_config import setup_logger

from kiteos import brand

logger = setup_logger()

KITEOS_VIEWER_TITLE = f"{brand.BRAND_NAME} Manipulation"
KITEOS_VIEWER_LABEL = brand.BRAND_NAME
KITEOS_BRAND_RGB = (47, 128, 237)  # Kite sky #2F80ED (the Quest page's --kite-sky)
_RUNTIME_THEME_HOOK = "apply_dimos_theme"


def kiteos_logo_data_url() -> str:
    """The KiteOS kite mark (the Quest page's logo) as a data URL."""
    svg = resources.files("kiteos").joinpath("quest", "static", "logo.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode("ascii")


def apply_kiteos_theme(server: Any) -> bool:
    """Theme a viser server as KiteOS: titlebar logo and name, dark mode, the Kite blue."""
    try:
        from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

        logo = kiteos_logo_data_url()
        titlebar = TitlebarConfig(
            buttons=(TitlebarButton(text=KITEOS_VIEWER_TITLE, icon=None, href=brand.WEBSITE),),
            image=TitlebarImage(image_url_light=logo, image_url_dark=logo, image_alt=brand.BRAND_NAME, href=brand.WEBSITE),
        )
        server.gui.configure_theme(
            titlebar_content=titlebar,
            control_layout="fixed",
            control_width="large",
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            brand_color=KITEOS_BRAND_RGB,
        )
        server.gui.set_panel_label(KITEOS_VIEWER_LABEL)
    except Exception:  # a look must never stop the viewer
        logger.warning("Skipping the KiteOS viewer theme; viser's theme API is unavailable", exc_info=True)
        return False
    return True


def install_kiteos_viewer_theme() -> bool:
    """Make the runtime's viser visualizer (in this process) start with the KiteOS theme."""
    from dimos.manipulation.visualization.viser import visualizer

    if not callable(getattr(visualizer, _RUNTIME_THEME_HOOK, None)):
        logger.warning(f"The runtime's viewer has no {_RUNTIME_THEME_HOOK}(); it keeps the runtime's theme")
        return False
    setattr(visualizer, _RUNTIME_THEME_HOOK, apply_kiteos_theme)
    return True
