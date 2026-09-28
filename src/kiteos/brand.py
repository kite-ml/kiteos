# Copyright 2026 Kite ML.
"""KiteOS brand constants: names, colours and the terminal banner.

Import-free on purpose (see ``kiteos/__init__.py``): terminal rendering lives in
``kiteos.cli``, web styling in ``kiteos/quest/static``.
"""

from __future__ import annotations

BRAND_NAME = "KiteOS"
COMPANY = "Kite ML"
TAGLINE = "Kite's robot runtime for data collection and evaluation, built on dimOS"
WEBSITE = "https://kiteml.com"

# Cotufa signal stops (Kite design system).
GOLD = "#E8D44D"
SENSOR = "#4DE8C1"
BLUE = "#3B82F6"
POLICY = "#8B5CF6"
MOTION = "#E84D8A"
INK = "#0B0F14"
PAPER = "#F7F7F5"
MUTED = "#8A8F98"

# Semantic roles used by the CLI, the TUIs and the Quest page.
PRIMARY = BLUE
ACCENT = POLICY
SUCCESS = SENSOR
WARNING = GOLD
ERROR = MOTION

_BANNER_ART = r"""
  _  ___ _        ___  ____
 | |/ (_) |_ ___ / _ \/ ___|
 | ' /| | __/ _ \ | | \___ \
 | . \| | ||  __/ |_| |___) |
 |_|\_\_|\__\___|\___/|____/
""".strip("\n")


def banner(version: str | None = None) -> str:
    """Multi-line ASCII banner with the version and tagline underneath."""
    tail = f"{BRAND_NAME} {version}" if version else BRAND_NAME
    return f"{_BANNER_ART}\n {tail} · {TAGLINE}"


def title(*parts: str) -> str:
    """Window/TUI title such as ``"KiteOS · close the box"``."""
    return " · ".join([BRAND_NAME, *(p for p in parts if p)])
