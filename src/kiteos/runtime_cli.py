# Copyright 2026 Kite ML.
"""``kiteos-runtime``: the base runtime's command line under the KiteOS name.

KiteOS starts every robot stack through this entry point
(``uv run --no-sync kiteos-runtime run kiteos.<setup> ...``), so process listings,
copy-pasteable commands and ``--help`` read KiteOS. The arguments reach the runtime's
own command line unchanged; nothing about how a stack runs is different.

People normally use ``kiteos``; this entry point is for KiteOS itself and for advanced
use (``kiteos-runtime --help``).
"""

from __future__ import annotations

RUNTIME_HELP = (
    "KiteOS runtime: start, inspect and stop robot stacks. "
    "Day-to-day use goes through `kiteos` (see `kiteos --help`)."
)


def main() -> None:
    # Imported here, not at module level: the runtime CLI loads heavy dependencies and
    # `import kiteos.runtime_cli` must stay cheap (tests, `kiteos doctor`).
    from dimos.cli import dimos as runtime_cli

    runtime_cli.main.info.help = RUNTIME_HELP
    runtime_cli.cli_main()


if __name__ == "__main__":
    main()
