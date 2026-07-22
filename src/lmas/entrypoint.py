from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Dispatch GUI launches without importing the full scientific CLI stack."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "gui":
        from .gui.app import main as gui_main

        return int(gui_main(arguments[1:]))
    from .cli import main as cli_main

    return int(cli_main(arguments))


__all__ = ["main"]
