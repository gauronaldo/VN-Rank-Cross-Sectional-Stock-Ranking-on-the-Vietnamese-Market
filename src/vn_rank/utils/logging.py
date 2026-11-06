"""Logging setup shared by all command-line entry points."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """Configure root logging once for a command-line run.

    Args:
        level: Standard level name such as ``"INFO"`` or ``"DEBUG"``.

    Raises:
        ValueError: If ``level`` is not a standard logging level name.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"Unknown log level: {level!r}")
    logging.basicConfig(level=numeric, format=LOG_FORMAT, force=True)
