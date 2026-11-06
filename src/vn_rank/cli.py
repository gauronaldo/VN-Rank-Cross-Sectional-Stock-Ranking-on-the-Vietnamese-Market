"""Thin command-line entry point (``vn-rank``).

Each sub-command parses arguments, loads the configuration, and delegates to library code.
Further sub-commands (download, features, train, backtest, report, update, lockbox) are added
in the phases that build them.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from vn_rank.config import ConfigError, load_config
from vn_rank.utils.logging import configure_logging

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all sub-commands.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="vn-rank", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config", help="Load and validate the configuration.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a configuration error.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("Invalid configuration %s: %s", args.config, exc)
        return 1
    logger.info(
        "Configuration %s is valid (lockbox_start=%s)",
        args.config,
        config.validation.lockbox_start.isoformat(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
