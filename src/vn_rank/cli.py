"""Thin command-line entry point (``vn-rank``).

Each sub-command parses arguments, loads the configuration, and delegates to library code.
Further sub-commands (features, train, backtest, report, update, lockbox) are added in the
phases that build them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from vn_rank.config import Config, ConfigError, load_config
from vn_rank.data import download
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
    commands.add_parser("download", help="Download or resume the raw vnstock snapshot.")
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
    handlers: dict[str, Callable[[Config], int]] = {
        "validate-config": _validate_config,
        "download": _download,
    }
    return handlers[args.command](config)


def _validate_config(config: Config) -> int:
    logger.info(
        "Configuration is valid (lockbox_start=%s)", config.validation.lockbox_start.isoformat()
    )
    return 0


def _download(config: Config) -> int:
    from vn_rank.data.vnstock_client import VnstockClient  # noqa: PLC0415 - needs vnstock

    data, raw_dir = config.data, config.paths.raw_dir
    spec = download.SnapshotSpec(
        source=data.source,
        index_symbol=data.index_symbol,
        exchanges=data.exchanges,
        start=data.history_start,
        end=download.resolve_snapshot_end(raw_dir, dt.datetime.now(dt.UTC)),
        max_rows=data.max_rows_per_request,
    )
    policy = download.RetryPolicy(
        data.request_interval_seconds, data.max_retries, data.retry_backoff_seconds
    )
    logger.info("Snapshot %s to %s into %s", spec.start, spec.end, raw_dir)
    counts = download.download_snapshot(
        VnstockClient(spec), download.RateLimitedCaller(policy), spec, raw_dir
    )
    logger.info("Download finished: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
