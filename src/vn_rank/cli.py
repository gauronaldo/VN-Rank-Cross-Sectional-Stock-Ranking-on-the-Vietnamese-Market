"""Thin command-line entry point (``vn-rank``).

Each sub-command parses arguments, loads the configuration, and delegates to library code.
Further sub-commands (features, train, backtest, report, update, lockbox) are added in the
phases that build them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from vn_rank.config import Config, ConfigError, config_to_dict, load_config
from vn_rank.data import download, storage, validate
from vn_rank.evaluation.ic import SignalSettings
from vn_rank.features import builder
from vn_rank.utils.logging import MlflowFoldLogger, configure_logging
from vn_rank.utils.seeds import set_global_seed
from vn_rank.validation import splits, walk_forward

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
QUALITY_REPORT_FILE = "data_quality.md"
TICKERS_FIGURE_FILE = "tickers_per_year.png"
QUALITY_FLAGS_FILE = "quality_flags.parquet"
COVERAGE_FILE = "coverage.parquet"
REPORT_TOP_N = 15
PANEL_FILE = "panel.parquet"
SCHEMA_FILE = "schema.json"
PREDICTIONS_FILE = "predictions.parquet"
RESULTS_STEM = "signal_results"
RESULTS_DIGITS = 4
EXPERIMENT_NAME = "walk_forward"
TRIPWIRE_EXIT_CODE = 2
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
    commands.add_parser("validate-data", help="Run data-quality checks on the raw snapshot.")
    commands.add_parser("features", help="Build the development panel (before the lockbox).")
    commands.add_parser("train", help="Tune, fit, and score all models on walk-forward folds.")
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
        "validate-data": _validate_data,
        "features": _features,
        "train": _train,
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


def _validate_data(config: Config) -> int:
    raw_dir, paths = config.paths.raw_dir, config.paths
    panel = storage.load_ohlcv_panel(raw_dir)
    symbols = storage.read_parquet(raw_dir / storage.SYMBOLS_FILE)
    index = storage.read_parquet(storage.index_path(raw_dir, config.data.index_symbol))
    settings = validate.QualitySettings(
        config.backtest.price_limits, config.data.band_check_tolerance
    )
    report = validate.run_quality_checks(panel, symbols, pd.Index(index["date"]), settings)
    metadata: dict[str, object] = storage.read_json(raw_dir / storage.METADATA_FILE)
    manifest = storage.read_json(raw_dir / storage.MANIFEST_FILE)
    for status in (download.STATUS_FAILED, download.STATUS_EMPTY):
        metadata[f"{status}_symbols"] = sorted(
            s for s, e in manifest.items() if e["status"] == status
        )
    text = validate.render_markdown(report, metadata, REPORT_TOP_N)
    storage.write_text(text, paths.reports_dir / QUALITY_REPORT_FILE)
    figure = validate.plot_tickers_per_year(report.tickers_per_year)
    storage.save_figure(figure, paths.reports_dir / TICKERS_FIGURE_FILE, config.project.figure_dpi)
    storage.write_parquet(report.flags, paths.interim_dir / QUALITY_FLAGS_FILE)
    storage.write_parquet(report.coverage.reset_index(), paths.interim_dir / COVERAGE_FILE)
    logger.info("Flags per check:\n%s", report.counts.to_string())
    return 0


def _features(config: Config) -> int:
    raw_dir, lockbox = config.paths.raw_dir, pd.Timestamp(config.validation.lockbox_start)
    raw = storage.load_ohlcv_panel(raw_dir)
    raw = raw.loc[raw["date"] < lockbox]  # lockbox prices never enter development
    index = storage.read_parquet(storage.index_path(raw_dir, config.data.index_symbol))
    calendar = pd.DatetimeIndex(index.loc[index["date"] < lockbox, "date"].sort_values())
    symbols = storage.read_parquet(raw_dir / storage.SYMBOLS_FILE)
    exchange = symbols.set_index("symbol")["exchange"]
    panel = builder.build_panel(raw, calendar, exchange, config)
    out_dir = config.paths.processed_dir
    storage.write_parquet(panel, out_dir / PANEL_FILE)
    schema = {
        "columns": builder.describe_columns(config.features),
        "dtypes": {c: str(t) for c, t in panel.dtypes.items()},
        "feature_groups": builder.feature_groups(config.features),
        "prices_before": lockbox.date().isoformat(),
        "snapshot": storage.read_json(raw_dir / storage.METADATA_FILE),
        "rows": len(panel),
    }
    storage.write_json(schema, out_dir / SCHEMA_FILE)
    per_day = panel.groupby("date").size()
    first, last = panel["date"].min().date(), panel["date"].max().date()
    symbols_used = panel["symbol"].nunique()
    logger.info("Panel: %d rows, %d symbols, %s to %s", len(panel), symbols_used, first, last)
    yearly = per_day.groupby(pd.DatetimeIndex(per_day.index).year).median()
    logger.info("Universe size per year (median):\n%s", yearly.to_string())
    return 0


def _run_info(config: Config) -> dict[str, str]:
    canonical = json.dumps(config_to_dict(config), sort_keys=True, default=str)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return {
        "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "git_commit": git.stdout.strip() or "unknown",
    }


def _train(config: Config) -> int:
    set_global_seed(config.project.seed)
    panel = storage.read_parquet(config.paths.processed_dir / PANEL_FILE)
    samples = splits.development_samples(panel, config.validation)
    info = _run_info(config)
    with MlflowFoldLogger(config.paths.mlruns_dir, EXPERIMENT_NAME, info) as log_fold:
        specs = walk_forward.model_specs(config)
        predictions = walk_forward.run_walk_forward(samples, specs, config, log_fold)
    storage.write_parquet(predictions, config.paths.processed_dir / PREDICTIONS_FILE)
    signal = SignalSettings(config.label.horizon_days, config.evaluation.n_quantiles)
    table = walk_forward.results_table(predictions, signal)
    reports = config.paths.reports_dir
    storage.write_text(table.to_csv(index=False), reports / f"{RESULTS_STEM}.csv")
    header = [
        "# Signal-level results on walk-forward folds",
        "",
        f"Generated by `vn-rank train` at commit `{info['git_commit'][:12]}`; config SHA-256"
        f" `{info['config_sha256'][:12]}`. Optuna configurations tried: {log_fold.n_trials}.",
    ]
    text = walk_forward.render_results_markdown(table, header, RESULTS_DIGITS)
    storage.write_text(text, reports / f"{RESULTS_STEM}.md")
    logger.info("Results:\n%s", text)
    evaluation = config.evaluation
    breaches = walk_forward.tripwire_breaches(
        table, evaluation.tripwire_mean_ic, evaluation.tripwire_fold_ic
    )
    if breaches:
        logger.error("Leakage tripwire breached; investigate before reporting: %s", breaches)
        return TRIPWIRE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
