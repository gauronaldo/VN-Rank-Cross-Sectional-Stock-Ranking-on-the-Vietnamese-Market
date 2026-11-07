"""Phase 7 commands (I/O layer): the one-time lockbox evaluation and the daily update.

``lockbox`` (``vn-rank lockbox --confirm-lockbox``) runs exactly once:

1. refuses without the flag, if ``reports/lockbox_run.lock`` exists, or on a dirty git
   working tree (so the recorded commit identifies the exact code);
2. writes the lock file **before** computing anything, so a crash still counts as the
   one run;
3. builds the panel from all prices, trains every model on all development samples
   (purged and embargoed against ``lockbox_start``, tuned exactly as in walk-forward),
   scores every lockbox date with a matured label, and reports Rank IC (criterion C2)
   and a backtest net of costs;
4. freezes the LightGBM boosters for ``update``.

``update`` scores the newest session with the frozen boosters, writes the buy list, and
tracks realized Rank IC. With ``--fetch`` it first downloads a complete fresh snapshot
into a dated directory: adjusted prices change with corporate actions, so appending new
sessions to an old snapshot would mix two adjustment bases.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import subprocess
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from vn_rank import pipeline
from vn_rank.backtest import runner
from vn_rank.config import Config
from vn_rank.data import storage
from vn_rank.evaluation.ic import SignalSettings
from vn_rank.features import builder, labels
from vn_rank.models.base import feature_columns
from vn_rank.models.gbm import LightGBMModel
from vn_rank.monitoring import update as monitor
from vn_rank.utils.markdown import markdown_table
from vn_rank.utils.seeds import set_global_seed
from vn_rank.validation import splits, walk_forward

LOCK_FILE = "lockbox_run.lock"
LOCKBOX_STEM = "lockbox_results"
LOCKBOX_FIGURE = "lockbox_equity.png"
FULL_PANEL_FILE = "panel_full.parquet"
LOCKBOX_PREDICTIONS_FILE = "lockbox_predictions.parquet"
MODEL_DIR = "final_model"
FEATURES_FILE = "features.json"
LIVE_DIR_NAME = "live"  # sibling of paths.processed_dir, i.e. data/live
SCORES_LOG = "scores_log.parquet"
REALIZED_FIGURE = "realized_ic.png"
ROLLING_WINDOW = 20
logger = logging.getLogger(__name__)


class LockboxRefusedError(RuntimeError):
    """Raised when a lockbox precondition fails; nothing has been computed."""


def live_dir(config: Config) -> Path:
    """Folder for the score log, buy lists, and fresh snapshots (gitignored, under data/)."""
    return config.paths.processed_dir.parent / LIVE_DIR_NAME


def check_preconditions(lock_path: Path, confirmed: bool) -> None:
    """Refuse unless confirmed, never run before, and the git tree is clean.

    Raises:
        LockboxRefusedError: If any precondition fails.
    """
    if not confirmed:
        raise LockboxRefusedError("the lockbox runs once; pass --confirm-lockbox to run it")
    if lock_path.exists():
        raise LockboxRefusedError(f"{lock_path} exists: the lockbox has already been run")
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    if status.stdout.strip():
        raise LockboxRefusedError("commit or stash every change first (dirty git working tree)")


def lockbox_split(panel: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training samples (all development data, purged and embargoed) and lockbox samples.

    Development dates are those whose label exits before ``lockbox_start``; the lockbox
    samples are signal dates from ``lockbox_start`` on with a matured label and at least
    ``min_stocks_per_date`` eligible stocks.
    """
    rule, boundary = config.validation, pd.Timestamp(config.validation.lockbox_start)
    development = splits.development_samples(panel.loc[panel["exit_date"] < boundary], rule)
    train = development.loc[splits.purged_train_mask(development, boundary, rule.embargo_days)]
    per_date = panel.groupby("date")["symbol"].transform("size")
    keep = (
        (panel["date"] >= boundary)
        & (per_date >= rule.min_stocks_per_date)
        & panel["target"].notna()
    )
    return train.reset_index(drop=True), panel.loc[keep].reset_index(drop=True)


def final_models(
    train: pd.DataFrame, test: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, list[lgb.Booster], dict[str, dict[str, float]]]:
    """Tune and refit every model on ``train``; score ``test``; keep LightGBM boosters."""
    frames, boosters, chosen = [], [], {}
    for spec in walk_forward.model_specs(config):
        params, _ = walk_forward.tune_spec(spec, train, config)
        chosen[spec.name] = params
        for seed in spec.seeds:
            model = spec.factory(params, seed).fit(train)
            frames.append(
                test.assign(score=model.predict(test), model=spec.name, seed=seed, fold="lockbox")
            )
            if isinstance(model, LightGBMModel) and model.booster is not None:
                boosters.append(model.booster)
        logger.info("Lockbox model %s: %s", spec.name, params)
    predictions = pd.concat(frames, ignore_index=True)[walk_forward.PREDICTION_COLUMNS]
    return predictions, boosters, chosen


def _write_lock(path: Path, info: dict[str, str], last_label: pd.Timestamp) -> None:
    record = {
        "run_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **info,
        "last_matured_signal_date": last_label.date().isoformat(),
    }
    storage.write_json(record, path)


def _save_boosters(boosters: list[lgb.Booster], features: list[str], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, booster in enumerate(boosters):
        booster.save_model(str(directory / f"lightgbm_{i}.txt"))
    storage.write_json({"features": features}, directory / FEATURES_FILE)


def _load_boosters(directory: Path) -> tuple[list[lgb.Booster], list[str]]:
    files = sorted(directory.glob("lightgbm_*.txt"))
    features: list[str] = storage.read_json(directory / FEATURES_FILE)["features"]
    return [lgb.Booster(model_file=str(f)) for f in files], features


def lockbox(config: Config, confirmed: bool) -> int:
    """Run the lockbox evaluation once (see the module docstring)."""
    lock_path = config.paths.reports_dir / LOCK_FILE
    try:
        check_preconditions(lock_path, confirmed)
    except LockboxRefusedError as exc:
        logger.error("Lockbox refused: %s", exc)
        return 1
    info, prices = pipeline.run_info(config), pipeline.load_prices(config, None)
    h = config.label.entry_lag_days + config.label.horizon_days
    _write_lock(lock_path, info, prices.calendar[-1 - h])
    set_global_seed(config.project.seed)
    panel = builder.build_panel(prices.raw, prices.calendar, prices.exchange, config)
    processed = config.paths.processed_dir
    storage.write_parquet(panel, processed / FULL_PANEL_FILE)
    train, test = lockbox_split(panel, config)
    predictions, boosters, chosen = final_models(train, test, config)
    storage.write_parquet(predictions, processed / LOCKBOX_PREDICTIONS_FILE)
    features = feature_columns(builder.feature_groups(config.features))
    _save_boosters(boosters, features, processed / MODEL_DIR)
    _report_lockbox(config, predictions, prices, (info, chosen, len(train)))
    lightgbm = predictions[predictions["model"] == runner.PRIMARY_MODEL]
    log = lightgbm.groupby(["date", "symbol"])["score"].mean().reset_index()  # ensemble score
    storage.write_parquet(log, live_dir(config) / SCORES_LOG)
    return 0


def _report_lockbox(
    config: Config,
    predictions: pd.DataFrame,
    prices: pipeline.Prices,
    context: tuple[dict[str, str], dict[str, dict[str, float]], int],
) -> None:
    info, chosen, n_train = context
    settings = pipeline.backtest_settings(config)
    signal = SignalSettings(config.label.horizon_days, config.evaluation.n_quantiles)
    table = walk_forward.results_table(predictions, signal)
    table = table.loc[table["fold"] == walk_forward.ALL_FOLDS]
    entry_dates = labels.label_entry_dates(prices.calendar, config.label)
    market = pipeline.market_data(config, prices)
    report = runner.backtest_all(predictions, entry_dates, (market, prices.index_close), settings)
    dev = storage.read_parquet(config.paths.processed_dir / pipeline.PREDICTIONS_FILE)
    c1 = runner.ic_confidence(runner.seed_averaged_ic(dev), runner.PRIMARY_MODEL, settings)
    c1_low, c1_high = c1.loc[runner.PRIMARY_MODEL, ["ci_low", "ci_high"]].to_numpy(dtype=float)
    lockbox_ic: pd.Series[float] = report.ic["mean_ic"].astype(float)
    c2_ic = float(lockbox_ic[runner.PRIMARY_MODEL])
    verdict = (
        "Supported"
        if c1_low > 0 and c2_ic > 0
        else ("Not supported" if c1_low <= 0 else "Inconclusive")
    )
    dates = predictions["date"]
    lines = [
        "# Lockbox evaluation (run once)",
        "",
        f"{pipeline.provenance(info, 'lockbox --confirm-lockbox')}",
        f"Signal dates {dates.min().date()} to {dates.max().date()} ({dates.nunique()} dates);"
        f" models trained on {n_train} development samples with walk-forward tuning.",
        "",
        "## Pre-registered criteria (LightGBM)",
        "",
        f"- C1 (walk-forward): 95% CI of mean Rank IC = [{c1_low:.4f}, {c1_high:.4f}]"
        f" -> {'holds' if c1_low > 0 else 'fails'}",
        f"- C2 (lockbox): mean Rank IC = {c2_ic:.4f} -> {'holds' if c2_ic > 0 else 'fails'}",
        f"- Outcome: **{verdict}**",
        "",
        "## Signal level",
        "",
        *markdown_table(table.drop(columns="fold"), pipeline.DIGITS),
        "",
        "## Final model parameters",
        "",
        *(f"- {name}: {json.dumps(params)}" for name, params in chosen.items()),
        "",
        runner.render_backtest_markdown(report, ["## Lockbox backtest"], pipeline.DIGITS),
    ]
    reports = config.paths.reports_dir
    storage.write_text("\n".join(lines), reports / f"{LOCKBOX_STEM}.md")
    storage.write_text(report.summary.to_csv(), reports / f"{LOCKBOX_STEM}.csv")
    figure = runner.plot_equity(report.curves, "Lockbox (2025-10 onward), net of costs")
    storage.save_figure(figure, reports / LOCKBOX_FIGURE, config.project.figure_dpi)
    logger.info("Lockbox:\n%s", "\n".join(lines))


def _fetch_snapshot(config: Config) -> Config:
    """Download a complete fresh snapshot into a dated directory and point the config at it."""
    from vn_rank.cli import download_snapshot  # noqa: PLC0415 - needs vnstock

    raw_dir = live_dir(config) / f"raw_{dt.date.today().isoformat()}"
    fresh = dataclasses.replace(config, paths=dataclasses.replace(config.paths, raw_dir=raw_dir))
    download_snapshot(fresh)
    return fresh


def update(config: Config, fetch: bool) -> int:
    """Score the newest session with the frozen model and track realized Rank IC."""
    processed = config.paths.processed_dir
    if not (processed / MODEL_DIR / FEATURES_FILE).exists():
        logger.error("No frozen model: run `vn-rank lockbox --confirm-lockbox` first")
        return 1
    config = _fetch_snapshot(config) if fetch else config
    boosters, features = _load_boosters(processed / MODEL_DIR)
    prices = pipeline.load_prices(config, None)
    panel = builder.build_panel(prices.raw, prices.calendar, prices.exchange, config)
    newest = panel["date"].max()
    scored = monitor.score_date(panel, newest, boosters, features)
    buy_list = monitor.top_list(scored, config.backtest.top_quantile)
    storage.write_text(
        buy_list.to_csv(index=False), live_dir(config) / f"top_list_{newest.date()}.csv"
    )
    log_path = live_dir(config) / SCORES_LOG
    log = (
        storage.read_parquet(log_path)
        if log_path.exists()
        else pd.DataFrame(columns=monitor.LOG_COLUMNS)
    )
    log = monitor.append_scores(log, scored)
    storage.write_parquet(log, log_path)
    ic = monitor.realized_ic(log, panel)
    references = _reference_ics(config)
    figure = monitor.plot_realized_ic(ic, references, ROLLING_WINDOW)
    storage.save_figure(figure, live_dir(config) / REALIZED_FIGURE, config.project.figure_dpi)
    logger.info(
        "Buy list for the close after %s (%d stocks):\n%s",
        newest.date(),
        len(buy_list),
        buy_list.head(10),
    )
    logger.info(
        "Realized Rank IC: %d matured dates, mean %.4f",
        len(ic),
        ic.mean() if len(ic) else float("nan"),
    )
    return 0


def _reference_ics(config: Config) -> dict[str, float]:
    reports = config.paths.reports_dir
    signal = pd.read_csv(reports / f"{pipeline.RESULTS_STEM}.csv")
    row = signal[
        (signal["model"] == runner.PRIMARY_MODEL) & (signal["fold"] == walk_forward.ALL_FOLDS)
    ]
    references = {"walk-forward mean": float(row["mean_ic"].iloc[0])}
    predictions_path = config.paths.processed_dir / LOCKBOX_PREDICTIONS_FILE
    if predictions_path.exists():
        lockbox_preds = storage.read_parquet(predictions_path)
        daily = runner.seed_averaged_ic(
            lockbox_preds[lockbox_preds["model"] == runner.PRIMARY_MODEL]
        )
        references["lockbox mean"] = float(daily[runner.PRIMARY_MODEL].mean())
    return references
