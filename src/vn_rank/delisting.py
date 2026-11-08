"""Exploratory sensitivity check: labels lost when a stock stops trading during the hold.

A label whose exit session is after the symbol's last row is missing (protocol Section 3),
so a stock delisted inside the 5-session holding period drops out of the evaluation and a
model is never penalized for ranking it high. This check finds those universe rows on the
walk-forward test dates and recomputes the mean Rank IC with them put back, under two
assumed returns:

- ``last close``: the return from the entry close to the symbol's last available close;
- ``total loss``: -100%, the most adverse return possible.

The rules (momentum, reversal, low volatility) score a stock from one feature rank, so
their scores for these rows are known exactly. The fold models of the learned models were
not saved, so they get the worst case: the lost row receives the highest score on its date.
Walk-forward only; the lockbox is spent.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from vn_rank import exploratory, pipeline
from vn_rank.backtest import runner
from vn_rank.config import Config
from vn_rank.data import storage
from vn_rank.features import builder
from vn_rank.models.base import FEATURE_PREFIX
from vn_rank.utils import report_paths
from vn_rank.utils.markdown import markdown_table

REPORT_STEM = "delisting_sensitivity"
LOW_VOLATILITY = "lowvol_vol60"
TOTAL_LOSS = -1.0
logger = logging.getLogger(__name__)


def lost_rows(panel: pd.DataFrame, test_dates: pd.Index) -> pd.DataFrame:
    """Universe rows on walk-forward test dates whose label is missing."""
    return panel.loc[panel["target"].isna() & panel["date"].isin(test_dates)]


def return_to_last_close(
    close: pd.DataFrame, alive: pd.DataFrame, rows: pd.DataFrame, entry_lag: int
) -> pd.Series:
    """Return from the entry close to the symbol's last real close, per lost row.

    The entry close is the close ``entry_lag`` sessions after the signal; when the symbol
    stopped trading before the entry, its last close is used, so the return is 0.
    """
    real = close.where(alive).ffill()
    last_close, entry = real.iloc[-1], real.shift(-entry_lag)
    values = [
        last_close[symbol] / entry.at[date, symbol] - 1
        for date, symbol in zip(rows["date"], rows["symbol"], strict=True)
    ]
    return pd.Series(values, index=rows.index, dtype=float)


def rule_scores(config: Config) -> dict[str, tuple[str, float]]:
    """Feature rank column and sign that define each rule's score."""
    models = config.models
    return {
        "momentum": (f"{FEATURE_PREFIX}ret_{models.momentum_window}", 1.0),
        "reversal": (f"{FEATURE_PREFIX}ret_{models.reversal_window}", -1.0),
        LOW_VOLATILITY: (exploratory.LOW_VOLATILITY_RULES[LOW_VOLATILITY], -1.0),
    }


def with_lost_rows(
    predictions: pd.DataFrame,
    lost: pd.DataFrame,
    returns: pd.Series,
    rules: dict[str, tuple[str, float]],
) -> pd.DataFrame:
    """Predictions plus the lost rows, scored by the rule or by the worst case."""
    extra = []
    for key, part in predictions.groupby(["model", "seed"], sort=False):
        model, seed = cast(tuple[str, int], key)
        rows = lost.loc[lost["date"].isin(part["date"].unique())]
        if model in rules:
            column, sign = rules[model]
            score = sign * rows[column]
        else:
            score = rows["date"].map(part.groupby("date")["score"].max()) + 1.0
        fold = rows["date"].map(part.groupby("date")["fold"].first())
        extra.append(
            rows[["date", "symbol"]].assign(
                forward_return=returns.loc[rows.index],
                score=score,
                model=model,
                fold=fold,
                seed=seed,
            )
        )
    return pd.concat([predictions, *extra], ignore_index=True)[predictions.columns]


def sensitivity_table(scenarios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean Rank IC per model under each scenario, and the change from the first one."""
    means = pd.DataFrame(
        {name: runner.seed_averaged_ic(frame).mean() for name, frame in scenarios.items()}
    )
    reported = means.columns[0]
    for name in means.columns[1:]:
        means[f"change, {name}"] = means[name] - means[reported]
    return means.rename_axis("model").reset_index()


def lost_summary(lost: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """One line per symbol: affected test dates and the return to its last close."""
    frame = lost.assign(return_to_last_close=returns)
    return (
        frame.groupby(["symbol", "exchange"])
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            rows=("date", "size"),
            worst_return=("return_to_last_close", "min"),
        )
        .reset_index()
        .assign(
            first_date=lambda f: f["first_date"].dt.date, last_date=lambda f: f["last_date"].dt.date
        )
    )


def delisting(config: Config) -> int:
    """Run the check and write ``reports/6_exploratory/delisting_sensitivity.md``."""
    processed = config.paths.processed_dir
    panel = storage.read_parquet(processed / pipeline.PANEL_FILE)
    original = storage.read_parquet(processed / pipeline.PREDICTIONS_FILE)
    rules_frame = exploratory.low_volatility_predictions(original, panel)
    predictions = pd.concat(
        [original, rules_frame[rules_frame["model"] == LOW_VOLATILITY]], ignore_index=True
    )
    lost = lost_rows(panel, pd.Index(original["date"].unique()))
    prices = pipeline.load_prices(config, pd.Timestamp(config.validation.lockbox_start))
    frames = builder.align_to_calendar(builder.repair_ohlc(prices.raw), prices.calendar)
    last = return_to_last_close(frames.close, frames.alive, lost, config.label.entry_lag_days)
    rules = rule_scores(config)
    scenarios = {
        "reported": predictions,
        "last close": with_lost_rows(predictions, lost, last, rules),
        "total loss": with_lost_rows(
            predictions, lost, pd.Series(TOTAL_LOSS, index=lost.index), rules
        ),
    }
    table = sensitivity_table(scenarios)
    n_rows, n_dates = len(original.loc[original["model"] == "momentum"]), original["date"].nunique()
    lines = [
        "# Exploratory: labels lost to delisting during the holding period (walk-forward)",
        "",
        f"{pipeline.provenance(pipeline.run_info(config), 'delisting')}",
        "",
        f"Universe rows on the {n_dates} test dates without a label because the stock stopped "
        f"trading inside the hold: **{len(lost)}**, against {n_rows:,} scored rows "
        f"({len(lost) / n_rows:.4%}), on {lost['date'].nunique()} dates.",
        "",
        *markdown_table(lost_summary(lost, last), pipeline.DIGITS),
        "",
        "## Mean Rank IC with the lost rows put back",
        "",
        "Rules use their real scores; learned models get the worst case (the lost row gets the",
        "highest score on its date). `total loss` assumes a -100% return.",
        "",
        *markdown_table(table, pipeline.DIGITS),
        "",
    ]
    folder = report_paths.section(config.paths.reports_dir, report_paths.EXPLORATORY)
    storage.write_text("\n".join(lines), folder / f"{REPORT_STEM}.md")
    logger.info("Delisting sensitivity:\n%s", "\n".join(lines))
    return 0
