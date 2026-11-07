"""Exploratory analysis after the lockbox (I/O layer): low-volatility baselines.

**Exploratory, not part of the pre-registered protocol.** The lockbox has been run, so
nothing here may change the primary result. The question is whether LightGBM adds
anything over a one-line low-volatility rule, the best-known effect in its features:

- ``lowvol_vol60``: minus the rank of the 60-day return volatility, the textbook
  definition of the low-volatility effect (known before this project);
- ``lowvol_atr14``: minus the rank of ATR(14) / close, picked **after** seeing that it
  was the strongest single feature on the walk-forward test blocks, so it is optimistic.

Both are scored on exactly the rows LightGBM was scored on, traded with the same engine,
costs, constraints, and benchmark, and compared on Rank IC (paired with LightGBM), net
excess return over equal weight, and annualized alpha after adjusting for beta to the
equal-weight benchmark (a low-volatility portfolio has a low beta, so its raw excess
return mixes skill with market exposure).

The definitions are module constants rather than config keys, so that the config hash
recorded in ``reports/lockbox_run.lock`` stays verifiable.
"""

from __future__ import annotations

import logging

import pandas as pd

from vn_rank import final, pipeline
from vn_rank.backtest import runner
from vn_rank.config import Config
from vn_rank.data import storage
from vn_rank.evaluation import bootstrap
from vn_rank.features import labels
from vn_rank.utils.markdown import markdown_table

LOW_VOLATILITY_RULES = {"lowvol_vol60": "rank_vol_60", "lowvol_atr14": "rank_atr_14"}
COMPARED_MODELS = (runner.PRIMARY_MODEL, "elastic_net")
REPORT_STEM = "exploratory_low_volatility"
logger = logging.getLogger(__name__)


def low_volatility_predictions(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Compared models' predictions plus one low-volatility score per rule.

    The low-volatility rows are the rows the baselines were scored on (one seed), so every
    strategy ranks the same stocks on the same dates.
    """
    keys = predictions.loc[
        predictions["model"] == "momentum", ["date", "symbol", "forward_return", "fold"]
    ]
    rows = keys.merge(
        panel[["date", "symbol", *LOW_VOLATILITY_RULES.values()]], on=["date", "symbol"]
    )
    rules = [
        rows.assign(score=-rows[column], model=name, seed=0)[predictions.columns]
        for name, column in LOW_VOLATILITY_RULES.items()
    ]
    compared = predictions[predictions["model"].isin(COMPARED_MODELS)]
    return pd.concat([compared, *rules], ignore_index=True)


def alpha_table(report: runner.BacktestReport, settings: runner.BacktestSettings) -> pd.DataFrame:
    """Beta-adjusted alpha against the equal-weight benchmark for every strategy curve."""
    benchmark = report.curves[runner.EQUAL_WEIGHT]
    rows = {}
    for name, daily in report.curves.items():
        if name == runner.EQUAL_WEIGHT:
            continue
        aligned = benchmark.reindex(daily.index).fillna(0.0)
        rows[name] = bootstrap.alpha_beta_ci(
            daily.to_numpy(), aligned.to_numpy(), settings.bootstrap, settings.trading_days
        )
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


_PORTFOLIO_COLUMNS = {
    "ann_return": "ann. return", "ann_turnover": "turnover / yr", "ann_cost": "cost / yr",
    "gross_excess_ann_return": "gross excess vs EW / yr", "excess_ann_return": "excess vs EW / yr",
    "excess_ci_low": "excess CI low", "excess_ci_high": "excess CI high", "max_drawdown": "max DD",
}  # fmt: skip


def period_section(
    title: str, report: runner.BacktestReport, settings: runner.BacktestSettings
) -> list[str]:
    """Markdown lines for one period: Rank IC, portfolio metrics, and alpha after beta."""
    portfolio = report.summary.reindex(columns=list(_PORTFOLIO_COLUMNS)).rename(
        columns=_PORTFOLIO_COLUMNS
    )
    return [
        f"## {title}",
        "",
        "Rank IC with 95% block-bootstrap intervals; `diff_vs_primary` is LightGBM minus the row.",
        "",
        *markdown_table(report.ic.reset_index(), pipeline.DIGITS),
        "",
        "Portfolios (top decile, net of costs):",
        "",
        *markdown_table(portfolio.reset_index(), pipeline.DIGITS),
        "",
        "Annualized alpha after beta to the equal-weight benchmark (net of costs):",
        "",
        *markdown_table(alpha_table(report, settings).reset_index(), pipeline.DIGITS),
        "",
    ]


def _period_report(
    config: Config, predictions: pd.DataFrame, before: pd.Timestamp | None
) -> runner.BacktestReport:
    prices = pipeline.load_prices(config, before)
    entry = labels.label_entry_dates(prices.calendar, config.label)
    market = pipeline.market_data(config, prices)
    settings = pipeline.backtest_settings(config)
    return runner.backtest_all(predictions, entry, (market, prices.index_close), settings)


def exploratory(config: Config) -> int:
    """Compare the low-volatility rules with LightGBM on walk-forward and lockbox rows."""
    processed, settings = config.paths.processed_dir, pipeline.backtest_settings(config)
    lockbox = pd.Timestamp(config.validation.lockbox_start)
    periods = {
        "Walk-forward test blocks (2022-01 to 2025-09)": (
            pipeline.PREDICTIONS_FILE, pipeline.PANEL_FILE, lockbox
        ),
        "Lockbox (2025-10 to 2026-10)": (
            final.LOCKBOX_PREDICTIONS_FILE, final.FULL_PANEL_FILE, None
        ),
    }  # fmt: skip
    lines = [
        "# Exploratory: low-volatility baselines versus LightGBM",
        "",
        "**Exploratory analysis after the lockbox; it does not change any pre-registered result.**",
        "`lowvol_vol60` is the textbook low-volatility rule; `lowvol_atr14` was chosen after",
        "seeing the test blocks and is optimistic. "
        + pipeline.provenance(pipeline.run_info(config), "exploratory"),
        "",
    ]
    for title, (predictions_file, panel_file, before) in periods.items():
        predictions = low_volatility_predictions(
            storage.read_parquet(processed / predictions_file),
            storage.read_parquet(processed / panel_file),
        )
        lines += period_section(title, _period_report(config, predictions, before), settings)
    storage.write_text("\n".join(lines), config.paths.reports_dir / f"{REPORT_STEM}.md")
    logger.info("Exploratory:\n%s", "\n".join(lines))
    return 0
