"""Backtest runner: market data, rebalance schedules, strategies, benchmarks, and tables.

Every model is traded the same way: on every ``h``-th signal date of the walk-forward
test blocks, buy the top quantile of its scores at the next close, equal weight, net of
costs and constraints. A stochastic model's portfolio is the average of its per-seed
portfolios (one fifth of capital in each); the seed standard deviation of Sharpe is also
reported. Benchmarks use the same schedule: the equal-weight portfolio of every scored
(eligible) stock, with the same costs and constraints, and VNINDEX (no costs).

Survivorship note: the scored stocks are universe members with a known label, so the
26 stocks whose label is missing because they delisted inside the window are never
selected; a delisted holding is carried at its last price until the next trade.
"""

from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from vn_rank.backtest.engine import (
    BacktestResult,
    EngineSettings,
    Market,
    TargetRule,
    band_rule,
    equal_weight_rule,
    run_backtest,
    top_quantile_rule,
)
from vn_rank.data import universe
from vn_rank.evaluation import bootstrap, performance
from vn_rank.evaluation.ic import SCORE_COLUMN, daily_rank_ic, rebalance_dates
from vn_rank.features.builder import MarketFrames
from vn_rank.utils.markdown import markdown_table

Schedule = dict[pd.Timestamp, pd.Series]
EQUAL_WEIGHT = "equal_weight"
VNINDEX = "vnindex"
PRIMARY_MODEL = "lightgbm"  # pre-registered primary model (protocol 7)
BAND_SUFFIX = "_band"  # secondary turnover-control variant of the primary model


@dataclasses.dataclass(frozen=True)
class BacktestSettings:
    """Engine settings plus portfolio, calendar, and bootstrap conventions.

    ``turnover_band`` is the hold band of the secondary turnover-control variant
    (``None`` skips it).
    """

    engine: EngineSettings
    top_quantile: float
    horizon: int
    trading_days: int
    bootstrap: bootstrap.BootstrapSettings
    turnover_band: float | None = None


def build_market(
    frames: MarketFrames, exchange: pd.Series, price_limits: dict[str, float], adv_window: int
) -> Market:
    """Daily returns, prior-session average traded value, and price band per symbol."""
    returns = frames.close.pct_change(fill_method=None)
    value = universe.traded_value_vnd(frames.close, frames.volume)
    adv = value.rolling(adv_window, min_periods=adv_window).mean().shift(1)
    return Market(returns, adv, exchange.reindex(frames.close.columns).map(price_limits))


def score_schedule(predictions: pd.DataFrame, entry_dates: pd.Series, horizon: int) -> Schedule:
    """Scores (symbol -> score) per trade day, every ``horizon`` signal dates."""
    signals = rebalance_dates(predictions["date"], horizon)
    chosen = predictions.loc[predictions["date"].isin(signals)]
    return {
        pd.Timestamp(entry_dates[date]): group.set_index("symbol")[SCORE_COLUMN]
        for date, group in chosen.groupby("date", sort=True)
    }


def run_model(
    predictions: pd.DataFrame, entry_dates: pd.Series, market: Market, settings: BacktestSettings
) -> list[BacktestResult]:
    """One top-quantile backtest per seed of one model's predictions."""
    rule = top_quantile_rule(settings.top_quantile)
    return _run_seeds(predictions, entry_dates, (market, settings), rule)


def run_band_model(
    predictions: pd.DataFrame, entry_dates: pd.Series, market: Market, settings: BacktestSettings
) -> list[BacktestResult]:
    """One turnover-band backtest per seed (requires ``settings.turnover_band``)."""
    if settings.turnover_band is None:
        raise ValueError("turnover_band is not set")
    rule = band_rule(settings.top_quantile, settings.turnover_band)
    return _run_seeds(predictions, entry_dates, (market, settings), rule)


def _run_seeds(
    predictions: pd.DataFrame,
    entry_dates: pd.Series,
    market_settings: tuple[Market, BacktestSettings],
    rule: TargetRule,
) -> list[BacktestResult]:
    market, settings = market_settings
    results = []
    for _, part in predictions.groupby("seed", sort=True):
        schedule = score_schedule(part, entry_dates, settings.horizon)
        end = _last_exit(schedule, market, settings)
        results.append(run_backtest(schedule, rule, end, (market, settings.engine)))
    return results


def run_equal_weight(
    predictions: pd.DataFrame, entry_dates: pd.Series, market: Market, settings: BacktestSettings
) -> BacktestResult:
    """Equal-weight portfolio of every scored stock on the same schedule."""
    schedule = score_schedule(predictions, entry_dates, settings.horizon)
    end = _last_exit(schedule, market, settings)
    return run_backtest(schedule, equal_weight_rule(), end, (market, settings.engine))


def _last_exit(schedule: Schedule, market: Market, settings: BacktestSettings) -> pd.Timestamp:
    """Session ``h`` after the last trade day: the end of the last holding period."""
    sessions = pd.DatetimeIndex(market.returns.index)
    position = int(sessions.searchsorted(max(schedule))) + settings.horizon
    return sessions[min(position, len(sessions) - 1)]


def average_portfolio(results: list[BacktestResult]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Seed-averaged daily net and gross returns, and mean turnover per trade."""
    net = pd.concat([r.daily["net_return"] for r in results], axis=1).mean(axis=1)
    gross = pd.concat([r.daily["gross_return"] for r in results], axis=1).mean(axis=1)
    moved = pd.concat([r.trades["turnover"] for r in results], axis=1).mean(axis=1)
    return net, gross, moved


def portfolio_row(
    results: list[BacktestResult], benchmark: BacktestResult, settings: BacktestSettings
) -> dict[str, float]:
    """Metrics of one (seed-averaged) portfolio, its excess over the benchmark, and CIs.

    The excess return is net of both portfolios' costs; ``gross_excess_ann_return``
    compares them before costs.
    """
    net, gross, moved = average_portfolio(results)
    days, boot = settings.trading_days, settings.bootstrap
    row = performance.performance_summary(net, moved, days, settings.horizon)
    excess = (net - benchmark.daily["net_return"].reindex(net.index)).fillna(0.0)
    gross_excess = (gross - benchmark.daily["gross_return"].reindex(net.index)).fillna(0.0)
    sharpe_low, sharpe_high = bootstrap.bootstrap_ci(
        net.to_numpy(), bootstrap.sharpe_statistic(days), boot
    )
    excess_low, excess_high = bootstrap.bootstrap_ci(
        excess.to_numpy(), bootstrap.mean_statistic, boot
    )
    seed_sharpes = [performance.sharpe_ratio(r.daily["net_return"], days) for r in results]
    row |= {
        "sharpe_ci_low": sharpe_low,
        "sharpe_ci_high": sharpe_high,
        "sharpe_seed_std": float(np.std(seed_sharpes, ddof=1))
        if len(results) > 1
        else float("nan"),
        "gross_ann_return": performance.annualized_return(gross, days),
        "gross_sharpe": performance.sharpe_ratio(gross, days),
        "ann_cost": float((gross - net).mean() * days),
        "gross_excess_ann_return": float(gross_excess.mean() * days),
        "excess_ann_return": float(excess.mean() * days),
        "excess_ci_low": excess_low * days,
        "excess_ci_high": excess_high * days,
        "avg_holdings": float(np.mean([r.trades["n_holdings"].mean() for r in results])),
        "avg_invested": float(np.mean([r.trades["invested"].mean() for r in results])),
    }
    return row


def index_row(index_returns: pd.Series, settings: BacktestSettings) -> dict[str, float]:
    """Buy-and-hold VNINDEX metrics over the same sessions (no costs, no turnover)."""
    days = settings.trading_days
    row = performance.performance_summary(
        index_returns, pd.Series(dtype=float), days, settings.horizon
    )
    low, high = bootstrap.bootstrap_ci(
        index_returns.to_numpy(), bootstrap.sharpe_statistic(days), settings.bootstrap
    )
    return row | {"sharpe_ci_low": low, "sharpe_ci_high": high}


def seed_averaged_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Daily Rank IC per model (dates x models), averaged over seeds on each date."""
    per_seed = {
        (model, seed): daily_rank_ic(part)
        for (model, seed), part in predictions.groupby(["model", "seed"], sort=False)
    }
    frame = pd.DataFrame(per_seed)
    return frame.T.groupby(level=0, sort=False).mean().T


def ic_confidence(daily_ic: pd.DataFrame, primary: str, settings: BacktestSettings) -> pd.DataFrame:
    """Mean Rank IC and paired differences against the primary model, with bootstrap CIs.

    The difference is ``primary - model`` on each date. Criterion C1 uses the primary
    model's own interval.
    """
    rows = {}
    for model in daily_ic.columns:
        ic = daily_ic[model].dropna().to_numpy()
        low, high = bootstrap.bootstrap_ci(ic, bootstrap.mean_statistic, settings.bootstrap)
        diff = (daily_ic[primary] - daily_ic[model]).dropna().to_numpy()
        row = {"mean_ic": float(ic.mean()), "ci_low": low, "ci_high": high}
        if model != primary:
            d_low, d_high = bootstrap.bootstrap_ci(
                diff, bootstrap.mean_statistic, settings.bootstrap
            )
            row |= {
                "diff_vs_primary": float(diff.mean()),
                "diff_ci_low": d_low,
                "diff_ci_high": d_high,
            }
        rows[model] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")


@dataclasses.dataclass(frozen=True)
class BacktestReport:
    """Portfolio metrics per strategy, Rank IC intervals, yearly returns, and curves."""

    summary: pd.DataFrame
    ic: pd.DataFrame
    yearly: pd.DataFrame
    curves: dict[str, pd.Series]


def backtest_all(
    predictions: pd.DataFrame,
    entry_dates: pd.Series,
    market_and_index: tuple[Market, pd.Series],
    settings: BacktestSettings,
) -> BacktestReport:
    """Backtest every model, the equal-weight benchmark, and VNINDEX on one schedule.

    Args:
        predictions: Walk-forward test predictions of all models and seeds.
        entry_dates: Entry session per signal session.
        market_and_index: Market data and VNINDEX closes on the same calendar.
        settings: Engine, portfolio, and bootstrap settings.
    """
    market, index_close = market_and_index
    models = list(dict.fromkeys(predictions["model"]))
    primary = PRIMARY_MODEL if PRIMARY_MODEL in models else models[0]
    by_model = {m: predictions[predictions["model"] == m] for m in models}
    results = {m: run_model(p, entry_dates, market, settings) for m, p in by_model.items()}
    if settings.turnover_band is not None:
        band = run_band_model(by_model[primary], entry_dates, market, settings)
        results[primary + BAND_SUFFIX] = band
    first = by_model[models[0]]
    benchmark = run_equal_weight(
        first[first["seed"] == first["seed"].min()], entry_dates, market, settings
    )
    ew_net = benchmark.daily["net_return"]
    rows = {m: portfolio_row(r, benchmark, settings) for m, r in results.items()}
    rows[EQUAL_WEIGHT] = portfolio_row([benchmark], benchmark, settings)
    index_returns = index_close.pct_change(fill_method=None).reindex(ew_net.index).fillna(0.0)
    index_returns.iloc[0] = 0.0  # positions are opened at the first close
    rows[VNINDEX] = index_row(index_returns, settings)
    curves = {m: average_portfolio(r)[0] for m, r in results.items()}
    curves |= {EQUAL_WEIGHT: ew_net, VNINDEX: index_returns}
    yearly = pd.concat(
        {
            name: performance.yearly_breakdown(daily, settings.trading_days)
            for name, daily in curves.items()
        }
    )
    ic = ic_confidence(seed_averaged_ic(predictions), primary, settings)
    summary = pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")
    return BacktestReport(summary, ic, yearly, curves)


_SUMMARY_COLUMNS = {
    "ann_return": "ann. return", "ann_volatility": "ann. vol", "sharpe": "Sharpe",
    "sharpe_ci_low": "Sharpe CI low", "sharpe_ci_high": "Sharpe CI high",
    "max_drawdown": "max DD", "ann_turnover": "turnover / yr", "ann_cost": "cost / yr",
    "gross_sharpe": "gross Sharpe", "gross_excess_ann_return": "gross excess vs EW / yr",
    "excess_ann_return": "excess vs EW / yr",
    "excess_ci_low": "excess CI low", "excess_ci_high": "excess CI high",
    "avg_holdings": "holdings", "avg_invested": "invested",
}  # fmt: skip


def render_backtest_markdown(report: BacktestReport, header: list[str], digits: int) -> str:
    """Markdown report: portfolio metrics, Rank IC intervals, and returns per year."""
    summary = report.summary.reindex(columns=list(_SUMMARY_COLUMNS)).rename(
        columns=_SUMMARY_COLUMNS
    )
    yearly = report.yearly["total_return"].unstack(level=0)
    lines = [*header, "", "## Portfolio metrics (net of costs)", ""]
    lines += markdown_table(summary.reset_index(), digits)
    lines += ["", "## Mean Rank IC with 95% block-bootstrap intervals", ""]
    lines += markdown_table(report.ic.reset_index(), digits)
    lines += ["", "## Compounded net return per calendar year", ""]
    lines += markdown_table(yearly[list(report.curves)].reset_index(), digits)
    return "\n".join([*lines, ""])


def plot_equity(curves: dict[str, pd.Series], title: str) -> Figure:
    """Compounded equity curves (log scale) of daily return series."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for name, daily in curves.items():
        axis.plot((1.0 + daily).cumprod(), label=name, linewidth=1.2)
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_ylabel("growth of 1 (log scale)")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    return figure
