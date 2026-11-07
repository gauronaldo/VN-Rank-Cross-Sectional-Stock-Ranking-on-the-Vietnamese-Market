"""Rebalance loop: long-only, equal-weight top quantile, close-to-close accounting.

Timeline per rebalance: scores at the close of signal day ``t``; trades at the close of
``t + 1`` (the label's entry session); the position is held until the next trade, ``h``
sessions later. Each session:

1. the drifted weights earn that session's close-to-close returns (cash earns 0);
2. on a trade day, the constrained target replaces them and costs are charged on the
   traded value, as a fraction of portfolio value.

On the last session the whole portfolio is sold, and that cost is charged too. The
liquidity cap uses the 20-day average traded value up to the previous session and the
current portfolio value in VND.

Leakage note: a target built from scores at ``t`` is only applied at the close of
``t + 1``, and returns from ``t + 1`` onward are earned after the trade.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from vn_rank.backtest.constraints import TradeDay, constrain
from vn_rank.backtest.costs import CostModel, trade_cost, turnover


@dataclasses.dataclass(frozen=True)
class Market:
    """Calendar-aligned market data (sessions x symbols).

    ``adv_vnd`` on a session must be known before that session's close (the caller
    passes the 20-day average traded value up to the previous session).
    """

    returns: pd.DataFrame
    adv_vnd: pd.DataFrame
    bands: pd.Series


@dataclasses.dataclass(frozen=True)
class EngineSettings:
    """Costs, liquidity cap, portfolio size, and the limit-detection margin."""

    costs: CostModel
    max_fraction_of_adv: float
    portfolio_value_vnd: float
    limit_hit_margin: float


@dataclasses.dataclass(frozen=True)
class BacktestResult:
    """Daily returns and per-trade statistics of one run.

    ``daily`` has ``gross_return``, ``cost``, ``net_return``, and ``invested``; ``trades``
    has ``turnover``, ``cost``, ``n_holdings``, and ``invested``.
    """

    daily: pd.DataFrame
    trades: pd.DataFrame


TargetRule = Callable[[pd.Series, pd.Index], pd.Series]
"""Maps (scores per symbol, symbols currently held) to target weights."""


def top_quantile_weights(scores: pd.Series, top_quantile: float) -> pd.Series:
    """Equal weights on the ``ceil(top_quantile * n)`` highest scores (NaN scores skipped)."""
    valid = scores.dropna()
    n = math.ceil(top_quantile * len(valid))
    chosen = valid.sort_values(ascending=False, kind="stable").index[:n]
    return pd.Series(1.0 / n, index=chosen) if n else pd.Series(dtype=float)


def equal_weights(symbols: pd.Index) -> pd.Series:
    """Equal weights on every symbol (the same-universe benchmark)."""
    return pd.Series(1.0 / len(symbols), index=symbols)


def top_quantile_rule(top_quantile: float) -> TargetRule:
    """Primary rule: the top quantile by score, ignoring current holdings."""
    return lambda scores, held: top_quantile_weights(scores, top_quantile)


def equal_weight_rule() -> TargetRule:
    """Benchmark rule: every scored stock at equal weight."""
    return lambda scores, held: equal_weights(scores.dropna().index)


def band_rule(top_quantile: float, band: float) -> TargetRule:
    """Turnover-control rule (AGENT.md 3.7, secondary variant).

    A held stock is kept while its score ranks within the top ``band`` of the universe;
    the remaining slots, up to ``ceil(top_quantile * n)``, are filled with the
    best-ranked stocks of the top quantile that are not already kept. Equal weights.
    """

    def rule(scores: pd.Series, held: pd.Index) -> pd.Series:
        order = scores.dropna().sort_values(ascending=False, kind="stable").index
        n, n_band = math.ceil(top_quantile * len(order)), math.ceil(band * len(order))
        keep = [s for s in order[:n_band] if s in held][:n]
        fill = [s for s in order[:n] if s not in keep][: n - len(keep)]
        chosen = keep + fill
        return pd.Series(1.0 / len(chosen), index=chosen) if chosen else pd.Series(dtype=float)

    return rule


class _Book:
    """Mutable portfolio state of one backtest run."""

    def __init__(self, n_symbols: int) -> None:
        self.weights = np.zeros(n_symbols)
        self.equity = 1.0

    def earn(self, returns: np.ndarray) -> float:
        gross = float(self.weights @ returns)
        if 1.0 + gross > 0:
            self.weights = self.weights * (1.0 + returns) / (1.0 + gross)
        self.equity *= 1.0 + gross
        return gross

    def trade(self, target: np.ndarray, costs: CostModel) -> tuple[float, float]:
        cost = trade_cost(self.weights, target, costs)
        moved = turnover(self.weights, target)
        self.weights = target
        self.equity *= 1.0 - cost
        return cost, moved


def _target_vector(target: pd.Series, symbols: pd.Index) -> np.ndarray:
    return target.reindex(symbols).fillna(0.0).to_numpy()


def _trade_day(
    market: Market, day: pd.Timestamp, equity: float, settings: EngineSettings
) -> TradeDay:
    adv = market.adv_vnd.loc[day].fillna(0.0).to_numpy()
    nav = settings.portfolio_value_vnd * equity
    return TradeDay(
        returns=market.returns.loc[day].to_numpy(),
        bands=market.bands.reindex(market.returns.columns).to_numpy(),
        cap_weights=settings.max_fraction_of_adv * adv / nav,
        limit_hit_margin=settings.limit_hit_margin,
    )


def run_backtest(
    schedule: dict[pd.Timestamp, pd.Series],
    rule: TargetRule,
    end: pd.Timestamp,
    market_settings: tuple[Market, EngineSettings],
) -> BacktestResult:
    """Run the rebalance loop from the first trade day through ``end``.

    Args:
        schedule: Scores per trade day (symbol -> score), known at the previous close.
        rule: Turns the scores and the current holdings into target weights.
        end: Last session; the portfolio is sold at its close.
        market_settings: Market data, and costs, cap, portfolio size, and limit margin.

    Returns:
        Daily returns and per-trade statistics.
    """
    market, settings = market_settings
    symbols = market.returns.columns
    sessions = market.returns.index
    sessions = sessions[(sessions >= min(schedule)) & (sessions <= end)]
    book, daily, trades = _Book(len(symbols)), [], []
    for day in sessions:
        start_equity = book.equity
        gross = book.earn(np.nan_to_num(market.returns.loc[day].to_numpy(), nan=0.0))
        invested_after_returns = float(book.weights.sum())
        if day in schedule:
            wanted = _target_vector(rule(schedule[day], symbols[book.weights > 0]), symbols)
            feasible = constrain(
                book.weights, wanted, _trade_day(market, day, book.equity, settings)
            )
            cost, moved = book.trade(feasible, settings.costs)
            held = int((feasible > 0).sum())
            trades.append({"date": day, "turnover": moved, "cost": cost, "n_holdings": held,
                           "invested": float(feasible.sum())})  # fmt: skip
        if day == sessions[-1]:
            book.trade(np.zeros(len(symbols)), settings.costs)
        net = book.equity / start_equity - 1.0
        daily.append({"date": day, "gross_return": gross, "cost": (1.0 + gross) - (1.0 + net),
                      "net_return": net, "invested": invested_after_returns})  # fmt: skip
    return BacktestResult(
        pd.DataFrame(daily).set_index("date"), pd.DataFrame(trades).set_index("date")
    )
