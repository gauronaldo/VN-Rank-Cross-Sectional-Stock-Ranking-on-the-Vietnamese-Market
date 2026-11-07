"""Portfolio performance: annualized return, volatility, Sharpe, drawdown, turnover.

Conventions (protocol 8), on daily net returns:

- annualized return = mean daily return x ``trading_days`` (arithmetic, 252);
- annualized volatility = standard deviation of daily returns (``ddof = 1``) x sqrt(252);
- Sharpe = annualized return / annualized volatility, risk-free rate 0;
- maximum drawdown = the largest peak-to-trough fall of the compounded equity curve;
- turnover = mean one-way turnover per rebalance x ``trading_days / h`` (per year).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(daily: pd.Series, trading_days: int) -> float:
    """Mean daily return times the number of trading days per year."""
    return float(daily.mean() * trading_days)


def annualized_volatility(daily: pd.Series, trading_days: int) -> float:
    """Sample standard deviation of daily returns times sqrt(trading days)."""
    return float(daily.std() * np.sqrt(trading_days))


def sharpe_ratio(daily: pd.Series, trading_days: int) -> float:
    """Annualized return over annualized volatility (risk-free rate 0); NaN if flat."""
    volatility = annualized_volatility(daily, trading_days)
    return annualized_return(daily, trading_days) / volatility if volatility > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    """Largest fall from a running peak of the compounded equity curve (a negative number)."""
    equity = (1.0 + daily).cumprod()
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity.to_numpy()]))[1:]
    return float((equity.to_numpy() / peak - 1.0).min())


def total_return(daily: pd.Series) -> float:
    """Compounded return over the whole series."""
    return float(np.prod(1.0 + daily.to_numpy(dtype=float)) - 1.0)


def performance_summary(
    daily: pd.Series, turnover_per_rebalance: pd.Series, trading_days: int, horizon: int
) -> dict[str, float]:
    """All portfolio metrics of one daily return series.

    Keys: ``ann_return``, ``ann_volatility``, ``sharpe``, ``max_drawdown``,
    ``total_return``, and ``ann_turnover`` (NaN when no turnover is given).
    """
    turnover = float(turnover_per_rebalance.mean() * trading_days / horizon)
    return {
        "ann_return": annualized_return(daily, trading_days),
        "ann_volatility": annualized_volatility(daily, trading_days),
        "sharpe": sharpe_ratio(daily, trading_days),
        "max_drawdown": max_drawdown(daily),
        "total_return": total_return(daily),
        "ann_turnover": turnover if len(turnover_per_rebalance) else float("nan"),
    }


def yearly_breakdown(daily: pd.Series, trading_days: int) -> pd.DataFrame:
    """Annualized return, volatility, Sharpe, and compounded return per calendar year."""
    years = pd.DatetimeIndex(daily.index).year
    rows = {
        int(year): {
            "ann_return": annualized_return(part, trading_days),
            "ann_volatility": annualized_volatility(part, trading_days),
            "sharpe": sharpe_ratio(part, trading_days),
            "total_return": total_return(part),
            "days": float(len(part)),
        }
        for year, part in daily.groupby(years)
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")
