"""Tests for the extension indicators: rolling beta, idiosyncratic volatility, MAX, skewness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vn_rank.features import indicators as ind

SESSIONS = pd.bdate_range("2024-01-01", periods=120)
WINDOW = 60


def prices(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Two stocks driven by a market factor with betas 0.5 and 1.5, plus noise."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, len(SESSIONS))
    stocks = {
        "A": 0.5 * market + rng.normal(0, 0.005, len(SESSIONS)),
        "B": 1.5 * market + rng.normal(0, 0.010, len(SESSIONS)),
    }
    close = pd.DataFrame({k: 10 * np.cumprod(1 + v) for k, v in stocks.items()}, index=SESSIONS)
    return close, pd.Series(1000 * np.cumprod(1 + market), index=SESSIONS)


def test_beta_and_ivol_match_an_ols_fit_on_the_window() -> None:
    close, market = prices()
    beta, ivol = ind.rolling_market_moments(close, market, WINDOW)
    r = close.pct_change().iloc[-WINDOW:]
    m = market.pct_change().iloc[-WINDOW:]
    for symbol in ("A", "B"):
        slope, intercept = np.polyfit(m, r[symbol], 1)
        residual = r[symbol] - (slope * m + intercept)
        assert beta[symbol].iloc[-1] == pytest.approx(slope, rel=1e-9)
        # sqrt(var(r) - beta * cov) equals the residual std with ddof = 1
        assert ivol[symbol].iloc[-1] == pytest.approx(residual.std(ddof=1), rel=1e-9)
    assert beta["B"].iloc[-1] > beta["A"].iloc[-1]


def test_market_moments_need_a_full_window_and_ignore_missing_stock_rows() -> None:
    close, market = prices()
    close.loc[SESSIONS[:30], "A"] = np.nan  # listed late
    beta, _ = ind.rolling_market_moments(close, market, WINDOW)
    assert beta["B"].iloc[:WINDOW].isna().all()
    assert beta["B"].iloc[WINDOW:].notna().all()
    assert beta["A"].iloc[: 30 + WINDOW].isna().all()
    assert np.isfinite(beta["A"].iloc[-1])


def test_max_and_skewness_use_only_the_trailing_window() -> None:
    close, _ = prices(1)
    returns = close.pct_change()
    max_20 = ind.rolling_max_return(close, 20)
    skew_60 = ind.rolling_skewness(close, 60)
    assert max_20.iloc[-1].to_numpy() == pytest.approx(returns.iloc[-20:].max().to_numpy())
    expected = stats.skew(returns.iloc[-60:], bias=False)
    assert skew_60.iloc[-1].to_numpy() == pytest.approx(expected)
    truncated = ind.rolling_max_return(close.iloc[:80], 20)
    assert truncated.iloc[-1].to_numpy() == pytest.approx(max_20.iloc[79].to_numpy())
