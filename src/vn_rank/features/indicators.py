"""Hand-implemented technical indicators on wide frames (dates x symbols).

Every function takes frames indexed by trading session (VNINDEX calendar) with one column
per symbol and returns a frame of the same shape. All operations are column-wise
``shift`` (positive lags only), ``rolling``, or ``ewm``, so the value on day ``t`` uses
only data up to and including ``t`` (leakage note for every function below). Windows
require a full history (``min_periods`` = window); earlier values are NaN.

Wilder smoothing (RSI, ATR) is an exponential average with ``alpha = 1 / n`` seeded with
the first observation instead of an ``n``-day simple average; the two agree after a few
multiples of ``n`` sessions, well inside the 252-session history requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AMIHUD_SCALE_VND = 1e9  # Amihud illiquidity is reported per billion VND traded
RSI_SCALE = 100.0


def trailing_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return ``close[t] / close[t - window] - 1``; uses data up to ``t``."""
    return close / close.shift(window) - 1


def daily_return(close: pd.DataFrame) -> pd.DataFrame:
    """Return the one-session close-to-close return; uses data up to ``t``."""
    return trailing_return(close, 1)


def rolling_volatility(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sample standard deviation of daily returns over ``window`` sessions ending at ``t``."""
    return daily_return(close).rolling(window, min_periods=window).std()


def wilder_average(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder's smoothed average (EMA with ``alpha = 1 / window``); causal."""
    return values.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def exponential_average(values: pd.DataFrame, span: int) -> pd.DataFrame:
    """EMA with ``alpha = 2 / (span + 1)``, as used by MACD; causal."""
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Largest of ``high - low``, ``|high - prev close|``, ``|low - prev close|``.

    On a symbol's first session there is no previous close and ``high - low`` is used.
    """
    previous = close.shift(1)
    ranges = np.fmax(high - low, np.fmax((high - previous).abs(), (low - previous).abs()))
    return pd.DataFrame(ranges, index=close.index, columns=close.columns)


def normalized_atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int
) -> pd.DataFrame:
    """ATR(window) divided by ``close``; uses data up to ``t``."""
    return wilder_average(true_range(high, low, close), window) / close


def rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Relative strength index in [0, 100] with Wilder smoothing.

    NaN when the price did not move over the smoothing history (no gains and no losses).
    """
    change = close.diff()
    gain = wilder_average(change.clip(lower=0), window)
    loss = wilder_average(-change.clip(upper=0), window)
    strength = gain / loss
    return RSI_SCALE - RSI_SCALE / (1 + strength)


def macd_histogram(close: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """MACD histogram (MACD line minus its signal line) divided by ``close``; causal."""
    line = exponential_average(close, fast) - exponential_average(close, slow)
    return (line - exponential_average(line, signal)) / close


def bollinger_percent_b(close: pd.DataFrame, window: int, num_std: float) -> pd.DataFrame:
    """Position of ``close`` inside the Bollinger band: 0 at the lower band, 1 at the upper.

    The band uses the population standard deviation (``ddof = 0``) over ``window``
    sessions ending at ``t``. NaN when the band has zero width.
    """
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    width = (2 * num_std * std).where(std > 0)
    return (close - (mean - num_std * std)) / width


def log_median_traded_value(traded_value: pd.DataFrame, window: int) -> pd.DataFrame:
    """Natural log of the median traded value (VND) over ``window`` sessions; NaN if zero."""
    median = traded_value.rolling(window, min_periods=window).median()
    return pd.DataFrame(
        np.log(median.where(median > 0)), index=median.index, columns=median.columns
    )


def volume_ratio(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """Volume on ``t`` divided by the mean volume of the ``window`` sessions ending at ``t``."""
    mean = volume.rolling(window, min_periods=window).mean()
    return volume / mean.where(mean > 0)


def amihud_illiquidity(
    close: pd.DataFrame, traded_value: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Mean of ``|daily return| / traded value`` over ``window`` sessions, per billion VND.

    Sessions without a trade (traded value 0) are skipped inside the window; the value is
    NaN only when no session in the window traded.
    """
    ratio = daily_return(close).abs() / traded_value.where(traded_value > 0)
    return ratio.rolling(window, min_periods=1).mean() * AMIHUD_SCALE_VND


def close_to_rolling_high(close: pd.DataFrame, high: pd.DataFrame, window: int) -> pd.DataFrame:
    """``close`` divided by the highest ``high`` of the ``window`` sessions ending at ``t``."""
    return close / high.rolling(window, min_periods=window).max()


def rolling_market_moments(
    close: pd.DataFrame, market_close: pd.Series, window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OLS beta of daily returns on the market and the residual volatility.

    Over the ``window`` sessions ending at ``t``: ``beta = cov(r, m) / var(m)`` and the
    idiosyncratic volatility is the sample standard deviation of the regression residual,
    ``sqrt(var(r) - beta * cov(r, m))`` (all with ``ddof = 1``). Uses data up to ``t``.

    Args:
        close: Calendar-aligned closes (sessions x symbols).
        market_close: Market index closes on the same sessions.
        window: Number of sessions.

    Returns:
        ``(beta, idiosyncratic_volatility)``, each shaped like ``close``.
    """
    stock = daily_return(close)
    market = market_close.pct_change(fill_method=None).reindex(close.index)
    present = stock.notna()
    market_wide = pd.DataFrame(
        np.broadcast_to(market.to_numpy()[:, None], stock.shape),
        index=stock.index,
        columns=stock.columns,
    ).where(present)

    w = window
    mean_product = (stock * market_wide).rolling(w, min_periods=w).mean()
    means = stock.rolling(w, min_periods=w).mean() * market_wide.rolling(w, min_periods=w).mean()
    covariance = (mean_product - means) * w / (w - 1)
    beta = covariance / market_wide.rolling(w, min_periods=w).var()
    stock_variance = stock.rolling(w, min_periods=w).var()
    residual_variance = (stock_variance - beta * covariance).clip(lower=0.0)
    return beta, pd.DataFrame(np.sqrt(residual_variance), index=close.index, columns=close.columns)


def rolling_max_return(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Largest daily return over the ``window`` sessions ending at ``t``."""
    return daily_return(close).rolling(window, min_periods=window).max()


def rolling_skewness(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Sample skewness of daily returns over the ``window`` sessions ending at ``t``."""
    return daily_return(close).rolling(window, min_periods=window).skew()
