"""Point-in-time universe eligibility on wide frames (dates x symbols).

A stock is eligible on session ``t`` when all of the following hold:

1. it has at least ``min_history_days`` sessions of history up to and including ``t``
   (sessions since its first row, no-trade sessions included);
2. the median traded value over the ``traded_value_window`` sessions ending at ``t`` is at
   least ``min_traded_value_vnd`` (no-trade sessions count as zero);
3. it traded on ``t`` (volume > 0).

Leakage note: eligibility on ``t`` uses only information up to and including ``t``.
Traded value uses adjusted prices, so it is only approximate in earlier years.
"""

from __future__ import annotations

import pandas as pd

from vn_rank.config import UniverseConfig

PRICE_UNIT_VND = 1000.0  # vnstock prices are quoted in thousand VND


class UndecidedParameterError(ValueError):
    """Raised when a configuration value needed here is still ``null``."""


def traded_value_vnd(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Return ``close x 1000 x volume`` in VND (approximate: prices are adjusted)."""
    return close * PRICE_UNIT_VND * volume


def history_length(close: pd.DataFrame) -> pd.DataFrame:
    """Number of sessions with a price up to and including each session, per symbol."""
    return close.notna().cumsum()


def median_traded_value(traded_value: pd.DataFrame, window: int) -> pd.DataFrame:
    """Median traded value over the ``window`` sessions ending at each session."""
    return traded_value.rolling(window, min_periods=window).median()


def eligibility(close: pd.DataFrame, volume: pd.DataFrame, rule: UniverseConfig) -> pd.DataFrame:
    """Boolean frame: ``True`` where the stock is in the universe on that session.

    Args:
        close: Calendar-aligned closes (thousand VND); NaN before a symbol's first row.
        volume: Calendar-aligned volumes; 0 on no-trade sessions.
        rule: History, window, and threshold settings.

    Returns:
        A boolean frame with the shape of ``close``.

    Raises:
        UndecidedParameterError: If ``min_traded_value_vnd`` is still unset.
    """
    if rule.min_traded_value_vnd is None:
        raise UndecidedParameterError("universe.min_traded_value_vnd must be set")
    liquidity = median_traded_value(traded_value_vnd(close, volume), rule.traded_value_window)
    eligible = (
        (history_length(close) >= rule.min_history_days)
        & (liquidity >= rule.min_traded_value_vnd)
        & (volume > 0)
    )
    return eligible.fillna(False).astype(bool)
