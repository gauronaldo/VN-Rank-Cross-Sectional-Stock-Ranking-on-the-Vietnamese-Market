"""Forward-return label on wide frames (dates x symbols).

Signal at the close of session ``t``; entry at the close of ``t + entry_lag`` (``t + 1``);
exit at the close of ``t + entry_lag + h``::

    forward_return[t] = close[t + 1 + h] / close[t + 1] - 1

Leakage note: this is the only module that looks forward. Its outputs must never be used
as features, and training windows must be purged on :func:`label_exit_dates`.
"""

from __future__ import annotations

import pandas as pd

from vn_rank.config import LabelConfig
from vn_rank.features.transforms import cross_sectional_rank, cross_sectional_zscore


def forward_return(close: pd.DataFrame, alive: pd.DataFrame, rule: LabelConfig) -> pd.DataFrame:
    """Return from the entry close to the exit close, in market sessions.

    Args:
        close: Calendar-aligned closes; no-trade sessions carry the previous close.
        alive: ``True`` on sessions up to a symbol's last real row; a label whose exit
            session is after that row (for example a delisting) is NaN.
        rule: Horizon and entry lag.
    """
    entry, exit_ = rule.entry_lag_days, rule.entry_lag_days + rule.horizon_days
    exit_close = close.shift(-exit_).where(alive.shift(-exit_, fill_value=False))
    return exit_close / close.shift(-entry) - 1


def label_exit_dates(sessions: pd.DatetimeIndex, rule: LabelConfig) -> pd.Series:
    """Exit session ``t + entry_lag + h`` per signal session (NaT near the end)."""
    exit_ = rule.entry_lag_days + rule.horizon_days
    return pd.Series(sessions, index=sessions).shift(-exit_)


def label_entry_dates(sessions: pd.DatetimeIndex, rule: LabelConfig) -> pd.Series:
    """Entry session ``t + entry_lag`` per signal session (NaT near the end)."""
    return pd.Series(sessions, index=sessions).shift(-rule.entry_lag_days)


def training_target(returns: pd.DataFrame, mask: pd.DataFrame, rule: LabelConfig) -> pd.DataFrame:
    """Cross-sectional target per date among universe members with a known return.

    ``rank`` scales to [0, 1]; ``zscore`` standardizes within the date.
    """
    if rule.target == "rank":
        return cross_sectional_rank(returns, mask)
    return cross_sectional_zscore(returns, mask)
