"""Tests for point-in-time universe eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank.config import UniverseConfig
from vn_rank.data import universe

RULE = UniverseConfig(min_history_days=3, traded_value_window=2, min_traded_value_vnd=1e6)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Prices in thousand VND: close 10 x volume 100 = 1,000,000 VND per session.
    close = pd.DataFrame({"A": [10.0] * 5, "B": [np.nan, np.nan, 10, 10, 10], "C": [10.0] * 5})
    volume = pd.DataFrame({"A": [100.0] * 5, "B": [np.nan, np.nan, 100, 100, 100]})
    volume["C"] = [100.0, 100, 100, 0, 100]
    return close, volume


def test_traded_value_converts_thousand_vnd() -> None:
    close, volume = _frames()
    assert universe.traded_value_vnd(close, volume)["A"].iloc[0] == pytest.approx(1e6)


def test_history_counts_sessions_since_first_row() -> None:
    close, _ = _frames()
    assert universe.history_length(close)["B"].tolist() == [0, 0, 1, 2, 3]


def test_eligibility_requires_history_liquidity_and_a_trade() -> None:
    close, volume = _frames()
    eligible = universe.eligibility(close, volume, RULE)
    assert eligible["A"].tolist() == [False, False, True, True, True]
    assert eligible["B"].tolist() == [False, False, False, False, True]
    # C: no trade on session 3; on session 4 the 2-session median is 0.5e6 < 1e6.
    assert eligible["C"].tolist() == [False, False, True, False, False]
    assert eligible.dtypes.eq(bool).all()


def test_eligibility_refuses_an_undecided_threshold() -> None:
    close, volume = _frames()
    rule = UniverseConfig(min_history_days=3, traded_value_window=2, min_traded_value_vnd=None)
    with pytest.raises(universe.UndecidedParameterError):
        universe.eligibility(close, volume, rule)
