"""Tests for per-date ranks, z-scores, and the forward-return label alignment."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from vn_rank.config import LabelConfig
from vn_rank.features import labels
from vn_rank.features.transforms import cross_sectional_rank, cross_sectional_zscore

RULE = LabelConfig(horizon_days=2, entry_lag_days=1, target="rank")


def _wide(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["A", "B", "C", "D"], dtype=float)


def test_rank_is_per_date_and_scaled_to_unit_interval() -> None:
    values = _wide([[3, 1, 2, 4], [10, 40, 30, 20]])
    ranks = cross_sectional_rank(values, values.notna())
    assert ranks.iloc[0].tolist() == pytest.approx([2 / 3, 0, 1 / 3, 1])
    assert ranks.iloc[1].tolist() == pytest.approx([0, 1, 2 / 3, 1 / 3])


def test_rank_ignores_masked_and_missing_and_averages_ties() -> None:
    values = _wide([[5, 5, np.nan, 100], [7, 1, 2, 3]])
    mask = pd.DataFrame(True, index=values.index, columns=values.columns)
    mask.loc[0, "D"] = False
    mask.loc[1, ["B", "C", "D"]] = False
    ranks = cross_sectional_rank(values, mask)
    assert ranks.iloc[0, :2].tolist() == pytest.approx([0.5, 0.5])
    assert ranks.iloc[0, 2:].isna().all()
    assert ranks.iloc[1].tolist()[0] == pytest.approx(0.5)  # single stock on the date
    assert ranks.iloc[1, 1:].isna().all()


def test_zscore_is_per_date() -> None:
    values = _wide([[1, 2, 3, 4], [5, 5, 5, 5]])
    z = cross_sectional_zscore(values, values.notna())
    assert z.iloc[0].mean() == pytest.approx(0)
    assert z.iloc[0].std() == pytest.approx(1)
    assert z.iloc[1].isna().all()  # zero dispersion


def test_forward_return_enters_next_close_and_exits_h_sessions_later() -> None:
    close = pd.DataFrame({"A": [10.0, 11, 12, 13, 14, 15]})
    alive = close.notna()
    forward = labels.forward_return(close, alive, RULE)["A"].tolist()
    # t=0: entry close[1]=11, exit close[3]=13
    assert forward[:3] == pytest.approx([13 / 11 - 1, 14 / 12 - 1, 15 / 13 - 1])
    assert np.isnan(forward[3:]).all()


def test_forward_return_is_missing_when_exit_is_after_last_row() -> None:
    close = pd.DataFrame({"A": [10.0, 11, 12, 13, 13, 13]})  # last real row at index 3
    alive = pd.DataFrame({"A": [True, True, True, True, False, False]})
    forward = labels.forward_return(close, alive, RULE)["A"]
    assert forward.iloc[0] == pytest.approx(13 / 11 - 1)  # exit at index 3, still alive
    assert forward.iloc[[1, 2]].isna().all()  # exits at 4 and 5 are after the last row


def test_entry_and_exit_dates() -> None:
    sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=5))
    exits = labels.label_exit_dates(sessions, RULE)
    entries = labels.label_entry_dates(sessions, RULE)
    assert exits.iloc[0] == sessions[3]
    assert entries.iloc[0] == sessions[1]
    assert exits.iloc[-3:].isna().all()


def test_training_target_rank_and_zscore() -> None:
    returns = _wide([[0.01, -0.02, 0.03, np.nan]])
    mask = returns.notna()
    assert labels.training_target(returns, mask, RULE).iloc[0, :3].tolist() == [0.5, 0, 1]
    zscore = labels.training_target(returns, mask, dataclasses.replace(RULE, target="zscore"))
    assert zscore.iloc[0, :3].mean() == pytest.approx(0)
