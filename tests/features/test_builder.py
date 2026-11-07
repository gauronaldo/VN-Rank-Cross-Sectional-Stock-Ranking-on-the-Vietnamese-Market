"""Tests for data treatment, calendar alignment, panel assembly, and look-ahead safety."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vn_rank.config import Config, load_config
from vn_rank.features import builder

N_SESSIONS = 330
CALENDAR = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=N_SESSIONS))
LABEL_COLUMNS = ["entry_date", "exit_date", "forward_return", "target"]


def _symbol(name: str, rng: np.random.Generator, volume_scale: float) -> pd.DataFrame:
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, N_SESSIONS)))
    spread = np.abs(rng.normal(0, 0.01, N_SESSIONS)) * close
    volume = rng.integers(1, 5, N_SESSIONS) * volume_scale
    volume[rng.random(N_SESSIONS) < 0.1] = 0
    return pd.DataFrame(
        {
            "date": CALENDAR,
            "symbol": name,
            "open": close + rng.normal(0, 0.005, N_SESSIONS) * close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": volume.astype(np.int64),
        }
    )


def synthetic_raw_panel() -> pd.DataFrame:
    """Six symbols covering missing rows, a delisting, a late listing, and bad prices."""
    rng = np.random.default_rng(11)
    frames = {f"S{i}": _symbol(f"S{i}", rng, 1000.0) for i in range(5)}
    frames["S5"] = _symbol("S5", rng, 1.0)  # too illiquid for the universe
    frames["S1"] = frames["S1"].drop(index=range(270, 273))  # missing sessions
    frames["S2"] = frames["S2"].iloc[:300]  # delisted after session 299
    frames["S3"] = frames["S3"].iloc[40:]  # listed at session 40
    frames["S4"].loc[280, "open"] = 0.0
    frames["S4"].loc[281, "high"] = frames["S4"].loc[281, "close"] * 0.9
    return pd.concat(frames.values(), ignore_index=True)


@pytest.fixture
def config(default_config_path: Path) -> Config:
    base = load_config(default_config_path)
    rule = dataclasses.replace(base.universe, min_traded_value_vnd=1e7)
    return dataclasses.replace(base, universe=rule)


EXCHANGE = pd.Series({f"S{i}": "HOSE" for i in range(6)})


def test_repair_ohlc_flattens_no_trade_bars_and_clips_high_low() -> None:
    raw = pd.DataFrame(
        {
            "open": [9.0, 0.0, 10.0],
            "high": [11.0, 12.0, 9.0],
            "low": [9.5, 9.0, 8.0],
            "close": [10.0, 11.0, 9.5],
            "volume": [0, 100, 100],
        }
    )
    repaired = builder.repair_ohlc(raw)
    assert repaired.iloc[0][["open", "high", "low", "close"]].tolist() == [10.0] * 4
    assert repaired.iloc[1][["open", "high", "low"]].tolist() == [11.0, 12.0, 9.0]
    assert repaired.iloc[2][["high", "low"]].tolist() == [10.0, 8.0]
    assert repaired["close"].tolist() == raw["close"].tolist()


def test_align_to_calendar_fills_missing_sessions_and_marks_last_row() -> None:
    raw = builder.repair_ohlc(synthetic_raw_panel())
    frames = builder.align_to_calendar(raw, CALENDAR)
    s1 = raw.loc[raw["symbol"] == "S1"].set_index("date")
    filled = CALENDAR[270:273]
    assert (frames.close.loc[filled, "S1"] == s1.loc[CALENDAR[269], "close"]).all()
    assert (frames.volume.loc[filled, "S1"] == 0).all()
    assert (frames.high.loc[filled, "S1"] == frames.close.loc[filled, "S1"]).all()
    assert frames.close["S3"].iloc[:40].isna().all()
    assert frames.alive["S2"].iloc[:300].all()
    assert not frames.alive["S2"].iloc[300:].any()


def test_align_to_calendar_rejects_duplicates() -> None:
    raw = synthetic_raw_panel()
    with pytest.raises(builder.DuplicateRowsError):
        builder.align_to_calendar(pd.concat([raw, raw.iloc[:1]]), CALENDAR)


def test_build_panel_rows_are_eligible_and_ranked(config: Config) -> None:
    panel = builder.build_panel(synthetic_raw_panel(), CALENDAR, EXCHANGE, config)
    groups = builder.feature_groups(config.features)
    names = [n for g in groups.values() for n in g]
    assert len(groups) == 5
    assert len(names) == 15
    assert set(names) <= set(panel.columns)
    assert "S5" not in set(panel["symbol"])
    assert panel.groupby("symbol")["date"].min()["S3"] >= CALENDAR[40 + 251]
    ranks = panel[[builder.RANK_PREFIX + n for n in names]]
    assert ranks.min().min() >= 0
    assert ranks.max().max() <= 1
    assert panel.loc[panel["date"] >= CALENDAR[-6], "target"].isna().all()
    assert (panel["exit_date"].dropna() > panel.loc[panel["exit_date"].notna(), "date"]).all()


def test_no_look_ahead_in_features(config: Config) -> None:
    """Mandatory: features on data truncated at t equal features at t on the full data."""
    raw = synthetic_raw_panel()
    full = builder.build_panel(raw, CALENDAR, EXCHANGE, config).drop(columns=LABEL_COLUMNS)
    for position in (262, 271, 290, 299, 300, N_SESSIONS - 1):
        t = CALENDAR[position]
        truncated = builder.build_panel(
            raw.loc[raw["date"] <= t], CALENDAR[: position + 1], EXCHANGE, config
        ).drop(columns=LABEL_COLUMNS)
        expected = full.loc[full["date"] <= t].reset_index(drop=True)
        assert len(expected) > 0
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected)


def test_labels_change_only_through_future_prices(config: Config) -> None:
    """Changing prices after t moves labels before t but never features before t."""
    raw = synthetic_raw_panel()
    shocked = raw.copy()
    after = shocked["date"] > CALENDAR[300]
    shocked.loc[after, ["open", "high", "low", "close"]] *= 2.0
    base = builder.build_panel(raw, CALENDAR, EXCHANGE, config)
    moved = builder.build_panel(shocked, CALENDAR, EXCHANGE, config)
    before = base["date"] <= CALENDAR[300]
    features = [c for c in base.columns if c not in LABEL_COLUMNS]
    pd.testing.assert_frame_equal(base.loc[before, features], moved.loc[before, features])
    assert not np.allclose(
        base.loc[before, "forward_return"].dropna().tail(20),
        moved.loc[before, "forward_return"].dropna().tail(20),
    )
