"""Tests for data-quality checks on hand-built panels with known anomalies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank.data import validate
from vn_rank.data.validate import QualitySettings

SETTINGS = QualitySettings(price_limits={"HOSE": 0.07, "DELISTED": 0.15}, band_tolerance=0.001)
CALENDAR = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=6))
SYMBOLS = pd.DataFrame({"symbol": ["AAA", "OLD"], "exchange": ["HOSE", "DELISTED"]})


def _panel(symbol: str, closes: list[float], start: int = 0) -> pd.DataFrame:
    dates = CALENDAR[start : start + len(closes)]
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": 1000,
        }
    )


def _flags(panel: pd.DataFrame) -> pd.DataFrame:
    exchange = SYMBOLS.set_index("symbol")["exchange"]
    return validate.flag_rows(panel, exchange, SETTINGS, CALENDAR)


def test_clean_panel_has_no_flags() -> None:
    panel = pd.concat([_panel("AAA", [10, 10.5, 10.2]), _panel("OLD", [5, 5.5])])
    assert not _flags(panel.reset_index(drop=True)).to_numpy().any()


def test_band_breach_uses_the_exchange_band() -> None:
    # +8% breaches HOSE (7%) but +14% does not breach DELISTED (15%).
    panel = pd.concat([_panel("AAA", [10, 10.8]), _panel("OLD", [10, 11.4])])
    flags = _flags(panel.reset_index(drop=True))
    assert flags["band_breach"].tolist() == [False, True, False, False]


def test_returns_do_not_cross_symbols() -> None:
    panel = pd.concat([_panel("AAA", [10.0]), _panel("OLD", [50.0])]).reset_index(drop=True)
    assert validate.daily_returns(panel).isna().all()


def test_row_level_anomalies_are_flagged() -> None:
    panel = _panel("AAA", [10, 10, 10, 10, 10])
    panel.loc[1, "high"] = 9.0  # high below close
    panel.loc[2, "volume"] = 0
    panel.loc[3, "close"] = np.nan
    panel.loc[4, "low"] = -1.0
    flags = _flags(panel)
    assert flags["ohlc_inconsistent"].tolist() == [False, True, False, False, False]
    assert flags["zero_volume"].tolist() == [False, False, True, False, False]
    assert flags["missing_value"].tolist() == [False, False, False, True, False]
    assert flags["non_positive_price"].tolist() == [False, False, False, False, True]


def test_duplicates_and_off_calendar_dates_are_flagged() -> None:
    panel = _panel("AAA", [10, 10, 10])
    panel.loc[1, "date"] = panel.loc[0, "date"]
    panel.loc[2, "date"] = pd.Timestamp("2024-01-06")  # a Saturday
    flags = _flags(panel)
    assert flags["duplicate"].tolist() == [True, True, False]
    assert flags["off_calendar"].tolist() == [False, False, True]


def test_calendar_coverage_counts_missing_sessions() -> None:
    panel = _panel("AAA", [10, 10, 10, 10, 10]).drop(index=[1, 2])
    panel.loc[4, "volume"] = 0
    coverage = validate.calendar_coverage(panel.reset_index(drop=True), CALENDAR)
    row = coverage.loc["AAA"]
    assert (row["rows"], row["traded_days"]) == (3, 2)
    assert (row["expected_sessions"], row["missing_sessions"]) == (5, 2)


def test_tickers_per_year_splits_listed_and_delisted() -> None:
    panel = pd.concat([_panel("AAA", [10, 10]), _panel("OLD", [5, 5])]).reset_index(drop=True)
    panel.loc[panel["symbol"] == "OLD", "volume"] = 0  # never traded: not counted
    table = validate.tickers_per_year(panel, SYMBOLS.set_index("symbol")["exchange"])
    assert table.loc[2024].to_dict() == {"listed": 1, "delisted": 0}


def test_run_quality_checks_summarizes_counts() -> None:
    panel = pd.concat([_panel("AAA", [10, 10.8, 10.8]), _panel("OLD", [5, 5])])
    report = validate.run_quality_checks(panel.reset_index(drop=True), SYMBOLS, CALENDAR, SETTINGS)
    assert report.counts.loc["band_breach"].tolist() == [1, 1]
    assert report.flags[["symbol", "exchange"]].values.tolist() == [["AAA", "HOSE"]]
    assert report.flags["ret"].iloc[0] == pytest.approx(0.08)
    assert set(report.coverage.index) == {"AAA", "OLD"}


def test_plot_tickers_per_year_returns_figure() -> None:
    table = pd.DataFrame({"listed": [10, 12], "delisted": [3, 1]}, index=[2019, 2020])
    figure = validate.plot_tickers_per_year(table)
    assert len(figure.axes[0].patches) == len(table) * 2


def test_markdown_table_formats_cells() -> None:
    frame = pd.DataFrame({"ret": [0.123456], "date": [pd.Timestamp("2024-01-02")]}, index=["A"])
    assert validate.markdown_table(frame).splitlines() == [
        "| index | ret | date |",
        "|---|---|---|",
        "| A | 0.1235 | 2024-01-02 |",
    ]


def test_render_markdown_lists_sections() -> None:
    panel = pd.concat([_panel("AAA", [10, 10.8, 10.8]), _panel("OLD", [5, 5])])
    report = validate.run_quality_checks(panel.reset_index(drop=True), SYMBOLS, CALENDAR, SETTINGS)
    text = validate.render_markdown(report, {"source": "vci"}, top_n=5)
    assert "- **source:** vci" in text
    assert "## Largest band breaches (top 5)" in text
    assert "| 2024-01-02 | AAA | HOSE | 0.0800 |" in text


def test_flags_are_numpy_bool_with_pyarrow_strings() -> None:
    panel = _panel("AAA", [10, 10.8]).astype({"symbol": "string[pyarrow]"})
    symbols = SYMBOLS.astype({"symbol": "string[pyarrow]", "exchange": "string[pyarrow]"})
    report = validate.run_quality_checks(panel, symbols, CALENDAR, SETTINGS)
    assert report.counts["rows"].dtype == np.int64
    assert report.counts.loc["band_breach", "rows"] == 1


def test_unknown_exchange_band_raises() -> None:
    exchange = pd.Series({"AAA": "HNX"})
    with pytest.raises(KeyError, match="HNX"):
        validate.flag_rows(_panel("AAA", [10, 11]), exchange, SETTINGS, CALENDAR)
