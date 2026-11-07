"""Known-answer tests for the hand-written indicators, against plain-loop references."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank.features import indicators as ind

RNG = np.random.default_rng(7)


def _frame(values: list[float] | np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def _walk(n: int) -> pd.DataFrame:
    return _frame(10 * np.exp(np.cumsum(RNG.normal(0, 0.02, n))))


def _loop_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def test_trailing_return() -> None:
    result = ind.trailing_return(_frame([10, 11, 12.1, 13.31]), 2)["A"].tolist()
    assert np.isnan(result[:2]).all()
    assert result[2:] == pytest.approx([0.21, 0.21])


def test_rolling_volatility_matches_sample_std() -> None:
    close = _walk(30)
    expected = close["A"].pct_change().iloc[-20:].std(ddof=1)
    assert ind.rolling_volatility(close, 20)["A"].iloc[-1] == pytest.approx(expected)
    assert ind.rolling_volatility(close, 20)["A"].iloc[:20].isna().all()


def test_true_range_uses_previous_close() -> None:
    high, low, close = _frame([11, 12, 10.5]), _frame([9, 11, 9.5]), _frame([10, 11.5, 10])
    assert ind.true_range(high, low, close)["A"].tolist() == [2.0, 2.0, 2.0]
    gap_down = ind.true_range(_frame([10, 8]), _frame([9, 7]), _frame([10, 7.5]))
    assert gap_down["A"].tolist() == [1.0, 3.0]


def test_wilder_average_matches_loop() -> None:
    values = RNG.normal(size=40)
    expected = _loop_ema(values, 1 / 14)
    result = ind.wilder_average(_frame(values), 14)["A"].to_numpy()
    assert np.isnan(result[:13]).all()
    assert result[13:] == pytest.approx(expected[13:])


def test_rsi_matches_loop_reference() -> None:
    close = _walk(60)["A"].to_numpy()
    change = np.diff(close)
    gain = _loop_ema(np.clip(change, 0, None), 1 / 14)
    loss = _loop_ema(np.clip(-change, 0, None), 1 / 14)
    expected = 100 - 100 / (1 + gain / loss)
    result = ind.rsi(_frame(close), 14)["A"].to_numpy()
    assert result[-1] == pytest.approx(expected[-1])
    assert np.nanmin(result) >= 0
    assert np.nanmax(result) <= 100


def test_rsi_is_100_when_price_only_rises() -> None:
    assert ind.rsi(_frame(np.arange(1, 30)), 14)["A"].iloc[-1] == pytest.approx(100.0)


def test_macd_histogram_matches_loop_reference() -> None:
    close = _walk(80)["A"].to_numpy()
    line = _loop_ema(close, 2 / 13) - _loop_ema(close, 2 / 27)
    expected = (line - _loop_ema(line[25:], 2 / 10)[-1:])[-1] / close[-1]
    assert ind.macd_histogram(_frame(close), 12, 26, 9)["A"].iloc[-1] == pytest.approx(expected)


def test_bollinger_percent_b_bounds() -> None:
    close = _frame([10.0] * 19 + [10.0])
    assert np.isnan(ind.bollinger_percent_b(close, 20, 2.0)["A"].iloc[-1])  # zero width
    values = np.array([10.0, 12.0] * 10)
    mean, std = values.mean(), values.std()
    expected = (values[-1] - (mean - 2 * std)) / (4 * std)
    assert ind.bollinger_percent_b(_frame(values), 20, 2.0)["A"].iloc[-1] == pytest.approx(expected)


def test_liquidity_features() -> None:
    volume = _frame([100, 100, 100, 400])
    assert ind.volume_ratio(volume, 4)["A"].iloc[-1] == pytest.approx(400 / 175)
    value = _frame([0, 2e9, 4e9, 6e9])
    assert ind.log_median_traded_value(value, 4)["A"].iloc[-1] == pytest.approx(np.log(3e9))
    assert np.isnan(ind.log_median_traded_value(_frame([0.0] * 4), 4)["A"].iloc[-1])


def test_amihud_skips_sessions_without_trades() -> None:
    close = _frame([10, 11, 11, 12.1])
    value = _frame([1e9, 2e9, 0, 4e9])
    expected = np.mean([0.1 / 2e9, 0.1 / 4e9]) * 1e9
    assert ind.amihud_illiquidity(close, value, 3)["A"].iloc[-1] == pytest.approx(expected)


def test_close_to_rolling_high() -> None:
    high, close = _frame([10, 15, 12, 11]), _frame([9, 14, 12, 10.5])
    result = ind.close_to_rolling_high(close, high, 3)["A"].tolist()
    assert result[2:] == pytest.approx([12 / 15, 10.5 / 15])
