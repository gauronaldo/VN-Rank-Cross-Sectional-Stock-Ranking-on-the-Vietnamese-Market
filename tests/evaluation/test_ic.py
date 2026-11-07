"""Tests for Rank IC, Newey-West t-stat, quantile returns, and signal autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from vn_rank.evaluation import ic

DATES = pd.bdate_range("2024-01-01", periods=12)
SYMBOLS = [f"S{i}" for i in range(10)]


def scored_frame(scores: np.ndarray, returns: np.ndarray) -> pd.DataFrame:
    """Long frame from (dates x symbols) arrays of scores and forward returns."""
    dates = DATES[: scores.shape[0]]
    return pd.DataFrame(
        {
            "date": np.repeat(dates, scores.shape[1]),
            "symbol": np.tile(SYMBOLS[: scores.shape[1]], len(dates)),
            "score": scores.ravel(),
            "forward_return": returns.ravel(),
        }
    )


def test_spearman_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    left = pd.Series(rng.integers(0, 5, 200).astype(float))
    right = pd.Series(left + rng.normal(0, 2, 200))
    expected = stats.spearmanr(left, right).statistic
    assert ic.spearman(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("right", "expected"),
    [([1.0, 2.0, 3.0, 4.0], 1.0), ([40.0, 30.0, 20.0, 10.0], -1.0)],
)
def test_spearman_monotone(right: list[float], expected: float) -> None:
    left = pd.Series([0.1, 0.2, 0.5, 0.9])
    assert ic.spearman(left, pd.Series(right)) == pytest.approx(expected)


def test_spearman_undefined_cases_are_nan() -> None:
    assert np.isnan(ic.spearman(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0])))
    assert np.isnan(ic.spearman(pd.Series([1.0, np.nan]), pd.Series([2.0, 3.0])))


def test_daily_rank_ic_known_answers() -> None:
    returns = np.array([np.arange(10.0), np.arange(10.0), np.arange(10.0)])
    scores = np.array([np.arange(10.0), -np.arange(10.0), np.ones(10)])
    daily = ic.daily_rank_ic(scored_frame(scores, returns))
    assert list(daily.index) == list(DATES[:3])
    assert daily.to_list() == pytest.approx([1.0, -1.0, 0.0])  # constant scores: IC 0


def test_daily_rank_ic_drops_dates_with_constant_returns() -> None:
    returns = np.array([np.arange(10.0), np.zeros(10)])
    daily = ic.daily_rank_ic(scored_frame(np.tile(np.arange(10.0), (2, 1)), returns))
    assert list(daily.index) == [DATES[0]]


def test_abstaining_model_summary_has_zero_ic_and_no_quantiles() -> None:
    returns = np.random.default_rng(5).normal(size=(12, 10))
    frame = scored_frame(np.ones((12, 10)), returns)
    summary = ic.signal_summary(frame, ic.SignalSettings(horizon_days=5, n_quantiles=5))
    assert summary["mean_ic"] == 0.0
    assert summary["n_dates"] == 12
    for key in ("ic_ir", "t_nw", "q1", "q5", "top_minus_bottom", "signal_autocorr"):
        assert np.isnan(summary[key])


def test_newey_west_degenerate_inputs_are_nan() -> None:
    assert np.isnan(ic.newey_west_tstat(pd.Series([0.1]), 4))
    assert np.isnan(ic.newey_west_tstat(pd.Series([0.0, 0.0, 0.0]), 4))


def test_newey_west_lag_zero_is_plain_tstat() -> None:
    values = pd.Series(np.random.default_rng(2).normal(0.02, 0.1, 300))
    plain = values.mean() / (values.std(ddof=0) / np.sqrt(len(values)))
    assert ic.newey_west_tstat(values, 0) == pytest.approx(plain)


@pytest.mark.parametrize("lag", [1, 4, 10])
def test_newey_west_matches_statsmodels(lag: int) -> None:
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.1, 400)
    values = pd.Series(np.convolve(noise, np.ones(5) / 5, mode="same") + 0.01)  # overlapping
    fit = sm.OLS(values.to_numpy(), np.ones(len(values))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    assert ic.newey_west_tstat(values, lag) == pytest.approx(fit.tvalues[0])


def test_quantile_returns_known_answer() -> None:
    returns = np.array([np.arange(1.0, 11.0), np.arange(1.0, 11.0)])
    scores = returns[:, ::-1] * -1  # same ordering as returns
    result = ic.quantile_returns(scored_frame(scores, returns), 5)
    assert result.to_list() == pytest.approx([1.5, 3.5, 5.5, 7.5, 9.5])
    assert list(result.index) == [1, 2, 3, 4, 5]


def test_rebalance_dates_take_every_hth_date() -> None:
    dates = pd.Series(np.repeat(DATES, 3))
    assert list(ic.rebalance_dates(dates, 5)) == [DATES[0], DATES[5], DATES[10]]


def test_signal_autocorrelation_known_answers() -> None:
    base = np.arange(10.0)
    stable = np.tile(base, (12, 1))
    flipping = np.array([base if (i // 5) % 2 == 0 else -base for i in range(12)])
    returns = np.zeros((12, 10))
    kept = ic.signal_autocorrelation(scored_frame(stable, returns), 5)
    assert kept.to_list() == pytest.approx([1.0, 1.0])
    flipped = ic.signal_autocorrelation(scored_frame(flipping, returns), 5)
    assert flipped.to_list() == pytest.approx([-1.0, -1.0])


def test_signal_summary_keys_and_values() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.05, (12, 10))
    frame = scored_frame(returns + rng.normal(0, 0.05, (12, 10)), returns)
    summary = ic.signal_summary(frame, ic.SignalSettings(horizon_days=5, n_quantiles=5))
    expected_keys = {"mean_ic", "ic_std", "ic_ir", "t_nw", "n_dates", "top_minus_bottom"}
    assert expected_keys | {f"q{i}" for i in range(1, 6)} | {"signal_autocorr"} == set(summary)
    daily = ic.daily_rank_ic(frame)
    assert summary["mean_ic"] == pytest.approx(daily.mean())
    assert summary["ic_ir"] == pytest.approx(daily.mean() / daily.std())
    assert summary["n_dates"] == 12
    assert summary["top_minus_bottom"] == pytest.approx(summary["q5"] - summary["q1"])
