"""Tests for portfolio performance metrics and the moving-block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank.evaluation import bootstrap, performance

DAYS = pd.bdate_range("2023-12-25", periods=10)


def test_performance_metrics_known_answers() -> None:
    daily = pd.Series([0.01, -0.02, 0.01, 0.0], index=DAYS[:4])
    assert performance.annualized_return(daily, 252) == pytest.approx(0.0 * 252)
    assert performance.annualized_volatility(daily, 252) == pytest.approx(
        daily.std() * np.sqrt(252)
    )
    equity = np.cumprod([1.01, 0.98, 1.01, 1.0])
    assert performance.max_drawdown(daily) == pytest.approx(equity[1] / equity[0] - 1)
    assert performance.total_return(daily) == pytest.approx(equity[-1] - 1)


def test_drawdown_counts_a_fall_from_the_start() -> None:
    assert performance.max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_sharpe_and_summary() -> None:
    daily = pd.Series([0.01, 0.02, 0.0, 0.01], index=DAYS[:4])
    expected = daily.mean() * 252 / (daily.std() * np.sqrt(252))
    assert performance.sharpe_ratio(daily, 252) == pytest.approx(expected)
    assert np.isnan(performance.sharpe_ratio(pd.Series([0.01, 0.01]), 252))
    summary = performance.performance_summary(daily, pd.Series([0.5, 0.3]), 252, 5)
    assert summary["ann_turnover"] == pytest.approx(0.4 * 252 / 5)
    empty = performance.performance_summary(daily, pd.Series(dtype=float), 252, 5)
    assert np.isnan(empty["ann_turnover"])


def test_yearly_breakdown_splits_calendar_years() -> None:
    daily = pd.Series(0.01, index=DAYS)
    table = performance.yearly_breakdown(daily, 252)
    assert list(table.index) == [2023, 2024]
    assert table["days"].sum() == len(DAYS)


def test_block_indices_are_consecutive_runs() -> None:
    idx = bootstrap.block_indices(10, 4, 3, np.random.default_rng(0))
    assert idx.shape == (3, 10)
    for row in idx:
        for start in (0, 4):
            block = row[start : start + 4]
            assert np.array_equal(np.diff(block), np.ones(3))
        assert row.max() <= 9


def test_bootstrap_ci_covers_the_mean_and_is_reproducible() -> None:
    values = np.random.default_rng(1).normal(0.05, 0.1, 500)
    settings = bootstrap.BootstrapSettings(block=20, n_resamples=2000, level=0.95, seed=7)
    low, high = bootstrap.bootstrap_ci(values, bootstrap.mean_statistic, settings)
    assert low < values.mean() < high
    assert (high - low) == pytest.approx(2 * 1.96 * 0.1 / np.sqrt(500), rel=0.3)
    assert bootstrap.bootstrap_ci(values, bootstrap.mean_statistic, settings) == (low, high)


def test_bootstrap_rejects_nan_and_sharpe_statistic_matches() -> None:
    settings = bootstrap.BootstrapSettings(5, 10, 0.9, 0)
    with pytest.raises(ValueError, match="no NaN"):
        bootstrap.bootstrap_ci(np.array([0.1, np.nan, 0.2]), bootstrap.mean_statistic, settings)
    sample = np.array([[0.01, 0.02, 0.0]])
    expected = sample.mean() / sample.std(ddof=1) * np.sqrt(252)
    assert bootstrap.sharpe_statistic(252)(sample)[0] == pytest.approx(expected)
