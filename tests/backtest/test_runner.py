"""Tests for the backtest runner: schedules, benchmarks, and the full report."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from vn_rank.backtest import costs, engine, runner
from vn_rank.evaluation.bootstrap import BootstrapSettings

SESSIONS = pd.bdate_range("2024-01-01", periods=80)
SYMBOLS = [f"S{i:02d}" for i in range(40)]
HORIZON = 5
SETTINGS = runner.BacktestSettings(
    engine=engine.EngineSettings(costs.CostModel(0.0015, 0.001, 0.001), 0.1, 1e10, 0.005),
    top_quantile=0.1,
    horizon=HORIZON,
    trading_days=252,
    bootstrap=BootstrapSettings(block=5, n_resamples=200, level=0.95, seed=0),
)


def synthetic_market() -> tuple[engine.Market, pd.Series]:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.02, (80, 40)), index=SESSIONS, columns=SYMBOLS)
    adv = pd.DataFrame(1e15, index=SESSIONS, columns=SYMBOLS)
    market = engine.Market(returns, adv, pd.Series(0.15, index=SYMBOLS))
    index_close = (1 + returns.mean(axis=1)).cumprod() * 1000
    return market, index_close


def predictions(market: engine.Market) -> tuple[pd.DataFrame, pd.Series]:
    """Oracle scores (the realized label) for 'lightgbm', noise for 'momentum'."""
    growth = (1 + market.returns).cumprod()
    forward = growth.shift(-(1 + HORIZON)) / growth.shift(-1) - 1  # close t+1 -> t+1+h
    signals = SESSIONS[: -(1 + HORIZON)]
    long = forward.loc[signals].stack().rename("forward_return").reset_index()
    long.columns = ["date", "symbol", "forward_return"]
    rng = np.random.default_rng(1)
    frames = [
        long.assign(score=long["forward_return"], model="lightgbm", seed=s, fold="f")
        for s in (0, 1)
    ]
    frames.append(long.assign(score=rng.random(len(long)), model="momentum", seed=42, fold="f"))
    entry = pd.Series(SESSIONS, index=SESSIONS).shift(-1)
    return pd.concat(frames, ignore_index=True), entry


def test_score_schedule_trades_the_next_session_every_h_dates() -> None:
    market, _ = synthetic_market()
    preds, entry = predictions(market)
    schedule = runner.score_schedule(preds[preds["model"] == "momentum"], entry, HORIZON)
    days = sorted(schedule)
    assert days[0] == SESSIONS[1]
    gaps = [SESSIONS.get_loc(b) - SESSIONS.get_loc(a) for a, b in itertools.pairwise(days)]
    assert set(gaps) == {HORIZON}
    assert len(schedule[days[0]]) == len(SYMBOLS)


def test_backtest_all_ranks_an_oracle_above_the_benchmark() -> None:
    market, index_close = synthetic_market()
    preds, entry = predictions(market)
    report = runner.backtest_all(preds, entry, (market, index_close), SETTINGS)
    assert list(report.summary.index) == ["lightgbm", "momentum", "equal_weight", "vnindex"]
    summary = report.summary
    assert summary.loc["lightgbm", "sharpe"] > summary.loc["equal_weight", "sharpe"] + 1
    assert summary.loc["lightgbm", "excess_ci_low"] > 0
    assert summary.loc["lightgbm", "avg_holdings"] == pytest.approx(4)
    assert summary.loc["equal_weight", "excess_ann_return"] == pytest.approx(0.0)
    assert report.ic.loc["lightgbm", "mean_ic"] == pytest.approx(1.0)
    assert report.ic.loc["momentum", "diff_ci_low"] > 0
    last_day = report.curves["lightgbm"].index.max()
    # last rebalance signal is session 70, traded at 71, held for h sessions until 76
    assert SESSIONS.get_loc(last_day) == 71 + HORIZON
    text = runner.render_backtest_markdown(report, ["# Title"], 4)
    assert "## Portfolio metrics (net of costs)" in text


def test_identical_seeds_average_to_the_same_portfolio() -> None:
    market, _ = synthetic_market()
    preds, entry = predictions(market)
    results = runner.run_model(preds[preds["model"] == "lightgbm"], entry, market, SETTINGS)
    net, _, _ = runner.average_portfolio(results)
    pd.testing.assert_series_equal(net, results[0].daily["net_return"], check_names=False)
