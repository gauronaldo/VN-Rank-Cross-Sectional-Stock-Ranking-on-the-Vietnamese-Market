"""Tests for costs, constraints, and the rebalance loop, including the cost-accounting test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank.backtest import constraints, costs, engine
from vn_rank.config import CostsConfig
from vn_rank.data.universe import UndecidedParameterError

COSTS = costs.CostModel(brokerage_fee=0.0015, selling_tax=0.001, slippage=0.001)
DAYS = pd.bdate_range("2024-01-01", periods=6)
SYMBOLS = pd.Index(["A", "B", "C"])


def market(returns: np.ndarray | None = None, adv: float = 1e15) -> engine.Market:
    values = np.zeros((len(DAYS), len(SYMBOLS))) if returns is None else returns
    return engine.Market(
        returns=pd.DataFrame(values, index=DAYS, columns=SYMBOLS),
        adv_vnd=pd.DataFrame(adv, index=DAYS, columns=SYMBOLS),
        bands=pd.Series(0.07, index=SYMBOLS),
    )


SETTINGS = engine.EngineSettings(
    COSTS, max_fraction_of_adv=0.1, portfolio_value_vnd=1e10, limit_hit_margin=0.005
)


def test_cost_accounting_matches_hand_computation() -> None:
    """Mandatory test 3: buy A, rotate the whole portfolio into B, then liquidate."""
    targets = {DAYS[1]: pd.Series({"A": 1.0}), DAYS[3]: pd.Series({"B": 1.0})}
    result = engine.run_backtest(targets, DAYS[4], market(), SETTINGS)
    buy, sell = 0.0015 + 0.001, 0.0015 + 0.001 + 0.001
    expected = pd.Series([-buy, 0.0, -(sell + buy), -sell], index=DAYS[1:5])
    assert result.daily["net_return"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-15)
    assert result.trades["turnover"].to_list() == pytest.approx([0.5, 1.0])
    assert result.trades["cost"].to_list() == pytest.approx([buy, sell + buy])
    final_equity = (1 - buy) * (1 - sell - buy) * (1 - sell)
    assert (1 + result.daily["net_return"]).prod() == pytest.approx(final_equity)


def test_weights_drift_with_returns_and_cash_earns_nothing() -> None:
    returns = np.zeros((len(DAYS), 3))
    returns[2] = [0.10, -0.05, 0.0]
    targets = {DAYS[1]: pd.Series({"A": 0.5, "B": 0.25})}  # 25% cash
    no_cost = engine.EngineSettings(costs.CostModel(0, 0, 0), 0.1, 1e10, 0.005)
    result = engine.run_backtest(targets, DAYS[3], market(returns), no_cost)
    assert result.daily.loc[DAYS[2], "gross_return"] == pytest.approx(0.5 * 0.10 - 0.25 * 0.05)
    assert result.trades["invested"].iloc[0] == pytest.approx(0.75)


def test_top_quantile_weights_take_the_highest_scores() -> None:
    scores = pd.Series({"A": 0.1, "B": 0.9, "C": 0.5, "D": np.nan, "E": 0.7})
    weights = engine.top_quantile_weights(scores, 0.4)
    assert weights.to_dict() == pytest.approx({"B": 0.5, "E": 0.5})
    assert engine.equal_weights(pd.Index(["X", "Y"])).to_list() == [0.5, 0.5]


def _day(returns: list[float], caps: list[float] | None = None) -> constraints.TradeDay:
    return constraints.TradeDay(
        returns=np.array(returns),
        bands=np.array([0.07, 0.07, 0.15]),
        cap_weights=np.array(caps or [1.0, 1.0, 1.0]),
        limit_hit_margin=0.005,
    )


def test_limit_up_blocks_buys_and_limit_down_blocks_sells() -> None:
    current = np.array([0.0, 0.5, 0.5])
    target = np.array([0.5, 0.0, 0.5])
    day = _day([0.069, 0.0, 0.0])  # A closes at its +7% limit (within the margin)
    assert constraints.constrain(current, target, day) == pytest.approx([0.0, 0.0, 0.5])
    day = _day([0.0, -0.068, 0.0])  # B is limit-down: cannot be sold
    assert constraints.constrain(current, target, day) == pytest.approx([0.0, 0.5, 0.5])


def test_blocked_sell_scales_down_buys_to_avoid_leverage() -> None:
    current = np.array([0.0, 1.0, 0.0])
    target = np.array([0.5, 0.0, 0.5])
    feasible = constraints.constrain(current, target, _day([0.0, -0.07, 0.0]))
    assert feasible == pytest.approx([0.0, 1.0, 0.0])
    current = np.array([0.0, 0.4, 0.0])
    feasible = constraints.constrain(current, np.array([0.5, 0.0, 0.5]), _day([0.0, -0.07, 0.0]))
    assert feasible.sum() == pytest.approx(1.0)
    assert feasible == pytest.approx([0.3, 0.4, 0.3])


def test_liquidity_cap_limits_increases_only() -> None:
    current = np.array([0.3, 0.0, 0.0])
    target = np.array([0.2, 0.4, 0.4])
    feasible = constraints.constrain(current, target, _day([0, 0, 0], caps=[0.1, 0.1, 0.5]))
    assert feasible == pytest.approx([0.2, 0.1, 0.4])  # A may shrink; B is capped
    held = constraints.constrain(
        np.array([0.3, 0, 0]), np.array([0.5, 0, 0]), _day([0, 0, 0], [0.1, 1, 1])
    )
    assert held == pytest.approx([0.3, 0.0, 0.0])  # a drifted position is not forced down


def test_cap_uses_portfolio_value_in_vnd() -> None:
    settings = engine.EngineSettings(costs.CostModel(0, 0, 0), 0.1, 1e10, 0.005)
    result = engine.run_backtest(
        {DAYS[1]: pd.Series({"A": 1.0})}, DAYS[2], market(adv=2e10), settings
    )
    assert result.trades["invested"].iloc[0] == pytest.approx(0.2)  # 10% of 20 bn / 10 bn


def test_cost_model_requires_decided_values() -> None:
    with pytest.raises(UndecidedParameterError):
        costs.cost_model(CostsConfig(None, 0.001, 0.001))
    model = costs.cost_model(CostsConfig(0.0015, 0.001, 0.001))
    assert model.sell_rate == pytest.approx(0.0035)
