"""Trading costs: brokerage fee, selling tax, and slippage.

Weights are fractions of portfolio value. Buying ``dw`` of the portfolio costs
``dw * (fee + slippage)``; selling ``dw`` costs ``dw * (fee + slippage + tax)``. Costs are
charged as a fraction of portfolio value on the trade day.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vn_rank.config import CostsConfig
from vn_rank.data.universe import UndecidedParameterError


@dataclasses.dataclass(frozen=True)
class CostModel:
    """Per-side cost rates as fractions of traded value."""

    brokerage_fee: float
    selling_tax: float
    slippage: float

    @property
    def buy_rate(self) -> float:
        """Cost of buying one unit of value."""
        return self.brokerage_fee + self.slippage

    @property
    def sell_rate(self) -> float:
        """Cost of selling one unit of value (includes the selling tax)."""
        return self.brokerage_fee + self.slippage + self.selling_tax


def cost_model(costs: CostsConfig) -> CostModel:
    """Build the cost model from config.

    Raises:
        UndecidedParameterError: If the brokerage fee or slippage is still unset.
    """
    if costs.brokerage_fee_per_side is None or costs.slippage_per_side is None:
        raise UndecidedParameterError("backtest.costs fee and slippage must be set")
    return CostModel(costs.brokerage_fee_per_side, costs.selling_tax, costs.slippage_per_side)


def trade_cost(current: np.ndarray, target: np.ndarray, model: CostModel) -> float:
    """Cost of moving from ``current`` to ``target`` weights, as a fraction of value."""
    change = target - current
    bought = float(np.clip(change, 0.0, None).sum())
    sold = float(np.clip(-change, 0.0, None).sum())
    return bought * model.buy_rate + sold * model.sell_rate


def turnover(current: np.ndarray, target: np.ndarray) -> float:
    """One-way turnover ``0.5 * sum |target - current|`` of one rebalance."""
    return 0.5 * float(np.abs(target - current).sum())
