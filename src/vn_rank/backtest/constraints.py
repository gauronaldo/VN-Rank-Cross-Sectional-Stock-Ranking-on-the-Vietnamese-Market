"""Trading constraints: limit-locked stocks and a liquidity cap on position size.

On a trade day (the close of ``t + 1``):

- a stock that closed at its **limit-up** price cannot be bought (no sellers), so its
  weight cannot increase;
- a stock that closed at its **limit-down** price cannot be sold (no buyers), so its
  weight cannot decrease;
- a position cannot be **increased** above ``max_fraction_of_adv`` of the stock's 20-day
  average traded value; a position that drifted above the cap is not forced down.

A stock is at a limit when its close-to-close return is within ``limit_hit_margin`` of
its exchange band (the limit price is rounded to the tick, so the realized move is a
little under the band). The band is today's exchange, a known limitation. If blocked
sells leave the weights summing above 1, the increases are scaled down so the portfolio
is fully invested without leverage; weights below 1 leave the rest in cash.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class TradeDay:
    """Market facts on one trade day, aligned with the weight vector's symbols."""

    returns: np.ndarray
    bands: np.ndarray
    cap_weights: np.ndarray
    limit_hit_margin: float


def limit_locks(day: TradeDay) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of limit-up and limit-down closes (NaN returns are not locked)."""
    returns = np.nan_to_num(day.returns, nan=0.0)
    threshold = day.bands - day.limit_hit_margin
    return returns >= threshold, returns <= -threshold


def constrain(current: np.ndarray, target: np.ndarray, day: TradeDay) -> np.ndarray:
    """Feasible weights closest to ``target`` given locks and the liquidity cap.

    Args:
        current: Drifted weights before trading (sum at most 1).
        target: Desired weights.
        day: Returns, bands, and cap weights on the trade day.

    Returns:
        Weights that respect the constraints and sum to at most 1.
    """
    limit_up, limit_down = limit_locks(day)
    capped = np.where(
        target > current, np.minimum(target, np.maximum(day.cap_weights, current)), target
    )
    feasible = np.where(limit_up & (capped > current), current, capped)
    feasible = np.where(limit_down & (feasible < current), current, feasible)
    excess = feasible.sum() - 1.0
    increases = np.clip(feasible - current, 0.0, None)
    if excess > 0 and increases.sum() > 0:
        scale = max(0.0, 1.0 - excess / increases.sum())
        feasible = np.where(feasible > current, current + increases * scale, feasible)
    return feasible
