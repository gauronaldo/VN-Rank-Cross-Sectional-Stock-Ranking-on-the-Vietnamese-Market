"""Moving-block bootstrap confidence intervals (protocol 8).

A resample of a daily series of length ``n`` concatenates blocks of ``block`` consecutive
observations whose start points are drawn uniformly (with replacement) from the
``n - block + 1`` possible starts, then truncates to ``n``. Blocks keep the serial
dependence that overlapping 5-day labels create. The interval is the percentile
interval of the statistic over ``n_resamples`` resamples; the generator is seeded.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

Statistic = Callable[[np.ndarray], np.ndarray]  # rows are resamples -> one value per row


@dataclasses.dataclass(frozen=True)
class BootstrapSettings:
    """Block length, number of resamples, confidence level, and seed."""

    block: int
    n_resamples: int
    level: float
    seed: int


def block_indices(n: int, block: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Index matrix (``n_resamples`` x ``n``) of moving-block resamples."""
    block = min(block, n)
    n_blocks = -(-n // block)  # ceiling division
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    return (starts[:, :, None] + offsets).reshape(n_resamples, -1)[:, :n]


def bootstrap_ci(
    values: np.ndarray, statistic: Statistic, settings: BootstrapSettings
) -> tuple[float, float]:
    """Percentile confidence interval of ``statistic`` under the moving-block bootstrap.

    Args:
        values: Daily observations in time order (NaN not allowed).
        statistic: Maps a (resamples x n) matrix to one value per resample.
        settings: Block length, resamples, level, and seed.

    Returns:
        ``(low, high)`` bounds of the interval.

    Raises:
        ValueError: If ``values`` contains NaN or has fewer than two observations.
    """
    if len(values) < 2 or np.isnan(values).any():  # noqa: PLR2004
        raise ValueError("bootstrap needs at least two observations and no NaN")
    rng = np.random.default_rng(settings.seed)
    samples = values[block_indices(len(values), settings.block, settings.n_resamples, rng)]
    stats = statistic(samples)
    tail = (1.0 - settings.level) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(low), float(high)


def alpha_beta_ci(
    strategy: np.ndarray, benchmark: np.ndarray, settings: BootstrapSettings, trading_days: int
) -> dict[str, float]:
    """Annualized alpha and beta of ``strategy`` on ``benchmark`` with a bootstrap CI.

    Daily OLS ``strategy = a + b * benchmark``; alpha is ``a * trading_days``. Both series
    are resampled with the same block indices, so their dependence is kept.

    Returns:
        ``alpha``, ``alpha_ci_low``, ``alpha_ci_high``, and ``beta``.

    Raises:
        ValueError: If the series differ in length, contain NaN, or are shorter than 2.
    """
    if len(strategy) != len(benchmark) or len(strategy) < 2:  # noqa: PLR2004
        raise ValueError("alpha_beta_ci needs two series of the same length >= 2")
    if np.isnan(strategy).any() or np.isnan(benchmark).any():
        raise ValueError("alpha_beta_ci does not accept NaN")
    rng = np.random.default_rng(settings.seed)
    idx = block_indices(len(strategy), settings.block, settings.n_resamples, rng)
    ys, xs = strategy[idx], benchmark[idx]
    x_dev = xs - xs.mean(axis=1, keepdims=True)
    betas = (x_dev * (ys - ys.mean(axis=1, keepdims=True))).sum(axis=1) / (x_dev**2).sum(axis=1)
    alphas = (ys.mean(axis=1) - betas * xs.mean(axis=1)) * trading_days
    beta = float(np.cov(strategy, benchmark)[0, 1] / np.var(benchmark, ddof=1))
    tail = (1.0 - settings.level) / 2.0
    low, high = np.quantile(alphas, [tail, 1.0 - tail])
    alpha = float((strategy.mean() - beta * benchmark.mean()) * trading_days)
    return {"alpha": alpha, "alpha_ci_low": float(low), "alpha_ci_high": float(high), "beta": beta}


def mean_statistic(samples: np.ndarray) -> np.ndarray:
    """Row means (for mean Rank IC or a mean difference)."""
    return np.asarray(samples.mean(axis=1))


def sharpe_statistic(trading_days: int) -> Statistic:
    """Annualized Sharpe ratio of each row (risk-free rate 0)."""

    def statistic(samples: np.ndarray) -> np.ndarray:
        return np.asarray(
            samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(trading_days)
        )

    return statistic
