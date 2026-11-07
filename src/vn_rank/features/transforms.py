"""Cross-sectional transforms on wide frames (dates x symbols).

Ranks and z-scores are computed **within each date** (row-wise) and only over the stocks
that are in the universe on that date. Nothing is computed across dates, so a transform on
day ``t`` uses only the values of day ``t``.
"""

from __future__ import annotations

import pandas as pd

SINGLE_STOCK_RANK = 0.5  # the rank of a date with a single valid stock


def cross_sectional_rank(values: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Rank values within each date among masked, non-missing entries, scaled to [0, 1].

    The lowest value gets 0 and the highest 1, i.e. ``(rank - 1) / (n - 1)`` with average
    ranks for ties; a date with one valid stock gets 0.5. Entries outside ``mask`` or
    missing are NaN.

    Args:
        values: Wide frame of raw values.
        mask: Boolean frame of the same shape; ``True`` marks universe members.
    """
    masked = values.where(mask)
    ranks = masked.rank(axis=1, method="average")
    count = masked.notna().sum(axis=1)
    scaled = ranks.sub(1).div((count - 1).where(count > 1), axis=0)
    single = ranks.notna() & count.eq(1).to_numpy()[:, None]
    return scaled.mask(single, SINGLE_STOCK_RANK)


def cross_sectional_zscore(values: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Standardize within each date among masked entries: ``(x - mean) / std`` (ddof 1).

    NaN outside ``mask`` and on dates where the standard deviation is zero or undefined.
    """
    masked = values.where(mask)
    std = masked.std(axis=1)
    return masked.sub(masked.mean(axis=1), axis=0).div(std.where(std > 0), axis=0)
