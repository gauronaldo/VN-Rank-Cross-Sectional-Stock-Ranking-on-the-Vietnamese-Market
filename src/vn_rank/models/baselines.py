"""Non-learning baselines: 20-day momentum and 5-day reversal.

Each scores a stock by one feature rank, with a fixed sign, and learns nothing. The
equal-weight universe baseline has no ranking (every stock gets the same weight), so it
has no Rank IC; it is the primary portfolio benchmark in Phase 5.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from vn_rank.models.base import FEATURE_PREFIX, RankingModel


class SingleFeatureBaseline(RankingModel):
    """Score = ``sign * rank`` of one feature; ``fit`` only runs the lockbox guard."""

    def __init__(self, feature: str, sign: float, lockbox_start: dt.date) -> None:
        """Use the rank column of ``feature``, multiplied by ``sign`` (+1 or -1)."""
        super().__init__([FEATURE_PREFIX + feature], lockbox_start)
        self.sign = sign

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        return None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the signed feature rank (NaN where the rank is missing)."""
        return self.sign * frame[self.features[0]].to_numpy(dtype=float)


def momentum_baseline(window: int, lockbox_start: dt.date) -> SingleFeatureBaseline:
    """Past winners rank highest: score = rank of the ``window``-day return."""
    return SingleFeatureBaseline(f"ret_{window}", 1.0, lockbox_start)


def reversal_baseline(window: int, lockbox_start: dt.date) -> SingleFeatureBaseline:
    """Past losers rank highest: score = minus the rank of the ``window``-day return."""
    return SingleFeatureBaseline(f"ret_{window}", -1.0, lockbox_start)
