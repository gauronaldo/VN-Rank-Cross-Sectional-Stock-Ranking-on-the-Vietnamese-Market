"""Elastic Net ranking model (scikit-learn).

Inputs are the feature ranks centred at 0 (``rank - 0.5``); a missing rank is filled with
``missing_rank_fill`` (0.5, the median rank) before centring. The target is the per-date
rank of the forward return. Coordinate descent with cyclic selection is deterministic,
so the model runs once.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from vn_rank.models.base import TARGET_COLUMN, RankingModel

RANK_CENTRE = 0.5


@dataclasses.dataclass(frozen=True)
class ElasticNetSettings:
    """Fixed settings: missing-rank fill value and the solver's iteration cap."""

    missing_rank_fill: float
    max_iter: int


class ElasticNetModel(RankingModel):
    """Elastic Net on centred feature ranks.

    Args:
        features: Feature rank columns.
        params: ``alpha`` and ``l1_ratio`` (tuned).
        settings: Fixed settings.
        lockbox_start: First lockbox date.
    """

    def __init__(
        self,
        features: Sequence[str],
        params: dict[str, float],
        settings: ElasticNetSettings,
        lockbox_start: dt.date,
    ) -> None:
        """Build the unfitted scikit-learn estimator."""
        super().__init__(features, lockbox_start)
        self.settings = settings
        self.estimator = ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=settings.max_iter,
            selection="cyclic",
        )

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.features].fillna(self.settings.missing_rank_fill)
        return values.to_numpy(dtype=float) - RANK_CENTRE

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        self.estimator.fit(self._design(train), train[TARGET_COLUMN].to_numpy(dtype=float))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the linear prediction of the target rank."""
        return np.asarray(self.estimator.predict(self._design(frame)), dtype=float)

    def coefficients(self) -> dict[str, float]:
        """Fitted coefficient per feature (zero means the feature was dropped)."""
        return dict(zip(self.features, map(float, self.estimator.coef_), strict=True))
