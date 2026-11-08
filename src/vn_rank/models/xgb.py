"""XGBoost ranking model for the exploratory extension (protocol Section 12).

It mirrors :mod:`vn_rank.models.gbm` so that only the library differs: L2 regression on the
per-date rank of the forward return, histogram trees grown leaf-wise (``lossguide``), the
same fixed learning rate, round cap, and early-stopping patience, and a refit for the
tuned number of rounds. Missing feature ranks stay NaN; XGBoost routes them natively.
Row and column subsampling make a fit depend on the seed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from vn_rank.config import LightGBMConfig, SearchRange
from vn_rank.models.base import TARGET_COLUMN, RankingModel
from vn_rank.models.gbm import ROUNDS_KEY, GBMRun

# Search space fixed in protocol Section 12, mirroring the LightGBM ranges.
SEARCH_SPACE = {
    "max_leaves": SearchRange(low=8, high=128, log=True, integer=True),
    "min_child_weight": SearchRange(low=100, high=5000, log=True, integer=True),
    "alpha": SearchRange(low=1e-3, high=10.0, log=True, integer=False),
    "lambda": SearchRange(low=1e-3, high=100.0, log=True, integer=False),
    "colsample_bytree": SearchRange(low=0.5, high=1.0, log=False, integer=False),
    "subsample": SearchRange(low=0.5, high=1.0, log=False, integer=False),
}
_INTEGER_PARAMS = frozenset({"max_leaves"})


class XGBoostModel(RankingModel):
    """XGBoost regressor on feature ranks.

    Args:
        features: Feature rank columns.
        run: Tuned parameters, seed, and optional fixed number of rounds.
        fixed: Learning rate, round cap, and early-stopping patience (shared with LightGBM).
        lockbox_start: First lockbox date.
    """

    def __init__(
        self, features: Sequence[str], run: GBMRun, fixed: LightGBMConfig, lockbox_start: dt.date
    ) -> None:
        """Store settings; the booster is created by ``fit``."""
        super().__init__(features, lockbox_start)
        self.run, self.fixed = run, fixed
        self.booster: xgb.Booster | None = None
        self.best_iteration = 0

    def _params(self) -> dict[str, Any]:
        tuned = {k: int(v) if k in _INTEGER_PARAMS else v for k, v in self.run.params.items()}
        return {
            "objective": "reg:squarederror",
            "eta": self.fixed.learning_rate,
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "max_depth": 0,
            "seed": self.run.seed,
            "verbosity": 0,
            **tuned,
        }

    def _matrix(self, frame: pd.DataFrame) -> xgb.DMatrix:
        label = frame[TARGET_COLUMN].to_numpy(dtype=float) if TARGET_COLUMN in frame else None
        return xgb.DMatrix(frame[self.features].to_numpy(dtype=float), label=label)

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        train_matrix = self._matrix(train)
        if valid is None:
            rounds = self.run.num_rounds or self.fixed.max_boost_rounds
            self.booster = xgb.train(self._params(), train_matrix, num_boost_round=rounds)
            self.best_iteration = rounds
            return
        self.booster = xgb.train(
            self._params(),
            train_matrix,
            num_boost_round=self.fixed.max_boost_rounds,
            evals=[(self._matrix(valid), "valid")],
            early_stopping_rounds=self.fixed.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration = int(self.booster.best_iteration) + 1

    def fitted_attributes(self) -> dict[str, float]:
        """The number of rounds a refit on the whole window should use."""
        return {ROUNDS_KEY: float(self.best_iteration)}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return predicted target ranks using the best iteration.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.booster is None:
            raise RuntimeError("XGBoostModel.predict called before fit")
        matrix = xgb.DMatrix(frame[self.features].to_numpy(dtype=float))
        scores = self.booster.predict(matrix, iteration_range=(0, self.best_iteration))
        return np.asarray(scores, dtype=float)
