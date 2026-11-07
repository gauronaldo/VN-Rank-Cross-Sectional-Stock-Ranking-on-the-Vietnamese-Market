"""LightGBM ranking model, the pre-registered primary model.

L2 regression on the per-date rank of the forward return. Missing feature ranks stay
NaN; LightGBM routes them natively. With a validation frame, training stops early on the
validation L2 loss and records ``best_iteration``; without one, it trains for a fixed
``num_rounds`` (the tuned ``best_iteration``). ``deterministic`` mode makes a run
reproducible for a given seed and thread count; feature and bagging subsampling make
results depend on the seed, so the final fit runs with several seeds.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from vn_rank.config import LightGBMConfig
from vn_rank.models.base import TARGET_COLUMN, RankingModel

ROUNDS_KEY = "num_rounds"


@dataclasses.dataclass(frozen=True)
class GBMRun:
    """Tuned parameters, seed, and (for a refit) the number of boosting rounds."""

    params: dict[str, float]
    seed: int
    num_rounds: int | None = None


class LightGBMModel(RankingModel):
    """LightGBM regressor on feature ranks.

    Args:
        features: Feature rank columns.
        run: Tuned parameters, seed, and optional fixed number of rounds.
        fixed: Settings that are not tuned.
        lockbox_start: First lockbox date.
    """

    def __init__(
        self,
        features: Sequence[str],
        run: GBMRun,
        fixed: LightGBMConfig,
        lockbox_start: dt.date,
    ) -> None:
        """Store settings; the booster is created by ``fit``."""
        super().__init__(features, lockbox_start)
        self.run, self.fixed = run, fixed
        self.booster: lgb.Booster | None = None
        self.best_iteration = 0

    def _params(self) -> dict[str, Any]:
        integers = {"num_leaves", "min_child_samples"}
        tuned = {k: int(v) if k in integers else v for k, v in self.run.params.items()}
        return {
            "objective": "regression",
            "learning_rate": self.fixed.learning_rate,
            "bagging_freq": self.fixed.bagging_freq,
            "seed": self.run.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
            **tuned,
        }

    def _dataset(self, frame: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        return lgb.Dataset(
            frame[self.features].to_numpy(dtype=float),
            label=frame[TARGET_COLUMN].to_numpy(dtype=float),
            reference=reference,
            free_raw_data=True,
        )

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        train_set = self._dataset(train)
        if valid is None:
            rounds = self.run.num_rounds or self.fixed.max_boost_rounds
            self.booster = lgb.train(self._params(), train_set, num_boost_round=rounds)
            self.best_iteration = rounds
            return
        stop = lgb.early_stopping(self.fixed.early_stopping_rounds, verbose=False)
        self.booster = lgb.train(
            self._params(),
            train_set,
            num_boost_round=self.fixed.max_boost_rounds,
            valid_sets=[self._dataset(valid, train_set)],
            callbacks=[stop],
        )
        self.best_iteration = max(int(self.booster.best_iteration), 1)

    def fitted_attributes(self) -> dict[str, float]:
        """The number of rounds a refit on the whole window should use."""
        return {ROUNDS_KEY: float(self.best_iteration)}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return predicted target ranks using the best iteration.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.booster is None:
            raise RuntimeError("LightGBMModel.predict called before fit")
        values = frame[self.features].to_numpy(dtype=float)
        return np.asarray(self.booster.predict(values, num_iteration=self.best_iteration))
