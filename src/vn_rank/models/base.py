"""Common interface of every ranking model.

A model maps the per-date feature ranks of a sample to a score; only the ordering of
scores within a date matters. :meth:`RankingModel.fit` runs the lockbox guard on every
frame it receives before any learning happens.
"""

from __future__ import annotations

import abc
import datetime as dt
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd

from vn_rank.validation.splits import ensure_before_lockbox

TARGET_COLUMN = "target"
FEATURE_PREFIX = "rank_"


def feature_columns(groups: dict[str, list[str]]) -> list[str]:
    """Rank column names of all features, in group order."""
    return [FEATURE_PREFIX + name for names in groups.values() for name in names]


class RankingModel(abc.ABC):
    """Base class: lockbox-guarded ``fit`` and a ``predict`` that returns scores.

    Args:
        features: Feature columns used as inputs.
        lockbox_start: First lockbox date; ``fit`` refuses data reaching it.
    """

    def __init__(self, features: Sequence[str], lockbox_start: dt.date) -> None:
        """Store the feature columns and the lockbox boundary."""
        self.features = list(features)
        self.lockbox_start = lockbox_start

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> Self:
        """Fit on ``train``; ``valid`` (if given) is used for early stopping only.

        Raises:
            LockboxViolationError: If either frame has a signal or label exit on or
                after ``lockbox_start``.
        """
        for frame in (train, valid):
            if frame is not None:
                ensure_before_lockbox(frame["date"], frame["exit_date"], self.lockbox_start)
        self._fit(train, valid)
        return self

    def fitted_attributes(self) -> dict[str, float]:
        """Values learned during ``fit`` that a refit should reuse (none by default)."""
        return {}

    @abc.abstractmethod
    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None: ...

    @abc.abstractmethod
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return one score per row of ``frame``; higher means a better expected rank."""
