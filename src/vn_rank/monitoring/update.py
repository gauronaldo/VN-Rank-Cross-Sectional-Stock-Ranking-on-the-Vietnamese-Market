"""Daily scoring with the frozen model and realized-IC tracking (pure functions).

The model is the LightGBM ensemble frozen by the lockbox run (one booster per seed,
trained on all data before ``lockbox_start``); it is never retrained afterwards. Each
update scores the newest session's universe, keeps the top quantile as the buy list for
the next close, and appends every score to a log. Once a logged date's label has matured
(``h`` sessions after entry), its realized Rank IC is computed and compared with the
walk-forward and lockbox Rank IC.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from vn_rank.evaluation.ic import RETURN_COLUMN, SCORE_COLUMN, daily_rank_ic

LOG_COLUMNS = ["date", "symbol", SCORE_COLUMN]


def ensemble_scores(
    frame: pd.DataFrame, boosters: Sequence[lgb.Booster], features: list[str]
) -> np.ndarray:
    """Mean prediction of the per-seed boosters (missing ranks stay NaN, as in training)."""
    values = frame[features].to_numpy(dtype=float)
    stacked = np.vstack([np.asarray(b.predict(values), dtype=float) for b in boosters])
    return np.asarray(stacked.mean(axis=0))


def score_date(
    panel: pd.DataFrame, date: pd.Timestamp, boosters: Sequence[lgb.Booster], features: list[str]
) -> pd.DataFrame:
    """Scores of every universe member on ``date`` (labels are not needed)."""
    rows = panel.loc[panel["date"] == date]
    return rows[["date", "symbol", "exchange"]].assign(
        score=ensemble_scores(rows, boosters, features)
    )


def top_list(scored: pd.DataFrame, top_quantile: float) -> pd.DataFrame:
    """The ``ceil(top_quantile * n)`` highest scores, ranked from 1, equal target weight."""
    n = math.ceil(top_quantile * len(scored))
    top = (
        scored.sort_values(SCORE_COLUMN, ascending=False, kind="stable")
        .head(n)
        .reset_index(drop=True)
    )
    return top.assign(rank=np.arange(1, n + 1), target_weight=1.0 / n)


def append_scores(log: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Add a date's scores to the log, replacing any earlier scores for that date."""
    kept = log.loc[~log["date"].isin(scored["date"].unique())]
    return pd.concat([kept, scored[LOG_COLUMNS]], ignore_index=True).sort_values(["date", "symbol"])


def realized_ic(log: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Daily Rank IC of logged scores against matured forward returns."""
    matured = panel.loc[panel[RETURN_COLUMN].notna(), ["date", "symbol", RETURN_COLUMN]]
    merged = log.merge(matured, on=["date", "symbol"], how="inner")
    return daily_rank_ic(merged)


def plot_realized_ic(ic: pd.Series, references: dict[str, float], window: int) -> Figure:
    """Daily realized Rank IC, its rolling mean, and reference mean Rank IC lines."""
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(ic.index, ic.to_numpy(), color="lightgray", label="daily realized Rank IC")
    rolling = ic.rolling(window, min_periods=1).mean()
    axis.plot(rolling.index, rolling.to_numpy(), color="black", label=f"{window}-day rolling mean")
    for (name, value), style in zip(references.items(), ("--", ":", "-."), strict=False):
        axis.axhline(value, linestyle=style, color="tab:blue", label=f"{name}: {value:.3f}")
    axis.axhline(0.0, color="gray", linewidth=0.8)
    axis.set_title("Frozen LightGBM: realized Rank IC after the lockbox start")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    return figure
