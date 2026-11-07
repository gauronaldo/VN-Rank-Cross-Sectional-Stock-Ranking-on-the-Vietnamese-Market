"""Rank IC, IC IR, Newey-West t-stat, quantile returns, signal autocorrelation.

"IC" always means Rank IC: the per-date Spearman correlation between predicted scores
and realized forward returns across the stocks scored on that date. Every function takes
a long frame with columns ``date``, ``symbol``, a score column, and ``forward_return``.

Conventions:

- A date on which the model gives every stock the same score has a Rank IC of 0: the
  model offers no ranking, which counts as no skill rather than being dropped (dropping
  would flatter a model that abstains on hard periods). Such dates have no quantiles.
- IC IR is ``mean / std`` of the daily Rank IC series (``ddof = 1``), per day, not
  annualized.
- The t-statistic of the mean Rank IC uses a Newey-West (Bartlett kernel) standard error
  with lag ``h - 1``, because 5-day labels on consecutive dates overlap.
- Quantile returns average the equal-weight forward return of each score quantile within
  a date, then across dates; quantile 1 has the lowest scores.
- Signal autocorrelation is the Spearman correlation of scores between consecutive
  rebalance dates (every ``h`` sessions), over the symbols scored on both dates.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

RETURN_COLUMN = "forward_return"
SCORE_COLUMN = "score"


@dataclasses.dataclass(frozen=True)
class SignalSettings:
    """Label horizon (sets the Newey-West lag and rebalance step) and quantile count."""

    horizon_days: int
    n_quantiles: int


def spearman(left: pd.Series, right: pd.Series) -> float:
    """Spearman correlation: Pearson correlation of average ranks (NaN pairs dropped).

    Returns NaN with fewer than two pairs or when either side is constant.
    """
    pairs = pd.DataFrame({"left": left, "right": right}).dropna()
    ranks = pairs.rank()
    if len(ranks) < 2 or ranks["left"].nunique() < 2 or ranks["right"].nunique() < 2:  # noqa: PLR2004
        return float("nan")
    return float(np.corrcoef(ranks["left"], ranks["right"])[0, 1])


def _date_rank_ic(group: pd.DataFrame, score: str) -> float:
    value = spearman(group[score], group[RETURN_COLUMN])
    pairs = group[[score, RETURN_COLUMN]].dropna()
    abstains = pairs[score].nunique() == 1 and pairs[RETURN_COLUMN].nunique() > 1
    return 0.0 if abstains else value


def daily_rank_ic(frame: pd.DataFrame, score: str = SCORE_COLUMN) -> pd.Series:
    """Rank IC per signal date, indexed by date.

    A date with constant scores (but varying returns) has Rank IC 0; dates that are
    otherwise undefined (fewer than two stocks, constant returns) are dropped.
    """
    ic = {date: _date_rank_ic(group, score) for date, group in frame.groupby("date", sort=True)}
    return pd.Series(ic, dtype=float, name="rank_ic").rename_axis("date").dropna()


def newey_west_tstat(values: pd.Series, lag: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett kernel) standard error.

    The long-run variance is ``g0 + 2 * sum_{k=1..lag} (1 - k / (lag + 1)) * gk``, where
    ``gk`` is the lag-``k`` autocovariance with divisor ``n`` (no small-sample correction).
    Returns NaN with fewer than two values or a zero long-run variance.
    """
    x = values.to_numpy(dtype=float)
    n = len(x)
    if n < 2:  # noqa: PLR2004
        return float("nan")
    deviations = x - x.mean()
    long_run = float(deviations @ deviations) / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1)
        long_run += 2.0 * weight * float(deviations[k:] @ deviations[:-k]) / n
    return float(x.mean() / np.sqrt(long_run / n)) if long_run > 0 else float("nan")


def quantile_returns(frame: pd.DataFrame, n_quantiles: int, score: str = SCORE_COLUMN) -> pd.Series:
    """Mean forward return per score quantile (1 = lowest scores), averaged over dates.

    Within a date, stocks are ordered by score (ties broken by row order) and cut into
    ``n_quantiles`` groups of near-equal size. Dates with constant scores have no order
    and are skipped; a quantile with no data is NaN.
    """
    valid = frame.dropna(subset=[score, RETURN_COLUMN])
    valid = valid.loc[valid.groupby("date")[score].transform("nunique") > 1]
    pct = valid.groupby("date")[score].rank(method="first", pct=True)
    bucket = np.ceil(pct * n_quantiles).clip(1, n_quantiles).astype(int)
    per_date = valid.groupby([valid["date"], bucket.rename("quantile")])[RETURN_COLUMN].mean()
    means = per_date.groupby(level="quantile").mean()
    return means.reindex(pd.RangeIndex(1, n_quantiles + 1, name="quantile"))


def rebalance_dates(dates: pd.Series, every: int) -> pd.DatetimeIndex:
    """Every ``every``-th distinct signal date, starting from the first one."""
    distinct = pd.DatetimeIndex(dates.unique()).sort_values()
    return distinct[::every]


def signal_autocorrelation(frame: pd.DataFrame, every: int, score: str = SCORE_COLUMN) -> pd.Series:
    """Spearman correlation of scores between consecutive rebalance dates.

    Indexed by the later rebalance date of each pair.
    """
    keep = frame["date"].isin(rebalance_dates(frame["date"], every))
    wide = frame.loc[keep].pivot_table(index="date", columns="symbol", values=score)
    pairs = {wide.index[i]: spearman(wide.iloc[i - 1], wide.iloc[i]) for i in range(1, len(wide))}
    return pd.Series(pairs, dtype=float, name="signal_autocorr").rename_axis("date")


def signal_summary(
    frame: pd.DataFrame, settings: SignalSettings, score: str = SCORE_COLUMN
) -> dict[str, float]:
    """All signal-level metrics of one scored frame, as a flat mapping.

    Keys: ``mean_ic``, ``ic_std``, ``ic_ir``, ``t_nw``, ``n_dates``, ``q1`` ...
    ``qN``, ``top_minus_bottom``, and ``signal_autocorr`` (mean over rebalance pairs).
    """
    ic = daily_rank_ic(frame, score)
    quantiles = quantile_returns(frame, settings.n_quantiles, score)
    autocorr = signal_autocorrelation(frame, settings.horizon_days, score)
    std = float(ic.std())
    summary = {
        "mean_ic": float(ic.mean()),
        "ic_std": std,
        "ic_ir": float(ic.mean()) / std if std > 0 else float("nan"),
        "t_nw": newey_west_tstat(ic, settings.horizon_days - 1),
        "n_dates": float(len(ic)),
    }
    labels = quantiles.index.astype(int)
    summary.update({f"q{q}": float(r) for q, r in zip(labels, quantiles, strict=True)})
    summary["top_minus_bottom"] = float(quantiles.iloc[-1] - quantiles.iloc[0])
    summary["signal_autocorr"] = float(autocorr.mean())
    return summary
