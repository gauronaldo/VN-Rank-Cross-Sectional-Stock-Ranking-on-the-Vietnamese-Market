"""Purged walk-forward splits with embargo, and the lockbox date guard.

Every function works on the long development panel (one row per eligible date and
symbol) and its ``date`` and ``exit_date`` columns, where ``exit_date`` is the session
whose close ends the label (``t + 1 + h``).

- **Purge:** a training sample is dropped if its ``exit_date`` is on or after the first
  date of the block it is evaluated against.
- **Embargo:** a further ``embargo_days`` sessions of training dates are dropped before
  the purged train end.
- **Lockbox guard:** :func:`ensure_before_lockbox` raises if any signal or exit date is
  on or after ``lockbox_start``; it is called on every split, fit, and tuning objective
  (see docs/design_notes.md, "Lockbox guard").
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd

from vn_rank.config import ValidationConfig


class LockboxViolationError(RuntimeError):
    """Raised when development data reaches into the lockbox period."""


@dataclasses.dataclass(frozen=True)
class Fold:
    """One walk-forward test block; both ends are inclusive calendar dates."""

    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclasses.dataclass(frozen=True)
class FoldSplit:
    """Training and test samples of one fold."""

    fold: Fold
    train: pd.DataFrame
    test: pd.DataFrame


def ensure_before_lockbox(
    signal_dates: pd.Series, exit_dates: pd.Series, lockbox_start: dt.date
) -> None:
    """Refuse data whose signal or label exit is on or after the lockbox start.

    Uses an explicit ``raise`` rather than ``assert``, so ``python -O`` cannot disable it.

    Args:
        signal_dates: Signal session of each sample.
        exit_dates: Label exit session of each sample; a missing exit is a violation,
            because the label cannot be shown to end before the lockbox.
        lockbox_start: First date of the lockbox period.

    Raises:
        LockboxViolationError: If any date is on or after ``lockbox_start`` or missing.
    """
    boundary = pd.Timestamp(lockbox_start)
    late_signals = int((signal_dates >= boundary).sum())
    late_exits = int((exit_dates.isna() | (exit_dates >= boundary)).sum())
    if late_signals or late_exits:
        raise LockboxViolationError(
            f"{late_signals} signal dates and {late_exits} label exits are missing or on"
            f" or after lockbox_start {boundary.date()}"
        )


def walk_forward_folds(rule: ValidationConfig) -> list[Fold]:
    """Consecutive test blocks of ``test_block_months`` over the walk-forward period.

    The last block is cut at ``walk_forward_end`` (2025-07 to 2025-09 with the defaults).
    """
    folds = []
    start, end = pd.Timestamp(rule.walk_forward_start), pd.Timestamp(rule.walk_forward_end)
    while start <= end:
        stop = min(start + pd.DateOffset(months=rule.test_block_months) - pd.Timedelta(days=1), end)
        folds.append(Fold(f"{start:%Y-%m}/{stop:%Y-%m}", start, stop))
        start = stop + pd.Timedelta(days=1)
    return folds


def development_samples(panel: pd.DataFrame, rule: ValidationConfig) -> pd.DataFrame:
    """Samples usable for training and evaluation, before any fold split.

    Keeps signal dates from ``train_start`` on with at least ``min_stocks_per_date``
    eligible stocks (counted before dropping missing labels) and a known target, then
    checks the lockbox guard.

    Raises:
        LockboxViolationError: If a kept sample reaches the lockbox.
    """
    per_date = panel.groupby("date")["symbol"].transform("size")
    keep = (
        (panel["date"] >= pd.Timestamp(rule.train_start))
        & (per_date >= rule.min_stocks_per_date)
        & panel["target"].notna()
    )
    samples = panel.loc[keep].reset_index(drop=True)
    ensure_before_lockbox(samples["date"], samples["exit_date"], rule.lockbox_start)
    return samples


def purged_train_mask(
    samples: pd.DataFrame, block_start: pd.Timestamp, embargo_days: int
) -> pd.Series:
    """Rows usable for training against a block that starts on ``block_start``.

    A row is kept when its label exits before ``block_start`` (purge) and its signal date
    is not among the last ``embargo_days`` purged sessions (embargo).
    """
    purged = samples["exit_date"].notna() & (samples["exit_date"] < block_start)
    sessions = pd.Index(samples.loc[purged, "date"].unique()).sort_values()
    if embargo_days >= len(sessions):
        return pd.Series(False, index=samples.index)
    last_kept = pd.Timestamp(sessions[len(sessions) - 1 - embargo_days])
    keep: pd.Series = purged & (samples["date"] <= last_kept)
    return keep


def split_fold(samples: pd.DataFrame, fold: Fold, rule: ValidationConfig) -> FoldSplit:
    """Expanding-window training set and test set of one fold.

    Test rows are the signal dates inside the block whose label exits before
    ``lockbox_start``. Both sets pass the lockbox guard.
    """
    train = samples.loc[purged_train_mask(samples, fold.test_start, rule.embargo_days)]
    in_block = samples["date"].between(fold.test_start, fold.test_end)
    test = samples.loc[in_block & (samples["exit_date"] < pd.Timestamp(rule.lockbox_start))]
    for part in (train, test):
        ensure_before_lockbox(part["date"], part["exit_date"], rule.lockbox_start)
    return FoldSplit(fold, train.reset_index(drop=True), test.reset_index(drop=True))


def inner_validation_split(
    train: pd.DataFrame, rule: ValidationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training window into inner training and validation sets for tuning.

    Validation is the last ``inner_validation_months`` of the window; inner training is
    purged and embargoed against it exactly like a training set against a test block.

    Returns:
        ``(inner_train, validation)``.
    """
    window_end = train["date"].max()
    valid_start = (
        window_end - pd.DateOffset(months=rule.inner_validation_months) + pd.Timedelta(days=1)
    )
    valid = train.loc[train["date"] >= valid_start]
    inner = train.loc[purged_train_mask(train, valid_start, rule.embargo_days)]
    return inner.reset_index(drop=True), valid.reset_index(drop=True)
