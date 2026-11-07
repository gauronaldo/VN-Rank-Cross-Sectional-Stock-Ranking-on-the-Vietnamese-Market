"""Tests for walk-forward folds, purge and embargo alignment, and the lockbox guard."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vn_rank.config import Config, ValidationConfig, load_config
from vn_rank.validation import splits

CALENDAR = pd.DatetimeIndex(pd.bdate_range("2019-11-01", "2025-12-31"))
N_SYMBOLS = 60


@pytest.fixture
def config(default_config_path: Path) -> Config:
    return load_config(default_config_path)


@pytest.fixture
def rule(config: Config) -> ValidationConfig:
    return config.validation


def synthetic_panel(config: Config, calendar: pd.DatetimeIndex = CALENDAR) -> pd.DataFrame:
    """Every symbol on every session; labels are missing after the last exit session."""
    label = config.label
    entry = pd.Series(calendar, index=calendar).shift(-label.entry_lag_days)
    exit_ = pd.Series(calendar, index=calendar).shift(-(label.entry_lag_days + label.horizon_days))
    dates = np.repeat(calendar, N_SYMBOLS)
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": np.tile([f"S{i:02d}" for i in range(N_SYMBOLS)], len(calendar)),
            "entry_date": entry.loc[dates].to_numpy(),
            "exit_date": exit_.loc[dates].to_numpy(),
            "target": np.random.default_rng(0).random(len(dates)),
        }
    )
    panel.loc[panel["exit_date"].isna(), "target"] = np.nan
    return panel


def dev_panel(config: Config) -> pd.DataFrame:
    """Synthetic panel built only from sessions before the lockbox, like the real one."""
    lockbox = pd.Timestamp(config.validation.lockbox_start)
    return synthetic_panel(config, CALENDAR[: CALENDAR.searchsorted(lockbox)])


def session_gap(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return int(CALENDAR.get_loc(later)) - int(CALENDAR.get_loc(earlier))


def test_default_folds_cover_walk_forward_period(rule: ValidationConfig) -> None:
    folds = splits.walk_forward_folds(rule)
    assert [f.name for f in folds] == [
        "2022-01/2022-06", "2022-07/2022-12", "2023-01/2023-06", "2023-07/2023-12",
        "2024-01/2024-06", "2024-07/2024-12", "2025-01/2025-06", "2025-07/2025-09",
    ]  # fmt: skip
    assert folds[0].test_start == pd.Timestamp(rule.walk_forward_start)
    assert folds[-1].test_end == pd.Timestamp(rule.walk_forward_end)
    for before, after in itertools.pairwise(folds):
        assert after.test_start == before.test_end + pd.Timedelta(days=1)


def test_purge_and_embargo_alignment(config: Config, rule: ValidationConfig) -> None:
    """Mandatory test 2: no training label window overlaps its test block; embargo holds."""
    samples = splits.development_samples(dev_panel(config), rule)
    label = config.label
    gap = label.entry_lag_days + label.horizon_days + 1 + rule.embargo_days
    for fold in splits.walk_forward_folds(rule):
        split = splits.split_fold(samples, fold, rule)
        train, test = split.train, split.test
        assert (train["exit_date"] < fold.test_start).all()
        overlaps = (train["entry_date"] <= fold.test_end) & (train["exit_date"] >= fold.test_start)
        assert not overlaps.any()
        assert session_gap(test["date"].min(), train["date"].max()) == gap
        assert test["date"].between(fold.test_start, fold.test_end).all()


def test_training_window_expands(config: Config, rule: ValidationConfig) -> None:
    samples = splits.development_samples(dev_panel(config), rule)
    folds = splits.walk_forward_folds(rule)
    sizes = [len(splits.split_fold(samples, f, rule).train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] > 0


def test_zero_embargo_keeps_every_purged_date(config: Config, rule: ValidationConfig) -> None:
    samples = splits.development_samples(dev_panel(config), rule)
    block = pd.Timestamp("2023-01-02")
    mask = splits.purged_train_mask(samples, block, 0)
    assert samples.loc[mask, "exit_date"].max() == CALENDAR[: CALENDAR.searchsorted(block)][-1]


def test_test_block_drops_labels_ending_in_lockbox(config: Config, rule: ValidationConfig) -> None:
    lockbox = pd.Timestamp(rule.lockbox_start)
    samples = synthetic_panel(config).dropna(subset=["target"])
    samples = samples.loc[samples["date"] < lockbox]  # the last labels exit in October
    fold = splits.walk_forward_folds(rule)[-1]
    test = splits.split_fold(samples, fold, rule).test
    assert test["exit_date"].max() < lockbox
    last_signal = CALENDAR[: CALENDAR.searchsorted(lockbox)][-1]
    assert session_gap(last_signal, test["date"].max()) == config.label.horizon_days + 1


def test_development_samples_refuse_lockbox_labels(config: Config, rule: ValidationConfig) -> None:
    panel = synthetic_panel(config)
    panel = panel.loc[panel["date"] < pd.Timestamp(rule.lockbox_start)]
    with pytest.raises(splits.LockboxViolationError, match="label exits"):
        splits.development_samples(panel, rule)


def test_development_samples_drop_thin_dates(config: Config, rule: ValidationConfig) -> None:
    panel = dev_panel(config)
    thin_day = pd.Timestamp("2020-03-02")
    panel = panel.loc[(panel["date"] != thin_day) | (panel["symbol"] < "S10")]
    samples = splits.development_samples(panel, rule)
    assert thin_day not in set(samples["date"])
    assert samples["date"].min() >= pd.Timestamp(rule.train_start)
    assert samples["target"].notna().all()


@pytest.mark.parametrize(
    ("signal", "exit_", "match"),
    [
        ("2025-10-01", "2025-09-30", "1 signal dates"),
        ("2025-09-22", "2025-10-01", "1 label exits"),
        ("2025-09-22", None, "1 label exits"),
    ],
)
def test_lockbox_guard_raises(
    rule: ValidationConfig, signal: str, exit_: str | None, match: str
) -> None:
    signals = pd.Series(pd.to_datetime(["2025-01-02", signal]))
    exits = pd.Series(pd.to_datetime(["2025-01-10", exit_]))
    with pytest.raises(splits.LockboxViolationError, match=match):
        splits.ensure_before_lockbox(signals, exits, rule.lockbox_start)


def test_lockbox_guard_accepts_development_dates(rule: ValidationConfig) -> None:
    signals = pd.Series(pd.to_datetime(["2025-09-22"]))
    exits = pd.Series(pd.to_datetime(["2025-09-30"]))
    splits.ensure_before_lockbox(signals, exits, rule.lockbox_start)


def test_inner_validation_is_purged_and_embargoed(config: Config, rule: ValidationConfig) -> None:
    samples = splits.development_samples(dev_panel(config), rule)
    train = splits.split_fold(samples, splits.walk_forward_folds(rule)[2], rule).train
    inner, valid = splits.inner_validation_split(train, rule)
    window_end = train["date"].max()
    assert valid["date"].max() == window_end
    assert valid["date"].min() > window_end - pd.DateOffset(months=rule.inner_validation_months)
    assert (inner["exit_date"] < valid["date"].min()).all()
    label = config.label
    gap = label.entry_lag_days + label.horizon_days + 1 + rule.embargo_days
    assert session_gap(valid["date"].min(), inner["date"].max()) == gap


def test_embargo_longer_than_history_gives_empty_train(config: Config) -> None:
    rule = dataclasses.replace(config.validation, embargo_days=10_000)
    samples = splits.development_samples(dev_panel(config), rule)
    assert not splits.purged_train_mask(samples, pd.Timestamp("2022-01-03"), 10_000).any()
