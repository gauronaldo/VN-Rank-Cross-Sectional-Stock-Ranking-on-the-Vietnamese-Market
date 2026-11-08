"""Tests for the XGBoost ranking model used by the exploratory extension."""

from __future__ import annotations

import numpy as np
import pytest

from tests.models.test_models import FEATURES, FIXED, LOCKBOX, synthetic_samples
from vn_rank.models import tuning, xgb
from vn_rank.models.gbm import ROUNDS_KEY, GBMRun

PARAMS = {
    "max_leaves": 8,
    "min_child_weight": 20,
    "alpha": 0.0,
    "lambda": 1.0,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
}


def test_early_stopping_learns_the_signal_and_records_rounds() -> None:
    train, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    model = xgb.XGBoostModel(FEATURES, GBMRun(PARAMS, seed=7), FIXED, LOCKBOX).fit(train, valid)
    rounds = model.fitted_attributes()[ROUNDS_KEY]
    assert 1 <= rounds <= FIXED.max_boost_rounds
    assert tuning.mean_rank_ic(model, valid) > 0.3


def test_refit_is_deterministic_for_a_seed_and_depends_on_it() -> None:
    train, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    run = GBMRun(PARAMS, seed=3, num_rounds=20)
    first = xgb.XGBoostModel(FEATURES, run, FIXED, LOCKBOX).fit(train).predict(valid)
    second = xgb.XGBoostModel(FEATURES, run, FIXED, LOCKBOX).fit(train).predict(valid)
    other = xgb.XGBoostModel(FEATURES, GBMRun(PARAMS, 4, 20), FIXED, LOCKBOX).fit(train)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other.predict(valid))


def test_predict_needs_fit_handles_missing_ranks_and_no_target() -> None:
    train = synthetic_samples(seed=1)
    model = xgb.XGBoostModel(FEATURES, GBMRun(PARAMS, 0, 10), FIXED, LOCKBOX)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(train)
    model.fit(train)
    scored = train.drop(columns="target").assign(rank_ret_20=np.nan)
    assert np.isfinite(model.predict(scored)).all()


def test_search_space_mirrors_the_registered_ranges() -> None:
    assert set(xgb.SEARCH_SPACE) == set(PARAMS)
    assert xgb.SEARCH_SPACE["max_leaves"].integer
    assert (
        xgb.SEARCH_SPACE["min_child_weight"].low,
        xgb.SEARCH_SPACE["min_child_weight"].high,
    ) == (100, 5000)
