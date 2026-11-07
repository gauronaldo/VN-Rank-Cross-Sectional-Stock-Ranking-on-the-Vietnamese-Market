"""Tests for the model interface, baselines, Elastic Net, LightGBM, and Optuna tuning."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vn_rank.config import LightGBMConfig, SearchRange
from vn_rank.models import base, baselines, gbm, linear, tuning
from vn_rank.utils.seeds import set_global_seed
from vn_rank.validation.splits import LockboxViolationError

LOCKBOX = dt.date(2030, 1, 1)
FEATURES = ["rank_ret_20", "rank_ret_5", "rank_noise"]
FIXED = LightGBMConfig(
    learning_rate=0.1, max_boost_rounds=200, early_stopping_rounds=10, bagging_freq=1
)
GBM_PARAMS = {
    "num_leaves": 8,
    "min_child_samples": 20,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}
EN_SETTINGS = linear.ElasticNetSettings(missing_rank_fill=0.5, max_iter=10_000)


def synthetic_samples(n_dates: int = 40, n_symbols: int = 80, seed: int = 0) -> pd.DataFrame:
    """Forward return driven by ``ret_20`` momentum; ``ret_5`` and noise carry nothing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, n_symbols),
            "symbol": np.tile([f"S{i}" for i in range(n_symbols)], n_dates),
            "momentum": rng.normal(size=n_dates * n_symbols),
            "rank_ret_5": rng.random(n_dates * n_symbols),
            "rank_noise": rng.random(n_dates * n_symbols),
        }
    )
    frame["exit_date"] = frame["date"] + pd.offsets.BDay(6)
    frame["forward_return"] = 0.02 * frame["momentum"] + rng.normal(0, 0.02, len(frame))
    by_date = frame.groupby("date")
    frame["rank_ret_20"] = by_date["momentum"].rank(pct=True)
    frame["target"] = by_date["forward_return"].rank(pct=True)
    return frame.drop(columns="momentum")


def test_feature_columns_follow_group_order() -> None:
    groups = {"momentum": ["ret_1", "ret_5"], "technical": ["rsi_14"]}
    assert base.feature_columns(groups) == ["rank_ret_1", "rank_ret_5", "rank_rsi_14"]


def test_baselines_score_by_signed_rank() -> None:
    frame = synthetic_samples(n_dates=2)
    momentum = baselines.momentum_baseline(20, LOCKBOX).fit(frame)
    reversal = baselines.reversal_baseline(5, LOCKBOX).fit(frame)
    assert np.array_equal(momentum.predict(frame), frame["rank_ret_20"].to_numpy())
    assert np.array_equal(reversal.predict(frame), -frame["rank_ret_5"].to_numpy())


def test_fit_refuses_lockbox_data() -> None:
    frame = synthetic_samples(n_dates=2)
    model = baselines.momentum_baseline(20, frame["date"].max().date())
    with pytest.raises(LockboxViolationError):
        model.fit(frame)
    with pytest.raises(LockboxViolationError):
        baselines.momentum_baseline(20, LOCKBOX).fit(frame.head(5), frame.assign(exit_date=pd.NaT))


def test_elastic_net_finds_the_informative_feature() -> None:
    train, test = synthetic_samples(seed=1), synthetic_samples(seed=2)
    params = {"alpha": 1e-4, "l1_ratio": 0.5}
    model = linear.ElasticNetModel(FEATURES, params, EN_SETTINGS, LOCKBOX).fit(train)
    coefs = model.coefficients()
    assert coefs["rank_ret_20"] > 10 * max(abs(coefs["rank_ret_5"]), abs(coefs["rank_noise"]))
    assert tuning.mean_rank_ic(model, test) > 0.3


def test_elastic_net_fills_missing_ranks_with_median() -> None:
    train = synthetic_samples(seed=1)
    model = linear.ElasticNetModel(FEATURES, {"alpha": 1e-4, "l1_ratio": 0.5}, EN_SETTINGS, LOCKBOX)
    model.fit(train)
    row = train.head(1).assign(rank_ret_20=np.nan)
    filled = row.assign(rank_ret_20=0.5)
    assert model.predict(row) == pytest.approx(model.predict(filled))


def test_elastic_net_heavy_penalty_gives_constant_scores() -> None:
    train = synthetic_samples(seed=1)
    model = linear.ElasticNetModel(FEATURES, {"alpha": 1.0, "l1_ratio": 1.0}, EN_SETTINGS, LOCKBOX)
    model.fit(train)
    assert all(c == 0 for c in model.coefficients().values())
    assert tuning.mean_rank_ic(model, train) == tuning.NO_SIGNAL


def test_lightgbm_early_stopping_and_seeded_refit() -> None:
    train, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    run = gbm.GBMRun(GBM_PARAMS, seed=7)
    tuned = gbm.LightGBMModel(FEATURES, run, FIXED, LOCKBOX).fit(train, valid)
    rounds = int(tuned.fitted_attributes()[gbm.ROUNDS_KEY])
    assert 1 <= rounds < FIXED.max_boost_rounds
    refit = gbm.GBMRun(GBM_PARAMS, seed=7, num_rounds=rounds)
    first = gbm.LightGBMModel(FEATURES, refit, FIXED, LOCKBOX).fit(train).predict(valid)
    second = gbm.LightGBMModel(FEATURES, refit, FIXED, LOCKBOX).fit(train).predict(valid)
    assert np.array_equal(first, second)
    assert tuning.mean_rank_ic(tuned, valid) > 0.3


def test_lightgbm_seed_changes_subsampled_fit() -> None:
    train = synthetic_samples(seed=1)
    runs = [gbm.GBMRun(GBM_PARAMS, seed=s, num_rounds=30) for s in (0, 1)]
    preds = [gbm.LightGBMModel(FEATURES, r, FIXED, LOCKBOX).fit(train).predict(train) for r in runs]
    assert not np.array_equal(preds[0], preds[1])


def test_lightgbm_handles_missing_ranks_and_requires_fit() -> None:
    train = synthetic_samples(seed=1)
    model = gbm.LightGBMModel(FEATURES, gbm.GBMRun(GBM_PARAMS, 0, 20), FIXED, LOCKBOX)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(train)
    train.loc[train.index[:50], "rank_ret_20"] = np.nan
    assert np.isfinite(model.fit(train).predict(train)).all()


SPACE = {
    "alpha": SearchRange(low=1e-5, high=1e-1, log=True, integer=False),
    "l1_ratio": SearchRange(low=0.05, high=1.0, log=False, integer=False),
}


def _en_factory(params: dict[str, float]) -> base.RankingModel:
    return linear.ElasticNetModel(FEATURES, params, EN_SETTINGS, LOCKBOX)


def test_tuning_is_reproducible_and_stays_in_bounds() -> None:
    inner, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    settings = tuning.TuningSettings(SPACE, n_trials=4, seed=3)
    first = tuning.tune(_en_factory, inner, valid, settings)
    second = tuning.tune(_en_factory, inner, valid, settings)
    assert first.params == second.params
    assert first.best_value == second.best_value
    assert len(first.trials) == 4
    for name, dim in SPACE.items():
        assert dim.low <= first.params[name] <= dim.high


def test_tuning_records_lightgbm_rounds() -> None:
    inner, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    space = {"num_leaves": SearchRange(low=4, high=16, log=True, integer=True)}

    def factory(params: dict[str, float]) -> base.RankingModel:
        run = gbm.GBMRun({**GBM_PARAMS, **params}, seed=0)
        return gbm.LightGBMModel(FEATURES, run, FIXED, LOCKBOX)

    result = tuning.tune(factory, inner, valid, tuning.TuningSettings(space, 2, 0))
    assert isinstance(result.params["num_leaves"], int)
    assert result.attributes[gbm.ROUNDS_KEY] >= 1


def test_set_global_seed_makes_numpy_reproducible() -> None:
    set_global_seed(5)
    first = np.random.rand(3)
    set_global_seed(5)
    assert np.array_equal(first, np.random.rand(3))
