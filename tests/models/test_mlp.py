"""Tests for the PyTorch MLP ranking model."""

from __future__ import annotations

import numpy as np
import pytest

from tests.models.test_models import FEATURES, LOCKBOX, synthetic_samples
from vn_rank.config import MLPConfig
from vn_rank.models import mlp, tuning

SETTINGS = mlp.MLPSettings(
    MLPConfig(batch_size=256, max_epochs=15, patience=3), missing_rank_fill=0.5
)
PARAMS = {
    "hidden_units": 16,
    "n_layers": 2,
    "dropout": 0.1,
    "weight_decay": 1e-4,
    "learning_rate": 3e-3,
}


def test_network_shape_follows_params() -> None:
    net = mlp.build_network(15, {**PARAMS, "n_layers": 3, "hidden_units": 8})
    linear_layers = [m for m in net if isinstance(m, mlp.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear_layers] == [
        (15, 8), (8, 8), (8, 8), (8, 1),
    ]  # fmt: skip


def test_early_stopping_learns_the_signal_and_records_epochs() -> None:
    train, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    model = mlp.MLPModel(FEATURES, mlp.MLPRun(PARAMS, seed=0), SETTINGS, LOCKBOX)
    model.fit(train, valid)
    epochs = model.fitted_attributes()[mlp.EPOCHS_KEY]
    assert 1 <= epochs <= SETTINGS.fixed.max_epochs
    assert tuning.mean_rank_ic(model, valid) > 0.3


def test_fit_is_deterministic_for_a_seed() -> None:
    train, valid = synthetic_samples(seed=1), synthetic_samples(seed=2)
    run = mlp.MLPRun(PARAMS, seed=3, num_epochs=4)
    first = mlp.MLPModel(FEATURES, run, SETTINGS, LOCKBOX).fit(train).predict(valid)
    second = mlp.MLPModel(FEATURES, run, SETTINGS, LOCKBOX).fit(train).predict(valid)
    other = mlp.MLPModel(FEATURES, mlp.MLPRun(PARAMS, 4, 4), SETTINGS, LOCKBOX)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other.fit(train).predict(valid))


def test_predict_needs_fit_and_no_target() -> None:
    train = synthetic_samples(seed=1)
    model = mlp.MLPModel(FEATURES, mlp.MLPRun(PARAMS, 0, 2), SETTINGS, LOCKBOX)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(train)
    model.fit(train)
    scores = model.predict(train.drop(columns="target").assign(rank_ret_20=np.nan))
    assert np.isfinite(scores).all()
