"""Tests for the exploratory low-volatility comparison and the alpha-beta interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn_rank import exploratory
from vn_rank.evaluation import bootstrap

SETTINGS = bootstrap.BootstrapSettings(block=10, n_resamples=500, level=0.95, seed=0)


def test_alpha_beta_recovers_a_known_line() -> None:
    rng = np.random.default_rng(1)
    bench = rng.normal(0, 0.01, 1000)
    strategy = 0.0002 + 0.5 * bench + rng.normal(0, 0.001, 1000)
    result = bootstrap.alpha_beta_ci(strategy, bench, SETTINGS, 252)
    assert result["beta"] == pytest.approx(0.5, abs=0.01)
    assert result["alpha"] == pytest.approx(0.0002 * 252, abs=0.01)
    assert result["alpha_ci_low"] < result["alpha"] < result["alpha_ci_high"]


def test_alpha_beta_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap.alpha_beta_ci(np.zeros(3), np.zeros(4), SETTINGS, 252)
    with pytest.raises(ValueError, match="NaN"):
        bootstrap.alpha_beta_ci(np.array([0.1, np.nan]), np.zeros(2), SETTINGS, 252)


def test_low_volatility_predictions_use_the_baseline_rows() -> None:
    columns = ["date", "symbol", "forward_return", "score", "model", "seed", "fold"]
    base = pd.DataFrame(
        {"date": pd.Timestamp("2024-01-02"), "symbol": ["A", "B"], "forward_return": [0.1, -0.1],
         "score": [0.0, 0.0], "model": "momentum", "seed": 42, "fold": "f"}
    )[columns]  # fmt: skip
    lightgbm = base.assign(model="lightgbm", seed=0, score=[0.6, 0.4])
    panel = base[["date", "symbol"]].assign(rank_vol_60=[0.9, 0.1], rank_atr_14=[0.2, 0.8])
    out = exploratory.low_volatility_predictions(pd.concat([base, lightgbm]), panel)
    assert set(out["model"]) == {"momentum", "lightgbm", "lowvol_vol60", "lowvol_atr14"}
    vol = out[out["model"] == "lowvol_vol60"].set_index("symbol")["score"]
    assert vol.to_dict() == {"A": -0.9, "B": -0.1}  # the calmer stock ranks higher
    assert list(out.columns) == columns
