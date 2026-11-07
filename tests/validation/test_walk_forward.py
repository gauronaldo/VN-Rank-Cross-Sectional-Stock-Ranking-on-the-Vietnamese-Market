"""Tests for the walk-forward runner, results table, tripwire, and the train command."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from vn_rank.cli import main
from vn_rank.config import Config, config_from_dict, config_to_dict, load_config
from vn_rank.evaluation.ic import SignalSettings
from vn_rank.features.builder import feature_groups
from vn_rank.models.base import feature_columns
from vn_rank.validation import splits, walk_forward

CALENDAR = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2021-12-31"))
N_SYMBOLS = 60


@pytest.fixture
def config(default_config_path: Path) -> Config:
    """Two 6-month folds in 2021, a small tuning budget, and two LightGBM seeds."""
    base = load_config(default_config_path)
    validation = dataclasses.replace(
        base.validation,
        train_start=dt.date(2020, 1, 1),
        walk_forward_start=dt.date(2021, 1, 1),
        walk_forward_end=dt.date(2021, 12, 31),
        lockbox_start=dt.date(2022, 1, 1),
        inner_validation_months=3,
    )
    models = dataclasses.replace(
        base.models,
        seeds=(0, 1),
        lightgbm=dataclasses.replace(base.models.lightgbm, max_boost_rounds=40),
        tuning=dataclasses.replace(base.models.tuning, n_trials=2),
    )
    return dataclasses.replace(base, validation=validation, models=models)


def synthetic_panel(config: Config) -> pd.DataFrame:
    """Feature ranks are noise except ``rank_ret_20``, which drives the forward return."""
    rng = np.random.default_rng(0)
    exit_ = pd.Series(CALENDAR, index=CALENDAR).shift(-6)
    dates = np.repeat(CALENDAR, N_SYMBOLS)
    n = len(dates)
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": np.tile([f"S{i:02d}" for i in range(N_SYMBOLS)], len(CALENDAR)),
            "exit_date": exit_.loc[dates].to_numpy(),
            **{c: rng.random(n) for c in feature_columns(feature_groups(config.features))},
        }
    )
    panel["forward_return"] = 0.05 * panel["rank_ret_20"] + rng.normal(0, 0.05, n)
    panel["target"] = panel.groupby("date")["forward_return"].rank(pct=True)
    late = panel["exit_date"].isna()
    panel.loc[late, ["forward_return", "target"]] = np.nan
    return panel


def test_run_walk_forward_scores_every_model_on_every_fold(config: Config) -> None:
    samples = splits.development_samples(synthetic_panel(config), config.validation)
    records: list[walk_forward.FoldRecord] = []
    specs = walk_forward.model_specs(config)
    predictions = walk_forward.run_walk_forward(
        samples, specs, config, lambda record, trials: records.append(record)
    )
    assert [s.name for s in specs] == ["momentum", "reversal", "elastic_net", "lightgbm"]
    assert len(records) == 8
    seeds = predictions.groupby("model")["seed"].nunique().to_dict()
    assert seeds == {"momentum": 1, "reversal": 1, "elastic_net": 1, "lightgbm": 2}
    by_model = {r.model: r for r in records if r.fold == "2021-07/2021-12"}
    assert by_model["momentum"].n_trials == 0
    assert by_model["lightgbm"].n_trials == config.models.tuning.n_trials
    assert "num_rounds" in by_model["lightgbm"].params
    assert by_model["momentum"].test_mean_ic[config.project.seed] > 0.2
    assert predictions["date"].min() >= pd.Timestamp(config.validation.walk_forward_start)


def _toy_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=20)
    rows = []
    for model, seeds in (("a", [0]), ("b", [0, 1])):
        for seed in seeds:
            returns = rng.normal(size=(20, 30))
            score = returns + rng.normal(scale=2.0, size=(20, 30))
            rows.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(dates, 30),
                        "symbol": np.tile([f"S{i}" for i in range(30)], 20),
                        "forward_return": returns.ravel(),
                        "score": score.ravel(),
                        "model": model,
                        "fold": np.where(np.repeat(dates, 30) < dates[10], "f1", "f2"),
                        "seed": seed,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_results_table_averages_over_seeds() -> None:
    table = walk_forward.results_table(_toy_predictions(), SignalSettings(5, 5))
    assert set(zip(table["model"], table["fold"], strict=True)) == {
        (m, f) for m in "ab" for f in ("all", "f1", "f2")
    }
    pooled = table.set_index(["model", "fold"])
    assert pooled.loc[("a", "all"), "n_seeds"] == 1
    assert np.isnan(pooled.loc[("a", "all"), "mean_ic_seed_std"])
    assert pooled.loc[("b", "all"), "n_seeds"] == 2
    text = walk_forward.render_results_markdown(table, ["# Title"], 4)
    assert "## All test folds pooled" in text
    assert "| fold | a | b |" in text


def test_tripwire_flags_high_ic() -> None:
    table = pd.DataFrame(
        {"model": ["m", "m", "m"], "fold": ["all", "f1", "f2"], "mean_ic": [0.11, 0.05, 0.21]}
    )
    assert walk_forward.tripwire_breaches(table, 0.10, 0.20) == [
        "m all: mean Rank IC 0.1100",
        "m f2: mean Rank IC 0.2100",
    ]
    assert walk_forward.tripwire_breaches(table, 0.5, 0.5) == []


def _write_config(config: Config, tmp_path: Path) -> Path:
    raw: dict[str, Any] = config_to_dict(config)
    raw["paths"].update(
        processed_dir=str(tmp_path / "processed"),
        reports_dir=str(tmp_path / "reports"),
        mlruns_dir=str(tmp_path / "mlruns"),
    )
    config_from_dict(raw)  # still valid
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_train_command_end_to_end_trips_on_planted_signal(config: Config, tmp_path: Path) -> None:
    """The synthetic signal (Rank IC about 0.27) is far above the tripwire: exit code 2."""
    path = _write_config(config, tmp_path)
    (tmp_path / "processed").mkdir()
    synthetic_panel(config).to_parquet(tmp_path / "processed" / "panel.parquet")
    assert main(["--config", str(path), "train"]) == 2
    report = (tmp_path / "reports" / "signal_results.md").read_text(encoding="utf-8")
    assert "Optuna configurations tried: 8." in report
    assert (tmp_path / "processed" / "predictions.parquet").exists()
    assert (tmp_path / "mlruns" / "mlflow.db").exists()
