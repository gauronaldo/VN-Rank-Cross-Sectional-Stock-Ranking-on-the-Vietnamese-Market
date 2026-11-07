"""Tests for the Phase 6 experiment helpers: variant rows, cost grid, and SHAP."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from tests.backtest.test_runner import SETTINGS, predictions, synthetic_market
from tests.validation.test_walk_forward import config as walk_forward_config  # noqa: F401
from tests.validation.test_walk_forward import synthetic_panel
from vn_rank import ablation
from vn_rank.backtest import runner
from vn_rank.config import Config, load_config
from vn_rank.validation import splits


@pytest.fixture
def context(default_config_path: Path) -> ablation.Context:
    base = load_config(default_config_path)
    grid = dataclasses.replace(base.ablation, brokerage_fees=(0.0, 0.003), slippages=(0.0, 0.003))
    market, _ = synthetic_market()
    _, entry = predictions(market)
    return ablation.Context(dataclasses.replace(base, ablation=grid), market, entry, SETTINGS)


def test_variant_row_against_itself_has_zero_difference(context: ablation.Context) -> None:
    preds, _ = predictions(context.market)
    oracle = preds[preds["model"] == "lightgbm"]
    reference_ic = runner.seed_averaged_ic(oracle)["lightgbm"]
    row = ablation.variant_row(oracle, reference_ic, context)
    assert row["mean_ic"] == pytest.approx(1.0)
    assert row["diff_vs_all"] == pytest.approx(0.0)
    assert row["sharpe"] > 1
    assert row["stocks_per_date"] == 40


def test_cost_grid_sharpe_falls_as_costs_rise(context: ablation.Context) -> None:
    preds, _ = predictions(context.market)
    grid = ablation.cost_grid(preds[preds["model"] == "lightgbm"], context)
    assert len(grid) == 4
    cheapest = grid.loc[grid["round_trip_cost"].idxmin(), "sharpe"]
    dearest = grid.loc[grid["round_trip_cost"].idxmax(), "sharpe"]
    assert cheapest > dearest
    assert grid["round_trip_cost"].max() == pytest.approx(2 * 0.006 + 0.001)


def test_shap_figure_explains_the_last_fold(walk_forward_config: Config) -> None:  # noqa: F811
    samples = splits.development_samples(
        synthetic_panel(walk_forward_config), walk_forward_config.validation
    )
    rule = dataclasses.replace(walk_forward_config.ablation, shap_sample_rows=300)
    figure = ablation.shap_figure(samples, dataclasses.replace(walk_forward_config, ablation=rule))
    assert isinstance(figure, Figure)
    plt.close(figure)
