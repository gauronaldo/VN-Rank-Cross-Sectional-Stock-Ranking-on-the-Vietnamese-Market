"""Tests for typed configuration loading, validation, and round-tripping."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any

import pytest
import yaml

from vn_rank.config import Config, ConfigError, config_from_dict, config_to_dict, load_config


def _set(raw: dict[str, Any], dotted: str, value: object) -> None:
    *parents, leaf = dotted.split(".")
    node = raw
    for key in parents:
        node = node[key]
    node[leaf] = value


def test_default_config_loads_with_typed_fields(default_config_path: Path) -> None:
    config = load_config(default_config_path)
    assert isinstance(config, Config)
    assert config.paths.raw_dir == Path("data/raw")
    assert config.validation.lockbox_start == dt.date(2025, 10, 1)
    assert config.features.return_windows == (1, 5, 20, 60, 120)
    assert config.label.horizon_days == 5
    assert config.backtest.costs.selling_tax == pytest.approx(0.001)
    assert config.universe.min_traded_value_vnd == pytest.approx(1e9)


def test_round_trip_through_dict(default_config_path: Path) -> None:
    config = load_config(default_config_path)
    assert config_from_dict(config_to_dict(config)) == config


def test_round_trip_through_yaml_text(
    default_config_path: Path, raw_config: dict[str, Any]
) -> None:
    plain = config_to_dict(load_config(default_config_path))
    assert plain == raw_config
    assert yaml.safe_load(yaml.safe_dump(plain)) == plain


def test_config_is_immutable(default_config_path: Path) -> None:
    config = load_config(default_config_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.label.horizon_days = 10  # type: ignore[misc]


def test_decided_values_are_accepted(raw_config: dict[str, Any]) -> None:
    _set(raw_config, "universe.min_traded_value_vnd", 1_000_000_000)
    _set(raw_config, "backtest.costs.brokerage_fee_per_side", 0.0015)
    config = config_from_dict(raw_config)
    assert config.universe.min_traded_value_vnd == pytest.approx(1e9)
    assert isinstance(config.universe.min_traded_value_vnd, float)


def test_zero_embargo_is_allowed(raw_config: dict[str, Any]) -> None:
    _set(raw_config, "validation.embargo_days", 0)
    assert config_from_dict(raw_config).validation.embargo_days == 0


@pytest.mark.parametrize(
    ("dotted", "value", "match"),
    [
        ("label.unexpected", 1, "unknown keys"),
        ("label.horizon_days", "5", "expected int"),
        ("label.horizon_days", True, "expected int"),
        ("label.horizon_days", 0, "horizon_days must be > 0"),
        ("label.target", "classification", "label.target"),
        ("features.return_windows", [1, -5], "return_windows must be > 0"),
        ("features.return_windows", 5, "expected a list"),
        ("features.macd_fast", 30, "macd_fast must be < macd_slow"),
        ("validation.embargo_days", -1, "embargo_days must be >= 0"),
        ("validation.min_stocks_per_date", 0, "min_stocks_per_date must be > 0"),
        ("validation.lockbox_start", "2025-10-01", "expected date"),
        ("validation.lockbox_start", dt.date(2025, 9, 30), "lockbox_start"),
        ("validation.walk_forward_start", dt.date(2019, 1, 1), "train_start"),
        ("data.history_start", dt.date(2020, 1, 1), "history_start"),
        ("data.max_rows_per_request", 0, "max_rows_per_request"),
        ("backtest.costs.selling_tax", 1.5, "selling_tax must be in"),
        ("backtest.costs.slippage_per_side", -0.001, "slippage_per_side must be in"),
        ("backtest.top_quantile", 0.0, "top_quantile"),
        ("backtest.price_limits", {"HOSE": 0.07}, "price_limits keys"),
        ("universe.min_traded_value_vnd", -1, "must be > 0"),
        ("models.seeds", [1, 1], "seeds"),
        ("models.seeds", [], "seeds"),
        ("evaluation.n_quantiles", 1, "n_quantiles"),
        ("models.missing_rank_fill", 1.5, "missing_rank_fill"),
        ("models.momentum_window", 30, "baseline windows must be features"),
        ("evaluation.tripwire_fold_ic", 0.0, "tripwire_fold_ic must be > 0"),
        ("evaluation.confidence_level", 1.0, "confidence_level"),
        ("backtest.portfolio_value_vnd", 0.0, "portfolio_value_vnd"),
        ("backtest.limit_hit_margin", 0.2, "limit_hit_margin"),
        ("backtest.turnover_band", 0.05, "turnover_band"),
        ("ablation.slippages", [], "must not be empty"),
        ("ablation.brokerage_fees", [1.5], "cost rates"),
        ("ablation.liquidity_thresholds_vnd", [-1.0], "thresholds"),
        ("ablation.shap_sample_rows", 0, "shap_sample_rows"),
        ("models.tuning.n_trials", 0, "n_trials must be > 0"),
        ("models.lightgbm.learning_rate", 0.0, "learning_rate must be > 0"),
        ("models.tuning.lightgbm.lambda_l1.low", 20.0, "low must be < high"),
        ("models.tuning.elastic_net.alpha.low", 0.0, "log scale needs low > 0"),
        ("models.tuning.lightgbm.num_leaves.low", 8.5, "integer bounds must be whole"),
        ("models.tuning.lightgbm.num_leaves.log", "yes", "expected bool"),
        ("paths", "data", "expected a mapping"),
    ],
)
def test_invalid_values_are_rejected(
    raw_config: dict[str, Any], dotted: str, value: object, match: str
) -> None:
    _set(raw_config, dotted, value)
    with pytest.raises(ConfigError, match=match):
        config_from_dict(raw_config)


def test_missing_key_is_rejected(raw_config: dict[str, Any]) -> None:
    del raw_config["validation"]["lockbox_start"]
    with pytest.raises(ConfigError, match="missing keys \\['lockbox_start'\\]"):
        config_from_dict(raw_config)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")
