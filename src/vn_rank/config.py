"""Typed configuration loading and validation.

The YAML file is mapped onto frozen dataclasses. Parsing is strict: unknown keys, missing
keys, and values of the wrong type raise :class:`ConfigError`. Semantic checks (date order,
positive windows, rate bounds) run after the dataclasses are built. The lockbox start date
lives here (``validation.lockbox_start``) and must come after the walk-forward period.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from vn_rank.utils.dataclass_parsing import SchemaError, build_dataclass, is_number, to_plain

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_TARGETS = ("rank", "zscore")
_NON_NEGATIVE_FIELDS = frozenset({"embargo_days"})
_MIN_QUANTILES = 2


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or violates a constraint."""


@dataclasses.dataclass(frozen=True)
class ProjectConfig:
    """Project-wide settings."""

    seed: int


@dataclasses.dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations, relative to the repository root."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    mlruns_dir: Path


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Data source and downloader settings."""

    source: str
    index_symbol: str
    exchanges: tuple[str, ...]
    history_start: dt.date
    request_interval_seconds: float
    max_rows_per_request: int
    max_retries: int
    retry_backoff_seconds: float


@dataclasses.dataclass(frozen=True)
class UniverseConfig:
    """Point-in-time eligibility rule; ``None`` threshold means not yet decided."""

    min_history_days: int
    traded_value_window: int
    min_traded_value_vnd: float | None


@dataclasses.dataclass(frozen=True)
class LabelConfig:
    """Forward-return label definition."""

    horizon_days: int
    entry_lag_days: int
    target: str


@dataclasses.dataclass(frozen=True)
class FeaturesConfig:
    """Indicator window lengths and parameters."""

    return_windows: tuple[int, ...]
    volatility_windows: tuple[int, ...]
    atr_window: int
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    bollinger_window: int
    bollinger_num_std: float
    traded_value_window: int
    volume_ratio_window: int
    amihud_window: int
    high_window: int


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    """Validation periods, walk-forward block sizes, and embargo."""

    train_start: dt.date
    walk_forward_start: dt.date
    walk_forward_end: dt.date
    lockbox_start: dt.date
    test_block_months: int
    inner_validation_months: int
    embargo_days: int


@dataclasses.dataclass(frozen=True)
class ModelsConfig:
    """Model-level settings shared by all models."""

    seeds: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Signal-evaluation settings."""

    n_quantiles: int


@dataclasses.dataclass(frozen=True)
class CostsConfig:
    """Trading costs as fractions of traded value; ``None`` means not yet decided."""

    brokerage_fee_per_side: float | None
    selling_tax: float
    slippage_per_side: float | None


@dataclasses.dataclass(frozen=True)
class BacktestConfig:
    """Portfolio construction, costs, and trading constraints."""

    top_quantile: float
    costs: CostsConfig
    price_limits: dict[str, float]
    position_cap_adv_window: int
    max_position_fraction_of_adv: float | None


@dataclasses.dataclass(frozen=True)
class Config:
    """Root configuration object."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    universe: UniverseConfig
    label: LabelConfig
    features: FeaturesConfig
    validation: ValidationConfig
    models: ModelsConfig
    evaluation: EvaluationConfig
    backtest: BacktestConfig


def load_config(path: Path) -> Config:
    """Read, parse, and validate a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is malformed or violates a constraint.
        FileNotFoundError: If ``path`` does not exist.
    """
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config_from_dict(raw)


def config_from_dict(raw: object) -> Config:
    """Build and validate a :class:`Config` from plain YAML-style data.

    Args:
        raw: Nested mappings as produced by ``yaml.safe_load``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: On unknown or missing keys, wrong types, or violated constraints.
    """
    try:
        config = build_dataclass(Config, raw, "config")
    except SchemaError as exc:
        raise ConfigError(str(exc)) from exc
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Check semantic constraints that types alone cannot express.

    Args:
        config: A structurally valid configuration.

    Raises:
        ConfigError: If a window is not positive, periods are out of order, a rate is out
            of bounds, or exchanges and price limits disagree.
    """
    for section in (config.universe, config.label, config.features, config.validation):
        _check_positive(section)
    _check_periods(config.data.history_start, config.validation)
    _check_rates(config.universe, config.backtest)
    _check_misc(config)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_positive(section: DataclassInstance) -> None:
    for name, value in dataclasses.asdict(section).items():
        values = value if isinstance(value, tuple) else (value,)
        numbers = [v for v in values if is_number(v)]
        if name in _NON_NEGATIVE_FIELDS:
            _require(all(v >= 0 for v in numbers), f"{name} must be >= 0, got {value}")
        else:
            _require(all(v > 0 for v in numbers), f"{name} must be > 0, got {value}")


def _check_periods(history_start: dt.date, periods: ValidationConfig) -> None:
    ordered = [
        history_start,
        periods.train_start,
        periods.walk_forward_start,
        periods.walk_forward_end,
        periods.lockbox_start,
    ]
    _require(
        all(a < b for a, b in itertools.pairwise(ordered)),
        "dates must satisfy history_start < train_start < walk_forward_start"
        f" < walk_forward_end < lockbox_start, got {[d.isoformat() for d in ordered]}",
    )


def _check_rates(universe: UniverseConfig, backtest: BacktestConfig) -> None:
    rates = {**dataclasses.asdict(backtest.costs), **backtest.price_limits}
    for name, rate in rates.items():
        _require(rate is None or 0 <= rate < 1, f"{name} must be in [0, 1), got {rate}")
    cap = backtest.max_position_fraction_of_adv
    _require(cap is None or 0 < cap <= 1, f"max_position_fraction_of_adv out of (0, 1]: {cap}")
    _require(0 < backtest.top_quantile <= 1, f"top_quantile out of (0, 1]: {backtest.top_quantile}")
    threshold = universe.min_traded_value_vnd
    _require(threshold is None or threshold > 0, f"min_traded_value_vnd must be > 0: {threshold}")


def _check_misc(config: Config) -> None:
    features, data = config.features, config.data
    _require(features.macd_fast < features.macd_slow, "macd_fast must be < macd_slow")
    _require(config.label.target in _TARGETS, f"label.target must be one of {_TARGETS}")
    _require(config.evaluation.n_quantiles >= _MIN_QUANTILES, "evaluation.n_quantiles must be >= 2")
    seeds = config.models.seeds
    _require(len(seeds) > 0 and len(set(seeds)) == len(seeds), "seeds must be non-empty, unique")
    _require(
        set(config.backtest.price_limits) == set(data.exchanges),
        "backtest.price_limits keys must equal data.exchanges",
    )
    _require(data.request_interval_seconds > 0, "request_interval_seconds must be > 0")
    _require(data.max_rows_per_request > 0, "max_rows_per_request must be > 0")
    _require(data.max_retries >= 0, "max_retries must be >= 0")


def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a configuration into plain data that ``yaml.safe_dump`` can write.

    Args:
        config: The configuration to convert.

    Returns:
        Nested dicts with paths as POSIX strings and tuples as lists; dates are kept as
        :class:`datetime.date`, which YAML round-trips natively.
    """
    plain: dict[str, Any] = to_plain(dataclasses.asdict(config))
    return plain
