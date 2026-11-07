"""Assemble the processed panel: data treatment, calendar alignment, features, labels.

Steps (pure; I/O lives in the CLI):

1. :func:`repair_ohlc` applies data-treatment rules 1 and 2 (``docs/protocol.md``, Section 2);
2. :func:`align_to_calendar` applies rule 5: wide frames on the VNINDEX calendar, missing
   sessions after a symbol's first row become no-trade bars;
3. :func:`compute_feature_groups` builds the 15 raw features in their 5 ablation groups;
4. :func:`build_panel` adds universe eligibility, per-date feature ranks, the forward return
   and its per-date target, and returns one row per eligible (date, symbol).

Leakage notes: features, ranks, and eligibility on ``t`` use data up to and including ``t``
(the truncation test checks this). The only forward-looking inputs are the label and
``alive`` (sessions up to a symbol's last real row), used for labels only. The caller passes
prices that end before ``lockbox_start`` during development.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from vn_rank.config import Config, FeaturesConfig
from vn_rank.data import universe
from vn_rank.features import indicators as ind
from vn_rank.features import labels
from vn_rank.features.transforms import cross_sectional_rank

PRICE_COLUMNS = ("open", "high", "low", "close")
RANK_PREFIX = "rank_"
KEY_COLUMNS = ("date", "symbol")


class DuplicateRowsError(ValueError):
    """Raised when the raw panel has more than one row per (symbol, date)."""


@dataclasses.dataclass(frozen=True)
class MarketFrames:
    """Calendar-aligned wide frames (sessions x symbols), NaN before a symbol's first row.

    ``alive`` is ``True`` up to a symbol's last real row; it looks forward and is used for
    labels only.
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    alive: pd.DataFrame


def repair_ohlc(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply data-treatment rules 1 and 2 to a long OHLCV panel; ``close`` is never changed.

    No-trade rows and non-positive ``open``/``high``/``low`` take the value of ``close``;
    then ``high`` becomes the row maximum and ``low`` the row minimum of the four prices.
    """
    repaired = panel.copy()
    no_trade = repaired["volume"].eq(0)
    for column in PRICE_COLUMNS[:-1]:
        bad = no_trade | repaired[column].le(0)
        repaired[column] = repaired[column].mask(bad, repaired["close"])
    prices = repaired[list(PRICE_COLUMNS)]
    repaired["high"], repaired["low"] = prices.max(axis=1), prices.min(axis=1)
    return repaired


def align_to_calendar(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> MarketFrames:
    """Pivot a repaired long panel to the calendar and fill missing sessions (rule 5).

    Raises:
        DuplicateRowsError: If a (symbol, date) pair appears more than once.
    """
    if panel.duplicated(list(KEY_COLUMNS)).any():
        raise DuplicateRowsError("raw panel has duplicate (symbol, date) rows")
    wide = {
        c: panel.pivot(index="date", columns="symbol", values=c).reindex(calendar)
        for c in (*PRICE_COLUMNS, "volume")
    }
    real = wide["close"].notna()
    started = real.cummax()
    close = wide["close"].ffill().where(started)
    prices = {c: wide[c].where(real, close) for c in PRICE_COLUMNS[:-1]}
    return MarketFrames(
        **prices,
        close=close,
        volume=wide["volume"].where(real, 0.0).where(started),
        alive=real.iloc[::-1].cummax().iloc[::-1],
    )


def compute_feature_groups(
    frames: MarketFrames, rule: FeaturesConfig
) -> dict[str, dict[str, pd.DataFrame]]:
    """Return the raw features by ablation group: ``{group: {feature: frame}}``."""
    return {
        "momentum": _momentum(frames, rule),
        "volatility": _volatility(frames, rule),
        "technical": _technical(frames, rule),
        "liquidity": _liquidity(frames, rule),
        "price_position": {
            f"close_to_high_{rule.high_window}": ind.close_to_rolling_high(
                frames.close, frames.high, rule.high_window
            ),
        },
    }


def _momentum(frames: MarketFrames, rule: FeaturesConfig) -> dict[str, pd.DataFrame]:
    return {f"ret_{w}": ind.trailing_return(frames.close, w) for w in rule.return_windows}


def _volatility(frames: MarketFrames, rule: FeaturesConfig) -> dict[str, pd.DataFrame]:
    close = frames.close
    result = {f"vol_{w}": ind.rolling_volatility(close, w) for w in rule.volatility_windows}
    atr = ind.normalized_atr(frames.high, frames.low, close, rule.atr_window)
    return {**result, f"atr_{rule.atr_window}": atr}


def _technical(frames: MarketFrames, rule: FeaturesConfig) -> dict[str, pd.DataFrame]:
    close = frames.close
    return {
        f"rsi_{rule.rsi_window}": ind.rsi(close, rule.rsi_window),
        "macd_hist": ind.macd_histogram(close, rule.macd_fast, rule.macd_slow, rule.macd_signal),
        "bb_pctb": ind.bollinger_percent_b(close, rule.bollinger_window, rule.bollinger_num_std),
    }


def _liquidity(frames: MarketFrames, rule: FeaturesConfig) -> dict[str, pd.DataFrame]:
    value = universe.traded_value_vnd(frames.close, frames.volume)
    tv_window, ratio_window, amihud_window = (
        rule.traded_value_window,
        rule.volume_ratio_window,
        rule.amihud_window,
    )
    return {
        f"log_tv_{tv_window}": ind.log_median_traded_value(value, tv_window),
        f"vol_ratio_{ratio_window}": ind.volume_ratio(frames.volume, ratio_window),
        f"amihud_{amihud_window}": ind.amihud_illiquidity(frames.close, value, amihud_window),
    }


def feature_groups(rule: FeaturesConfig) -> dict[str, list[str]]:
    """Return the feature names of each ablation group (computed on a 1 x 1 dummy frame)."""
    dummy = pd.DataFrame([[1.0]])
    frames = MarketFrames(dummy, dummy, dummy, dummy, dummy, dummy.astype(bool))
    return {g: list(f) for g, f in compute_feature_groups(frames, rule).items()}


def build_panel(
    raw: pd.DataFrame, calendar: pd.DatetimeIndex, exchange: pd.Series, config: Config
) -> pd.DataFrame:
    """Build the processed panel: one row per eligible (date, symbol).

    Args:
        raw: Long OHLCV panel (``storage.OHLCV_COLUMNS``).
        calendar: VNINDEX sessions to align on.
        exchange: Exchange code per symbol (today's exchange).
        config: Universe, feature, and label settings.

    Returns:
        Columns ``date, symbol, exchange, entry_date, exit_date, close,
        median_traded_value``, the raw features, their ``rank_`` versions,
        ``forward_return``, and ``target`` (NaN where the label has not matured).
    """
    frames = align_to_calendar(repair_ohlc(raw), calendar)
    eligible = universe.eligibility(frames.close, frames.volume, config.universe)
    value = universe.traded_value_vnd(frames.close, frames.volume)
    features = {
        name: frame
        for group in compute_feature_groups(frames, config.features).values()
        for name, frame in group.items()
    }
    forward = labels.forward_return(frames.close, frames.alive, config.label)
    columns = {
        "close": frames.close,
        "median_traded_value": universe.median_traded_value(
            value, config.universe.traded_value_window
        ),
        **features,
        **{RANK_PREFIX + n: cross_sectional_rank(f, eligible) for n, f in features.items()},
        "forward_return": forward,
        "target": labels.training_target(forward, eligible & forward.notna(), config.label),
    }
    panel = _to_long(columns, eligible)
    panel.insert(2, "exchange", panel["symbol"].map(exchange))
    panel.insert(
        3, "entry_date", panel["date"].map(labels.label_entry_dates(calendar, config.label))
    )
    panel.insert(4, "exit_date", panel["date"].map(labels.label_exit_dates(calendar, config.label)))
    return panel


def _to_long(columns: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    rows, cols = np.nonzero(mask.to_numpy())
    data = {name: frame.to_numpy()[rows, cols] for name, frame in columns.items()}
    keys = {"date": mask.index[rows], "symbol": mask.columns[cols].astype(str)}
    return pd.DataFrame({**keys, **data})
