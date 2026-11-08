"""Exploratory extension (I/O layer): literature features and XGBoost, protocol Section 12.

Pre-registered before this code was written. Walk-forward only (the lockbox is spent):

- four new price features (``beta_60``, ``ivol_60``, ``max_20``, ``skew_60``), ranked
  within each date among universe members like the original 15;
- three new runs under the unchanged protocol: LightGBM with 19 features, XGBoost with 15,
  XGBoost with 19; the original LightGBM predictions (15 features) are reused;
- two rules: minus the 60-day volatility rank and minus the ``ivol_60`` rank;
- decision rules H1-H3 on paired mean Rank IC differences (95% block bootstrap).

Windows and names are module constants, not config keys, so that the config hash in the
lockbox lock file stays valid.
"""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd

from vn_rank import exploratory, pipeline
from vn_rank.backtest import runner
from vn_rank.config import Config
from vn_rank.data import storage, universe
from vn_rank.evaluation import bootstrap
from vn_rank.evaluation.ic import SignalSettings, daily_rank_ic
from vn_rank.features import builder, labels
from vn_rank.features import indicators as ind
from vn_rank.features.transforms import cross_sectional_rank
from vn_rank.models import xgb
from vn_rank.models.base import FEATURE_PREFIX, RankingModel, feature_columns
from vn_rank.models.gbm import ROUNDS_KEY, GBMRun
from vn_rank.utils.logging import MlflowFoldLogger
from vn_rank.utils.markdown import markdown_table
from vn_rank.validation import splits, walk_forward

MARKET_WINDOW, MAX_WINDOW, SKEW_WINDOW = 60, 20, 60
NEW_FEATURES = ("beta_60", "ivol_60", "max_20", "skew_60")
RULES = {"lowvol_vol60": "rank_vol_60", "ivol_rule": "rank_ivol_60"}
PRIMARY = runner.PRIMARY_MODEL  # run 1: the original LightGBM predictions (15 features)
EXPERIMENT_NAME = "extension"
PREDICTIONS_FILE = "extension_predictions.parquet"
REPORT_STEM = "extension_features_xgboost"
# (name, run A, run B): the pre-registered paired difference is A minus B.
HYPOTHESES = (
    ("H1 features", "lightgbm_19", PRIMARY),
    ("H2 library", "xgboost_15", PRIMARY),
    ("H3 beats rule", "lightgbm_19", "lowvol_vol60"),
)
DESCRIPTIVE = (("XGBoost 19 vs LightGBM 15", "xgboost_19", PRIMARY),
               ("IVOL rule vs LightGBM 15", "ivol_rule", PRIMARY))  # fmt: skip
logger = logging.getLogger(__name__)


def new_feature_frames(
    frames: builder.MarketFrames, market_close: pd.Series
) -> dict[str, pd.DataFrame]:
    """The four pre-registered features as wide frames; each uses data up to ``t``."""
    beta, ivol = ind.rolling_market_moments(frames.close, market_close, MARKET_WINDOW)
    return {
        "beta_60": beta,
        "ivol_60": ivol,
        "max_20": ind.rolling_max_return(frames.close, MAX_WINDOW),
        "skew_60": ind.rolling_skewness(frames.close, SKEW_WINDOW),
    }


def extend_panel(
    panel: pd.DataFrame, frames: builder.MarketFrames, market_close: pd.Series, config: Config
) -> pd.DataFrame:
    """Add the per-date ranks of the new features to the development panel."""
    eligible = universe.eligibility(frames.close, frames.volume, config.universe)
    ranks = []
    for name, frame in new_feature_frames(frames, market_close).items():
        ranked = cross_sectional_rank(frame, eligible).where(eligible)
        ranks.append(pd.Series(ranked.stack(), name=FEATURE_PREFIX + name))
    extra = pd.concat(ranks, axis=1).rename_axis(["date", "symbol"]).reset_index()
    return panel.merge(extra, on=["date", "symbol"], how="left")


def extension_specs(config: Config) -> list[walk_forward.ModelSpec]:
    """LightGBM with 19 features, and XGBoost with 15 and with 19 features."""
    base = feature_columns(builder.feature_groups(config.features))
    extended = base + [FEATURE_PREFIX + name for name in NEW_FEATURES]
    lightgbm = next(s for s in walk_forward.model_specs(config, extended) if s.name == PRIMARY)
    fixed, lockbox, seeds = (
        config.models.lightgbm,
        config.validation.lockbox_start,
        config.models.seeds,
    )

    def xgb_spec(name: str, features: list[str]) -> walk_forward.ModelSpec:
        def factory(params: dict[str, float], seed: int) -> RankingModel:
            tuned, rounds = walk_forward.split_count(params, ROUNDS_KEY)
            return xgb.XGBoostModel(features, GBMRun(tuned, seed, rounds), fixed, lockbox)

        return walk_forward.ModelSpec(name, factory, xgb.SEARCH_SPACE, seeds)

    return [
        dataclasses.replace(lightgbm, name="lightgbm_19"),
        xgb_spec("xgboost_15", base),
        xgb_spec("xgboost_19", extended),
    ]


def rule_predictions(keys: pd.DataFrame, panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-seed predictions of each rule (minus a feature rank) on the given test rows."""
    rows = keys.merge(panel[["date", "symbol", *RULES.values()]], on=["date", "symbol"])
    return pd.concat(
        [rows.assign(score=-rows[c], model=name, seed=0)[columns] for name, c in RULES.items()],
        ignore_index=True,
    )


def paired_tests(
    daily_ic: pd.DataFrame,
    tests: tuple[tuple[str, str, str], ...],
    settings: runner.BacktestSettings,
) -> pd.DataFrame:
    """Mean of ``A - B`` daily Rank IC with its bootstrap CI and the decision at 95%."""
    rows = []
    for name, first, second in tests:
        diff = (daily_ic[first] - daily_ic[second]).dropna().to_numpy()
        low, high = bootstrap.bootstrap_ci(diff, bootstrap.mean_statistic, settings.bootstrap)
        verdict = "A better" if low > 0 else ("B better" if high < 0 else "no difference")
        rows.append({"test": name, "A": first, "B": second, "mean_A_minus_B": float(diff.mean()),
                     "ci_low": low, "ci_high": high, "result": verdict})  # fmt: skip
    return pd.DataFrame(rows)


def feature_ic(samples: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Univariate mean Rank IC of each new feature on the walk-forward test rows."""
    rows = samples.merge(keys[["date", "symbol"]], on=["date", "symbol"])
    means = {
        name: float(daily_rank_ic(rows.assign(score=rows[FEATURE_PREFIX + name])).mean())
        for name in NEW_FEATURES
    }
    return pd.DataFrame({"feature": list(means), "mean_rank_ic": list(means.values())})


def extension(config: Config) -> int:
    """Run the pre-registered extension once and write ``reports/extension_*.md``."""
    lockbox = pd.Timestamp(config.validation.lockbox_start)
    prices, processed = pipeline.load_prices(config, lockbox), config.paths.processed_dir
    frames = builder.align_to_calendar(builder.repair_ohlc(prices.raw), prices.calendar)
    panel = extend_panel(
        storage.read_parquet(processed / pipeline.PANEL_FILE), frames, prices.index_close, config
    )
    samples = splits.development_samples(panel, config.validation)
    original = storage.read_parquet(processed / pipeline.PREDICTIONS_FILE)
    info = pipeline.run_info(config)
    with MlflowFoldLogger(config.paths.mlruns_dir, EXPERIMENT_NAME, info) as log_fold:
        new_runs = walk_forward.run_walk_forward(samples, extension_specs(config), config, log_fold)
    keys = original.loc[
        original["model"] == "momentum", ["date", "symbol", "forward_return", "fold"]
    ]
    columns = list(original.columns)
    predictions = pd.concat(
        [
            original[original["model"] == PRIMARY],
            new_runs[columns],
            rule_predictions(keys, panel, columns),
        ],
        ignore_index=True,
    )
    storage.write_parquet(predictions, processed / PREDICTIONS_FILE)
    _report(config, predictions, (samples, keys, prices), (info, log_fold.n_trials))
    return 0


def _report(
    config: Config,
    predictions: pd.DataFrame,
    inputs: tuple[pd.DataFrame, pd.DataFrame, pipeline.Prices],
    provenance: tuple[dict[str, str], int],
) -> None:
    samples, keys, prices = inputs
    info, n_trials = provenance
    settings = dataclasses.replace(pipeline.backtest_settings(config), turnover_band=None)
    entry = labels.label_entry_dates(prices.calendar, config.label)
    market = pipeline.market_data(config, prices)
    report = runner.backtest_all(predictions, entry, (market, prices.index_close), settings)
    daily_ic = runner.seed_averaged_ic(predictions)
    signal = SignalSettings(config.label.horizon_days, config.evaluation.n_quantiles)
    table = walk_forward.results_table(predictions, signal)
    per_fold = table[table["fold"] != walk_forward.ALL_FOLDS].pivot(
        index="fold", columns="model", values="mean_ic"
    )
    portfolio = report.summary[
        ["ann_turnover", "ann_cost", "gross_excess_ann_return", "excess_ann_return"]
    ]
    portfolio = portfolio.join(
        exploratory.alpha_table(report, settings)[
            ["alpha", "alpha_ci_low", "alpha_ci_high", "beta"]
        ]
    )
    lines = [
        "# Exploratory extension: literature features and XGBoost (walk-forward)",
        "",
        "**Pre-registered in protocol Section 12 before any code was written; exploratory, the",
        "lockbox is not used, and no pre-registered result changes.** "
        + f"{pipeline.provenance(info, 'extension')} Optuna configurations tried: {n_trials}.",
        "",
        "## Pre-registered tests (paired mean Rank IC, A minus B, 95% block bootstrap)",
        "",
        *markdown_table(paired_tests(daily_ic, HYPOTHESES, settings), pipeline.DIGITS),
        "",
        "Descriptive comparisons (no claim):",
        "",
        *markdown_table(paired_tests(daily_ic, DESCRIPTIVE, settings), pipeline.DIGITS),
        "",
        "## Mean Rank IC per model (95% CI; `diff_vs_primary` is LightGBM 15 minus the row)",
        "",
        *markdown_table(report.ic.reset_index(), pipeline.DIGITS),
        "",
        "## Mean Rank IC per fold",
        "",
        *markdown_table(per_fold.reset_index(), pipeline.DIGITS),
        "",
        "## New features on their own (univariate mean Rank IC, test rows)",
        "",
        *markdown_table(feature_ic(samples, keys), pipeline.DIGITS),
        "",
        "## Portfolios (top decile, net of costs; alpha after beta to equal weight)",
        "",
        *markdown_table(portfolio.reset_index(), pipeline.DIGITS),
        "",
    ]
    storage.write_text("\n".join(lines), config.paths.reports_dir / f"{REPORT_STEM}.md")
    logger.info("Extension:\n%s", "\n".join(lines))
