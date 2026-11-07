"""Phase 6 experiments (I/O layer): feature-group ablation, sensitivity checks, and SHAP.

Every experiment reuses the unchanged walk-forward protocol for LightGBM (same folds,
tuning budget, seeds, and evaluation). None of them feeds back into the primary result:

- **feature-group ablation:** drop each of the 5 groups in turn;
- **liquidity sensitivity:** rebuild the panel at the alternative traded-value
  thresholds (protocol changelog 2026-11-07) and rerun;
- **cost sensitivity:** re-trade the primary LightGBM predictions over a grid of
  brokerage fees and slippage (the equal-weight benchmark pays the same costs);
- **SHAP:** the last fold's LightGBM (same tuning, seed 0) explained on its test rows.
"""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from vn_rank import pipeline
from vn_rank.backtest import runner
from vn_rank.backtest.costs import CostModel
from vn_rank.backtest.engine import Market
from vn_rank.config import Config
from vn_rank.data import storage
from vn_rank.evaluation import bootstrap
from vn_rank.features import builder, labels
from vn_rank.models.base import feature_columns
from vn_rank.models.gbm import LightGBMModel
from vn_rank.utils.logging import MlflowFoldLogger
from vn_rank.utils.markdown import markdown_table
from vn_rank.utils.seeds import set_global_seed
from vn_rank.validation import splits, walk_forward

EXPERIMENT_NAME = "ablation"
PRIMARY = runner.PRIMARY_MODEL
REFERENCE = "all features"
ABLATION_STEM = "ablation_results"
SHAP_FIGURE = "shap_summary.png"
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Context:
    """Inputs shared by every experiment of one ``ablation`` run."""

    config: Config
    market: Market
    entry_dates: pd.Series
    settings: runner.BacktestSettings


def _lightgbm(config: Config, features: list[str] | None = None) -> walk_forward.ModelSpec:
    specs = walk_forward.model_specs(config, features)
    return next(spec for spec in specs if spec.name == PRIMARY)


def run_variant(
    samples: pd.DataFrame,
    config: Config,
    features: list[str] | None,
    log_fold: walk_forward.FoldLogger,
) -> pd.DataFrame:
    """Walk-forward LightGBM predictions on ``samples`` with the given features."""
    return walk_forward.run_walk_forward(samples, [_lightgbm(config, features)], config, log_fold)


def variant_row(
    predictions: pd.DataFrame, reference_ic: pd.Series, ctx: Context
) -> dict[str, float]:
    """Mean Rank IC with CI, paired difference to the reference, and portfolio metrics."""
    boot = ctx.settings.bootstrap
    daily = runner.seed_averaged_ic(predictions)[PRIMARY]
    low, high = bootstrap.bootstrap_ci(daily.to_numpy(), bootstrap.mean_statistic, boot)
    diff = (daily - reference_ic).dropna()
    d_low, d_high = bootstrap.bootstrap_ci(diff.to_numpy(), bootstrap.mean_statistic, boot)
    first_seed = predictions[predictions["seed"] == predictions["seed"].min()]
    benchmark = runner.run_equal_weight(first_seed, ctx.entry_dates, ctx.market, ctx.settings)
    results = runner.run_model(predictions, ctx.entry_dates, ctx.market, ctx.settings)
    portfolio = runner.portfolio_row(results, benchmark, ctx.settings)
    return {
        "mean_ic": float(daily.mean()), "ic_ci_low": low, "ic_ci_high": high,
        "diff_vs_all": float(diff.mean()), "diff_ci_low": d_low, "diff_ci_high": d_high,
        "sharpe": portfolio["sharpe"], "excess_ann_return": portfolio["excess_ann_return"],
        "ann_turnover": portfolio["ann_turnover"], "stocks_per_date": float(
            first_seed.groupby("date").size().mean()),
    }  # fmt: skip


def cost_grid(predictions: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Net Sharpe and excess return over the equal-weight benchmark per fee and slippage."""
    rows = []
    ablation, tax = ctx.config.ablation, ctx.settings.engine.costs.selling_tax
    first_seed = predictions[predictions["seed"] == predictions["seed"].min()]
    for fee in ablation.brokerage_fees:
        for slip in ablation.slippages:
            engine = dataclasses.replace(ctx.settings.engine, costs=CostModel(fee, tax, slip))
            settings = dataclasses.replace(ctx.settings, engine=engine)
            ew = runner.run_equal_weight(first_seed, ctx.entry_dates, ctx.market, settings)
            results = runner.run_model(predictions, ctx.entry_dates, ctx.market, settings)
            row = runner.portfolio_row(results, ew, settings)
            keep = {k: row[k] for k in ("sharpe", "excess_ann_return", "ann_return")}
            round_trip = 2 * (fee + slip) + tax
            rows.append(
                {"brokerage_fee": fee, "slippage": slip, "round_trip_cost": round_trip, **keep}
            )
    return pd.DataFrame(rows)


def shap_figure(samples: pd.DataFrame, config: Config) -> Figure:
    """SHAP summary of the last fold's LightGBM (same tuning as walk-forward, seed 0)."""
    fold = splits.walk_forward_folds(config.validation)[-1]
    split = splits.split_fold(samples, fold, config.validation)
    spec = _lightgbm(config)
    params, _ = walk_forward.tune_spec(spec, split.train, config)
    model = spec.factory(params, config.models.seeds[0]).fit(split.train)
    if not isinstance(model, LightGBMModel) or model.booster is None:
        raise TypeError("SHAP expects a fitted LightGBM model")
    rows = split.test.sample(
        min(config.ablation.shap_sample_rows, len(split.test)), random_state=config.project.seed
    )
    values = shap.TreeExplainer(model.booster).shap_values(rows[model.features].to_numpy(float))
    names = [f.removeprefix("rank_") for f in model.features]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(values, rows[model.features].to_numpy(float), feature_names=names, show=False)
    figure = plt.gcf()
    figure.suptitle(f"LightGBM SHAP values, test block {fold.name} (feature ranks)", fontsize=10)
    figure.tight_layout()
    return figure


def _feature_variants(config: Config) -> dict[str, list[str]]:
    groups = builder.feature_groups(config.features)
    return {
        f"without {group}": feature_columns({g: f for g, f in groups.items() if g != group})
        for group in groups
    }


def _liquidity_samples(config: Config, prices: pipeline.Prices, threshold: float) -> pd.DataFrame:
    rule = dataclasses.replace(config.universe, min_traded_value_vnd=threshold)
    panel = builder.build_panel(
        prices.raw, prices.calendar, prices.exchange, dataclasses.replace(config, universe=rule)
    )
    return splits.development_samples(panel, config.validation)


def ablation(config: Config) -> int:
    """Run every Phase 6 experiment and write ``reports/ablation_results.md`` and figures."""
    set_global_seed(config.project.seed)
    lockbox = pd.Timestamp(config.validation.lockbox_start)
    prices = pipeline.load_prices(config, lockbox)
    ctx = Context(
        config,
        pipeline.market_data(config, prices),
        labels.label_entry_dates(prices.calendar, config.label),
        pipeline.backtest_settings(config),
    )
    processed = config.paths.processed_dir
    samples = splits.development_samples(
        storage.read_parquet(processed / pipeline.PANEL_FILE), config.validation
    )
    all_predictions = storage.read_parquet(processed / pipeline.PREDICTIONS_FILE)
    reference = all_predictions[all_predictions["model"] == PRIMARY]
    reference_ic = runner.seed_averaged_ic(reference)[PRIMARY]
    variants = {REFERENCE: reference}
    info = pipeline.run_info(config)
    with MlflowFoldLogger(config.paths.mlruns_dir, EXPERIMENT_NAME, info) as log_fold:
        for name, features in _feature_variants(config).items():
            logger.info("Ablation: %s", name)
            variants[name] = run_variant(samples, config, features, log_fold)
        for threshold in config.ablation.liquidity_thresholds_vnd:
            name = f"liquidity >= {threshold / 1e9:g} bn VND"
            logger.info("Sensitivity: %s", name)
            variants[name] = run_variant(
                _liquidity_samples(config, prices, threshold), config, None, log_fold
            )
    table = pd.DataFrame(
        {name: variant_row(p, reference_ic, ctx) for name, p in variants.items()}
    ).T
    costs = cost_grid(reference, ctx)
    _write_reports(config, table.rename_axis("variant"), costs, (info, log_fold.n_trials))
    storage.save_figure(
        shap_figure(samples, config),
        config.paths.reports_dir / SHAP_FIGURE,
        config.project.figure_dpi,
    )
    return 0


def _write_reports(
    config: Config, table: pd.DataFrame, costs: pd.DataFrame, provenance: tuple[dict[str, str], int]
) -> None:
    info, n_trials = provenance
    reports = config.paths.reports_dir
    storage.write_text(table.to_csv(), reports / f"{ABLATION_STEM}.csv")
    storage.write_text(costs.to_csv(index=False), reports / f"{ABLATION_STEM}_costs.csv")
    sharpe = costs.pivot(index="brokerage_fee", columns="slippage", values="sharpe")
    excess = costs.pivot(index="brokerage_fee", columns="slippage", values="excess_ann_return")
    lines = [
        "# Ablation and sensitivity (LightGBM, walk-forward, unchanged protocol)",
        "",
        f"{pipeline.provenance(info, 'ablation')} Optuna configurations tried: {n_trials}.",
        "",
        "## Feature groups and liquidity threshold",
        "",
        *markdown_table(table.reset_index(), pipeline.DIGITS),
        "",
        "## Net Sharpe by brokerage fee (rows) and slippage (columns), per side",
        "",
        *markdown_table(sharpe.reset_index(), pipeline.DIGITS),
        "",
        "## Net excess return over equal weight per year, same grid",
        "",
        *markdown_table(excess.reset_index(), pipeline.DIGITS),
        "",
    ]
    storage.write_text("\n".join(lines), reports / f"{ABLATION_STEM}.md")
    logger.info("Ablation:\n%s", "\n".join(lines))
