"""Walk-forward runner: tune, refit, and score every model on every fold.

For each fold and model:

1. if the model has a search space, run Optuna on the inner split of the training window
   (:func:`~vn_rank.validation.splits.inner_validation_split`);
2. refit on the whole purged and embargoed training window with the best parameters,
   once per seed (several seeds only for stochastic models);
3. score the test block.

The test block is used only in step 3. Results are handed to a ``log_fold`` callback
(MLflow in the CLI), so this module does no I/O.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd

from vn_rank.config import Config, SearchRange
from vn_rank.evaluation.ic import RETURN_COLUMN, SCORE_COLUMN, SignalSettings, signal_summary
from vn_rank.features.builder import feature_groups
from vn_rank.models import baselines, gbm, linear, mlp, tuning
from vn_rank.models.base import RankingModel, feature_columns
from vn_rank.validation import splits

ALL_FOLDS = "all"
PREDICTION_COLUMNS = ["date", "symbol", RETURN_COLUMN, SCORE_COLUMN, "model", "fold", "seed"]
Factory = Callable[[dict[str, float], int], RankingModel]


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """A named model: how to build it, what to tune, and which seeds to refit with."""

    name: str
    factory: Factory
    space: dict[str, SearchRange]
    seeds: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class FoldRecord:
    """What one model did on one fold (logged per fold)."""

    model: str
    fold: str
    params: dict[str, float]
    valid_mean_ic: float | None
    n_trials: int
    n_train: int
    n_test: int
    test_mean_ic: dict[int, float]


FoldLogger = Callable[[FoldRecord, pd.DataFrame | None], None]


def _split_count(params: dict[str, float], key: str) -> tuple[dict[str, float], int | None]:
    """Separate a learned count (rounds, epochs) from the tuned parameters."""
    tuned = {k: v for k, v in params.items() if k != key}
    return tuned, int(params[key]) if key in params else None


def model_specs(config: Config, features: list[str] | None = None) -> list[ModelSpec]:
    """Baselines, Elastic Net, LightGBM, and the MLP, built from the configuration.

    Args:
        config: Model, tuning, and validation settings.
        features: Feature rank columns for the learned models; all features by default
            (the feature-group ablation passes a subset).
    """
    models, lockbox = config.models, config.validation.lockbox_start
    features = features or feature_columns(feature_groups(config.features))
    en_settings = linear.ElasticNetSettings(models.missing_rank_fill, models.elastic_net_max_iter)
    mlp_settings = mlp.MLPSettings(models.mlp, models.missing_rank_fill)

    def gbm_factory(params: dict[str, float], seed: int) -> RankingModel:
        tuned, rounds = _split_count(params, gbm.ROUNDS_KEY)
        run = gbm.GBMRun(tuned, seed, rounds)
        return gbm.LightGBMModel(features, run, models.lightgbm, lockbox)

    def mlp_factory(params: dict[str, float], seed: int) -> RankingModel:
        tuned, epochs = _split_count(params, mlp.EPOCHS_KEY)
        return mlp.MLPModel(features, mlp.MLPRun(tuned, seed, epochs), mlp_settings, lockbox)

    one_seed = (config.project.seed,)
    return [
        ModelSpec(
            "momentum",
            lambda p, s: baselines.momentum_baseline(models.momentum_window, lockbox),
            {},
            one_seed,
        ),
        ModelSpec(
            "reversal",
            lambda p, s: baselines.reversal_baseline(models.reversal_window, lockbox),
            {},
            one_seed,
        ),
        ModelSpec(
            "elastic_net",
            lambda p, s: linear.ElasticNetModel(features, p, en_settings, lockbox),
            models.tuning.elastic_net,
            one_seed,
        ),
        ModelSpec("lightgbm", gbm_factory, models.tuning.lightgbm, models.seeds),
        ModelSpec("mlp", mlp_factory, models.tuning.mlp, models.seeds),
    ]


def _tune(
    spec: ModelSpec, train: pd.DataFrame, config: Config
) -> tuple[dict[str, float], tuning.TuningResult | None]:
    if not spec.space:
        return {}, None
    inner, valid = splits.inner_validation_split(train, config.validation)
    seed = config.project.seed
    settings = tuning.TuningSettings(spec.space, config.models.tuning.n_trials, seed)
    result = tuning.tune(lambda p: spec.factory(p, seed), inner, valid, settings)
    return {**result.params, **result.attributes}, result


def run_fold(
    split: splits.FoldSplit, spec: ModelSpec, config: Config, log_fold: FoldLogger
) -> pd.DataFrame:
    """Tune, refit per seed, and score one model on one fold; returns test predictions."""
    params, result = _tune(spec, split.train, config)
    frames, test_ic = [], {}
    for seed in spec.seeds:
        model = spec.factory(params, seed).fit(split.train)
        scored = split.test.assign(score=model.predict(split.test), model=spec.name, seed=seed)
        frames.append(scored.assign(fold=split.fold.name)[PREDICTION_COLUMNS])
        test_ic[seed] = tuning.mean_rank_ic(model, split.test)
    record = FoldRecord(
        model=spec.name,
        fold=split.fold.name,
        params=params,
        valid_mean_ic=None if result is None else result.best_value,
        n_trials=0 if result is None else len(result.trials),
        n_train=len(split.train),
        n_test=len(split.test),
        test_mean_ic=test_ic,
    )
    log_fold(record, None if result is None else result.trials)
    return pd.concat(frames, ignore_index=True)


def run_walk_forward(
    samples: pd.DataFrame, specs: list[ModelSpec], config: Config, log_fold: FoldLogger
) -> pd.DataFrame:
    """Run every model on every walk-forward fold; returns all test predictions."""
    predictions: list[pd.DataFrame] = []
    for fold in splits.walk_forward_folds(config.validation):
        split = splits.split_fold(samples, fold, config.validation)
        predictions.extend(run_fold(split, spec, config, log_fold) for spec in specs)
    return pd.concat(predictions, ignore_index=True)


def results_table(predictions: pd.DataFrame, signal: SignalSettings) -> pd.DataFrame:
    """Signal metrics per model and fold (and ``all`` folds pooled), averaged over seeds.

    ``n_seeds`` counts the seeds and ``mean_ic_seed_std`` is the standard deviation of
    the mean Rank IC across them (NaN for a single seed).
    """
    rows = []
    for (model, seed), part in predictions.groupby(["model", "seed"], sort=False):
        rows.append(
            {"model": model, "fold": ALL_FOLDS, "seed": seed, **signal_summary(part, signal)}
        )
        for fold, fold_part in part.groupby("fold", sort=True):
            rows.append(
                {"model": model, "fold": fold, "seed": seed, **signal_summary(fold_part, signal)}
            )
    per_seed = pd.DataFrame(rows)
    grouped = per_seed.drop(columns="seed").groupby(["model", "fold"], sort=False)
    table = grouped.mean()
    table.insert(0, "n_seeds", grouped.size())
    table.insert(2, "mean_ic_seed_std", grouped["mean_ic"].std())
    return table.reset_index()


_POOLED_COLUMNS = {
    "n_seeds": "seeds", "mean_ic": "mean Rank IC", "mean_ic_seed_std": "seed std",
    "ic_std": "daily IC std", "ic_ir": "IC IR", "t_nw": "NW t", "n_dates": "dates",
}  # fmt: skip
_TAIL_COLUMNS = {"top_minus_bottom": "Q5-Q1", "signal_autocorr": "signal autocorr"}


def _markdown_table(frame: pd.DataFrame, digits: int) -> list[str]:
    cells = frame.map(lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    return [header, rule, *("| " + " | ".join(row) + " |" for row in cells.to_numpy())]


def render_results_markdown(table: pd.DataFrame, header: list[str], digits: int) -> str:
    """Markdown report: pooled metrics per model, then mean Rank IC per fold."""
    pooled = table.loc[table["fold"] == ALL_FOLDS].set_index("model")
    quantiles = [c for c in pooled.columns if c.startswith("q") and c[1:].isdigit()]
    columns = {**_POOLED_COLUMNS, **{q: q.upper() for q in quantiles}, **_TAIL_COLUMNS}
    pooled_view = pooled[list(columns)].rename(columns=columns).reset_index()
    pooled_view["dates"] = pooled_view["dates"].astype(int)
    folds = table.loc[table["fold"] != ALL_FOLDS]
    per_fold = folds.pivot(index="fold", columns="model", values="mean_ic")
    per_fold = per_fold[list(pooled.index)].reset_index()
    lines = [*header, "", "## All test folds pooled", ""]
    lines += _markdown_table(pooled_view, digits)
    lines += ["", "## Mean Rank IC per fold", ""]
    return "\n".join([*lines, *_markdown_table(per_fold, digits), ""])


def tripwire_breaches(table: pd.DataFrame, mean_limit: float, fold_limit: float) -> list[str]:
    """Leakage tripwire (protocol 9): pooled mean Rank IC or any fold above its limit."""
    pooled = table["fold"] == ALL_FOLDS
    limits = pooled.map({True: mean_limit, False: fold_limit})
    breached = table.loc[table["mean_ic"] > limits]
    return [f"{r.model} {r.fold}: mean Rank IC {r.mean_ic:.4f}" for r in breached.itertuples()]
