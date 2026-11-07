"""Optuna tuning on the inner validation window only.

Each trial fits a model on the inner training set (validation frame passed for early
stopping) and scores it by the mean daily Rank IC on the inner validation set. The TPE
sampler is seeded, so a study is reproducible.

Leakage note: tuning never sees walk-forward test blocks or the lockbox. The caller
passes only the two parts of a training window built by
:func:`vn_rank.validation.splits.inner_validation_split`, and every ``fit`` runs the
lockbox guard.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import optuna
import pandas as pd

from vn_rank.config import SearchRange
from vn_rank.evaluation.ic import SCORE_COLUMN, daily_rank_ic
from vn_rank.models.base import RankingModel

ModelFactory = Callable[[dict[str, float]], RankingModel]
NO_SIGNAL = 0.0  # objective value when no validation date has a defined Rank IC


@dataclasses.dataclass(frozen=True)
class TuningSettings:
    """Search space, trial budget, and sampler seed of one study."""

    space: dict[str, SearchRange]
    n_trials: int
    seed: int


@dataclasses.dataclass(frozen=True)
class TuningResult:
    """Best parameters, what the best fit learned (for example rounds), and all trials."""

    params: dict[str, float]
    attributes: dict[str, float]
    best_value: float
    trials: pd.DataFrame


def suggest_params(trial: optuna.Trial, space: dict[str, SearchRange]) -> dict[str, float]:
    """Draw one value per search dimension."""
    params: dict[str, float] = {}
    for name, dim in space.items():
        if dim.integer:
            params[name] = trial.suggest_int(name, int(dim.low), int(dim.high), log=dim.log)
        else:
            params[name] = trial.suggest_float(name, dim.low, dim.high, log=dim.log)
    return params


def mean_rank_ic(model: RankingModel, frame: pd.DataFrame) -> float:
    """Mean daily Rank IC of a fitted model on ``frame`` (``NO_SIGNAL`` if undefined)."""
    scored = frame.assign(**{SCORE_COLUMN: model.predict(frame)})
    daily = daily_rank_ic(scored)
    return float(daily.mean()) if len(daily) else NO_SIGNAL


def tune(
    factory: ModelFactory, inner: pd.DataFrame, valid: pd.DataFrame, settings: TuningSettings
) -> TuningResult:
    """Run one Optuna study and return the best trial.

    Args:
        factory: Builds an unfitted model from a parameter mapping.
        inner: Inner training set (purged and embargoed against ``valid``).
        valid: Inner validation set (last months of the training window).
        settings: Search space, number of trials, and sampler seed.

    Returns:
        The best parameters, the fitted attributes of the best trial, its validation
        mean Rank IC, and the table of all trials.
    """

    def objective(trial: optuna.Trial) -> float:
        model = factory(suggest_params(trial, settings.space)).fit(inner, valid)
        for key, value in model.fitted_attributes().items():
            trial.set_user_attr(key, value)
        return mean_rank_ic(model, valid)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=settings.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=settings.n_trials)
    best = study.best_trial
    attributes = {k: float(v) for k, v in best.user_attrs.items()}
    return TuningResult(dict(best.params), attributes, study.best_value, study.trials_dataframe())
