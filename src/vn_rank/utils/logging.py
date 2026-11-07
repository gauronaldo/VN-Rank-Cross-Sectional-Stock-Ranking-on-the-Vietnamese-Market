"""Logging setup shared by all command-line entry points, and MLflow experiment logging.

MLflow uses a SQLite backend (``<mlruns_dir>/mlflow.db``) with file artifacts under
``<mlruns_dir>/artifacts``; MLflow 3 refuses the plain file-store backend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    import pandas as pd

    from vn_rank.validation.walk_forward import FoldRecord

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MLFLOW_DB_FILE = "mlflow.db"
MLFLOW_ARTIFACT_DIR = "artifacts"


def configure_logging(level: str) -> None:
    """Configure root logging once for a command-line run.

    Args:
        level: Standard level name such as ``"INFO"`` or ``"DEBUG"``.

    Raises:
        ValueError: If ``level`` is not a standard logging level name.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"Unknown log level: {level!r}")
    logging.basicConfig(level=numeric, format=LOG_FORMAT, force=True)


class MlflowFoldLogger:
    """Context manager: one parent run per command, one nested run per model and fold.

    Every fold run records the chosen parameters, validation and test mean Rank IC, the
    sample counts, and the table of Optuna trials; :attr:`n_trials` counts every tuning
    configuration tried, as AGENT.md 2.3 requires.

    Args:
        mlruns_dir: Directory for the SQLite database and artifacts.
        experiment: Experiment name.
        run_info: Parameters of the parent run (for example the config hash).
    """

    def __init__(self, mlruns_dir: Path, experiment: str, run_info: dict[str, Any]) -> None:
        """Store settings; nothing is written until the context is entered."""
        self.mlruns_dir, self.experiment, self.run_info = mlruns_dir, experiment, run_info
        self.n_trials = 0

    def __enter__(self) -> Self:
        """Open the tracking database and start the parent run."""
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow  # noqa: PLC0415 - heavy import, only needed for training commands

        root = self.mlruns_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(root / MLFLOW_DB_FILE).as_posix()}")
        if mlflow.get_experiment_by_name(self.experiment) is None:
            artifacts = (root / MLFLOW_ARTIFACT_DIR).as_uri()
            mlflow.create_experiment(self.experiment, artifact_location=artifacts)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run()
        mlflow.log_params(self.run_info)
        return self

    def __call__(self, record: FoldRecord, trials: pd.DataFrame | None) -> None:
        """Log one model on one fold as a nested run."""
        import mlflow  # noqa: PLC0415

        with mlflow.start_run(run_name=f"{record.model} {record.fold}", nested=True):
            mlflow.set_tags({"model": record.model, "fold": record.fold})
            mlflow.log_params({**record.params, "seeds": sorted(record.test_mean_ic)})
            metrics: dict[str, float] = {
                "n_trials": record.n_trials,
                "n_train": record.n_train,
                "n_test": record.n_test,
            }
            metrics |= {f"test_mean_ic_seed_{s}": v for s, v in record.test_mean_ic.items()}
            if record.valid_mean_ic is not None:
                metrics["valid_mean_ic"] = record.valid_mean_ic
            mlflow.log_metrics(metrics)
            if trials is not None:
                mlflow.log_text(trials.to_csv(index=False), "optuna_trials.csv")
        self.n_trials += record.n_trials

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record the total number of configurations tried and close the parent run."""
        import mlflow  # noqa: PLC0415

        mlflow.log_metric("n_trials_total", self.n_trials)
        mlflow.end_run(status="FINISHED" if exc is None else "FAILED")
