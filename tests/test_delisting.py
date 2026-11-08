"""Tests for the delisting sensitivity check."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.test_final import config_path  # noqa: F401 - fixture with a synthetic snapshot
from vn_rank import delisting, final
from vn_rank.cli import main
from vn_rank.config import load_config
from vn_rank.utils import report_paths

DATES = pd.bdate_range("2024-01-01", periods=6)


def test_return_to_last_close_uses_the_last_real_close() -> None:
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 9.0, 9.0, 9.0]}, index=DATES)
    alive = pd.DataFrame({"A": [True, True, True, True, False, False]}, index=DATES)
    rows = pd.DataFrame({"date": [DATES[0], DATES[3]], "symbol": ["A", "A"]})
    returns = delisting.return_to_last_close(close, alive, rows, entry_lag=1)
    assert returns.iloc[0] == pytest.approx(9.0 / 11.0 - 1)  # entry at the next close
    assert returns.iloc[1] == pytest.approx(0.0)  # stopped before the entry: no move


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for model, seeds in (("momentum", (0,)), ("lightgbm", (0, 1))):
        for seed in seeds:
            returns = rng.normal(0, 0.02, 20)
            frames.append(
                pd.DataFrame(
                    {
                        "date": DATES[0],
                        "symbol": [f"S{i}" for i in range(20)],
                        "forward_return": returns,
                        "score": returns + rng.normal(0, 0.01, 20),
                        "model": model,
                        "fold": 1,
                        "seed": seed,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_with_lost_rows_scores_rules_exactly_and_learned_models_at_the_top() -> None:
    predictions = _predictions()
    lost = pd.DataFrame({"date": [DATES[0]], "symbol": ["X"], "rank_ret_20": [0.25]})
    rules = {"momentum": ("rank_ret_20", 1.0)}
    loss = pd.Series(-1.0, index=lost.index)
    frame = delisting.with_lost_rows(predictions, lost, loss, rules)
    added = frame[frame["symbol"] == "X"].set_index(["model", "seed"])["score"]
    assert added.loc[("momentum", 0)] == pytest.approx(0.25)
    for seed in (0, 1):
        top = predictions.query("model == 'lightgbm' and seed == @seed")["score"].max()
        assert added.loc[("lightgbm", seed)] == pytest.approx(top + 1.0)
    table = delisting.sensitivity_table({"reported": predictions, "total loss": frame})
    change = table.set_index("model")["change, total loss"]
    assert change["lightgbm"] < 0  # the worst case can only lower the learned model's IC


def test_delisting_end_to_end(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    def clean(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(final.subprocess, "run", clean)
    args = ["--config", str(config_path)]
    assert main([*args, "features"]) == 0
    assert main([*args, "train"]) in (0, 2)
    assert main([*args, "delisting"]) == 0
    config = load_config(config_path)
    folder = report_paths.section(config.paths.reports_dir, report_paths.EXPLORATORY)
    report = (folder / f"{delisting.REPORT_STEM}.md").read_text(encoding="utf-8")
    assert "Mean Rank IC with the lost rows put back" in report
