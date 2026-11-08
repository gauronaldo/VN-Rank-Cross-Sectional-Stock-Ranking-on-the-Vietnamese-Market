"""Tests for the pre-registered extension: paired tests and the command end to end."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.backtest.test_runner import SETTINGS
from tests.test_final import config_path  # noqa: F401 - fixture with a synthetic snapshot
from vn_rank import extension, final
from vn_rank.cli import main
from vn_rank.config import load_config
from vn_rank.data import storage


def test_paired_tests_report_direction_and_verdict() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    base = rng.normal(0.05, 0.1, 300)
    daily = pd.DataFrame(
        {"a": base + 0.05, "b": base, "c": base + rng.normal(0, 0.01, 300)}, index=dates
    )
    table = extension.paired_tests(daily, (("better", "a", "b"), ("same", "c", "b")), SETTINGS)
    better = table.set_index("test").loc["better"]
    assert better["mean_A_minus_B"] == pytest.approx(0.05)
    assert better["result"] == "A better"
    assert table.set_index("test").loc["same", "result"] == "no difference"


def test_extension_end_to_end(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    def clean(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(final.subprocess, "run", clean)
    args = ["--config", str(config_path)]
    assert main([*args, "features"]) == 0
    assert main([*args, "train"]) in (0, 2)
    assert main([*args, "extension"]) == 0
    config = load_config(config_path)
    predictions = storage.read_parquet(config.paths.processed_dir / extension.PREDICTIONS_FILE)
    expected = {"lightgbm", "lightgbm_19", "xgboost_15", "xgboost_19", "lowvol_vol60", "ivol_rule"}
    assert set(predictions["model"]) == expected
    rows = predictions.groupby("model").size()
    assert rows["lowvol_vol60"] == rows["ivol_rule"]
    report = (config.paths.reports_dir / f"{extension.REPORT_STEM}.md").read_text(encoding="utf-8")
    for test in ("H1 features", "H2 library", "H3 beats rule"):
        assert test in report
