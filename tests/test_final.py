"""Tests for the lockbox guard command, the frozen-model update, and monitoring helpers.

The end-to-end test runs ``train``, ``lockbox``, and ``update`` through the CLI on a
synthetic snapshot, so the one real lockbox run cannot fail on a code path never executed.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from vn_rank import final
from vn_rank.cli import main
from vn_rank.config import config_from_dict, config_to_dict, load_config
from vn_rank.data import storage
from vn_rank.monitoring import update as monitor
from vn_rank.utils import report_paths

SESSIONS = pd.bdate_range("2019-01-01", "2021-12-31")
N_SYMBOLS = 60


def write_snapshot(raw_dir: Path) -> None:
    """Liquid synthetic stocks on a business-day calendar, plus VNINDEX and symbols."""
    rng = np.random.default_rng(3)
    n = len(SESSIONS)
    for i in range(N_SYMBOLS):
        close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        frame = pd.DataFrame(
            {"date": SESSIONS, "symbol": f"S{i:02d}", "open": close, "high": close * 1.01,
             "low": close * 0.99, "close": close, "volume": rng.integers(100_000, 200_000, n)}
        )  # fmt: skip
        storage.write_parquet(frame, storage.ohlcv_path(raw_dir, f"S{i:02d}"))
    index = pd.DataFrame({"date": SESSIONS, "symbol": "VNINDEX", "close": 1000 + np.arange(n)})
    storage.write_parquet(index, storage.index_path(raw_dir, "VNINDEX"))
    symbols = pd.DataFrame({"symbol": [f"S{i:02d}" for i in range(N_SYMBOLS)], "exchange": "HOSE"})
    storage.write_parquet(symbols, raw_dir / storage.SYMBOLS_FILE)
    storage.write_json({"source": "synthetic"}, raw_dir / storage.METADATA_FILE)


@pytest.fixture
def config_path(tmp_path: Path, default_config_path: Path) -> Path:
    raw: dict[str, Any] = config_to_dict(load_config(default_config_path))
    data = tmp_path / "data"
    raw["paths"].update(
        raw_dir=str(data / "raw"), interim_dir=str(data / "interim"),
        processed_dir=str(data / "processed"), reports_dir=str(tmp_path / "reports"),
        mlruns_dir=str(tmp_path / "mlruns"),
    )  # fmt: skip
    raw["validation"].update(
        train_start=dt.date(2020, 1, 1), walk_forward_start=dt.date(2020, 7, 1),
        walk_forward_end=dt.date(2021, 6, 30), lockbox_start=dt.date(2021, 7, 1),
        inner_validation_months=3,
    )  # fmt: skip
    raw["models"]["seeds"] = [0, 1]
    raw["models"]["tuning"]["n_trials"] = 2
    raw["models"]["lightgbm"]["max_boost_rounds"] = 30
    raw["models"]["mlp"].update(max_epochs=3, patience=1)
    raw["evaluation"]["bootstrap_resamples"] = 200
    config_from_dict(raw)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    write_snapshot(data / "raw")
    return path


def _clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(final.subprocess, "run", fake_run)


def test_lockbox_refuses_without_flag_existing_lock_or_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lockbox_run.lock"
    with pytest.raises(final.LockboxRefusedError, match="--confirm-lockbox"):
        final.check_preconditions(lock, confirmed=False)

    def dirty(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=" M src/x.py\n", stderr="")

    monkeypatch.setattr(final.subprocess, "run", dirty)
    with pytest.raises(final.LockboxRefusedError, match="dirty"):
        final.check_preconditions(lock, confirmed=True)
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(final.LockboxRefusedError, match="already been run"):
        final.check_preconditions(lock, confirmed=True)


def test_train_lockbox_update_end_to_end(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_git(monkeypatch)
    args = ["--config", str(config_path)]
    assert main([*args, "features"]) == 0
    assert main([*args, "train"]) in (0, 2)  # 2 only if the synthetic data trips the tripwire
    assert main([*args, "update"]) == 1  # no frozen model before the lockbox
    assert main([*args, "lockbox"]) == 1  # flag missing
    assert main([*args, "lockbox", "--confirm-lockbox"]) == 0
    config = load_config(config_path)
    reports, processed = final.lockbox_reports(config), config.paths.processed_dir
    lock = storage.read_json(reports / final.LOCK_FILE)
    assert lock["last_matured_signal_date"] == SESSIONS[-7].date().isoformat()
    text = (reports / "lockbox_results.md").read_text(encoding="utf-8")
    assert "- Outcome: **" in text
    predictions = storage.read_parquet(processed / final.LOCKBOX_PREDICTIONS_FILE)
    assert predictions["date"].min() >= pd.Timestamp("2021-07-01")
    assert len(list((processed / final.MODEL_DIR).glob("lightgbm_*.txt"))) == 2
    assert main([*args, "lockbox", "--confirm-lockbox"]) == 1  # never twice
    assert main([*args, "update"]) == 0
    live = final.live_dir(config)
    assert (live / f"top_list_{SESSIONS[-1].date()}.csv").exists()
    log = storage.read_parquet(live / final.SCORES_LOG)
    assert log["date"].max() == SESSIONS[-1]
    monitoring = report_paths.section(config.paths.reports_dir, report_paths.MONITORING)
    assert (monitoring / final.REALIZED_FIGURE).exists()


def test_lockbox_split_purges_against_the_lockbox_start(config_path: Path) -> None:
    config = load_config(config_path)
    assert main(["--config", str(config_path), "features"]) == 0
    prices = final.pipeline.load_prices(config, None)
    panel = final.builder.build_panel(prices.raw, prices.calendar, prices.exchange, config)
    train, test = final.lockbox_split(panel, config)
    boundary = pd.Timestamp(config.validation.lockbox_start)
    assert (train["exit_date"] < boundary).all()
    gap = SESSIONS.get_loc(test["date"].min()) - SESSIONS.get_loc(train["date"].max())
    assert gap == 1 + config.label.horizon_days + 1 + config.validation.embargo_days
    assert test["target"].notna().all()


def test_monitoring_helpers() -> None:
    scored = pd.DataFrame(
        {"date": pd.Timestamp("2024-01-02"), "symbol": list("ABCDE"), "exchange": "HOSE",
         "score": [0.1, 0.5, 0.3, 0.9, 0.2]}
    )  # fmt: skip
    top = monitor.top_list(scored, 0.4)
    assert top["symbol"].to_list() == ["D", "B"]
    assert top["rank"].to_list() == [1, 2]
    log = monitor.append_scores(pd.DataFrame(columns=monitor.LOG_COLUMNS), scored)
    log = monitor.append_scores(log, scored.assign(score=1.0 - scored["score"]))
    assert len(log) == 5
    assert log.set_index("symbol").loc["D", "score"] == pytest.approx(0.1)
    panel = scored.assign(forward_return=[0.1, 0.5, 0.3, 0.9, np.nan])
    ic = monitor.realized_ic(scored[monitor.LOG_COLUMNS], panel)
    assert ic.iloc[0] == pytest.approx(1.0)
