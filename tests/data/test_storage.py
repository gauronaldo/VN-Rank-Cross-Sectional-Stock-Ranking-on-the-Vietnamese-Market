"""Tests for Parquet, DuckDB, and JSON storage helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vn_rank.data import storage


def _ohlcv(symbol: str, days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
        }
    )


def test_parquet_round_trip_is_atomic(tmp_path: Path) -> None:
    path = storage.ohlcv_path(tmp_path, "FPT")
    frame = _ohlcv("FPT", 3)
    storage.write_parquet(frame, path)
    pd.testing.assert_frame_equal(storage.read_parquet(path), frame)
    assert not list(path.parent.glob(".*.tmp"))


def test_downloaded_symbols_lists_complete_files_only(tmp_path: Path) -> None:
    assert storage.downloaded_symbols(tmp_path) == set()
    storage.write_parquet(_ohlcv("FPT", 2), storage.ohlcv_path(tmp_path, "FPT"))
    (tmp_path / storage.OHLCV_DIR / ".VNM.parquet.tmp").write_bytes(b"partial")
    assert storage.downloaded_symbols(tmp_path) == {"FPT"}


def test_load_panel_concatenates_and_sorts(tmp_path: Path) -> None:
    for symbol, days in (("VNM", 2), ("FPT", 3)):
        storage.write_parquet(_ohlcv(symbol, days), storage.ohlcv_path(tmp_path, symbol))
    panel = storage.load_ohlcv_panel(tmp_path)
    assert list(panel.columns) == list(storage.OHLCV_COLUMNS)
    assert panel["symbol"].tolist() == ["FPT"] * 3 + ["VNM"] * 2
    assert panel.groupby("symbol")["date"].apply(lambda s: s.is_monotonic_increasing).all()


def test_load_panel_without_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.load_ohlcv_panel(tmp_path)


def test_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / storage.METADATA_FILE
    storage.write_json({"source": "vci", "symbols": 2}, path)
    assert storage.read_json(path) == {"source": "vci", "symbols": 2}


def test_read_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        storage.read_json(path)


def test_content_sha256_ignores_row_and_column_order() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "symbol": ["B", "A"],
            "x": [1.0, 2.0],
        }
    )
    shuffled = frame.iloc[::-1][["x", "symbol", "date"]]
    assert storage.content_sha256(frame) == storage.content_sha256(shuffled)
    changed = frame.assign(x=[1.0, 2.5])
    assert storage.content_sha256(frame) != storage.content_sha256(changed)
