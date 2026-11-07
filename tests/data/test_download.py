"""Tests for the resumable, rate-limited downloader (no network: a fake client is used)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from vn_rank.data import download, storage
from vn_rank.data.download import (
    RateLimitedCaller,
    RetryPolicy,
    SnapshotMismatchError,
    SnapshotSpec,
    SymbolDataError,
    TransientSourceError,
)

SPEC = SnapshotSpec(
    source="vci",
    index_symbol="VNINDEX",
    exchanges=("HOSE", "DELISTED"),
    start=dt.date(2024, 1, 1),
    end=dt.date(2024, 1, 10),
    max_rows=100,
)


def _raw(days: int, start: str = "2024-01-02") -> pd.DataFrame:
    times = pd.date_range(start, periods=days, freq="B", tz="Asia/Ho_Chi_Minh") + pd.Timedelta(
        hours=7
    )
    frame = pd.DataFrame(
        {"time": times, "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100}
    )
    frame.attrs["cap_rows_dropped"] = 3
    return frame


class FakeClient:
    """In-memory PriceClient that records every request."""

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.calls: list[str] = []
        self.failures = failures or {}

    def library_version(self) -> str:
        return "5.2.2"

    def list_symbols(self) -> pd.DataFrame:
        self.calls.append("<symbols>")
        return pd.DataFrame(
            {
                "symbol": ["FPT", "VNM", "OLD", "EMPTY", "BAD", "HNX1", "FPT"],
                "exchange": ["HOSE", "HOSE", "DELISTED", "HOSE", "HOSE", "HNX", "HOSE"],
                "organ_name": "x",
            }
        )

    def equity_ohlcv(self, symbol: str) -> pd.DataFrame:
        self.calls.append(symbol)
        if symbol in self.failures:
            raise self.failures[symbol]
        return _raw(0) if symbol == "EMPTY" else _raw(3)

    def index_ohlcv(self, symbol: str) -> pd.DataFrame:
        self.calls.append(f"<index {symbol}>")
        return _raw(5)


def _caller() -> RateLimitedCaller:
    return RateLimitedCaller(RetryPolicy(0.0, 2, 0.0), sleep=lambda _: None)


def test_to_canonical_ohlcv_converts_time_to_naive_date() -> None:
    frame = download.to_canonical_ohlcv(_raw(2), "FPT")
    assert list(frame.columns) == list(storage.OHLCV_COLUMNS)
    assert frame["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert frame["date"].dt.tz is None
    assert (frame["symbol"] == "FPT").all()


def test_to_canonical_ohlcv_rejects_missing_columns() -> None:
    with pytest.raises(SymbolDataError, match="volume"):
        download.to_canonical_ohlcv(_raw(2).drop(columns="volume"), "FPT")


def test_select_symbols_filters_exchanges_and_deduplicates() -> None:
    selected = download.select_symbols(FakeClient().list_symbols(), ("HOSE", "DELISTED"))
    assert selected["symbol"].tolist() == ["BAD", "EMPTY", "FPT", "OLD", "VNM"]
    assert list(selected.columns) == list(storage.SYMBOL_COLUMNS)


@pytest.mark.parametrize(
    ("utc_time", "expected"),
    [
        (dt.datetime(2026, 11, 7, 2, 0, tzinfo=dt.UTC), dt.date(2026, 11, 6)),
        (dt.datetime(2026, 11, 7, 18, 0, tzinfo=dt.UTC), dt.date(2026, 11, 7)),
    ],
)
def test_snapshot_end_is_yesterday_in_market_time(utc_time: dt.datetime, expected: dt.date) -> None:
    assert download.snapshot_end(utc_time) == expected


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_caller_spaces_consecutive_calls() -> None:
    clock = FakeClock()
    caller = RateLimitedCaller(RetryPolicy(1.2, 0, 0.0), sleep=clock.sleep, clock=lambda: clock.now)
    assert [caller(lambda: 1), caller(lambda: 2)] == [1, 2]
    assert clock.sleeps == [pytest.approx(1.2)]


def _flaky(failures: int) -> Callable[[], str]:
    state = {"left": failures}

    def call() -> str:
        if state["left"] > 0:
            state["left"] -= 1
            raise TransientSourceError("rate limited")
        return "ok"

    return call


def test_caller_retries_with_linear_backoff() -> None:
    clock = FakeClock()
    caller = RateLimitedCaller(
        RetryPolicy(0.0, 3, 10.0), sleep=clock.sleep, clock=lambda: clock.now
    )
    assert caller(_flaky(2)) == "ok"
    assert clock.sleeps == [10.0, 20.0]


def test_caller_raises_when_retries_are_exhausted() -> None:
    caller = RateLimitedCaller(RetryPolicy(0.0, 1, 0.0), sleep=lambda _: None)
    with pytest.raises(TransientSourceError):
        caller(_flaky(2))


def test_download_writes_snapshot_manifest_and_metadata(tmp_path: Path) -> None:
    client = FakeClient({"BAD": SymbolDataError("unknown symbol")})
    counts = download.download_snapshot(client, _caller(), SPEC, tmp_path)
    assert counts == {"ok": 3, "empty": 1, "failed": 1}
    assert storage.downloaded_symbols(tmp_path) == {"FPT", "VNM", "OLD"}
    assert storage.index_path(tmp_path, "VNINDEX").exists()
    manifest = storage.read_json(tmp_path / storage.MANIFEST_FILE)
    assert manifest["FPT"] == {
        "cap_rows_dropped": 3,
        "status": "ok",
        "rows": 3,
        "first": "2024-01-02",
        "last": "2024-01-04",
    }
    metadata = storage.read_json(tmp_path / storage.METADATA_FILE)
    assert metadata["end"] == "2024-01-10"
    assert metadata["vnstock_version"] == "5.2.2"
    assert metadata["status_counts"] == counts


def test_resume_requests_only_missing_and_failed_symbols(tmp_path: Path) -> None:
    download.download_snapshot(FakeClient({"BAD": SymbolDataError("x")}), _caller(), SPEC, tmp_path)
    storage.ohlcv_path(tmp_path, "VNM").unlink()
    client = FakeClient()
    counts = download.download_snapshot(client, _caller(), SPEC, tmp_path)
    assert client.calls == ["BAD", "VNM"]
    assert counts == {"ok": 4, "empty": 1}


def test_interrupted_download_keeps_progress(tmp_path: Path) -> None:
    client = FakeClient({"OLD": TransientSourceError("down")})
    with pytest.raises(TransientSourceError):
        download.download_snapshot(client, _caller(), SPEC, tmp_path)
    manifest = storage.read_json(tmp_path / storage.MANIFEST_FILE)
    assert set(manifest) == {"BAD", "EMPTY", "FPT"}


def test_resume_with_different_settings_is_refused(tmp_path: Path) -> None:
    download.download_snapshot(FakeClient(), _caller(), SPEC, tmp_path)
    other = SnapshotSpec(**{**SPEC.__dict__, "end": dt.date(2024, 2, 1)})
    with pytest.raises(SnapshotMismatchError):
        download.download_snapshot(FakeClient(), _caller(), other, tmp_path)


def test_resolve_snapshot_end_reuses_stored_end(tmp_path: Path) -> None:
    now = dt.datetime(2026, 11, 7, 2, 0, tzinfo=dt.UTC)
    assert download.resolve_snapshot_end(tmp_path, now) == dt.date(2026, 11, 6)
    download.download_snapshot(FakeClient(), _caller(), SPEC, tmp_path)
    assert download.resolve_snapshot_end(tmp_path, now) == SPEC.end
