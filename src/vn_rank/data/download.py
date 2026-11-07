"""Download daily OHLCV from vnstock into per-symbol Parquet files.

The download is resumable: ``manifest.json`` records the status of every symbol, and a rerun
only requests symbols that are missing or failed. The snapshot end date is fixed in
``metadata.json`` on the first run, so a resumed download never mixes two snapshots. Calls
are spaced by a minimum interval and transient failures are retried with linear backoff.
One price source only (``vci``); sources are never mixed.

The orchestration here depends only on the :class:`PriceClient` protocol; the vnstock
adapter lives in ``vnstock_client.py``. Clients raise :class:`TransientSourceError` (retry)
or :class:`SymbolDataError` (skip one symbol); any other error, such as an authentication or
entitlement failure, stops the run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import logging
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import pandas as pd

from vn_rank.data import storage

logger = logging.getLogger(__name__)
T = TypeVar("T")

MARKET_TZ = "Asia/Ho_Chi_Minh"
RAW_COLUMNS = ("time", "open", "high", "low", "close", "volume")
STATUS_OK, STATUS_EMPTY, STATUS_FAILED = "ok", "empty", "failed"
CAP_ATTRS = ("cap_rows_dropped", "tier_cap_history_days", "tier_cap_rows")
_RESUME_KEYS = ("source", "index_symbol", "exchanges", "start", "end")


class TransientSourceError(Exception):
    """The source failed in a way a later retry may fix (rate limit, network)."""


class SymbolDataError(Exception):
    """The source cannot serve one symbol (unknown symbol, unreadable response)."""


class SnapshotMismatchError(Exception):
    """Resuming would mix two snapshots downloaded with different settings."""


@dataclasses.dataclass(frozen=True)
class SnapshotSpec:
    """What one snapshot contains; fixed for the lifetime of a download."""

    source: str
    index_symbol: str
    exchanges: tuple[str, ...]
    start: dt.date
    end: dt.date
    max_rows: int


@dataclasses.dataclass(frozen=True)
class RetryPolicy:
    """Request spacing and retry settings."""

    interval_seconds: float
    max_retries: int
    backoff_seconds: float


class PriceClient(Protocol):
    """Source of symbol lists and daily OHLCV frames."""

    def library_version(self) -> str:
        """Return the version string of the underlying data library."""

    def list_symbols(self) -> pd.DataFrame:
        """Return at least the columns ``symbol`` and ``exchange``."""

    def equity_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return the raw daily OHLCV frame (columns :data:`RAW_COLUMNS`) of one stock."""

    def index_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return the raw daily OHLCV frame of one index."""


class RateLimitedCaller:
    """Run zero-argument calls with a minimum spacing and retries on transient errors."""

    def __init__(
        self,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a caller; ``sleep`` and ``clock`` are injectable for tests."""
        self._policy, self._sleep, self._clock = policy, sleep, clock
        self._last_call: float | None = None

    def __call__(self, fn: Callable[[], T]) -> T:
        """Call ``fn``, retrying :class:`TransientSourceError` up to ``max_retries`` times.

        Raises:
            TransientSourceError: If every attempt fails transiently.
        """
        attempt = 0
        while True:
            self._wait_turn()
            try:
                return fn()
            except TransientSourceError as exc:
                if attempt >= self._policy.max_retries:
                    raise
                attempt += 1
                delay = self._policy.backoff_seconds * attempt
                logger.warning("Transient error (%s); retry %d in %.0fs", exc, attempt, delay)
                self._sleep(delay)

    def _wait_turn(self) -> None:
        if self._last_call is not None:
            remaining = self._policy.interval_seconds - (self._clock() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()


def snapshot_end(now: dt.datetime) -> dt.date:
    """Return the last fully completed trading-calendar day: yesterday in market time.

    Args:
        now: A timezone-aware current time.
    """
    return (now.astimezone(dt.timezone(dt.timedelta(hours=7))) - dt.timedelta(days=1)).date()


def to_canonical_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert a raw vnstock frame to :data:`storage.OHLCV_COLUMNS`.

    ``time`` (07:00 market time) becomes a timezone-naive ``date`` at midnight. Values are
    not altered; duplicates and anomalies are left for ``validate.py`` to report.

    Raises:
        SymbolDataError: If a required column is missing.
    """
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    if missing:
        raise SymbolDataError(f"{symbol}: missing columns {missing}")
    times = pd.to_datetime(raw["time"])
    if times.dt.tz is None:
        times = times.dt.tz_localize(MARKET_TZ)
    frame = raw.loc[:, list(RAW_COLUMNS[1:])].copy()
    frame.insert(0, "symbol", symbol)
    frame.insert(0, "date", times.dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize())
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def select_symbols(symbols: pd.DataFrame, exchanges: tuple[str, ...]) -> pd.DataFrame:
    """Keep symbols on the configured exchanges, one row per symbol, sorted."""
    keep = symbols.loc[symbols["exchange"].isin(exchanges), list(storage.SYMBOL_COLUMNS)]
    return keep.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def resolve_snapshot_end(raw_dir: Path, now: dt.datetime) -> dt.date:
    """Return the end date stored by an earlier run, or :func:`snapshot_end` for a new one."""
    path = raw_dir / storage.METADATA_FILE
    if path.exists():
        return dt.date.fromisoformat(str(storage.read_json(path)["end"]))
    return snapshot_end(now)


def download_snapshot(
    client: PriceClient, caller: RateLimitedCaller, spec: SnapshotSpec, raw_dir: Path
) -> dict[str, int]:
    """Download (or resume) the symbol list, the index, and every stock's OHLCV.

    Returns:
        Number of symbols per status (``ok``, ``empty``, ``failed``).

    Raises:
        SnapshotMismatchError: If ``raw_dir`` holds a snapshot with different settings.
        TransientSourceError: If retries are exhausted; rerun to resume.
    """
    metadata = _start_metadata(raw_dir, spec, client.library_version())
    symbols = _load_or_fetch_symbols(client, caller, spec, raw_dir)
    path = storage.index_path(raw_dir, spec.index_symbol)
    if not path.exists():
        raw = caller(functools.partial(client.index_ohlcv, spec.index_symbol))
        storage.write_parquet(to_canonical_ohlcv(raw, spec.index_symbol), path)
    manifest = _download_equities(client, caller, symbols["symbol"].tolist(), raw_dir)
    counts = dict(Counter(entry["status"] for entry in manifest.values()))
    metadata.update(status_counts=counts, completed_at=_utc_now())
    storage.write_json(metadata, raw_dir / storage.METADATA_FILE)
    return counts


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _start_metadata(raw_dir: Path, spec: SnapshotSpec, version: str) -> dict[str, Any]:
    path = raw_dir / storage.METADATA_FILE
    wanted = storage.read_json(path) if path.exists() else {}
    current = {k: v for k, v in dataclasses.asdict(spec).items() if k in _RESUME_KEYS}
    current = {k: list(v) if isinstance(v, tuple) else str(v) for k, v in current.items()}
    stored = {k: wanted[k] for k in _RESUME_KEYS if k in wanted}
    if stored and stored != current:
        raise SnapshotMismatchError(f"{path} holds {stored}, requested {current}")
    if not stored:
        wanted = {**current, "vnstock_version": version, "started_at": _utc_now()}
    wanted["last_run_vnstock_version"] = version
    storage.write_json(wanted, path)
    return wanted


def _load_or_fetch_symbols(
    client: PriceClient, caller: RateLimitedCaller, spec: SnapshotSpec, raw_dir: Path
) -> pd.DataFrame:
    path = raw_dir / storage.SYMBOLS_FILE
    if path.exists():
        return storage.read_parquet(path)
    symbols = select_symbols(caller(client.list_symbols), spec.exchanges)
    storage.write_parquet(symbols, path)
    logger.info("Symbol list: %s", symbols["exchange"].value_counts().to_dict())
    return symbols


def _download_equities(
    client: PriceClient, caller: RateLimitedCaller, symbols: list[str], raw_dir: Path
) -> dict[str, dict[str, Any]]:
    path = raw_dir / storage.MANIFEST_FILE
    manifest = storage.read_json(path) if path.exists() else {}
    pending = [s for s in symbols if not _is_done(manifest.get(s), raw_dir, s)]
    logger.info("%d of %d symbols to download", len(pending), len(symbols))
    for number, symbol in enumerate(pending, start=1):
        manifest[symbol] = _download_one(client, caller, symbol, raw_dir)
        storage.write_json(manifest, path)
        logger.debug("%d/%d %s: %s", number, len(pending), symbol, manifest[symbol]["status"])
    return manifest


def _is_done(entry: dict[str, Any] | None, raw_dir: Path, symbol: str) -> bool:
    if entry is None or entry["status"] == STATUS_FAILED:
        return False
    return entry["status"] == STATUS_EMPTY or storage.ohlcv_path(raw_dir, symbol).exists()


def _download_one(
    client: PriceClient, caller: RateLimitedCaller, symbol: str, raw_dir: Path
) -> dict[str, Any]:
    try:
        raw = caller(functools.partial(client.equity_ohlcv, symbol))
        frame = to_canonical_ohlcv(raw, symbol)
    except SymbolDataError as exc:
        logger.warning("Skipping %s: %s", symbol, exc)
        return {"status": STATUS_FAILED, "error": str(exc)}
    entry: dict[str, Any] = {k: raw.attrs[k] for k in CAP_ATTRS if k in raw.attrs}
    if frame.empty:
        return {**entry, "status": STATUS_EMPTY, "rows": 0}
    storage.write_parquet(frame, storage.ohlcv_path(raw_dir, symbol))
    first, last = (d.date().isoformat() for d in frame["date"].iloc[[0, -1]])
    return {**entry, "status": STATUS_OK, "rows": len(frame), "first": first, "last": last}
