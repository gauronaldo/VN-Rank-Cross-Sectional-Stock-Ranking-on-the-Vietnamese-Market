"""Parquet, DuckDB, and JSON read/write helpers for the raw data snapshot.

The only data module that touches disk besides ``download.py``. Writes are atomic (write to
a temporary file, then rename), so an interrupted download never leaves a truncated file
that looks complete; this is what makes the downloader safely resumable.

Layout under ``raw_dir``::

    ohlcv/<SYMBOL>.parquet   one file per stock, canonical OHLCV schema
    index/<SYMBOL>.parquet   index series (VNINDEX), same schema
    symbols.parquet          symbol list with exchange
    manifest.json            per-symbol download status (resume state)
    metadata.json            snapshot metadata (download date, source, vnstock version)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from matplotlib.figure import Figure

OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")
SYMBOL_COLUMNS = ("symbol", "exchange")
OHLCV_DIR = "ohlcv"
INDEX_DIR = "index"
SYMBOLS_FILE = "symbols.parquet"
MANIFEST_FILE = "manifest.json"
METADATA_FILE = "metadata.json"


def ohlcv_path(raw_dir: Path, symbol: str) -> Path:
    """Return the Parquet path of one stock's OHLCV file."""
    return raw_dir / OHLCV_DIR / f"{symbol}.parquet"


def index_path(raw_dir: Path, symbol: str) -> Path:
    """Return the Parquet path of one index series (for example VNINDEX)."""
    return raw_dir / INDEX_DIR / f"{symbol}.parquet"


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet atomically, creating parent folders.

    Args:
        frame: Data to write; the index is not stored.
        path: Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read one Parquet file into a DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def downloaded_symbols(raw_dir: Path) -> set[str]:
    """Return the symbols that already have a complete OHLCV file."""
    folder = raw_dir / OHLCV_DIR
    return {p.stem for p in folder.glob("*.parquet")} if folder.exists() else set()


def load_ohlcv_panel(raw_dir: Path) -> pd.DataFrame:
    """Load every per-symbol OHLCV file into one long panel with DuckDB.

    Args:
        raw_dir: Root of the raw snapshot.

    Returns:
        Columns :data:`OHLCV_COLUMNS`, sorted by ``symbol`` then ``date``.

    Raises:
        FileNotFoundError: If no OHLCV file exists yet.
    """
    if not downloaded_symbols(raw_dir):
        raise FileNotFoundError(f"No OHLCV files under {raw_dir / OHLCV_DIR}")
    pattern = (raw_dir / OHLCV_DIR / "*.parquet").as_posix()
    columns = ", ".join(OHLCV_COLUMNS)
    query = f"SELECT {columns} FROM read_parquet(?) ORDER BY symbol, date"
    with duckdb.connect() as con:
        panel: pd.DataFrame = con.execute(query, [pattern]).df()
    return panel


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dict as pretty-printed UTF-8 JSON, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object written by :func:`write_json`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain a JSON object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def write_text(text: str, path: Path) -> None:
    """Write a UTF-8 text file (reports), creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_figure(figure: Figure, path: Path, dpi: int) -> None:
    """Save a Matplotlib figure, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def content_sha256(frame: pd.DataFrame, keys: tuple[str, ...] = ("date", "symbol")) -> str:
    """SHA-256 of a frame's values, independent of row order and of the file format.

    Rows are sorted by ``keys`` and columns by name, then hashed row by row with pandas, so
    two panels built from the same prices give the same hash even if written differently.
    """
    ordered = frame.sort_values(list(keys), kind="stable")[sorted(frame.columns)]
    rows = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(rows.tobytes()).hexdigest()
