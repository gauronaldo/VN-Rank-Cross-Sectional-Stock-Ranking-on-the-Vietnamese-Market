"""Download daily OHLCV from vnstock into per-symbol Parquet files (Phase 1).

Resumable, rate-limited, and retrying. Records download date, source, and vnstock
version as metadata. One price source only (``vci``); sources are never mixed.
"""
