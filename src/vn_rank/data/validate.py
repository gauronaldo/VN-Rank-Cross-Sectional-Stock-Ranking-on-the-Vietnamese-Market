"""Data-quality checks on the raw OHLCV snapshot.

Pure functions: they take the OHLCV panel (``storage.OHLCV_COLUMNS``, sorted by symbol and
date), the symbol list, and the VNINDEX trading calendar, and return flags and summary
tables. Nothing is dropped or changed here; how flagged rows are treated is decided by the
author and applied when the processed panel is built (Phase 2).

Checks, one boolean column each:

- ``duplicate``: more than one row for the same symbol and date;
- ``missing_value``: a price or the volume is missing;
- ``non_positive_price``: a price is zero or negative;
- ``ohlc_inconsistent``: ``high`` below, or ``low`` above, another price of the row;
- ``zero_volume``: a row with volume 0 (no trade that day);
- ``band_breach``: ``|close / previous close - 1|`` exceeds the exchange price band plus a
  tolerance (the previous row may be several sessions earlier);
- ``off_calendar``: a date that is not a VNINDEX trading day.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

PRICE_COLUMNS = ("open", "high", "low", "close")
DELISTED = "DELISTED"
STATUS_LISTED, STATUS_DELISTED = "listed", "delisted"
_SERIES_COLORS = {STATUS_LISTED: "#2a78d6", STATUS_DELISTED: "#eb6834"}
_TEXT_SECONDARY = "#52514e"


@dataclasses.dataclass(frozen=True)
class QualitySettings:
    """Exchange price bands and the tolerance of the band check."""

    price_limits: dict[str, float]
    band_tolerance: float


@dataclasses.dataclass(frozen=True)
class QualityReport:
    """Result of :func:`run_quality_checks`.

    Attributes:
        flags: Rows with at least one flag: ``date``, ``symbol``, ``exchange``, ``ret``, and
            one boolean column per check.
        counts: Flagged rows and symbols per check.
        coverage: Per symbol: first and last date, rows, traded days, VNINDEX sessions
            between first and last date, and sessions with no row.
        tickers_per_year: Symbols with at least one trade, per year and listing status.
    """

    flags: pd.DataFrame
    counts: pd.DataFrame
    coverage: pd.DataFrame
    tickers_per_year: pd.DataFrame


def daily_returns(panel: pd.DataFrame) -> pd.Series:
    """Close-to-close return versus the previous row of the same symbol (NaN for the first)."""
    return panel.groupby("symbol", sort=False)["close"].pct_change()


def flag_rows(
    panel: pd.DataFrame, exchange: pd.Series, settings: QualitySettings, calendar: pd.Index
) -> pd.DataFrame:
    """Compute one boolean column per check for every panel row.

    Args:
        panel: OHLCV panel sorted by symbol and date.
        exchange: Exchange code indexed by symbol.
        settings: Price bands per exchange code and band tolerance.
        calendar: VNINDEX trading dates.

    Returns:
        A boolean frame (NumPy ``bool``) aligned with ``panel``.

    Raises:
        KeyError: If a symbol has no exchange or an exchange has no price band.
    """
    prices = panel[list(PRICE_COLUMNS)]
    codes = panel["symbol"].map(exchange).astype(object)
    unknown = set(codes.dropna()) - set(settings.price_limits)
    if codes.isna().any() or unknown:
        raise KeyError(f"no price band for exchanges {sorted(unknown)} or symbols without one")
    band = codes.map(settings.price_limits).astype(float)
    return pd.DataFrame(
        {
            "duplicate": panel.duplicated(["symbol", "date"], keep=False),
            "missing_value": panel[[*PRICE_COLUMNS, "volume"]].isna().any(axis=1),
            "non_positive_price": (prices <= 0).any(axis=1),
            "ohlc_inconsistent": (panel["high"] < prices.max(axis=1))
            | (panel["low"] > prices.min(axis=1)),
            "zero_volume": panel["volume"] == 0,
            "band_breach": daily_returns(panel).abs() > band + settings.band_tolerance,
            "off_calendar": ~panel["date"].isin(calendar),
        },
        index=panel.index,
    )


def calendar_coverage(panel: pd.DataFrame, calendar: pd.Index) -> pd.DataFrame:
    """Compare each symbol's rows with the VNINDEX sessions between its first and last date.

    Returns:
        Columns ``first``, ``last``, ``rows``, ``traded_days``, ``expected_sessions``, and
        ``missing_sessions``, indexed by symbol.
    """
    dates = panel.groupby("symbol")["date"]
    sessions = np.sort(calendar.to_numpy())
    coverage = pd.DataFrame({"first": dates.min(), "last": dates.max(), "rows": dates.size()})
    coverage["traded_days"] = panel["volume"].gt(0).groupby(panel["symbol"]).sum()
    first = np.searchsorted(sessions, coverage["first"].to_numpy(), side="left")
    last = np.searchsorted(sessions, coverage["last"].to_numpy(), side="right")
    coverage["expected_sessions"] = last - first
    on_calendar = panel.loc[panel["date"].isin(calendar)].drop_duplicates(["symbol", "date"])
    observed = on_calendar.groupby("symbol").size().reindex(coverage.index, fill_value=0)
    coverage["missing_sessions"] = coverage["expected_sessions"] - observed
    return coverage


def tickers_per_year(panel: pd.DataFrame, exchange: pd.Series) -> pd.DataFrame:
    """Count symbols with at least one traded day per calendar year, by listing status."""
    traded = panel.loc[panel["volume"] > 0, ["symbol", "date"]]
    delisted = traded["symbol"].map(exchange).eq(DELISTED)
    table = (
        traded.assign(
            year=traded["date"].dt.year,
            status=np.where(delisted, STATUS_DELISTED, STATUS_LISTED),
        )
        .groupby(["year", "status"])["symbol"]
        .nunique()
        .unstack(fill_value=0)
    )
    return table.reindex(columns=[STATUS_LISTED, STATUS_DELISTED], fill_value=0)


def run_quality_checks(
    panel: pd.DataFrame, symbols: pd.DataFrame, calendar: pd.Index, settings: QualitySettings
) -> QualityReport:
    """Run every check and build the summary tables.

    Args:
        panel: OHLCV panel sorted by symbol and date.
        symbols: Symbol list with ``symbol`` and ``exchange``.
        calendar: VNINDEX trading dates.
        settings: Price bands and band tolerance.

    Returns:
        The quality report; the panel itself is not modified.
    """
    exchange = symbols.set_index("symbol")["exchange"]
    flags = flag_rows(panel, exchange, settings, calendar)
    counts = pd.DataFrame(
        {
            "rows": flags.sum(),
            "symbols": [panel.loc[flags[c], "symbol"].nunique() for c in flags.columns],
        }
    )
    flagged = panel.loc[flags.any(axis=1), ["date", "symbol"]].assign(
        exchange=lambda f: f["symbol"].map(exchange), ret=daily_returns(panel)
    )
    return QualityReport(
        flags=pd.concat([flagged, flags.loc[flagged.index]], axis=1).reset_index(drop=True),
        counts=counts,
        coverage=calendar_coverage(panel, calendar),
        tickers_per_year=tickers_per_year(panel, exchange),
    )


def plot_tickers_per_year(table: pd.DataFrame) -> Figure:
    """Stacked bars of traded symbols per year, listed versus delisted (survivorship view)."""
    figure = Figure(figsize=(8, 4))
    axes = figure.add_subplot()
    positions = np.arange(len(table))
    bottom = np.zeros(len(table))
    for status in table.columns:
        bars = axes.bar(
            positions, table[status], bottom=bottom, width=0.6, label=status,
            color=_SERIES_COLORS[status], edgecolor="white", linewidth=2,
        )  # fmt: skip
        axes.bar_label(bars, label_type="center", color="white", fontsize=8)
        bottom = bottom + table[status].to_numpy()
    axes.set_title("Symbols with at least one trade per year (first and last years are partial)")
    axes.set_xticks(positions, [str(year) for year in table.index])
    axes.set_ylabel("symbols")
    axes.legend(frameon=False, loc="upper left")
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(colors=_TEXT_SECONDARY)
    axes.grid(axis="y", alpha=0.3)
    axes.set_axisbelow(True)
    return figure


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame (index included) as a GitHub Markdown table."""
    table = frame.reset_index()
    lines = ["| " + " | ".join(map(str, table.columns)) + " |"]
    lines.append("|" + "---|" * len(table.columns))
    lines += ["| " + " | ".join(map(_cell, row)) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def _cell(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: QualityReport, metadata: dict[str, object], top_n: int) -> str:
    """Render the data-quality report as Markdown.

    Args:
        report: Output of :func:`run_quality_checks`.
        metadata: Snapshot metadata (source, dates, vnstock version, status counts).
        top_n: Number of largest band breaches and least-covered symbols to list.
    """
    coverage, flags = report.coverage, report.flags
    breaches = flags.loc[flags["band_breach"], ["date", "symbol", "exchange", "ret"]]
    worst = breaches.reindex(breaches["ret"].abs().sort_values(ascending=False).index)
    gaps = coverage.assign(share=coverage["missing_sessions"] / coverage["expected_sessions"])
    sections = [
        "# Data-quality report\n",
        "## Snapshot\n",
        "\n".join(f"- **{k}:** {v}" for k, v in sorted(metadata.items())),
        "\n## Flags per check\n",
        markdown_table(report.counts),
        f"\n## Coverage\n\n{len(coverage)} symbols with data; median rows "
        f"{coverage['rows'].median():.0f}; symbols with at least one missing VNINDEX "
        f"session: {int((coverage['missing_sessions'] > 0).sum())}.\n",
        markdown_table(gaps.nlargest(top_n, "share")),
        "\n## Symbols traded per year\n",
        markdown_table(report.tickers_per_year),
        f"\n## Largest band breaches (top {top_n})\n",
        markdown_table(worst.head(top_n).set_index("date")),
    ]
    return "\n".join(sections) + "\n"
