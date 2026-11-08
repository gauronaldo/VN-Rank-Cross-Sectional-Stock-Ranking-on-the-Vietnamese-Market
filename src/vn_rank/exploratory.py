"""Exploratory analysis after the lockbox (I/O layer): low-volatility baselines.

**Exploratory, not part of the protocol locked before results.** The lockbox has been run, so
nothing here may change the primary result. The question is whether LightGBM adds
anything over a one-line low-volatility rule, the best-known effect in its features:

- ``lowvol_vol60``: minus the rank of the 60-day return volatility, the textbook
  definition of the low-volatility effect (known before this project);
- ``lowvol_atr14``: minus the rank of ATR(14) / close, picked **after** seeing that it
  was the strongest single feature on the walk-forward test blocks, so it is optimistic.

Both are scored on exactly the rows LightGBM was scored on, traded with the same engine,
costs, constraints, and benchmark, and compared on Rank IC (paired with LightGBM), net
excess return over equal weight, and annualized alpha after adjusting for beta to the
equal-weight benchmark (a low-volatility portfolio has a low beta, so its raw excess
return mixes skill with market exposure).

The definitions are module constants rather than config keys, so that the config hash
recorded in ``reports/4_lockbox/lockbox_run.lock`` stays verifiable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from vn_rank import final, pipeline
from vn_rank.backtest import runner
from vn_rank.config import Config
from vn_rank.data import storage
from vn_rank.evaluation import bootstrap
from vn_rank.features import labels
from vn_rank.utils import report_paths
from vn_rank.utils.markdown import markdown_table

LOW_VOLATILITY_RULES = {"lowvol_vol60": "rank_vol_60", "lowvol_atr14": "rank_atr_14"}
COMPARED_MODELS = ("momentum", "elastic_net", runner.PRIMARY_MODEL, "mlp")
REPORT_STEM = "exploratory_low_volatility"
SUMMARY_FIGURE = "summary.png"
# Summary figure: models on the Rank IC panel and portfolios on the excess-return panel.
SUMMARY_MODELS = ("momentum", "elastic_net", runner.PRIMARY_MODEL, "mlp", "lowvol_vol60")
SUMMARY_PORTFOLIOS = (
    runner.PRIMARY_MODEL,
    runner.PRIMARY_MODEL + runner.BAND_SUFFIX,
    "lowvol_vol60",
)
DISPLAY_NAMES = {
    "momentum": "Momentum", "elastic_net": "Elastic Net", "lightgbm": "LightGBM", "mlp": "MLP",
    "lowvol_vol60": "Low-vol rule", "lightgbm_band": "LightGBM\nturnover band",
}  # fmt: skip
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, validated palette
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"
BAR_WIDTH, BAR_GAP = 0.26, 0.03  # bar width and the surface gap between adjacent bars
logger = logging.getLogger(__name__)


def low_volatility_predictions(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Compared models' predictions plus one low-volatility score per rule.

    The low-volatility rows are the rows the baselines were scored on (one seed), so every
    strategy ranks the same stocks on the same dates.
    """
    keys = predictions.loc[
        predictions["model"] == "momentum", ["date", "symbol", "forward_return", "fold"]
    ]
    rows = keys.merge(
        panel[["date", "symbol", *LOW_VOLATILITY_RULES.values()]], on=["date", "symbol"]
    )
    rules = [
        rows.assign(score=-rows[column], model=name, seed=0)[predictions.columns]
        for name, column in LOW_VOLATILITY_RULES.items()
    ]
    compared = predictions[predictions["model"].isin(COMPARED_MODELS)]
    return pd.concat([compared, *rules], ignore_index=True)


def alpha_table(report: runner.BacktestReport, settings: runner.BacktestSettings) -> pd.DataFrame:
    """Beta-adjusted alpha against the equal-weight benchmark for every strategy curve."""
    benchmark = report.curves[runner.EQUAL_WEIGHT]
    rows = {}
    for name, daily in report.curves.items():
        if name == runner.EQUAL_WEIGHT:
            continue
        aligned = benchmark.reindex(daily.index).fillna(0.0)
        rows[name] = bootstrap.alpha_beta_ci(
            daily.to_numpy(), aligned.to_numpy(), settings.bootstrap, settings.trading_days
        )
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


_PORTFOLIO_COLUMNS = {
    "ann_return": "ann. return", "ann_turnover": "turnover / yr", "ann_cost": "cost / yr",
    "gross_excess_ann_return": "gross excess vs EW / yr", "excess_ann_return": "excess vs EW / yr",
    "excess_ci_low": "excess CI low", "excess_ci_high": "excess CI high", "max_drawdown": "max DD",
}  # fmt: skip


def _style_axis(axis: Axes, title: str) -> None:
    axis.set_title(title, fontsize=10, color=TEXT_COLOR, loc="left")
    axis.axhline(0.0, color=MUTED_COLOR, linewidth=0.8)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED_COLOR, labelsize=8)


def _grouped_bars(axis: Axes, groups: list[str], series: dict[str, pd.DataFrame], fmt: str) -> None:
    """Grouped bars; each series frame has ``value`` and optional ``low``/``high`` columns."""
    x = np.arange(len(groups))
    step = BAR_WIDTH + BAR_GAP
    start = -step * (len(series) - 1) / 2
    label_style: dict[str, Any] = {"ha": "center", "fontsize": 6.5, "color": TEXT_COLOR}
    for i, (label, frame) in enumerate(series.items()):
        values = frame["value"].to_numpy(dtype=float)
        low = frame["low"].to_numpy(dtype=float) if "low" in frame else values
        high = frame["high"].to_numpy(dtype=float) if "high" in frame else values
        positions = x + start + i * step
        axis.bar(positions, values, BAR_WIDTH, color=SERIES_COLORS[i], label=label,
                 yerr=[values - low, high - values] if "low" in frame else None,
                 error_kw={"ecolor": MUTED_COLOR, "elinewidth": 0.8, "capsize": 2})  # fmt: skip
        for position, value, top, bottom in zip(positions, values, high, low, strict=True):
            anchor, offset = (top, (0, 3)) if value >= 0 else (bottom, (0, -9))
            axis.annotate(format(value, fmt), (position, anchor), xytext=offset,
                          textcoords="offset points", **label_style)  # fmt: skip
    axis.set_xticks(x, [DISPLAY_NAMES.get(g, g) for g in groups])
    axis.margins(y=0.12)
    axis.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                ncol=len(series))  # fmt: skip


def summary_figure(
    walk_forward: runner.BacktestReport, lockbox: runner.BacktestReport, alphas: pd.DataFrame
) -> Figure:
    """Two panels: mean Rank IC with CIs per period, and the walk-forward cost story."""
    figure, (ic_axis, money_axis) = plt.subplots(1, 2, figsize=(13, 4.8))
    models = list(SUMMARY_MODELS)
    ic = {
        label: report.ic.reindex(models).rename(
            columns={"mean_ic": "value", "ci_low": "low", "ci_high": "high"}
        )
        for label, report in (
            ("Walk-forward 2022-2025", walk_forward),
            ("Lockbox 2025-2026", lockbox),
        )
    }
    _grouped_bars(ic_axis, models, ic, ".3f")
    _style_axis(ic_axis, "Mean Rank IC with 95% CI: a one-line low-volatility rule ties the models")
    portfolios = list(SUMMARY_PORTFOLIOS)
    summary = walk_forward.summary.reindex(portfolios)
    money = {
        "Excess vs equal weight, before costs": summary[["gross_excess_ann_return"]].set_axis(
            ["value"], axis=1
        ),
        "Excess vs equal weight, after costs": summary[["excess_ann_return"]].set_axis(
            ["value"], axis=1
        ),
        "Alpha after beta, after costs": alphas.reindex(portfolios)[["alpha"]].set_axis(
            ["value"], axis=1
        ),
    }
    _grouped_bars(money_axis, portfolios, {k: v * 100 for k, v in money.items()}, "+.1f")
    _style_axis(money_axis, "Walk-forward, % per year: costs erase the edge; no alpha after beta")
    figure.tight_layout()
    return figure


def period_section(title: str, report: runner.BacktestReport, alphas: pd.DataFrame) -> list[str]:
    """Markdown lines for one period: Rank IC, portfolio metrics, and alpha after beta."""
    portfolio = report.summary.reindex(columns=list(_PORTFOLIO_COLUMNS)).rename(
        columns=_PORTFOLIO_COLUMNS
    )
    return [
        f"## {title}",
        "",
        "Rank IC with 95% block-bootstrap intervals; `diff_vs_primary` is LightGBM minus the row.",
        "",
        *markdown_table(report.ic.reset_index(), pipeline.DIGITS),
        "",
        "Portfolios (top decile, net of costs):",
        "",
        *markdown_table(portfolio.reset_index(), pipeline.DIGITS),
        "",
        "Annualized alpha after beta to the equal-weight benchmark (net of costs):",
        "",
        *markdown_table(alphas.reset_index(), pipeline.DIGITS),
        "",
    ]


def _period_report(
    config: Config, predictions: pd.DataFrame, before: pd.Timestamp | None
) -> runner.BacktestReport:
    prices = pipeline.load_prices(config, before)
    entry = labels.label_entry_dates(prices.calendar, config.label)
    market = pipeline.market_data(config, prices)
    settings = pipeline.backtest_settings(config)
    return runner.backtest_all(predictions, entry, (market, prices.index_close), settings)


def exploratory(config: Config) -> int:
    """Compare the low-volatility rules with LightGBM on walk-forward and lockbox rows."""
    processed, settings = config.paths.processed_dir, pipeline.backtest_settings(config)
    lockbox = pd.Timestamp(config.validation.lockbox_start)
    periods = {
        "Walk-forward test blocks (2022-01 to 2025-09)": (
            pipeline.PREDICTIONS_FILE, pipeline.PANEL_FILE, lockbox
        ),
        "Lockbox (2025-10 to 2026-10)": (
            final.LOCKBOX_PREDICTIONS_FILE, final.FULL_PANEL_FILE, None
        ),
    }  # fmt: skip
    lines = [
        "# Exploratory: low-volatility baselines versus LightGBM",
        "",
        "**Exploratory analysis after the lockbox; it does not change any pre-registered result.**",
        "`lowvol_vol60` is the textbook low-volatility rule; `lowvol_atr14` was chosen after",
        "seeing the test blocks and is optimistic. "
        + pipeline.provenance(pipeline.run_info(config), "exploratory"),
        "",
    ]
    reports, alphas = [], []
    for title, (predictions_file, panel_file, before) in periods.items():
        predictions = low_volatility_predictions(
            storage.read_parquet(processed / predictions_file),
            storage.read_parquet(processed / panel_file),
        )
        reports.append(_period_report(config, predictions, before))
        alphas.append(alpha_table(reports[-1], settings))
        lines += period_section(title, reports[-1], alphas[-1])
    reports_dir = config.paths.reports_dir
    exploratory_dir = report_paths.section(reports_dir, report_paths.EXPLORATORY)
    storage.write_text("\n".join(lines), exploratory_dir / f"{REPORT_STEM}.md")
    figure = summary_figure(reports[0], reports[1], alphas[0])
    storage.save_figure(figure, reports_dir / SUMMARY_FIGURE, config.project.figure_dpi)
    logger.info("Exploratory:\n%s", "\n".join(lines))
    return 0
