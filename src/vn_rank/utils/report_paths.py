"""Where each report lives: one folder per project stage under ``paths.reports_dir``.

The folders are numbered so that a directory listing follows the order of the project. The
summary figure stays at the top of ``reports/`` because the README opens with it.
"""

from __future__ import annotations

from pathlib import Path

DATA = "1_data"  # validate-data: data-quality report, tickers per year
WALK_FORWARD = "2_walk_forward"  # train and backtest: signal and portfolio results
ABLATION = "3_ablation"  # ablation: feature groups, sensitivity, SHAP
LOCKBOX = "4_lockbox"  # lockbox: one-time evaluation and its lock file
MONITORING = "5_monitoring"  # update: realized Rank IC of the frozen model
EXPLORATORY = "6_exploratory"  # post-lockbox checks, labelled exploratory


def section(reports_dir: Path, name: str) -> Path:
    """Folder of one project stage inside the reports directory."""
    return reports_dir / name
