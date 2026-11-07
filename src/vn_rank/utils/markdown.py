"""Plain Markdown tables for the committed reports (no extra dependency)."""

from __future__ import annotations

import pandas as pd


def markdown_table(frame: pd.DataFrame, digits: int) -> list[str]:
    """Render ``frame`` (columns as headers, index ignored) as Markdown table lines.

    Floats are shown with ``digits`` decimals; everything else with ``str``.
    """
    cells = frame.map(lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    return [header, rule, *("| " + " | ".join(row) + " |" for row in cells.to_numpy())]
