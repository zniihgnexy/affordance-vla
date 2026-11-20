"""Tables and the figure of the weekly review notebooks (report/2026-11-09_weekly_review.ipynb).

The counts come from the per-task summaries that scripts/eval/rollout_libero_logged.py writes next to the
rollout logs (`task<k>_summary.json`, one row per rollout with the benchmark's own `success`), so no log has
to be opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colours checked with the dataviz validator (categorical slots 1 and 2, light surface); ink for text and marks
SERIES = ("#2a78d6", "#eb6834")
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def summary_counts(folder: Path, suite: str) -> tuple[int, int]:
    """(solved, rollouts) of one suite of a run folder; (0, 0) if it has no summaries."""
    rows = [row for path in sorted((Path(folder) / suite).glob("task*_summary.json"))
            for row in json.loads(path.read_text())]
    return sum(bool(row["success"]) for row in rows), len(rows)


def count_table(runs: dict[str, str], root: Path, suites: dict[str, str]) -> pd.DataFrame:
    """One row per run (label -> folder under `root`): solved / rollouts and the rate for each suite
    (column label -> suite name). A run without summaries is marked "not run"."""
    table = {}
    for label, folder in runs.items():
        row = {}
        for column, suite in suites.items():
            done, total = summary_counts(Path(root) / folder, suite)
            row[column] = f"{done} / {total}" if total else "not run"
            row[f"{column} rate"] = done / total if total else np.nan
        table[label] = row
    return pd.DataFrame.from_dict(table, orient="index")


def progression_figure(table: pd.DataFrame, columns: tuple[str, str], bounds: dict[str, float] | None = None,
                       title: str = "") -> plt.Figure:
    """Horizontal grouped bars of the success rates of `table` (one row per configuration, top to bottom),
    one bar per column of `columns` (e.g. the two perturbations), the rate printed at each bar's tip.
    `bounds` (column -> rate) draws an upper bound as a dark tick across the chart for each column."""
    rows = list(table.index)[::-1]
    fig, axis = plt.subplots(figsize=(10.5, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    height, positions = 0.36, np.arange(len(rows))
    for k, (column, colour) in enumerate(zip(columns, SERIES)):
        rates = table.loc[rows, f"{column} rate"].to_numpy(dtype=float)
        offset = (0.5 - k) * height
        axis.barh(positions + offset, np.nan_to_num(rates), height=height * 0.9, color=colour, label=column)
        for y, rate in zip(positions + offset, rates):
            if np.isfinite(rate):
                axis.text(rate + 0.008, y, f"{rate:.2f}", va="center", fontsize=8, color=INK)
        if bounds and column in bounds:
            axis.axvline(bounds[column], color=MUTED, linestyle=(0, (4, 3)) if k else "-", linewidth=1.2,
                         label=f"oracle plan, {column}: {bounds[column]:.2f}")
    axis.set_yticks(positions, rows, fontsize=9, color=INK)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("success rate (benchmark goal check)", color=MUTED)
    axis.tick_params(colors=MUTED)
    axis.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="lower right", fontsize=8, frameon=False)
    if title:
        axis.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    return fig
