"""Success rates of RoboCasa365 evaluation episodes: per (task, condition) cell and per condition, with intervals."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a success rate."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _entry(wins: int, n: int) -> dict[str, Any]:
    return {"n": n, "wins": wins, "rate": wins / n, "wilson95": [round(x, 3) for x in wilson(wins, n)]}


def summarise(rows: Iterable[dict]) -> dict[str, dict]:
    """Counts, rates and Wilson 95 % intervals: `cells["Task/Cond"]` and `by_condition["Cond"]` (tasks pooled)."""
    cells: dict[str, list[int]] = {}
    for row in rows:
        counts = cells.setdefault(f"{row['task']}/{row['condition']}", [0, 0])
        counts[0] += int(row["success"])
        counts[1] += 1
    pooled: dict[str, list[int]] = {}
    for key, (wins, n) in cells.items():
        total = pooled.setdefault(key.split("/")[1], [0, 0])
        total[0] += wins
        total[1] += n
    return {
        "cells": {key: _entry(*counts) for key, counts in sorted(cells.items())},
        "by_condition": {key: _entry(*counts) for key, counts in sorted(pooled.items())},
    }


def markdown_table(summary: dict[str, dict], tasks: list[str], conditions: list[str]) -> str:
    """Rows = tasks, columns = conditions; each cell `wins/n (rate; Wilson interval)`; last row pools the tasks."""
    def show(entry: dict | None) -> str:
        if entry is None:
            return "-"
        low, high = entry["wilson95"]
        return f"{entry['wins']}/{entry['n']} ({entry['rate']:.0%}; {low:.0%}–{high:.0%})"

    lines = ["| task | " + " | ".join(conditions) + " |", "|---|" + "---|" * len(conditions)]
    for task in tasks:
        lines.append(f"| {task} | " + " | ".join(show(summary["cells"].get(f"{task}/{c}")) for c in conditions) + " |")
    lines.append("| all tasks | " + " | ".join(show(summary["by_condition"].get(c)) for c in conditions) + " |")
    return "\n".join(lines)
