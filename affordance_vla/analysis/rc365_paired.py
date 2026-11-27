"""Paired comparisons of RoboCasa365 evaluation runs (AFFO-19): the same (task, index) seed rebuilt the same kitchen,
so two runs, two checkpoints or two object splits can be compared episode by episode. Pure Python."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

Key = tuple[str, int]


def outcomes(rows: Iterable[dict], condition: str | None = None) -> dict[Key, bool]:
    """{(task, index): success} of the rows (of one condition, when given)."""
    return {(r["task"], r["index"]): bool(r["success"]) for r in rows if condition in (None, r["condition"])}


def paired_counts(first: dict[Key, bool], second: dict[Key, bool]) -> dict[str, int]:
    """Over the keys both have: both succeeded / only the first / only the second / neither."""
    counts = Counter(
        "both" if first[k] and second[k] else "only_first" if first[k] else "only_second" if second[k] else "neither"
        for k in first.keys() & second.keys()
    )
    return {name: counts.get(name, 0) for name in ("both", "only_first", "only_second", "neither")}


def paired_by_task(first: dict[Key, bool], second: dict[Key, bool]) -> dict[str, dict[str, int]]:
    tasks = sorted({task for task, _ in first.keys() & second.keys()})
    return {
        task: paired_counts(
            {k: v for k, v in first.items() if k[0] == task}, {k: v for k, v in second.items() if k[0] == task}
        )
        for task in tasks
    }


def sign_test_p(only_first: int, only_second: int) -> float:
    """Exact two-sided sign test (McNemar, exact) on the discordant pairs."""
    n = only_first + only_second
    if n == 0:
        return 1.0
    low = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(low + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_difference(only_first: int, only_second: int, n: int, z: float = 1.96) -> dict[str, float]:
    """The difference in success rate (first minus second) over `n` pairs, with a normal-approximation 95 % interval
    for paired proportions (the variance comes from the discordant pairs only)."""
    diff = (only_first - only_second) / n
    half = z * math.sqrt(max(only_first + only_second - (only_first - only_second) ** 2 / n, 0.0)) / n
    return {"difference": diff, "low": diff - half, "high": diff + half}


def smallest_detectable_difference(n: int, discordant: int, alpha: float = 0.05) -> float | None:
    """The smallest difference in success rate (as a fraction of the `n` pairs) the exact sign test calls significant
    when `discordant` of the pairs differ: how big a gap this many episodes can show. None if no split of the
    discordant pairs reaches `alpha`."""
    for larger in range((discordant + 1) // 2 + (discordant % 2 == 0), discordant + 1):
        if sign_test_p(larger, discordant - larger) < alpha:
            return (2 * larger - discordant) / n
    return None


def agreement(new: Sequence[dict], old: Sequence[dict]) -> dict[str, Any]:
    """How far two runs of the same cells agree, episode by episode (matched on task, condition, index): the same
    success, the same step count, and the same recorded end-effector trace."""
    index = {(r["task"], r["condition"], r["index"]): r for r in old}
    matched = [
        (r, index[(r["task"], r["condition"], r["index"])])
        for r in new
        if (r["task"], r["condition"], r["index"]) in index
    ]
    if not matched:
        return {"matched": 0}
    count = lambda test: sum(test(a, b) for a, b in matched)  # noqa: E731
    return {
        "matched": len(matched),
        "same_success": count(lambda a, b: bool(a["success"]) == bool(b["success"])),
        "same_steps": count(lambda a, b: a["steps"] == b["steps"]),
        "same_trace": count(lambda a, b: a["trace"] == b["trace"]),
        "success_flips": [
            (a["task"], a["index"], bool(b["success"]), bool(a["success"]))
            for a, b in matched
            if bool(a["success"]) != bool(b["success"])
        ],
    }
