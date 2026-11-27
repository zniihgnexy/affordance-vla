"""Tables of the AFFO-19 base-policy checks: success against training steps, and seen against unseen object instances
with the question "how big a gap can this many episodes show". Pure Python on lists of episode records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from affordance_vla.analysis.rc365_paired import (
    outcomes,
    paired_by_task,
    paired_counts,
    paired_difference,
    sign_test_p,
    smallest_detectable_difference,
)
from affordance_vla.analysis.rc365_stats import _entry

GRASP_OBJECT = "obj"  # the config name of the object a pick task moves


def cell(rows: Sequence[dict]) -> dict[str, Any]:
    """Wins, n, rate and Wilson 95 % interval of the rows."""
    return _entry(sum(bool(r["success"]) for r in rows), len(rows))


def by_task(rows: Iterable[dict]) -> dict[str, dict[str, Any]]:
    """{task: cell, "all": cell over every row}."""
    rows = list(rows)
    out = {task: cell([r for r in rows if r["task"] == task]) for task in sorted({r["task"] for r in rows})}
    return {**out, "all": cell(rows)}


def comparison(first: Sequence[dict], second: Sequence[dict]) -> dict[str, Any]:
    """Paired comparison of two row sets with matching (task, index): counts, exact sign-test p, the difference in
    success rate with its interval, and the smallest difference the sign test could call significant with this many
    discordant pairs."""
    a, b = outcomes(first), outcomes(second)
    counts = paired_counts(a, b)
    n = sum(counts.values())
    discordant = counts["only_first"] + counts["only_second"]
    return {
        "pairs": n,
        "counts": counts,
        "p_sign_test": sign_test_p(counts["only_first"], counts["only_second"]),
        **({"paired_difference": paired_difference(counts["only_first"], counts["only_second"], n)} if n else {}),
        "smallest_detectable_difference": smallest_detectable_difference(n, discordant) if n else None,
    }


def steps_report(runs: Mapping[int, Sequence[dict]]) -> dict[str, Any]:
    """`runs` maps a checkpoint's step count to its condition-A rows. Success per task and in total at each step count,
    and the paired comparison of each checkpoint with the next one."""
    steps = sorted(runs)
    return {
        "success": {str(s): by_task(runs[s]) for s in steps},
        "adjacent_pairs": {
            f"{s}->{t}": {
                "all": comparison(runs[s], runs[t]),
                "by_task": paired_by_task(outcomes(runs[s]), outcomes(runs[t])),
            }
            for s, t in zip(steps[:-1], steps[1:])
        },
    }


def instance_category(row: dict) -> str | None:
    """The category of the object the episode moves, from the objects probe (or record_objects)."""
    found = row.get("objects") or {}
    info = found.get(GRASP_OBJECT)
    return info["category"] if isinstance(info, dict) else info


def instances_report(seen: Sequence[dict], unseen: Sequence[dict]) -> dict[str, Any]:
    """Seen (A) against unseen (B) object instances, same seeds: per task and pooled, the success of each side, the
    paired comparison, and the success by category of the moved object (counts only; a small category is a count)."""
    tasks = sorted({r["task"] for r in seen})
    per_task = {
        task: {
            "seen": cell([r for r in seen if r["task"] == task]),
            "unseen": cell([r for r in unseen if r["task"] == task]),
            **comparison([r for r in seen if r["task"] == task], [r for r in unseen if r["task"] == task]),
        }
        for task in tasks
    }
    pooled = {"seen": cell(seen), "unseen": cell(unseen), **comparison(seen, unseen)}
    return {"per_task": per_task, "pooled": pooled, "by_category": category_table(seen, unseen)}


def category_table(seen: Sequence[dict], unseen: Sequence[dict]) -> dict[str, dict[str, Any]]:
    """{category: {"seen": [wins, n], "unseen": [wins, n]}} over the rows whose moved object's category is known."""
    table: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"seen": [0, 0], "unseen": [0, 0]})
    for side, rows in (("seen", seen), ("unseen", unseen)):
        for row in rows:
            category = instance_category(row)
            if category is not None:
                table[category][side][0] += int(bool(row["success"]))
                table[category][side][1] += 1
    return dict(sorted(table.items()))
