"""Tables of a finished (or running) AFFO-16 evaluation run: success per task x condition, the A-vs-B pairing, time.

Reads <run>/episodes.jsonl; writes <run>/summary.json and <run>/table.md and prints the table.
Episode i of A and of B has the same seed, hence the same kitchen: the pairs say how often the new object instances
flip an episode (A only = succeeded with seen instances and failed with unseen ones).

  python scripts/rc365/summarise_eval.py <run folder under $W/runs/robocasa365_eval>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.analysis.rc365_stats import markdown_table, summarise  # noqa: E402


def paired_ab(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per task, over episodes present in both A and B: both succeeded / A only / B only / neither."""
    outcome = {(r["task"], r["condition"], r["index"]): r["success"] for r in rows}
    pairs: dict[str, Counter] = {}
    for (task, condition, index), success in outcome.items():
        if condition == "A" and (task, "B", index) in outcome:
            key = ("both" if success and outcome[(task, "B", index)] else "A only" if success
                   else "B only" if outcome[(task, "B", index)] else "neither")
            pairs.setdefault(task, Counter())[key] += 1
    return {task: dict(counter) for task, counter in pairs.items()}


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    config = json.loads((run / "config.json").read_text())
    summary = summarise(rows)
    summary["paired_AB"] = paired_ab(rows)
    summary["seconds_per_episode_mean"] = round(sum(r["seconds"] for r in rows) / len(rows), 1)
    summary["episodes"] = len(rows)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    conditions = sorted({r["condition"] for r in rows})
    table = markdown_table(summary, config["tasks"], conditions)
    (run / "table.md").write_text(table + "\n")
    print(table)
    print(f"\n{len(rows)} episodes, {summary['seconds_per_episode_mean']} s per episode on average")
    print("paired A/B (same kitchen):", json.dumps(summary["paired_AB"]))


if __name__ == "__main__":
    main()
