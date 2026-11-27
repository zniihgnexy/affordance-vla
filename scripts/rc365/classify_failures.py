"""Failure classes per task and condition of a finished AFFO-16 evaluation run (rules in analysis/rc365_failures.py).

  python scripts/rc365/classify_failures.py <run folder under $W/runs/robocasa365_eval>   # writes <run>/failures.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.analysis.rc365_failures import count_failures  # noqa: E402


def main() -> None:
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    failures = count_failures(rows)
    timeouts = sum(1 for r in rows if not r["success"] and r["steps"] >= r["horizon"])
    (run / "failures.json").write_text(json.dumps({"failed": sum(not r["success"] for r in rows),
                                                   "timeouts": timeouts, "classes": failures}, indent=2))
    for key, counts in failures.items():
        print(f"{key:40s} {counts}")
    print(f"failed episodes {sum(not r['success'] for r in rows)}, of which timeouts {timeouts}")


if __name__ == "__main__":
    main()
