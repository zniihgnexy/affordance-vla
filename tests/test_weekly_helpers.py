"""Tables of report/weekly_helpers.py, read from the per-task summaries that the rollout script writes."""
import json

import numpy as np
import pytest

from report.weekly_helpers import count_table, summary_counts

SWAP, TASK = "libero_object_swap", "libero_object_task"


def write_run(root, name: str, solved: dict[str, list[int]], per_task: int = 4) -> None:
    """A run folder whose task k of each suite has `per_task` rollouts, of which solved[suite][k] succeed."""
    for suite, counts in solved.items():
        folder = root / name / suite
        folder.mkdir(parents=True)
        for task, done in enumerate(counts):
            rows = [{"success": k < done, "episode": k} for k in range(per_task)]
            (folder / f"task{task}_summary.json").write_text(json.dumps(rows))


def test_the_counts_of_a_suite_add_up_the_summaries_of_its_tasks(tmp_path):
    # Arrange
    write_run(tmp_path, "run", {SWAP: [4, 1, 0]})

    # Act / Assert
    assert summary_counts(tmp_path / "run", SWAP) == (5, 12)


def test_a_suite_without_summaries_has_no_counts(tmp_path):
    write_run(tmp_path, "run", {SWAP: [4]})

    assert summary_counts(tmp_path / "run", TASK) == (0, 0)


def test_the_count_table_gives_each_run_its_counts_and_rates_and_marks_missing_runs(tmp_path):
    # Arrange
    write_run(tmp_path, "a", {SWAP: [4, 2], TASK: [1, 1]})

    # Act
    table = count_table({"first": "a", "second": "missing"}, tmp_path, {"position": SWAP, "task": TASK})

    # Assert
    assert table.loc["first", "position"] == "6 / 8"
    assert table.loc["first", "position rate"] == pytest.approx(0.75)
    assert table.loc["first", "task rate"] == pytest.approx(0.25)
    assert table.loc["second", "position"] == "not run"
    assert np.isnan(table.loc["second", "position rate"])
