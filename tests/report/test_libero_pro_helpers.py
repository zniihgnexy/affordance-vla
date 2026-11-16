"""Tables of report/libero_pro_helpers.py over several evaluation seeds."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from report.libero_pro_helpers import SUITES, per_task_table, seed_table

SWAP, TASK = SUITES.values()


def logs(solved: int, total: int) -> list[SimpleNamespace]:
    """Stand-ins for rollout logs of which the first `solved` reached the benchmark's goal."""
    return [SimpleNamespace(meta={"success": k < solved}, path=Path(f"task0_ep{k}.npz")) for k in range(total)]


def test_the_seed_table_gives_each_seed_and_the_mean_and_spread_over_seeds():
    # Arrange: one policy, three seeds of 500 rollouts per perturbation
    runs = {"policy": {7: {SWAP: logs(300, 500), TASK: logs(250, 500)},
                       8: {SWAP: logs(320, 500), TASK: logs(260, 500)},
                       9: {SWAP: logs(310, 500), TASK: logs(270, 500)}}}

    # Act
    table = seed_table(runs)

    # Assert
    row = table.loc[("policy", list(SUITES)[0].split(":")[0])]
    assert row["seed 7"] == "300 / 500" and row["seed 9"] == "310 / 500"
    assert row["mean"] == pytest.approx(0.62)
    assert row["std over seeds"] == pytest.approx(np.std([0.60, 0.64, 0.62], ddof=1))
    assert table.loc[("policy", list(SUITES)[1].split(":")[0]), "mean"] == pytest.approx(0.52)


def test_a_seed_that_is_missing_or_incomplete_is_shown_and_left_out_of_the_mean():
    # Arrange: seed 8 has only part of its rollouts, seed 9 has none
    runs = {"policy": {7: {SWAP: logs(300, 500), TASK: logs(250, 500)},
                       8: {SWAP: logs(100, 200), TASK: logs(90, 200)},
                       9: {SWAP: [], TASK: []}}}

    # Act
    table = seed_table(runs, expected=500)

    # Assert
    row = table.loc[("policy", list(SUITES)[0].split(":")[0])]
    assert row["seed 8"] == "100 / 200 (incomplete)"
    assert row["seed 9"] == "not run"
    assert row["mean"] == pytest.approx(0.60)
    assert row["seeds in the mean"] == 1
    assert np.isnan(row["std over seeds"])


def test_the_per_task_table_shows_a_dash_where_a_policy_has_no_rollouts_of_a_task():
    # Arrange: the second policy has not run task 1 yet
    def task_logs(task: int, solved: int, total: int) -> list[SimpleNamespace]:
        return [SimpleNamespace(meta={"success": k < solved, "instruction": f"pick up thing {task}"},
                                path=Path(f"task{task}_ep{k}.npz")) for k in range(total)]

    runs = {"first": {SWAP: task_logs(0, 3, 4) + task_logs(1, 1, 4)}, "second": {SWAP: task_logs(0, 2, 4)}}

    # Act
    table = per_task_table(runs, SWAP)

    # Assert
    assert table.loc[1, "first"] == "1 / 4"
    assert table.loc[0, "second"] == "2 / 4"
    assert table.loc[1, "second"] == "–"
