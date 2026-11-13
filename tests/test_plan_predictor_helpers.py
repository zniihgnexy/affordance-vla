"""Tables of report/plan_predictor_helpers.py about where the plans of closed-loop rollouts pointed."""
import numpy as np

from affordance3d.plan_audit import ELSEWHERE, NO_PLAN, PlanPoint
from report.plan_predictor_helpers import GraspPlans, heights_when_near, per_task_pointing, pointing_table


def rollout(task: int, solved: bool, first: str, error_cm: float = 1.0, target: str = "soup_1") -> GraspPlans:
    error = np.nan if first == NO_PLAN else error_cm / 100.0
    return GraspPlans(task, solved, target, (PlanPoint(first, error),))


def test_the_pointing_table_counts_first_plans_by_where_they_pointed():
    # Arrange: two right (one solved), one at another object, one nowhere near, one without a plan
    group = [rollout(0, True, "soup_1", 1.0), rollout(0, False, "soup_1", 3.0), rollout(1, False, "milk_1", 20.0),
             rollout(1, False, ELSEWHERE, 40.0), rollout(2, False, NO_PLAN)]

    # Act
    row = pointing_table({"policy": group}).loc["policy"]

    # Assert
    assert row["rollouts"] == 5
    assert row["first plan is a grasp plan"] == "4 / 5"
    assert row["first plan points at the named object"] == "2 / 4"
    assert row["first plan points at another object"] == "1 / 4"
    assert row["first plan: median error (cm)"] == 11.5
    assert row["first plan within 2 cm"] == "1 / 4"
    assert row["solved"] == "1 / 5"
    assert row["solved when the first plan pointed at the named object"] == "1 / 2"
    assert row["solved when it did not"] == "0 / 3"


def test_a_rollout_without_any_query_before_a_grasp_counts_as_having_no_grasp_plan():
    row = pointing_table({"policy": [GraspPlans(0, False, "soup_1", ())]}).loc["policy"]

    assert row["first plan is a grasp plan"] == "0 / 1"
    assert row["first plan points at the named object"] == "–"


def test_the_per_task_table_shows_a_dash_for_a_task_a_policy_did_not_run():
    # Arrange: the second policy has no rollout of task 1
    audits = {"first": [rollout(0, True, "soup_1"), rollout(1, False, "milk_1")],
              "second": [rollout(0, False, "soup_1")]}

    # Act
    table = per_task_pointing(audits)

    # Assert
    assert table.loc[0, "first"] == "plan right 1 · solved 1 / 1"
    assert table.loc[1, "first"] == "plan right 0 · solved 0 / 1"
    assert table.loc[1, "second"] == "–"


def test_the_height_is_read_where_the_gripper_first_comes_within_each_horizontal_distance():
    # Arrange: 10 cm to the side at a height of 20 cm, across to above the point, then straight down
    across = np.stack([np.linspace(0.10, 0.0, 11), np.zeros(11), np.full(11, 0.20)], axis=1)
    down = np.stack([np.zeros(10), np.zeros(10), np.linspace(0.18, 0.0, 10)], axis=1)
    ee_pos = np.vstack([across, down])

    # Act
    heights = heights_when_near(ee_pos, np.zeros(3), marks=(0.06, 0.02))

    # Assert: the gripper is above the point before it comes down
    np.testing.assert_allclose(heights, [0.20, 0.20])


def test_a_diagonal_descent_is_lower_when_it_comes_near_and_a_distance_never_reached_has_no_height():
    # Arrange: from 10 cm to the side and 20 cm up, straight to a point 3 cm beside the target
    ee_pos = np.linspace([0.10, 0.0, 0.20], [0.03, 0.0, 0.0], 8)

    # Act
    heights = heights_when_near(ee_pos, np.zeros(3), marks=(0.06, 0.02))

    # Assert
    assert 0.0 < heights[0] < 0.20
    assert np.isnan(heights[1])
