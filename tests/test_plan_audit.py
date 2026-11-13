"""Where the plans given to a rollout pointed, against the oracle grasp points of the scene."""
import numpy as np

from affordance3d.plan_audit import ELSEWHERE, NO_PLAN, RELEASE_PLAN, plan_point, plans_before_grasp
from affordance3d.plan_input import no_plan_features, plan_features

DOWN = np.array([0.0, 0.0, -1.0])
GRIPPER = np.array([0.0, 0.0, 0.30])
POINTS = {"soup_1": np.array([0.10, 0.00, 0.05]), "milk_1": np.array([0.10, 0.20, 0.08])}


def grasp_plan(contact: np.ndarray, gripper: np.ndarray = GRIPPER) -> np.ndarray:
    return plan_features(gripper, contact, DOWN, closing=True)


def test_a_grasp_plan_at_the_named_objects_grasp_point_points_at_it():
    found = plan_point(grasp_plan(POINTS["soup_1"] + [0.0, 0.01, 0.0]), GRIPPER, POINTS, "soup_1")

    assert found.pointed == "soup_1"
    assert np.isclose(found.error_m, 0.01, atol=1e-6)


def test_a_grasp_plan_nearer_to_another_object_points_at_that_object():
    found = plan_point(grasp_plan(POINTS["milk_1"] + [0.0, -0.02, 0.0]), GRIPPER, POINTS, "soup_1")

    assert found.pointed == "milk_1"
    assert np.isclose(found.error_m, np.linalg.norm(POINTS["milk_1"] + [0.0, -0.02, 0.0] - POINTS["soup_1"]), atol=1e-6)


def test_a_grasp_plan_far_from_every_object_points_elsewhere():
    found = plan_point(grasp_plan(np.array([0.60, -0.40, 0.05])), GRIPPER, POINTS, "soup_1")

    assert found.pointed == ELSEWHERE
    assert found.error_m > 0.5


def test_a_release_plan_and_an_empty_plan_point_at_no_object():
    release = plan_features(GRIPPER, POINTS["soup_1"], DOWN, closing=False)

    assert plan_point(release, GRIPPER, POINTS, "soup_1").pointed == RELEASE_PLAN
    assert plan_point(no_plan_features(), GRIPPER, POINTS, "soup_1").pointed == NO_PLAN
    assert np.isnan(plan_point(release, GRIPPER, POINTS, "soup_1").error_m)


def test_only_the_queries_before_the_first_grasp_of_any_object_are_judged():
    # Arrange: 4 queries every 8 steps; the milk (not the named object) is held from step 12 on
    steps = 32
    names = ("milk_1", "soup_1")
    object_pos = np.tile(np.stack([POINTS["milk_1"], POINTS["soup_1"]]), (steps + 1, 1, 1))
    ee_pos = np.tile(GRIPPER, (steps + 1, 1))
    grasped = np.zeros((steps + 1, 2), dtype=bool)
    grasped[12:, 0] = True
    plans = np.stack([grasp_plan(POINTS["soup_1"])] * 4)
    offsets = {"milk_1": np.zeros(3), "soup_1": np.zeros(3)}

    # Act
    points = plans_before_grasp(plans, np.array([0, 8, 16, 24]), ee_pos, object_pos, names, offsets, "soup_1", grasped)

    # Assert: the queries at steps 0 and 8 come before the grasp that starts at step 12
    assert [point.pointed for point in points] == ["soup_1", "soup_1"]


def test_objects_without_a_known_grasp_point_are_not_candidates():
    # Arrange: the basket has no grasp offset; the plan lies right at it
    names = ("basket_1", "soup_1")
    object_pos = np.tile(np.stack([np.array([0.5, 0.5, 0.0]), POINTS["soup_1"]]), (9, 1, 1))
    plans = np.stack([grasp_plan(np.array([0.5, 0.5, 0.0]))])

    # Act
    points = plans_before_grasp(plans, np.array([0]), np.tile(GRIPPER, (9, 1)), object_pos, names,
                                {"soup_1": np.zeros(3)}, "soup_1", np.zeros((9, 2), dtype=bool))

    # Assert
    assert points[0].pointed == ELSEWHERE
