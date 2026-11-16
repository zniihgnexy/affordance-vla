"""Plan–action consistency: an action chunk's path compared with the CAW plan still ahead."""
import numpy as np
import pytest

from affordance_vla.plan.caw_labels import GripperEvent, caw_labels
from affordance_vla.plan.plan_consistency import (
    PlanAhead,
    chunk_consistency,
    chunk_path,
    consistency_along_episode,
    distance_to_polyline,
    plan_ahead,
    point_along,
)


def straight_plan(length: float = 1.0) -> PlanAhead:
    polyline = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]])
    return PlanAhead(polyline=polyline, approach=np.array([1.0, 0.0, 0.0]), steps_to_event=10, closing=True)


def test_chunk_path_starts_at_the_current_position_and_sums_scaled_actions():
    # Arrange
    actions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    # Act
    path = chunk_path(np.array([0.5, 0.0, 0.0]), actions, scale=0.01)

    # Assert
    np.testing.assert_allclose(path, [[0.5, 0, 0], [0.51, 0, 0], [0.52, 0, 0], [0.52, 0.02, 0]])


def test_chunk_path_rejects_full_action_vectors():
    with pytest.raises(ValueError, match=r"\(H, 3\)"):
        chunk_path(np.zeros(3), np.zeros((8, 7)), scale=0.01)


def test_plan_ahead_keeps_only_waypoints_after_the_current_step():
    # Arrange: move along +x for 8 steps, grasp at step 8; waypoints sit at steps 2, 4, 6, 8
    ee = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    labels = caw_labels(ee, [GripperEvent(step=8, closing=True)], n_waypoints=4)

    # Act
    plan = plan_ahead(labels, ee[5], step=5)

    # Assert
    np.testing.assert_allclose(plan.polyline[:, 0], [5.0, 6.0, 8.0])
    assert plan.steps_to_event == 3
    assert plan.closing is True
    np.testing.assert_allclose(plan.approach, [1.0, 0.0, 0.0])


def test_plan_ahead_switches_to_the_next_segment_at_an_event_and_ends_after_the_last():
    ee = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    labels = caw_labels(ee, [GripperEvent(step=3, closing=True), GripperEvent(step=7, closing=False)], n_waypoints=2)

    assert plan_ahead(labels, ee[3], step=3).closing is False  # the grasp has happened: the release plan is next
    assert plan_ahead(labels, ee[7], step=7) is None
    assert plan_ahead([], ee[0], step=0) is None


def test_point_along_walks_across_corners_and_stops_at_the_end():
    corner = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    np.testing.assert_allclose(point_along(corner, 0.25), [0.25, 0.0, 0.0])
    np.testing.assert_allclose(point_along(corner, 1.5), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(point_along(corner, 9.0), [1.0, 1.0, 0.0])


def test_distance_to_polyline_uses_the_nearest_segment_or_end_point():
    corner = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    points = np.array([[0.5, 0.2, 0.0], [1.3, 0.5, 0.0], [-0.3, 0.4, 0.0]])

    np.testing.assert_allclose(distance_to_polyline(points, corner), [0.2, 0.3, 0.5])


def test_chunk_along_the_plan_is_fully_consistent():
    path = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])

    result = chunk_consistency(path, straight_plan())

    assert result.path_deviation == pytest.approx(0.0)
    assert result.direction_cosine == pytest.approx(1.0)
    assert result.net_motion == pytest.approx(0.2)


def test_chunk_sideways_or_backwards_is_not_consistent():
    sideways = np.array([[0.0, 0, 0], [0.0, 0.1, 0], [0.0, 0.2, 0]])
    backwards = np.array([[0.0, 0, 0], [-0.1, 0, 0], [-0.2, 0, 0]])

    side = chunk_consistency(sideways, straight_plan())
    back = chunk_consistency(backwards, straight_plan())

    assert side.direction_cosine == pytest.approx(0.0, abs=1e-12)
    assert side.path_deviation == pytest.approx(0.15)  # mean of 0.1 and 0.2
    assert back.direction_cosine == pytest.approx(-1.0)


def test_direction_follows_the_plan_around_a_corner():
    # Arrange: the plan turns 90° after 0.1 m; a chunk that cuts the corner travels 0.2·√2/… along the diagonal
    plan = PlanAhead(polyline=np.array([[0.0, 0, 0], [0.1, 0, 0], [0.1, 0.5, 0]]), approach=np.array([0.0, 1, 0]),
                     steps_to_event=20, closing=True)
    path = np.array([[0.0, 0, 0], [0.1, 0.1, 0]])

    # Act
    result = chunk_consistency(path, plan)

    # Assert: walking the same distance (0.1414 m) along the plan reaches (0.1, 0.0414)
    wanted = np.array([0.1, np.hypot(0.1, 0.1) - 0.1, 0.0])
    expected = np.dot(path[1], wanted) / (np.linalg.norm(path[1]) * np.linalg.norm(wanted))
    assert result.direction_cosine == pytest.approx(expected)


def test_direction_is_undefined_when_the_chunk_barely_moves():
    path = np.array([[0.0, 0, 0], [1e-5, 0, 0]])

    assert np.isnan(chunk_consistency(path, straight_plan()).direction_cosine)


def test_chunk_consistency_needs_at_least_two_path_points():
    with pytest.raises(ValueError, match="at least two"):
        chunk_consistency(np.zeros((1, 3)), straight_plan())


def test_consistency_along_episode_scores_each_chunk_against_its_own_plan():
    # Arrange: the end effector moves 1 cm per step along +x and grasps at step 8; actions are 1 unit = 1 cm
    ee = np.stack([0.01 * np.arange(12.0), np.zeros(12), np.zeros(12)], axis=1)
    labels = caw_labels(ee, [GripperEvent(step=8, closing=True)], n_waypoints=4)
    along = np.tile([1.0, 0.0, 0.0], (4, 1))
    sideways = np.tile([0.0, 1.0, 0.0], (4, 1))

    # Act: a chunk along the plan at step 0, one sideways at step 2, one after the last event
    results = consistency_along_episode(labels, ee, [0, 2, 9], np.stack([along, sideways, along]), scale=0.01)

    # Assert
    assert results[0].direction_cosine == pytest.approx(1.0)
    assert results[0].path_deviation == pytest.approx(0.0, abs=1e-12)
    assert results[1].direction_cosine == pytest.approx(0.0, abs=1e-12)
    assert results[2] is None


def test_consistency_along_episode_needs_one_chunk_per_query_step():
    with pytest.raises(ValueError, match="query steps"):
        consistency_along_episode([], np.zeros((5, 3)), [0, 1], np.zeros((1, 4, 3)), scale=0.01)


def test_consistency_along_episode_ignores_the_part_of_a_chunk_after_the_event():
    # Arrange: approach along +x until the grasp at step 4, then lift along +z. A chunk requested at step 2
    # that follows the demonstration turns upwards after two steps; only those two steps belong to this plan.
    ee = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.04, 0, 0.01],
                   [0.04, 0, 0.02], [0.04, 0, 0.03]])
    labels = caw_labels(ee, [GripperEvent(step=4, closing=True)], n_waypoints=2)
    chunk = np.array([[1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1.0], [0.0, 0, 1.0]])

    # Act
    (result,) = consistency_along_episode(labels, ee, [2], chunk[None], scale=0.01)

    # Assert
    assert result.direction_cosine == pytest.approx(1.0)
    assert result.net_motion == pytest.approx(0.02)
    assert result.path_deviation == pytest.approx(0.0, abs=1e-12)
