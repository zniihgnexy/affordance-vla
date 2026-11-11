"""Tests for moving a demonstrated approach to another contact point."""
import numpy as np
import pytest

from affordance3d.counterfactual import counterfactual_chunk, retarget_path, walk


def approach_path() -> np.ndarray:
    """20 cm sideways at a height of 20 cm, then 20 cm straight down: 5 mm per step."""
    across = np.stack([np.linspace(0.0, 0.2, 41), np.zeros(41), np.full(41, 0.2)], axis=1)
    down = np.stack([np.full(40, 0.2), np.zeros(40), np.linspace(0.2, 0.0, 41)[1:]], axis=1)
    return np.vstack([across, down])


def test_retargeted_path_keeps_its_start_and_ends_at_the_new_contact():
    # Arrange
    path, new_end = approach_path(), np.array([0.3, 0.1, 0.0])

    # Act
    moved = retarget_path(path, new_end)

    # Assert
    np.testing.assert_allclose(moved[0], path[0])
    np.testing.assert_allclose(moved[-1], new_end)


def test_retargeted_path_moves_the_final_approach_as_a_whole():
    # Arrange
    path, shift = approach_path(), np.array([0.1, 0.1, 0.0])

    # Act
    moved = retarget_path(path, path[-1] + shift, rigid_tail=0.08)

    # Assert: the last 8 cm (16 steps of 5 mm) are the demonstrated descent, shifted
    np.testing.assert_allclose(moved[-17:], path[-17:] + shift, atol=1e-6)


def test_retargeting_to_the_same_end_returns_the_same_path():
    # Arrange
    path = approach_path()

    # Act
    moved = retarget_path(path, path[-1])

    # Assert
    np.testing.assert_allclose(moved, path)


def test_retarget_rejects_a_path_without_length():
    # Act / Assert
    with pytest.raises(ValueError, match="no length"):
        retarget_path(np.zeros((5, 3)), np.ones(3))


def test_walk_returns_points_at_the_given_distances_and_stops_at_the_end():
    # Arrange
    polyline = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    # Act
    points = walk(polyline, np.array([0.5, 1.5, 9.0]))

    # Assert
    np.testing.assert_allclose(points, [[0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0]])


def test_chunk_targets_follow_the_demonstration_when_the_contact_is_unchanged():
    # Arrange
    path = approach_path()

    # Act
    target = counterfactual_chunk(path, path[-1], horizon=8, flip_remaining=0.015)

    # Assert
    np.testing.assert_allclose(target.points, path[1:9], atol=1e-9)
    assert target.valid.all() and not target.closed.any()


def test_chunk_targets_never_ask_for_more_than_the_fastest_step():
    # Arrange: a contact 1.2 m further away would need more than three times the demonstrated speed
    path = approach_path()

    # Act
    target = counterfactual_chunk(path, path[-1] + np.array([1.2, 0.0, 0.0]), horizon=8, flip_remaining=0.015,
                                  max_step=0.0125)

    # Assert
    steps = np.linalg.norm(np.diff(np.vstack([path[0], target.points]), axis=0), axis=1)
    assert steps.max() <= 0.0125 + 1e-9
    assert steps.min() > 0.005  # and it does move faster than the demonstration did


def test_chunk_targets_stop_at_the_contact():
    # Arrange: three steps of the demonstration are left
    path = approach_path()[-4:]

    # Act
    target = counterfactual_chunk(path, path[-1], horizon=8, flip_remaining=0.0)

    # Assert: steps after the arrival are not supervised and stay at the contact
    assert target.valid.tolist() == [True, True, True, False, False, False, False, False]
    np.testing.assert_allclose(target.points[2:], np.tile(path[-1], (6, 1)), atol=1e-9)


def test_gripper_closes_within_the_flip_distance_of_the_new_contact():
    # Arrange: 4 cm above the contact, 5 mm per step, the demonstration closed 1.6 cm before contact
    path = approach_path()[-9:]

    # Act
    target = counterfactual_chunk(path, path[-1], horizon=8, flip_remaining=0.016)

    # Assert: a step that starts 4.0, 3.5, 3.0, 2.5 or 2.0 cm from the contact is open; from 1.5 cm on, closed
    assert target.closed.tolist() == [False, False, False, False, False, True, True, True]


def test_a_path_shorter_than_the_rigid_tail_takes_the_whole_shift_in_its_first_step():
    # Arrange: 2 cm of descent are left, less than the 8 cm that are moved as a whole
    path, shift = approach_path()[-5:], np.array([0.05, 0.0, 0.0])

    # Act
    moved = retarget_path(path, path[-1] + shift, rigid_tail=0.08)

    # Assert: the start stays, every later point is the demonstrated one, shifted
    np.testing.assert_allclose(moved[0], path[0])
    np.testing.assert_allclose(moved[1:], path[1:] + shift, atol=1e-9)
