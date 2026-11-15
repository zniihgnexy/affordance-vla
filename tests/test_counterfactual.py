"""Tests for moving a demonstrated approach to another contact point."""
import numpy as np
import pytest

from affordance3d.counterfactual import Retarget, counterfactual_chunk, retarget_path, walk


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
    moved = retarget_path(path, path[-1] + shift, Retarget(rigid_tail=0.08))

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
    moved = retarget_path(path, path[-1] + shift, Retarget(rigid_tail=0.08))

    # Assert: the start stays, every later point is the demonstrated one, shifted
    np.testing.assert_allclose(moved[0], path[0])
    np.testing.assert_allclose(moved[1:], path[1:] + shift, atol=1e-9)


ALIGN = Retarget(rigid_tail=0.18, align_first=True)


def test_a_late_shift_is_made_at_the_current_height_before_the_descent_goes_on():
    # Arrange: 10 cm of the descent are left, less than the 18 cm that are moved as a whole
    path, shift = approach_path()[-21:], np.array([0.05, 0.0, 0.0])

    # Act
    moved = retarget_path(path, path[-1] + shift, ALIGN)

    # Assert: first across to above the new contact, then the demonstrated descent, shifted
    np.testing.assert_allclose(moved[0], path[0])
    np.testing.assert_allclose(moved[1], path[0] + shift)
    np.testing.assert_allclose(moved[1:], path + shift, atol=1e-9)


def test_a_shift_longer_than_the_free_part_of_the_path_is_made_first():
    # Arrange: 40 cm of path, 18 cm rigid, so 22 cm are free; the contact moves by 30 cm
    path, shift = approach_path(), np.array([0.0, 0.3, 0.0])

    # Act
    moved = retarget_path(path, path[-1] + shift, ALIGN)

    # Assert
    np.testing.assert_allclose(moved[1:], path + shift, atol=1e-9)


def test_a_shift_that_fits_into_the_free_part_of_the_path_is_blended_in_before_the_rigid_tail():
    # Arrange: 22 cm are free, the contact moves by 10 cm
    path, shift = approach_path(), np.array([0.0, 0.1, 0.0])

    # Act
    moved = retarget_path(path, path[-1] + shift, ALIGN)

    # Assert: same number of points; the last 18 cm (36 steps) are the demonstrated ones, shifted
    assert len(moved) == len(path)
    np.testing.assert_allclose(moved[0], path[0])
    np.testing.assert_allclose(moved[-37:], path[-37:] + shift, atol=1e-9)
    assert 0.0 < moved[20, 1] < 0.1  # half-way along the free part the shift is partly taken up


def test_aligning_first_leaves_an_unmoved_contact_as_demonstrated():
    path = approach_path()[-21:]

    moved = retarget_path(path, path[-1], ALIGN)

    np.testing.assert_allclose(moved, path)


def test_chunk_targets_of_a_late_shift_keep_the_height_until_the_gripper_is_above_the_new_contact():
    # Arrange: 10 cm above the contact, which moves 5 cm to the side; the demonstration walks 5 mm a step
    path, shift = approach_path()[-21:], np.array([0.05, 0.0, 0.0])

    # Act
    target = counterfactual_chunk(path, path[-1] + shift, horizon=8, flip_remaining=0.0, max_step=0.0125,
                                  retarget=ALIGN)

    # Assert: the pace is stretched to the longer path (7.5 mm a step); six steps go across at the starting
    # height, and the descent only starts once the gripper is above the new contact
    np.testing.assert_allclose(target.points[:6, 2], path[0, 2])
    np.testing.assert_allclose(target.points[:6, 0], path[0, 0] + 0.0075 * np.arange(1, 7))
    np.testing.assert_allclose(target.points[6:, 0], path[0, 0] + 0.05)
    assert target.points[7, 2] < target.points[6, 2] < path[0, 2]


def test_the_default_retargeting_is_the_one_of_the_earlier_runs():
    # Arrange
    path, new_end = approach_path(), np.array([0.3, 0.1, 0.0])

    # Act
    default = retarget_path(path, new_end)
    explicit = retarget_path(path, new_end, Retarget(rigid_tail=0.08, align_first=False))

    # Assert
    np.testing.assert_array_equal(default, explicit)


def test_a_shift_as_long_as_the_free_part_of_the_path_is_still_blended_and_a_longer_one_is_made_first():
    # Arrange: 22 cm of the path are free
    path = approach_path()

    # Act
    fits = retarget_path(path, path[-1] + [0.0, 0.22, 0.0], ALIGN)
    too_long = retarget_path(path, path[-1] + [0.0, 0.221, 0.0], ALIGN)

    # Assert
    assert len(fits) == len(path)
    assert len(too_long) == len(path) + 1


def test_a_shift_made_first_also_covers_the_difference_in_height():
    # Arrange: the new contact is 5 cm to the side and 4 cm higher, with 10 cm of the descent left
    path, shift = approach_path()[-21:], np.array([0.05, 0.0, 0.04])

    # Act
    moved = retarget_path(path, path[-1] + shift, ALIGN)

    # Assert: a straight leg to the shifted start, then the demonstrated descent, moved
    np.testing.assert_allclose(moved[1], path[0] + shift)
    np.testing.assert_allclose(moved[-1], path[-1] + shift)


def test_the_gripper_closes_at_the_same_distance_from_the_contact_when_the_shift_is_made_first():
    # Arrange: 10 cm above the contact, the demonstration closed 1.6 cm before it; the contact moves 5 cm
    path, shift = approach_path()[-21:], np.array([0.05, 0.0, 0.0])

    # Act: the moved path is 15 cm long and is walked in 20 steps of 7.5 mm
    target = counterfactual_chunk(path, path[-1] + shift, horizon=24, flip_remaining=0.016, max_step=0.0125,
                                  retarget=ALIGN)

    # Assert: steps that start 1.6 cm or less from the new contact are closed (the last two of the 20, which
    # start 1.5 and 0.75 cm from it), and the steps after the arrival are not valid
    assert target.closed.tolist() == [False] * 18 + [True] * 6
    assert target.valid.tolist() == [True] * 20 + [False] * 4


SCALE = Retarget(rigid_tail=0.18, scale_from_start=True)


def detour_path() -> np.ndarray:
    """From (0, 0, 16 cm) 30 cm along -y at that height, then 16 cm straight down: 1 cm per step."""
    across = np.stack([np.zeros(31), np.linspace(0.0, -0.30, 31), np.full(31, 0.16)], axis=1)
    down = np.stack([np.zeros(17), np.full(17, -0.30), np.linspace(0.16, 0.0, 17)], axis=1)[1:]
    return np.vstack([across, down])


def test_scaling_about_the_start_leaves_an_unmoved_contact_as_demonstrated():
    path = detour_path()

    moved = retarget_path(path, path[-1], SCALE)

    np.testing.assert_allclose(moved, path, atol=1e-12)


def test_scaling_about_the_start_never_moves_away_from_a_contact_close_to_the_start():
    # Arrange: the new contact lies 5 cm beside the start, below it; the demonstration went 30 cm the other way
    path, new = detour_path(), np.array([0.0, 0.05, 0.0])

    # Act
    moved = retarget_path(path, new, SCALE)

    # Assert: horizontally the gripper only comes closer, and the moved path ends at the new contact
    horizontal = np.hypot(*(moved - new)[:, :2].T)
    assert horizontal.max() <= horizontal[0] + 1e-9
    np.testing.assert_allclose(moved[-1], new)


def test_the_last_part_is_moved_as_a_whole_and_shortened_when_the_contact_is_close():
    # Arrange: the straight distance is 16.8 cm, so the last part is half of it, 8.4 cm: the last 9 points
    path, new = detour_path(), np.array([0.0, 0.05, 0.0])

    # Act
    moved = retarget_path(path, new, SCALE)

    # Assert: the last 9 points are only moved; with the full 18 cm the corner of the path would be too
    np.testing.assert_allclose(moved[-9:], path[-9:] + (new - path[-1]), atol=1e-12)
    assert not np.allclose(moved[-19:], path[-19:] + (new - path[-1]))


def test_the_last_part_keeps_the_full_rigid_length_when_the_contact_is_far():
    # Arrange: 60 cm away, so the last part is the full 18 cm (here the whole 16 cm descent and 2 cm before it)
    path, new = detour_path(), np.array([0.0, -0.60, 0.0])

    # Act
    moved = retarget_path(path, new, SCALE)

    # Assert
    np.testing.assert_allclose(moved[-19:], path[-19:] + (new - path[-1]), atol=1e-12)


def test_a_given_final_approach_replaces_the_demonstrated_one():
    # Arrange: an approach low from the side, 10 cm long, given relative to its own contact
    path, new = detour_path(), np.array([0.0, -0.60, 0.0])
    side = np.stack([np.linspace(-0.10, 0.0, 11), np.zeros(11), np.full(11, 0.02)], axis=1)
    side[-1, 2] = 0.0

    # Act
    moved = retarget_path(path, new, SCALE, tail=side)

    # Assert: the moved path ends with the given approach, placed at the new contact
    np.testing.assert_allclose(moved[-11:], new + side, atol=1e-12)


def test_the_free_part_ends_where_the_last_part_begins():
    # Arrange
    path, new = detour_path(), np.array([0.10, -0.15, 0.03])

    # Act
    moved = retarget_path(path, new, SCALE)

    # Assert: no jump anywhere along the moved path
    steps = np.linalg.norm(np.diff(moved, axis=0), axis=1)
    assert steps.max() < 0.03


def test_a_rule_cannot_both_align_first_and_scale_from_the_start():
    with pytest.raises(ValueError, match="one of"):
        Retarget(align_first=True, scale_from_start=True)
