"""Measurements of report/task_perturbation_helpers.py on rollouts and on the paths of training labels."""
from types import SimpleNamespace

import numpy as np
import pytest

from affordance3d.counterfactual import Retarget
from report.task_perturbation_helpers import first_reach_step, label_path_metrics, stall_steps, tail_shape

POINT = np.zeros(3)


def log_with(ee_pos: np.ndarray) -> SimpleNamespace:
    """A stand-in for a rollout log with one object, the named one, whose grasp point is at the origin."""
    steps = len(ee_pos)
    return SimpleNamespace(ee_pos=np.asarray(ee_pos, dtype=float), object_pos=np.zeros((steps, 1, 3)), target_index=0,
                           target="milk_1")


OFFSETS = SimpleNamespace(grasp={"milk_1": np.zeros(3)})


def test_the_first_reach_is_the_first_step_within_two_centimetres_of_the_grasp_point():
    # Arrange: straight down from 10.5 cm in steps of 1 cm
    log = log_with(np.stack([np.zeros(11), np.zeros(11), np.linspace(0.105, 0.005, 11)], axis=1))

    # Act / Assert: step 8 is 2.5 cm above, step 9 is 1.5 cm above
    assert first_reach_step(log, OFFSETS) == 9
    assert first_reach_step(log_with(np.tile([0.1, 0.0, 0.1], (5, 1))), OFFSETS) is None


def test_stalling_counts_the_steps_spent_almost_still_far_from_the_grasp_point():
    # Arrange: 20 steps still at 20 cm, then 10 steps moving down quickly, then 5 still at the grasp point
    still_far = np.tile([0.0, 0.0, 0.20], (20, 1))
    moving = np.stack([np.zeros(10), np.zeros(10), np.linspace(0.18, 0.0, 10)], axis=1)
    still_near = np.zeros((5, 3))
    log = log_with(np.vstack([still_far, moving, still_near]))

    # Act
    stalled = stall_steps(log, OFFSETS)

    # Assert: the 19 still steps far away count; the still steps at the grasp point do not
    assert stalled == 19


def test_a_path_moved_to_a_contact_near_its_start_makes_a_detour_away_from_it():
    # Arrange: the demonstration goes 30 cm sideways and down; the new contact is 5 cm from the start, below it
    across = np.stack([np.zeros(31), np.linspace(0.0, -0.30, 31), np.full(31, 0.16)], axis=1)
    down = np.stack([np.zeros(16), np.full(16, -0.30), np.linspace(0.16, 0.0, 16)], axis=1)[1:]
    path = np.vstack([across, down])
    new = np.array([0.0, 0.05, 0.0])

    # Act
    metrics = label_path_metrics(path, new, Retarget(rigid_tail=0.18, align_first=True))

    # Assert: much longer than the straight line, and it first moves away from the new contact
    assert metrics["detour"] > 2.0
    assert metrics["away"] > 0.10


def test_the_tail_shape_says_where_the_last_part_of_an_approach_starts_and_how_low_it_moves_sideways():
    # Arrange: 20 cm sideways at 3 cm above the contact, then 3 cm straight down
    side = np.stack([np.linspace(-0.20, 0.0, 21), np.zeros(21), np.full(21, 0.03)], axis=1)
    down = np.stack([np.zeros(4), np.zeros(4), np.linspace(0.03, 0.0, 4)], axis=1)[1:]

    # Act
    shape = tail_shape(np.vstack([side, down]), rigid_tail=0.10)

    # Assert: the last 10 cm start 7 cm aside at 3 cm height, and all of that sideways travel is low
    assert shape["start_height"] == pytest.approx(0.03)
    assert shape["start_aside"] == pytest.approx(0.07)
    assert shape["low_aside"] == pytest.approx(0.07)
