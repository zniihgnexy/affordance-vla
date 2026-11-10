"""Outcome label of a closed-loop rollout from per-step grasp flags."""
import numpy as np
import pytest

from affordance3d.rollout_outcome import (NO_GRASP, NOT_DELIVERED, SUCCESS, WRONG_OBJECT, classify, first_grasp_steps)

NAMES = ["soup", "milk"]


def grasp_log(steps: int, **runs: tuple[int, int]) -> np.ndarray:
    """(T, 2) flags with one grasp interval [start, end) per named object."""
    log = np.zeros((steps, len(NAMES)), dtype=bool)
    for name, (start, end) in runs.items():
        log[start:end, NAMES.index(name)] = True
    return log


def test_first_grasp_needs_a_run_of_consecutive_steps():
    # Arrange: soup is touched for 2 steps at step 3 (a brush), then held from step 10
    log = grasp_log(30, soup=(10, 25))
    log[3:5, 0] = True

    # Act
    steps = first_grasp_steps(log, min_steps=5)

    # Assert
    assert steps == [10, None]


def test_delivering_the_target_after_grasping_it_first_is_a_success():
    outcome = classify(NAMES, grasp_log(40, soup=(10, 30)), target="soup", delivered=True)

    assert outcome.label == SUCCESS
    assert outcome.first_grasped == "soup"
    assert outcome.first_grasp_step == 10
    assert outcome.grasped == ("soup",)


def test_grasping_another_object_first_is_a_wrong_object_error():
    outcome = classify(NAMES, grasp_log(40, milk=(8, 20), soup=(25, 35)), target="soup", delivered=True)

    assert outcome.label == WRONG_OBJECT  # the first grasp decides, even if the target is delivered later
    assert outcome.first_grasped == "milk"
    assert outcome.grasped == ("milk", "soup")
    assert outcome.delivered is True


def test_never_grasping_anything_is_a_no_grasp_error():
    outcome = classify(NAMES, grasp_log(40), target="soup", delivered=False)

    assert outcome.label == NO_GRASP
    assert outcome.first_grasped is None
    assert outcome.first_grasp_step is None


def test_grasping_the_target_without_delivering_it_is_not_delivered():
    outcome = classify(NAMES, grasp_log(40, soup=(10, 20)), target="soup", delivered=False)

    assert outcome.label == NOT_DELIVERED


def test_classify_rejects_unknown_targets_and_mismatched_logs():
    with pytest.raises(ValueError, match="target"):
        classify(NAMES, grasp_log(10), target="ketchup", delivered=False)
    with pytest.raises(ValueError, match="columns"):
        classify(NAMES, np.zeros((10, 3), dtype=bool), target="soup", delivered=False)
