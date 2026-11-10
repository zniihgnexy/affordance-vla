"""Episode affordance state: per-object slots updated by rules over one episode."""
import numpy as np
import pytest

from affordance3d.episode_state import (Status, StepObservation, initial_state, observations_from_log, run_episode,
                                        update)

EE = np.array([0.0, 0.0, 0.5])


def slot_of(state, name):
    return next(slot for slot in state if slot.name == name)


def test_objects_start_unseen_with_no_position():
    state = initial_state(["cup", "bowl"])

    assert [slot.name for slot in state] == ["cup", "bowl"]
    assert all(slot.status is Status.UNSEEN and slot.position is None for slot in state)


def test_a_seen_object_is_refreshed_and_its_age_reset():
    state = initial_state(["cup"])

    state = update(state, StepObservation(ee_pos=EE, seen={"cup": np.array([0.1, 0.2, 0.0])}))

    cup = slot_of(state, "cup")
    assert cup.status is Status.VISIBLE
    np.testing.assert_allclose(cup.position, [0.1, 0.2, 0.0])
    assert cup.steps_since_seen == 0


def test_an_object_that_disappears_keeps_its_last_position_and_ages():
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": np.array([0.1, 0.2, 0.0])}))

    for _ in range(3):
        state = update(state, StepObservation(ee_pos=EE, seen={}))

    cup = slot_of(state, "cup")
    assert cup.status is Status.OCCLUDED
    np.testing.assert_allclose(cup.position, [0.1, 0.2, 0.0])
    assert cup.steps_since_seen == 3


def test_an_object_never_seen_stays_unseen():
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={}))

    cup = slot_of(state, "cup")
    assert cup.status is Status.UNSEEN
    assert cup.position is None


def test_a_grasped_object_follows_the_end_effector_while_hidden():
    # Arrange: seen 0.1 m below the gripper, then grasped and hidden while the gripper moves 0.3 m along x
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": EE - [0.0, 0.0, 0.1]}))
    state = update(state, StepObservation(ee_pos=EE, seen={}, grasped=frozenset({"cup"})))

    # Act
    moved_ee = EE + [0.3, 0.0, 0.0]
    state = update(state, StepObservation(ee_pos=moved_ee, seen={}, grasped=frozenset({"cup"})))

    # Assert
    cup = slot_of(state, "cup")
    assert cup.status is Status.GRASPED
    np.testing.assert_allclose(cup.position, moved_ee - [0.0, 0.0, 0.1])
    assert cup.steps_since_seen == 2
    assert cup.moved is True


def test_a_grasped_object_that_is_seen_uses_the_observation():
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": np.array([0.0, 0.0, 0.4])}))

    state = update(state, StepObservation(ee_pos=EE, seen={"cup": np.array([0.0, 0.0, 0.45])},
                                          grasped=frozenset({"cup"})))

    cup = slot_of(state, "cup")
    assert cup.status is Status.GRASPED
    np.testing.assert_allclose(cup.position, [0.0, 0.0, 0.45])
    assert cup.steps_since_seen == 0


def test_a_released_object_stays_where_it_was_let_go():
    # Arrange: the grasp starts at the object, which is then carried 0.2 m along x while hidden
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": EE.copy()}))
    state = update(state, StepObservation(ee_pos=EE, seen={}, grasped=frozenset({"cup"})))
    carried_to = EE + [0.2, 0.0, 0.0]
    state = update(state, StepObservation(ee_pos=carried_to, seen={}, grasped=frozenset({"cup"})))

    # Act: the gripper lets go and moves up
    state = update(state, StepObservation(ee_pos=carried_to + [0.0, 0.0, 0.2], seen={}))

    # Assert

    cup = slot_of(state, "cup")
    assert cup.status is Status.OCCLUDED
    np.testing.assert_allclose(cup.position, carried_to)


def test_moved_compares_the_belief_with_where_the_object_was_first_seen():
    state = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": np.zeros(3)}))
    assert slot_of(state, "cup").moved is False

    state = update(state, StepObservation(ee_pos=EE, seen={"cup": np.array([0.01, 0.0, 0.0])}))
    assert slot_of(state, "cup").moved is False  # 1 cm is within the tolerance

    state = update(state, StepObservation(ee_pos=EE, seen={"cup": np.array([0.10, 0.0, 0.0])}))
    assert slot_of(state, "cup").moved is True


def test_update_does_not_change_the_previous_state():
    before = update(initial_state(["cup"]), StepObservation(ee_pos=EE, seen={"cup": np.zeros(3)}))
    kept = before[0].position.copy()

    update(before, StepObservation(ee_pos=EE, seen={"cup": np.ones(3)}))

    np.testing.assert_allclose(before[0].position, kept)


def test_run_episode_returns_one_state_per_step():
    observations = [StepObservation(ee_pos=EE, seen={"cup": np.zeros(3)}), StepObservation(ee_pos=EE, seen={})]

    history = run_episode(["cup"], observations)

    assert [slot_of(state, "cup").status for state in history] == [Status.VISIBLE, Status.OCCLUDED]


def test_observations_from_log_apply_the_pixel_threshold():
    # Arrange: two steps, two objects; the bowl has too few visible pixels at step 1
    object_pos = np.arange(12, dtype=float).reshape(2, 2, 3)
    visible_px = np.array([[500, 400], [500, 10]])
    grasped = np.array([[False, False], [True, False]])

    # Act
    first, second = observations_from_log(["cup", "bowl"], np.zeros((2, 3)), object_pos, visible_px, grasped,
                                          min_pixels=50)

    # Assert
    assert set(first.seen) == {"cup", "bowl"}
    assert set(second.seen) == {"cup"}
    np.testing.assert_allclose(second.seen["cup"], object_pos[1, 0])
    assert second.grasped == frozenset({"cup"})


def test_observations_from_log_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="steps"):
        observations_from_log(["cup"], np.zeros((2, 3)), np.zeros((3, 1, 3)), np.zeros((3, 1)), np.zeros((3, 1), bool),
                              min_pixels=1)
