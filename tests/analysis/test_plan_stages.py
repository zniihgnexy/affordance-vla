"""How far a rollout that was given a plan got, and how jerky its commands were."""
import numpy as np
import pytest

from affordance_vla.analysis.plan_stages import REACH_TOLERANCE_M, command_roughness, plan_stages, reach_along_rollout

STEPS = 60


def held(start: int, end: int) -> np.ndarray:
    """(STEPS,) two-finger contact flags of the target, true in [start, end)."""
    flags = np.zeros(STEPS, dtype=bool)
    flags[start:end] = True
    return flags


def test_a_rollout_that_follows_its_plan_passes_every_stage():
    # Arrange: the gripper closes in on the planned grasp, holds the object and delivers it
    reach = np.array([0.20, 0.12, 0.05, 0.01, 0.30, 0.10])
    closing = np.array([True, True, True, True, False, False])
    height = np.full(STEPS, 0.90)

    # Act
    stages = plan_stages(reach, closing, held(25, 50), height, delivered=True)

    # Assert
    assert (stages.reached, stages.held, stages.toppled, stages.delivered) == (True, True, False, True)
    assert stages.holding_at_end is False  # it was let go at step 50


def test_an_object_still_in_the_gripper_in_the_last_steps_is_reported():
    # Arrange: picked up at step 20 and never released
    stages = plan_stages(np.array([0.01]), np.array([True]), held(20, STEPS), np.full(STEPS, 0.90), delivered=False)

    # Assert
    assert stages.held is True
    assert stages.holding_at_end is True


def test_reaching_counts_only_while_the_plan_is_a_grasp():
    # Arrange: the gripper is close to the planned contact only at queries whose plan is a release
    reach = np.array([0.20, 0.05, 0.01, 0.01])
    closing = np.array([True, True, False, False])

    # Act
    stages = plan_stages(reach, closing, held(0, 0), np.full(STEPS, 0.90), delivered=False)

    # Assert
    assert stages.reached is False


def test_a_brush_of_two_steps_is_not_holding():
    stages = plan_stages(np.array([0.01]), np.array([True]), held(20, 22), np.full(STEPS, 0.90), delivered=False)

    assert stages.reached is True
    assert stages.held is False


def test_an_object_that_sinks_before_it_is_held_was_toppled():
    # Arrange: a bottle falls over at step 10 (5 cm lower) and is picked up lying down from step 30
    height = np.full(STEPS, 0.90)
    height[10:] = 0.85

    # Act
    stages = plan_stages(np.array([0.01]), np.array([True]), held(30, 50), height, delivered=False)

    # Assert
    assert stages.toppled is True


def test_an_object_that_ends_lower_after_being_held_was_not_toppled():
    # Arrange: held from step 20, lifted, and released into a basket whose floor is lower than the start
    height = np.full(STEPS, 0.90)
    height[25:45] = 1.05
    height[45:] = 0.84

    # Act
    stages = plan_stages(np.array([0.01]), np.array([True]), held(20, 45), height, delivered=True)

    # Assert
    assert stages.toppled is False


def test_plan_stages_rejects_arrays_of_different_lengths():
    with pytest.raises(ValueError, match="queries"):
        plan_stages(np.array([0.1, 0.2]), np.array([True]), held(0, 0), np.full(STEPS, 0.9), delivered=False)
    with pytest.raises(ValueError, match="steps"):
        plan_stages(np.array([0.1]), np.array([True]), held(0, 0), np.full(STEPS - 1, 0.9), delivered=False)


def test_constant_commands_have_no_roughness():
    chunks = np.full((3, 8, 7), 0.4)

    assert command_roughness(chunks) == pytest.approx(0.0)


def test_roughness_is_the_mean_change_of_the_translation_between_steps():
    # Arrange: x alternates between 0 and 0.3; y and z stay; rotation and gripper change but do not count
    chunks = np.zeros((1, 4, 7))
    chunks[0, :, 0] = [0.0, 0.3, 0.0, 0.3]
    chunks[0, :, 3:] = np.arange(4)[:, None]

    # Act
    roughness = command_roughness(chunks)

    # Assert: three changes of 0.3 in x, none in y and z -> 0.9 / 9
    assert roughness == pytest.approx(0.1)


def test_command_roughness_needs_chunks_of_at_least_two_steps():
    with pytest.raises(ValueError, match="chunks"):
        command_roughness(np.zeros((3, 1, 7)))
    with pytest.raises(ValueError, match="chunks"):
        command_roughness(np.zeros((8, 7)))


def test_a_brush_in_the_very_last_steps_is_not_holding_at_the_end():
    stages = plan_stages(np.array([0.01]), np.array([True]), held(STEPS - 2, STEPS), np.full(STEPS, 0.90),
                         delivered=False)

    assert stages.holding_at_end is False


def test_a_drop_that_starts_when_the_object_is_first_held_is_not_toppling():
    # Arrange: the height falls from the first held step on (the gripper presses the object down)
    height = np.full(STEPS, 0.90)
    height[30:] = 0.85

    # Act
    stages = plan_stages(np.array([0.01]), np.array([True]), held(30, 50), height, delivered=False)

    # Assert
    assert stages.toppled is False


def test_an_object_held_from_the_first_step_was_not_toppled():
    stages = plan_stages(np.array([0.01]), np.array([True]), held(0, 40), np.full(STEPS, 0.90), delivered=False)

    assert (stages.held, stages.toppled) == (True, False)


def test_an_object_that_sinks_and_is_never_held_was_toppled():
    # Arrange
    height = np.full(STEPS, 0.90)
    height[20:] = 0.85

    # Act
    stages = plan_stages(np.array([0.01]), np.array([True]), held(0, 0), height, delivered=False)

    # Assert
    assert (stages.held, stages.toppled) == (False, True)


def test_a_distance_equal_to_the_tolerance_counts_as_reached():
    stages = plan_stages(np.array([0.10, REACH_TOLERANCE_M]), np.array([True, True]), held(0, 0),
                         np.full(STEPS, 0.90), delivered=False)

    assert stages.reached is True


def test_command_roughness_rejects_an_empty_set_of_chunks():
    with pytest.raises(ValueError, match="chunks"):
        command_roughness(np.zeros((0, 8, 7)))


def straight_line(steps: int = 21) -> np.ndarray:
    """(steps, 3) gripper positions moving 1 cm per step along x."""
    path = np.zeros((steps, 3))
    path[:, 0] = 0.01 * np.arange(steps)
    return path


def test_reach_is_measured_at_every_step_between_two_queries():
    # Arrange: one query at step 0 whose planned contact lies 10 cm ahead; the gripper passes it at step 10
    ee_pos = straight_line()

    # Act
    reach, query = reach_along_rollout(ee_pos, np.array([0]), np.array([[0.10, 0.0, 0.0]]))

    # Assert
    assert reach[0] == pytest.approx(0.10)
    assert reach[10] == pytest.approx(0.0)
    assert reach[20] == pytest.approx(0.10)
    assert query.tolist() == [0] * 21


def test_reach_uses_the_contact_of_the_latest_query():
    # Arrange: at step 8 the plan changes to a contact 30 cm to the side of where the gripper then is
    ee_pos = straight_line()
    to_contact = np.array([[0.10, 0.0, 0.0], [0.0, 0.30, 0.0]])

    # Act
    reach, query = reach_along_rollout(ee_pos, np.array([0, 8]), to_contact)

    # Assert
    assert reach[7] == pytest.approx(0.03)  # still the first contact, at x = 0.10
    assert reach[8] == pytest.approx(0.30)
    assert reach[10] == pytest.approx(np.hypot(0.02, 0.30))
    assert query[7] == 0 and query[8] == 1


def test_steps_before_the_first_query_have_no_reach():
    reach, query = reach_along_rollout(straight_line(), np.array([4]), np.array([[0.0, 0.0, 0.0]]))

    assert np.isinf(reach[:4]).all()
    assert reach[4] == pytest.approx(0.0)
    assert query[:4].tolist() == [-1] * 4


def test_reach_along_rollout_rejects_queries_that_do_not_fit_the_rollout():
    with pytest.raises(ValueError, match="query"):
        reach_along_rollout(straight_line(), np.array([8, 0]), np.zeros((2, 3)))  # not increasing
    with pytest.raises(ValueError, match="query"):
        reach_along_rollout(straight_line(), np.array([0, 40]), np.zeros((2, 3)))  # beyond the last step
    with pytest.raises(ValueError, match="query"):
        reach_along_rollout(straight_line(), np.array([0]), np.zeros((2, 3)))  # one vector too many
