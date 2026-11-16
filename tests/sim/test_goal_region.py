"""Which object owns the goal region of a LIBERO task, and the other simulator-free parts of an episode."""
import numpy as np
import pytest

from affordance_vla.sim.episode import env_gripper_command, max_steps_of, model_view, owns_region


def test_an_object_owns_its_named_regions():
    assert owns_region("basket_1", "basket_1_contain_region")


def test_an_object_that_is_itself_the_goal_owns_it():
    assert owns_region("plate_1", "plate_1")


def test_an_object_does_not_own_a_region_of_another_object_with_a_longer_name():
    assert not owns_region("plate_1", "plate_10_contain_region")


def test_a_perturbed_suite_takes_the_step_limit_of_the_suite_it_is_built_on():
    assert max_steps_of("libero_object_swap") == max_steps_of("libero_object") == 280
    assert max_steps_of("libero_spatial_task") == 220
    assert max_steps_of("libero_10") == 520


def test_an_unknown_suite_has_no_step_limit():
    with pytest.raises(SystemExit):
        max_steps_of("maniskill_pick")


def test_the_policy_gripper_output_maps_to_the_simulator_command():
    assert env_gripper_command(1.0) == -1.0  # policy: open -> simulator: open
    assert env_gripper_command(0.0) == 1.0  # policy: close -> simulator: close


def test_the_policy_sees_the_frame_rotated_by_half_a_turn():
    frame = np.arange(12).reshape(2, 3, 2)
    np.testing.assert_array_equal(model_view(frame), frame[::-1, ::-1])
