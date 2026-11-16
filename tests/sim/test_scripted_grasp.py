"""The scripted pick-and-place controller (affordance_vla/sim/scripted_grasp.py), without a simulator."""
import numpy as np

from affordance_vla.sim.scripted_grasp import (
    CLOSE,
    OPEN,
    GraspSettings,
    GraspState,
    Targets,
    next_action,
    phase_goal,
    phase_order,
    position_action,
)

TARGETS = Targets(grasp=np.array([0.1, 0.2, 0.1]), release=np.array([0.0, 0.3, 0.2]))


def test_the_phases_end_with_a_release_only_when_placing():
    assert phase_order(True) == ("above", "descend", "close", "lift", "carry", "lower", "release", "retreat", "done")
    assert phase_order(False) == ("above", "descend", "close", "lift", "done")


def test_the_gripper_comes_in_from_straight_above_the_grasp_point():
    settings = GraspSettings(above_m=0.1)
    above, gripper = phase_goal("above", TARGETS, settings)
    np.testing.assert_allclose(above, TARGETS.grasp + [0.0, 0.0, 0.1])
    assert gripper == OPEN
    assert phase_goal("close", TARGETS, settings) == (TARGETS.grasp, CLOSE)


def test_translation_is_proportional_to_the_error_in_units_of_5_cm_and_clipped():
    np.testing.assert_allclose(position_action(np.zeros(3), [0.025, 0.0, -1.0], gain=1.0), [0.5, 0.0, -1.0])


def test_a_phase_ends_once_the_gripper_is_within_tolerance():
    settings = GraspSettings(tolerance_m=0.01)
    action, state = next_action(GraspState("descend", 3), TARGETS.grasp + [0.0, 0.0, 0.005], TARGETS, settings)
    assert state == GraspState("close", 0)
    assert action[6] == OPEN and np.allclose(action[3:6], 0.0)


def test_closing_lasts_a_fixed_number_of_steps():
    settings = GraspSettings(close_steps=3)
    state = GraspState("close", 0)
    for _ in range(2):
        _, state = next_action(state, TARGETS.grasp, TARGETS, settings)
        assert state.phase == "close"
    action, state = next_action(state, TARGETS.grasp, TARGETS, settings)
    assert state.phase == "lift" and action[6] == CLOSE


def test_a_phase_that_cannot_reach_its_target_times_out():
    settings = GraspSettings(max_phase_steps=2)
    _, state = next_action(GraspState("above", 1), np.zeros(3), TARGETS, settings)
    assert state == GraspState("descend", 0)


def test_when_done_it_holds_still_with_the_gripper_of_the_last_phase():
    action, state = next_action(GraspState("done"), np.zeros(3), TARGETS, GraspSettings(place=False))
    assert state.phase == "done" and np.allclose(action[:6], 0.0) and action[6] == CLOSE
