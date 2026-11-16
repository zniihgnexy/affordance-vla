"""What a plan predictor is trained to output at one step of a demonstration."""
import numpy as np
import pytest

from affordance_vla.plan.plan_input import ContactOffsets
from affordance_vla.training.plan_samples import track_from_arrays
from affordance_vla.training.plan_targets import GRASP, NONE, RELEASE, oracle_target

GRASP_STEP, RELEASE_STEP, STEPS = 40, 70, 90
BOTTLE, BASKET, CAN = np.array([0.2, 0.0, 0.0]), np.array([0.2, 0.3, 0.0]), np.array([0.1, -0.2, 0.0])
DOWN = np.array([0.0, 0.0, -1.0])
OFFSETS = ContactOffsets(grasp={"bottle_1": np.array([0.0, 0.0, 0.05]), "can_1": np.array([0.01, 0.0, 0.03])},
                         grasp_approach={"bottle_1": DOWN, "can_1": DOWN},
                         release=np.array([0.0, 0.0, 0.15]), release_approach=np.array([0.0, 1.0, 0.0]))


def make_track():
    """Down onto the bottle (held from step 40 to 70), over to the basket, then away."""
    start, grasp, drop = np.array([0.0, 0.0, 0.3]), BOTTLE + [0.0, 0.0, 0.05], BASKET + [0.0, 0.0, 0.15]
    ee = np.vstack([np.linspace(start, grasp, GRASP_STEP + 1),
                    np.linspace(grasp, drop, RELEASE_STEP - GRASP_STEP + 1)[1:],
                    np.linspace(drop, drop + [0.0, 0.0, 0.1], STEPS - RELEASE_STEP)[1:]])
    actions = np.zeros((STEPS - 1, 7))
    actions[:, 6] = -1.0
    actions[GRASP_STEP - 4:RELEASE_STEP - 2, 6] = 1.0
    grasped = np.zeros(STEPS, dtype=bool)
    grasped[GRASP_STEP:RELEASE_STEP] = True
    object_pos = np.tile(np.stack([BASKET, BOTTLE, CAN]), (STEPS, 1, 1))
    return track_from_arrays(ee_pos=ee, actions=actions, grasped=grasped, object_pos=object_pos,
                             names=("basket_1", "bottle_1", "can_1"), target="bottle_1",
                             instruction="pick up the bottle and place it in the basket")


def test_before_the_grasp_the_target_is_the_grasp_point_of_the_instructed_object():
    target = oracle_target(make_track(), 10, OFFSETS, container="basket_1")

    assert target.event == GRASP
    np.testing.assert_allclose(target.contact, BOTTLE + [0.0, 0.0, 0.05])
    np.testing.assert_allclose(target.approach, DOWN)


def test_while_the_object_is_held_the_target_is_the_release_point_over_the_container():
    target = oracle_target(make_track(), 55, OFFSETS, container="basket_1")

    assert target.event == RELEASE
    np.testing.assert_allclose(target.contact, BASKET + [0.0, 0.0, 0.15])
    np.testing.assert_allclose(target.approach, [0.0, 1.0, 0.0])


def test_after_the_release_there_is_no_event():
    target = oracle_target(make_track(), 80, OFFSETS, container="basket_1")

    assert target.event == NONE
    np.testing.assert_allclose(target.contact, 0.0)
    np.testing.assert_allclose(target.approach, 0.0)


def test_another_object_can_be_asked_for_before_the_grasp():
    # the same frame under an instruction that names the can
    target = oracle_target(make_track(), 10, OFFSETS, container="basket_1", name="can_1")

    assert target.event == GRASP
    np.testing.assert_allclose(target.contact, CAN + [0.01, 0.0, 0.03])


def test_another_object_cannot_be_asked_for_once_the_demonstration_holds_its_own():
    with pytest.raises(ValueError, match="before the grasp"):
        oracle_target(make_track(), 55, OFFSETS, container="basket_1", name="can_1")


def test_objects_without_a_known_grasp_point_are_rejected():
    with pytest.raises(KeyError, match="basket_1"):
        oracle_target(make_track(), 10, OFFSETS, container="basket_1", name="basket_1")
