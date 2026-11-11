"""Tests for the numbers handed to the action head as its plan input."""
import numpy as np
import pytest

from affordance3d.plan_input import (
    PLAN_DIM,
    ContactOffsets,
    no_plan_features,
    offsets_from_json,
    offsets_to_json,
    oracle_plan,
    plan_features,
)

DOWN = np.array([0.0, 0.0, -1.0])


def make_offsets() -> ContactOffsets:
    return ContactOffsets(grasp={"bottle_1": np.array([0.0, 0.0, 0.05])}, grasp_approach={"bottle_1": DOWN},
                          release=np.array([0.0, 0.0, 0.15]), release_approach=DOWN)


def test_plan_features_encode_the_vector_from_gripper_to_contact():
    # Arrange
    ee, contact = np.array([0.0, 0.0, 0.2]), np.array([0.1, 0.0, 0.0])

    # Act
    features = plan_features(ee, contact, approach=DOWN, closing=True)

    # Assert: 10 cm counts as 1 in the coarse part; the fine part saturates beyond a few centimetres
    assert features.shape == (PLAN_DIM,) and features.dtype == np.float32
    np.testing.assert_allclose(features[:3], [1.0, 0.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(features[3:6], np.tanh([5.0, 0.0, -10.0]), atol=1e-6)
    np.testing.assert_allclose(features[6:9], DOWN)
    assert features[9] == 1.0


def test_a_release_plan_carries_a_negative_event_code():
    # Act
    features = plan_features(np.zeros(3), np.ones(3), approach=DOWN, closing=False)

    # Assert
    assert features[9] == -1.0


def test_no_plan_is_all_zeros():
    # Act
    features = no_plan_features()

    # Assert
    assert features.shape == (PLAN_DIM,) and not features.any()


def test_oracle_plan_points_at_the_target_grasp_before_the_grasp():
    # Arrange
    ee, target_pos, container_pos = np.array([0.0, 0.0, 0.3]), np.array([0.2, 0.1, 0.0]), np.array([0.5, 0.5, 0.0])

    # Act
    features = oracle_plan(make_offsets(), ee, target_pos, container_pos, target="bottle_1", grasped=False,
                           delivered=False)

    # Assert: the contact is the object's position plus its grasp offset
    expected = plan_features(ee, np.array([0.2, 0.1, 0.05]), approach=DOWN, closing=True)
    np.testing.assert_allclose(features, expected)


def test_oracle_plan_points_at_the_container_while_the_target_is_held():
    # Arrange
    ee, target_pos, container_pos = np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 0.25]), np.array([0.5, 0.5, 0.0])

    # Act
    features = oracle_plan(make_offsets(), ee, target_pos, container_pos, target="bottle_1", grasped=True,
                           delivered=False)

    # Assert
    expected = plan_features(ee, np.array([0.5, 0.5, 0.15]), approach=DOWN, closing=False)
    np.testing.assert_allclose(features, expected)


def test_oracle_plan_is_empty_once_the_target_is_delivered():
    # Act
    features = oracle_plan(make_offsets(), np.zeros(3), np.ones(3), np.ones(3), target="bottle_1", grasped=False,
                           delivered=True)

    # Assert
    assert not features.any()


def test_oracle_plan_rejects_a_target_without_a_known_grasp_offset():
    # Act / Assert
    with pytest.raises(KeyError, match="no grasp offset for 'box_1'"):
        oracle_plan(make_offsets(), np.zeros(3), np.ones(3), np.ones(3), target="box_1", grasped=False,
                    delivered=False)


def test_offsets_survive_a_json_round_trip():
    # Arrange
    offsets = make_offsets()

    # Act
    restored = offsets_from_json(offsets_to_json(offsets))

    # Assert
    np.testing.assert_allclose(restored.grasp["bottle_1"], offsets.grasp["bottle_1"])
    np.testing.assert_allclose(restored.grasp_approach["bottle_1"], DOWN)
    np.testing.assert_allclose(restored.release, offsets.release)
    np.testing.assert_allclose(restored.release_approach, DOWN)
