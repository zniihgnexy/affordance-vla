"""Tests for the numbers handed to the action head as its plan input."""
import numpy as np
import pytest

from affordance_vla.plan.plan_input import (
    PLAN_DIM,
    ContactOffsets,
    no_plan_features,
    offsets_from_json,
    offsets_to_json,
    oracle_plan,
    plan_contact,
    plan_features,
    with_same_kind,
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


def test_the_contact_is_recovered_from_a_plan_input_and_the_gripper_position():
    # Arrange
    ee, contact = np.array([0.30, -0.10, 0.25]), np.array([0.12, 0.05, 0.02])
    features = plan_features(ee, contact, DOWN, closing=True)

    # Act
    found = plan_contact(features, ee)

    # Assert
    np.testing.assert_allclose(found, contact, atol=1e-6)


def test_an_empty_plan_points_at_no_contact():
    assert plan_contact(no_plan_features(), np.zeros(3)) is None


def test_a_vector_of_the_wrong_length_is_not_a_plan():
    with pytest.raises(ValueError, match="plan"):
        plan_contact(np.zeros(PLAN_DIM + 1), np.zeros(3))


def test_objects_of_the_same_kind_get_the_offsets_of_the_one_that_was_demonstrated():
    # Arrange: only the first bowl was ever grasped in the demonstrations
    offsets = ContactOffsets(grasp={"black_bowl_1": np.array([0.0, 0.0, 0.03])}, grasp_approach={"black_bowl_1": DOWN},
                             release=np.zeros(3), release_approach=DOWN)

    # Act
    found = with_same_kind(offsets, ["black_bowl_1", "black_bowl_2", "plate_1"])

    # Assert: the second bowl is added, the plate is not, and the original is left as it was
    assert sorted(found.grasp) == ["black_bowl_1", "black_bowl_2"]
    np.testing.assert_allclose(found.grasp["black_bowl_2"], [0.0, 0.0, 0.03])
    assert sorted(offsets.grasp) == ["black_bowl_1"]


def test_the_oracle_contacts_are_the_object_or_container_position_plus_the_demonstrated_offset():
    from affordance_vla.plan.plan_input import ContactOffsets, grasp_contact, release_contact

    offsets = ContactOffsets(grasp={"cup_1": np.array([0.0, 0.0, 0.04])}, grasp_approach={"cup_1": np.zeros(3)},
                             release=np.array([0.0, 0.01, 0.1]), release_approach=np.zeros(3))
    np.testing.assert_allclose(grasp_contact(offsets, "cup_1", [0.1, 0.2, 0.0]), [0.1, 0.2, 0.04])
    np.testing.assert_allclose(release_contact(offsets, [0.0, 0.3, 0.0]), [0.0, 0.31, 0.1])
    with pytest.raises(KeyError):
        grasp_contact(offsets, "plate_1", [0.0, 0.0, 0.0])
