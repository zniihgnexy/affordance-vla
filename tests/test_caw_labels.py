"""Contact–approach–waypoint (CAW) labels derived from end-effector paths and gripper commands."""
import numpy as np
import pytest

from affordance3d.caw_labels import GripperEvent, caw_labels, fit_action_path, gripper_events


def test_gripper_events_mark_each_sign_change():
    # Arrange: open (-1) → close (+1) at step 3 → open again at step 6
    command = np.array([-1, -1, -1, 1, 1, 1, -1, -1], dtype=float)

    # Act
    events = gripper_events(command)

    # Assert
    assert events == [GripperEvent(step=3, closing=True), GripperEvent(step=6, closing=False)]


def test_gripper_events_ignore_zero_commands_between_equal_signs():
    command = np.array([-1, 0, -1, 1], dtype=float)
    assert gripper_events(command) == [GripperEvent(step=3, closing=True)]


def test_gripper_events_empty_when_command_never_changes():
    assert gripper_events(np.full(5, -1.0)) == []


def test_caw_label_uses_event_position_as_contact_and_motion_as_approach():
    # Arrange: straight descent along -z for 10 steps, gripper closes at step 10
    ee = np.zeros((12, 3))
    ee[:11, 2] = np.linspace(1.0, 0.0, 11)
    ee[11] = ee[10]
    events = [GripperEvent(step=10, closing=True)]

    # Act
    (label,) = caw_labels(ee, events, n_waypoints=5, approach_steps=4)

    # Assert
    np.testing.assert_allclose(label.contact, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(label.approach, [0.0, 0.0, -1.0])
    assert label.waypoints.shape == (5, 3)
    np.testing.assert_allclose(label.waypoints[-1], label.contact)
    np.testing.assert_allclose(label.waypoints[:, 2], [0.8, 0.6, 0.4, 0.2, 0.0])


def test_caw_label_segment_starts_at_previous_event():
    # Arrange: release at step 4 (after a grasp at step 2); the release segment starts at step 2
    ee = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], axis=1)
    events = [GripperEvent(step=2, closing=True), GripperEvent(step=4, closing=False)]

    # Act
    labels = caw_labels(ee, events, n_waypoints=2, approach_steps=10)

    # Assert
    np.testing.assert_allclose(labels[1].waypoints[:, 0], [3.0, 4.0])
    np.testing.assert_allclose(labels[1].approach, [1.0, 0.0, 0.0])  # clipped to the segment start


def test_caw_label_approach_is_zero_when_end_effector_did_not_move():
    ee = np.ones((5, 3))
    (label,) = caw_labels(ee, [GripperEvent(step=3, closing=True)], n_waypoints=2, approach_steps=2)
    np.testing.assert_allclose(label.approach, [0.0, 0.0, 0.0])


def test_caw_labels_reject_event_outside_path():
    with pytest.raises(ValueError, match="outside"):
        caw_labels(np.zeros((3, 3)), [GripperEvent(step=5, closing=True)])


def test_fit_action_path_recovers_scale_of_integrated_deltas():
    # Arrange: positions follow 0.02 m per unit action exactly
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(50, 3))
    ee = np.vstack([np.zeros(3), np.cumsum(0.02 * actions, axis=0)])[:50]

    # Act
    fit = fit_action_path(actions, ee)

    # Assert
    assert fit.scale == pytest.approx(0.02)
    assert fit.r2 == pytest.approx(1.0)
    np.testing.assert_allclose(fit.predicted, ee, atol=1e-12)


def test_fit_action_path_requires_matching_lengths():
    with pytest.raises(ValueError, match="same number of steps"):
        fit_action_path(np.zeros((4, 3)), np.zeros((5, 3)))


def test_chunk_endpoint_errors_are_zero_for_exactly_integrated_paths():
    from affordance3d.caw_labels import chunk_endpoint_errors

    # Arrange: positions follow 0.01 m per unit action exactly
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1, 1, size=(40, 3))
    ee = np.vstack([np.zeros(3), np.cumsum(0.01 * actions, axis=0)])[:40]

    # Act
    errors, travel = chunk_endpoint_errors(actions, ee, scale=0.01, horizon=8, stride=4)

    # Assert
    assert len(errors) == len(travel) == len(range(0, 40 - 8, 4))
    np.testing.assert_allclose(errors, 0.0, atol=1e-12)
    assert np.all(travel > 0)


def test_chunk_endpoint_errors_reject_horizon_longer_than_path():
    from affordance3d.caw_labels import chunk_endpoint_errors

    with pytest.raises(ValueError, match="horizon"):
        chunk_endpoint_errors(np.zeros((5, 3)), np.zeros((5, 3)), scale=0.01, horizon=8)


def test_gripper_events_detect_release_when_command_starts_closed():
    assert gripper_events(np.array([1, 1, -1, -1], dtype=float)) == [GripperEvent(step=2, closing=False)]


def test_caw_labels_reject_unsorted_events():
    events = [GripperEvent(step=4, closing=True), GripperEvent(step=2, closing=False)]
    with pytest.raises(ValueError, match="sorted"):
        caw_labels(np.zeros((6, 3)), events)


def test_fit_action_path_rejects_full_action_arrays_and_empty_input():
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        fit_action_path(np.zeros((5, 7)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="empty"):
        fit_action_path(np.zeros((0, 3)), np.zeros((0, 3)))


def test_chunk_endpoint_errors_validate_lengths_and_step_sizes():
    from affordance3d.caw_labels import chunk_endpoint_errors

    with pytest.raises(ValueError, match="same number of steps"):
        chunk_endpoint_errors(np.zeros((8, 3)), np.zeros((10, 3)), scale=0.01, horizon=4)
    with pytest.raises(ValueError, match="stride"):
        chunk_endpoint_errors(np.zeros((10, 3)), np.zeros((10, 3)), scale=0.01, horizon=4, stride=0)
    with pytest.raises(ValueError, match="horizon"):
        chunk_endpoint_errors(np.zeros((10, 3)), np.zeros((10, 3)), scale=0.01, horizon=0)
