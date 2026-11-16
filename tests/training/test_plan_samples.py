"""Tests for turning a replayed demonstration into training samples for a plan-conditioned action head:
the grasp side, contact offsets and the checks on demonstrations (releases: test_plan_samples_release.py)."""
import numpy as np
import pytest

from affordance_vla.plan.plan_input import no_plan_features, plan_features
from affordance_vla.training.plan_samples import (
    contact_offsets,
    demo_sample,
    shifted_sample,
    single_pick_and_place,
    track_from_arrays,
)
from tests.training.plan_tracks import (
    GRASP,
    HORIZON,
    RELEASE,
    SCALE,
    STEPS,
    make_track,
)


def test_demo_sample_describes_the_demonstrated_chunk_and_its_plan():
    # Arrange
    track = make_track()

    # Act
    sample = demo_sample(track, step=10, horizon=HORIZON)

    # Assert
    expected = plan_features(track.ee_pos[10], track.ee_pos[GRASP], approach=track.labels[0].approach, closing=True)
    np.testing.assert_allclose(sample.features, expected, atol=1e-6)
    np.testing.assert_allclose(sample.actions[:, :6], track.actions[10:18, :6])
    np.testing.assert_allclose(sample.actions[:, 6], 1.0)  # policy convention: 1 = open
    np.testing.assert_allclose(sample.start, track.ee_pos[10])
    assert sample.action_valid.all() and not sample.path_valid.any()


def test_demo_sample_converts_a_closed_gripper_to_the_policy_convention():
    # Act
    sample = demo_sample(make_track(), step=GRASP, horizon=HORIZON)

    # Assert
    np.testing.assert_allclose(sample.actions[:, 6], 0.0)


def test_demo_sample_after_the_last_event_has_no_plan():
    # Act
    sample = demo_sample(make_track(), step=RELEASE + 2, horizon=HORIZON)

    # Assert
    np.testing.assert_allclose(sample.features, no_plan_features())


def test_demo_sample_adds_the_given_noise_to_the_contact_only():
    # Arrange
    track, noise = make_track(), np.array([0.01, 0.0, 0.0])

    # Act
    sample = demo_sample(track, step=10, horizon=HORIZON, contact_noise=noise)

    # Assert
    expected = plan_features(track.ee_pos[10], track.ee_pos[GRASP] + noise, approach=track.labels[0].approach,
                             closing=True)
    np.testing.assert_allclose(sample.features, expected, atol=1e-6)
    np.testing.assert_allclose(sample.actions[:, :6], track.actions[10:18, :6])


def test_demo_sample_rejects_a_step_without_a_full_chunk():
    # Act / Assert
    with pytest.raises(ValueError, match="chunk of 8 steps"):
        demo_sample(make_track(), step=STEPS - 4, horizon=HORIZON)


def test_shifted_sample_points_at_the_new_contact():
    # Arrange
    track, new_contact = make_track(), np.array([0.3, 0.1, 0.05])

    # Act
    sample = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE)

    # Assert
    expected = plan_features(track.ee_pos[10], new_contact, approach=track.labels[0].approach, closing=True)
    np.testing.assert_allclose(sample.features, expected, atol=1e-6)


def test_shifted_sample_supervises_translation_by_the_path_and_the_rest_by_the_demonstration():
    # Arrange
    track = make_track()

    # Act
    sample = shifted_sample(track, step=10, horizon=HORIZON, new_contact=np.array([0.3, 0.1, 0.05]),
                            motion_scale=SCALE)

    # Assert
    assert sample.path_valid.all()
    assert not sample.action_valid[:, :3].any() and sample.action_valid[:, 3:].all()
    np.testing.assert_allclose(sample.actions[:, 3:6], track.actions[10:18, 3:6])
    np.testing.assert_allclose(sample.actions[:, 6], 1.0)  # still far from the contact: open


def test_shifted_sample_actions_integrate_to_the_path_target():
    # Arrange
    track = make_track()

    # Act
    sample = shifted_sample(track, step=10, horizon=HORIZON, new_contact=np.array([0.25, 0.05, 0.05]),
                            motion_scale=SCALE)

    # Assert: the clean chunk used for the noisy input asks for exactly the supervised path
    path = sample.start + SCALE * np.cumsum(sample.actions[:, :3], axis=0)
    np.testing.assert_allclose(path, sample.path_target, atol=1e-9)
    assert np.abs(sample.actions[:, :3]).max() <= 1.0 + 1e-9


def test_shifted_sample_with_the_demonstrated_contact_reproduces_the_demonstrated_path():
    # Arrange
    track = make_track()

    # Act
    sample = shifted_sample(track, step=10, horizon=HORIZON, new_contact=track.ee_pos[GRASP], motion_scale=SCALE)

    # Assert
    np.testing.assert_allclose(sample.path_target, track.ee_pos[11:19], atol=1e-9)


def test_shifted_sample_closes_the_gripper_at_the_demonstrated_distance_from_the_contact():
    # Arrange: at step 34 the demonstration is six steps from the contact and closes four steps before it
    track = make_track()

    # Act
    sample = shifted_sample(track, step=34, horizon=HORIZON, new_contact=track.ee_pos[GRASP], motion_scale=SCALE,
                            min_remaining=0.0)

    # Assert: as in the demonstration, the first two commands are open and the rest closed
    assert sample.actions[:, 6].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(sample.actions[:, 6], (1.0 - track.actions[34:42, 6]) / 2.0)
    assert sample.path_valid.tolist() == [True] * 6 + [False] * 2
    assert not sample.action_valid[6:, 3:6].any() and sample.action_valid[:, 6].all()


def test_shifted_sample_is_unavailable_outside_a_grasp_approach():
    # Arrange
    track = make_track()

    # Act
    while_holding = shifted_sample(track, step=GRASP + 5, horizon=HORIZON, new_contact=np.ones(3),
                                   motion_scale=SCALE)
    too_close = shifted_sample(track, step=GRASP - 1, horizon=HORIZON, new_contact=np.ones(3), motion_scale=SCALE)

    # Assert
    assert while_holding is None and too_close is None


def test_contact_offsets_average_grasp_and_release_over_demonstrations():
    # Arrange: the same bottle grasped 5 cm and 7 cm above its origin
    tracks = [make_track(lift=0.05), make_track(lift=0.07)]

    # Act
    offsets = contact_offsets(tracks, container="basket_1")

    # Assert
    np.testing.assert_allclose(offsets.grasp["bottle_1"], [0.0, 0.0, 0.06], atol=1e-9)
    np.testing.assert_allclose(offsets.release, [0.0, 0.0, 0.15], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(offsets.grasp_approach["bottle_1"]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(offsets.release_approach), 1.0)


def test_contact_offsets_reject_tracks_without_the_container():
    # Act / Assert
    with pytest.raises(ValueError, match="'crate_1' is not an object"):
        contact_offsets([make_track()], container="crate_1")


def test_shifted_sample_keeps_the_gripper_closed_once_the_close_command_was_given():
    # Arrange: step 38 lies between the close command (step 36) and the contact (step 40)
    track = make_track()

    # Act
    sample = shifted_sample(track, step=38, horizon=HORIZON, new_contact=track.ee_pos[GRASP], motion_scale=SCALE,
                            min_remaining=0.0)

    # Assert
    np.testing.assert_allclose(sample.actions[:, 6], 0.0)


def test_shifted_sample_without_a_close_command_closes_only_at_the_contact():
    # Arrange: a demonstration whose gripper command never changes
    track = make_track()
    actions = track.actions.copy()
    actions[:, 6] = -1.0
    silent = track_from_arrays(track.ee_pos, actions, track.grasped, track.object_pos, track.names, track.target,
                               track.instruction)

    # Act
    sample = shifted_sample(silent, step=36, horizon=HORIZON, new_contact=silent.ee_pos[GRASP], motion_scale=SCALE,
                            min_remaining=0.0)

    # Assert: four steps to the contact stay open; steps that start at the contact are closed
    assert sample.actions[:, 6].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_track_from_arrays_rejects_mismatched_lengths():
    # Arrange
    track = make_track()

    # Act / Assert
    with pytest.raises(ValueError, match=r"actions \(T, 7\) and ee_pos \(T \+ 1, 3\)"):
        track_from_arrays(track.ee_pos[:-1], track.actions, track.grasped, track.object_pos, track.names,
                          track.target, track.instruction)


def test_track_from_arrays_rejects_a_target_that_is_not_an_object():
    # Arrange
    track = make_track()

    # Act / Assert
    with pytest.raises(ValueError, match="'cup_1' is not one of the objects"):
        track_from_arrays(track.ee_pos, track.actions, track.grasped, track.object_pos, track.names, "cup_1",
                          track.instruction)


def test_a_demonstration_with_one_grasp_and_one_release_is_a_single_pick_and_place():
    # Act / Assert
    assert single_pick_and_place(make_track())


def test_a_demonstration_that_loses_and_regains_the_object_is_not_a_single_pick_and_place():
    # Arrange: two-finger contact is lost for three steps while carrying
    track = make_track()
    grasped = track.grasped.copy()
    grasped[50:53] = False
    regrasp = track_from_arrays(track.ee_pos, track.actions, grasped, track.object_pos, track.names, track.target,
                                track.instruction)

    # Act / Assert: such a track would add a mid-air "release" to the labels
    assert len(regrasp.labels) == 4
    assert not single_pick_and_place(regrasp)


def test_a_demonstration_that_never_releases_is_not_a_single_pick_and_place():
    # Arrange
    track = make_track()
    grasped = track.grasped.copy()
    grasped[GRASP:] = True
    held = track_from_arrays(track.ee_pos, track.actions, grasped, track.object_pos, track.names, track.target,
                             track.instruction)

    # Act / Assert
    assert not single_pick_and_place(held)
