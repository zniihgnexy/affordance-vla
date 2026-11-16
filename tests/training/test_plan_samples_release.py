"""Tests for training samples around releases, command timing and final approaches (plan_samples.py)."""
import numpy as np

from affordance_vla.plan.plan_input import no_plan_features, plan_features
from affordance_vla.training.counterfactual import Retarget, counterfactual_chunk
from affordance_vla.training.plan_samples import (
    approaches_by_target,
    demo_sample,
    shifted_sample,
    track_from_arrays,
    without_plan,
)
from tests.training.plan_tracks import (
    BASKET,
    GRASP,
    HORIZON,
    RELEASE,
    SCALE,
    STEPS,
    make_lagging_track,
    make_track,
    own_approach,
    track_with_commands,
)


def test_shifted_sample_with_the_demonstrated_contact_keeps_the_demonstrated_commands():
    # Arrange
    track = make_lagging_track()

    # Act
    sample = shifted_sample(track, step=10, horizon=HORIZON, new_contact=track.ee_pos[GRASP], motion_scale=SCALE)

    # Assert: an unchanged plan must not ask for other commands than the demonstration gave
    np.testing.assert_allclose(sample.actions[:, :3], track.actions[10:18, :3], atol=1e-9)
    commanded_path = track.ee_pos[10] + SCALE * np.cumsum(track.actions[10:18, :3], axis=0)
    np.testing.assert_allclose(sample.path_target, commanded_path, atol=1e-9)


def test_shifted_sample_adds_the_displacement_of_the_moved_path_to_the_demonstrated_commands():
    # Arrange: the same moved contact for a track without lag and for one whose commands lead its motion
    exact, lagging = make_track(), make_lagging_track()
    new_contact = exact.ee_pos[GRASP] + np.array([0.05, 0.05, 0.0])

    # Act
    plain = shifted_sample(exact, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE)
    sample = shifted_sample(lagging, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE)

    # Assert: on top of its own commands, the lagging track asks for the displacement the moved path adds
    added = SCALE * np.cumsum(sample.actions[:, :3] - lagging.actions[10:18, :3], axis=0)
    np.testing.assert_allclose(added, plain.path_target - exact.ee_pos[11:19], atol=1e-9)


def test_a_sample_without_its_plan_keeps_the_demonstrated_actions():
    # Arrange: a sample on the way to the grasp, which has a plan
    sample = demo_sample(make_track(), step=10, horizon=HORIZON)

    # Act
    dropped = without_plan(sample)

    # Assert: the plan input is the "no plan" input, everything supervised stays, the original is untouched
    np.testing.assert_array_equal(dropped.features, no_plan_features())
    np.testing.assert_array_equal(dropped.actions, sample.actions)
    np.testing.assert_array_equal(dropped.action_valid, sample.action_valid)
    np.testing.assert_array_equal(dropped.start, sample.start)
    assert np.abs(sample.features).sum() > 0


DROP = BASKET + [0.0, 0.0, 0.15]  # where make_track releases the bottle


def test_a_step_on_the_way_to_a_release_is_only_paired_with_another_contact_when_asked():
    # Arrange
    track, new_contact = make_track(), DROP + [0.05, -0.10, 0.0]

    # Act
    default = shifted_sample(track, step=50, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE)
    asked = shifted_sample(track, step=50, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE,
                           allow_release=True)

    # Assert
    assert default is None
    expected = plan_features(track.ee_pos[50], new_contact, approach=track.labels[1].approach, closing=False)
    np.testing.assert_allclose(asked.features, expected, atol=1e-6)


def test_a_shifted_release_with_the_demonstrated_contact_reproduces_the_demonstrated_path():
    # Arrange
    track = make_track()

    # Act
    sample = shifted_sample(track, step=50, horizon=HORIZON, new_contact=track.ee_pos[RELEASE], motion_scale=SCALE,
                            allow_release=True)

    # Assert
    np.testing.assert_allclose(sample.path_target, track.ee_pos[51:59], atol=1e-9)
    np.testing.assert_allclose(sample.actions[:, 6], 0.0)  # far from the release: the gripper stays closed


def test_a_shifted_release_carries_the_object_toward_the_new_contact():
    # Arrange: the release is moved 10 cm along x, across the direction of the carry
    track, shift = make_track(), np.array([0.10, 0.0, 0.0])

    # Act
    moved = shifted_sample(track, step=45, horizon=HORIZON, new_contact=DROP + shift, motion_scale=SCALE,
                           allow_release=True)
    unmoved = shifted_sample(track, step=45, horizon=HORIZON, new_contact=DROP, motion_scale=SCALE,
                             allow_release=True)

    # Assert: the chunk ends further along x than the demonstrated one, and the commands ask for that path
    assert moved.path_target[-1, 0] > unmoved.path_target[-1, 0] + 0.005
    path = moved.start + SCALE * np.cumsum(moved.actions[:, :3], axis=0)
    np.testing.assert_allclose(path, moved.path_target, atol=1e-9)
    assert moved.path_valid.all() and not moved.action_valid[:, :3].any()


def test_a_shifted_release_opens_the_gripper_at_the_demonstrated_distance_from_the_contact():
    # Arrange: at step 64 the demonstration is six steps from the release and opens two steps before it
    track = make_track()

    # Act
    sample = shifted_sample(track, step=64, horizon=HORIZON, new_contact=track.ee_pos[RELEASE], motion_scale=SCALE,
                            min_remaining=0.0, allow_release=True)

    # Assert: as in the demonstration, four commands closed and then open (policy convention: 1 = open)
    assert sample.actions[:, 6].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(sample.actions[:, 6], (1.0 - track.actions[64:72, 6]) / 2.0)
    assert sample.path_valid.tolist() == [True] * 6 + [False] * 2


def test_a_shifted_release_keeps_the_gripper_open_once_the_open_command_was_given():
    # Arrange: step 69 lies between the open command (step 68) and the loss of contact (step 70)
    track = make_track()

    # Act
    sample = shifted_sample(track, step=69, horizon=HORIZON, new_contact=track.ee_pos[RELEASE], motion_scale=SCALE,
                            min_remaining=0.0, allow_release=True)

    # Assert
    np.testing.assert_allclose(sample.actions[:, 6], 1.0)


def test_an_open_command_from_before_the_grasp_does_not_open_the_gripper_during_the_carry():
    # Arrange: a first closing attempt (steps 10-19) is given up; after the real grasp no open command comes
    track = track_with_commands([(10, 20), (GRASP - 4, STEPS - 1)])

    # Act
    sample = shifted_sample(track, step=50, horizon=HORIZON, new_contact=DROP + [0.05, 0.0, 0.0], motion_scale=SCALE,
                            allow_release=True)

    # Assert: the open command of step 20 belongs to the first attempt, not to this release
    np.testing.assert_allclose(sample.actions[:, 6], 0.0)


def test_a_release_after_a_failed_first_attempt_opens_at_the_distance_of_its_own_open_command():
    # Arrange: closed 10-19 (given up), closed again from 36, opened at 68 for the release at 70
    track = track_with_commands([(10, 20), (GRASP - 4, RELEASE - 2)])

    # Act
    sample = shifted_sample(track, step=64, horizon=HORIZON, new_contact=track.ee_pos[RELEASE], motion_scale=SCALE,
                            min_remaining=0.0, allow_release=True)

    # Assert
    assert sample.actions[:, 6].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_asking_for_releases_changes_nothing_on_the_way_to_a_grasp():
    # Arrange
    track, new_contact = make_track(), np.array([0.3, 0.1, 0.05])

    # Act
    default = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE)
    asked = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE,
                           allow_release=True)

    # Assert
    np.testing.assert_array_equal(asked.actions, default.actions)
    np.testing.assert_array_equal(asked.features, default.features)


def test_a_shifted_sample_can_move_above_the_new_contact_before_it_goes_on_down():
    # Arrange: at step 30 the gripper has 8 cm left, less than the 18 cm that are moved as a whole
    track, step = make_track(), 30
    new_contact = track.ee_pos[GRASP] + [0.0, 0.05, 0.0]

    # Act
    sample = shifted_sample(track, step=step, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE,
                            retarget=Retarget(rigid_tail=0.18, align_first=True))

    # Assert: four full-speed steps cross the 5 cm at the height of the step, then the approach goes on
    np.testing.assert_allclose(sample.path_target[:4, 2], track.ee_pos[step, 2], atol=1e-9)
    np.testing.assert_allclose(sample.path_target[3, 1], 0.05, atol=1e-9)
    assert sample.path_target[4, 2] < track.ee_pos[step, 2]


def test_steps_after_the_demonstrated_contact_follow_the_moved_path_alone():
    # Arrange: four steps before the grasp the contact moves 8 cm aside, so the moved path needs more steps
    # than the demonstration had left; after its grasp the demonstration carries the bottle away
    track, step = make_track(), GRASP - 4
    new_contact = track.ee_pos[GRASP] + [0.08, 0.0, 0.0]
    path = track.ee_pos[step:GRASP + 1]

    # Act
    sample = shifted_sample(track, step=step, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE,
                            min_remaining=0.0)

    # Assert: every step of the chunk is on the moved path; the carry of the demonstration is not mixed in
    walked = counterfactual_chunk(path, new_contact, HORIZON, flip_remaining=float("inf"), max_step=SCALE)
    np.testing.assert_allclose(sample.path_target, walked.points, atol=1e-9)
    assert walked.valid[4:].any()  # the moved path is still being walked after the demonstrated grasp step


def test_rotation_is_not_supervised_after_the_step_of_the_demonstrated_contact():
    # Arrange: as above; the rotation commands after the grasp belong to the carry
    track, step = make_track(), GRASP - 4

    # Act
    sample = shifted_sample(track, step=step, horizon=HORIZON, new_contact=track.ee_pos[GRASP] + [0.08, 0.0, 0.0],
                            motion_scale=SCALE, min_remaining=0.0)

    # Assert
    assert sample.action_valid[:4, 3:6].all()
    assert not sample.action_valid[4:, 3:6].any()


def test_handing_in_the_demonstrations_own_final_approach_changes_nothing():
    # Arrange
    track, new_contact = make_track(), np.array([0.3, 0.1, 0.05])
    rule = Retarget(rigid_tail=0.18, scale_from_start=True)

    # Act
    plain = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE, retarget=rule)
    handed = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE, retarget=rule,
                            tail=own_approach(track))

    # Assert
    np.testing.assert_allclose(handed.actions, plain.actions, atol=1e-12)


def test_another_final_approach_changes_the_labels_near_the_contact():
    # Arrange: eight steps before the grasp, with a final approach that comes in low from the side instead
    track, step = make_track(), GRASP - 8
    rule = Retarget(rigid_tail=0.18, scale_from_start=True)
    side = np.stack([np.linspace(-0.10, 0.0, 11), np.zeros(11), np.zeros(11)], axis=1)

    # Act
    plain = shifted_sample(track, step=step, horizon=HORIZON, new_contact=track.ee_pos[GRASP] + [0.0, 0.05, 0.0],
                           motion_scale=SCALE, min_remaining=0.0, retarget=rule)
    handed = shifted_sample(track, step=step, horizon=HORIZON, new_contact=track.ee_pos[GRASP] + [0.0, 0.05, 0.0],
                            motion_scale=SCALE, min_remaining=0.0, retarget=rule, tail=side)

    # Assert: only with the side approach is the gripper at the contact's height while still more than 2 cm aside
    def low_aside(sample) -> bool:
        relative = sample.path_target - (track.ee_pos[GRASP] + [0.0, 0.05, 0.0])
        return bool(np.any((np.abs(relative[:, 2]) < 0.005) & (np.hypot(relative[:, 0], relative[:, 1]) > 0.02)))

    assert low_aside(handed)
    assert not low_aside(plain)


def test_the_final_approaches_of_the_demonstrations_are_kept_per_object_relative_to_the_grasp_point():
    # Arrange: two demonstrations of the bottle and one of the can
    bottle, can = make_track(), make_track(target="can_1")

    # Act
    found = approaches_by_target([bottle, bottle, can])

    # Assert
    assert sorted(found) == ["bottle_1", "can_1"] and len(found["bottle_1"]) == 2
    np.testing.assert_allclose(found["bottle_1"][0], bottle.ee_pos[:GRASP + 1] - bottle.ee_pos[GRASP])
    np.testing.assert_allclose(found["bottle_1"][0][-1], 0.0)


def test_a_demonstration_that_does_not_start_on_the_way_to_a_grasp_gives_no_final_approach():
    # Arrange: the same track cut so that it starts while the bottle is already held
    track = make_track()
    held = track_from_arrays(track.ee_pos[GRASP + 2:], track.actions[GRASP + 2:], track.grasped[GRASP + 2:],
                             track.object_pos[GRASP + 2:], track.names, track.target, track.instruction)

    # Act / Assert
    assert approaches_by_target([held]) == {}


def test_the_noise_on_a_moved_plan_moves_the_plan_input_and_not_the_labels():
    # Arrange: the same moved contact, once with 3 cm of noise on the plan input
    track, new_contact = make_track(), np.array([0.3, 0.1, 0.05])
    rule = Retarget(rigid_tail=0.18, scale_from_start=True)

    # Act
    clean = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE, retarget=rule)
    noisy = shifted_sample(track, step=10, horizon=HORIZON, new_contact=new_contact, motion_scale=SCALE, retarget=rule,
                           contact_noise=np.array([0.03, 0.0, 0.0]))

    # Assert: the labels still lead to the contact itself; only the plan input points 3 cm away
    np.testing.assert_allclose(noisy.actions, clean.actions)
    np.testing.assert_allclose(noisy.path_target, clean.path_target)
    expected = plan_features(track.ee_pos[10], new_contact + [0.03, 0.0, 0.0], approach=track.labels[0].approach,
                             closing=True)
    np.testing.assert_allclose(noisy.features, expected, atol=1e-6)
