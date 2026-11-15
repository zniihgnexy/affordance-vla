"""Tests for turning a replayed demonstration into training samples for a plan-conditioned action head."""
import numpy as np
import pytest

from affordance3d.counterfactual import Retarget, counterfactual_chunk
from affordance3d.plan_input import no_plan_features, plan_features
from affordance3d.plan_samples import (
    approaches_by_target,
    contact_offsets,
    demo_sample,
    shifted_sample,
    single_pick_and_place,
    track_from_arrays,
    without_plan,
)

SCALE = 0.0125  # metres of gripper motion per unit of translation action in these tests
HORIZON = 8
GRASP, RELEASE, STEPS = 40, 70, 90
BOTTLE, BASKET = np.array([0.2, 0.0, 0.0]), np.array([0.2, 0.3, 0.0])


def make_track(target: str = "bottle_1", lift: float = 0.05):
    """Down onto the bottle (grasp at step 40), over to the basket (release at step 70), then back up."""
    start, grasp, drop = np.array([0.0, 0.0, 0.3]), BOTTLE + [0.0, 0.0, lift], BASKET + [0.0, 0.0, 0.15]
    ee = np.vstack([np.linspace(start, grasp, GRASP + 1), np.linspace(grasp, drop, RELEASE - GRASP + 1)[1:],
                    np.linspace(drop, drop + [0.0, 0.0, 0.1], STEPS - RELEASE)[1:]])
    actions = np.zeros((STEPS - 1, 7))
    actions[:, :3] = np.diff(ee, axis=0) / SCALE
    actions[:, 3] = 0.1
    actions[:, 6] = -1.0  # simulator convention: -1 open, +1 close
    actions[GRASP - 4:RELEASE - 2, 6] = 1.0  # the command flips a few steps before the contacts change
    grasped = np.zeros(STEPS, dtype=bool)
    grasped[GRASP:RELEASE] = True
    object_pos = np.tile(np.stack([BASKET, BOTTLE, np.array([0.1, -0.2, 0.0])]), (STEPS, 1, 1))
    return track_from_arrays(ee_pos=ee, actions=actions, grasped=grasped, object_pos=object_pos,
                             names=("basket_1", "bottle_1", "can_1"), target=target,
                             instruction="pick up the bottle and place it in the basket")


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


def make_lagging_track():
    """As make_track, but the commands are 1.25 times the motion they produce (a controller that lags)."""
    track = make_track()
    actions = track.actions.copy()
    actions[:, :3] *= 1.25
    return track_from_arrays(track.ee_pos, actions, track.grasped, track.object_pos, track.names, track.target,
                             track.instruction)


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


def track_with_commands(closed: list[tuple[int, int]]):
    """make_track with the gripper commanded closed over the given step ranges and open elsewhere."""
    track = make_track()
    actions = track.actions.copy()
    actions[:, 6] = -1.0
    for start, stop in closed:
        actions[start:stop, 6] = 1.0
    return track_from_arrays(track.ee_pos, actions, track.grasped, track.object_pos, track.names, track.target,
                             track.instruction)


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


def own_approach(track) -> np.ndarray:
    """The demonstrated way to the grasp, relative to the grasp point."""
    return track.ee_pos[:GRASP + 1] - track.ee_pos[GRASP]


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
