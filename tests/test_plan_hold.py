"""Keeping a predicted plan for a whole phase of the task instead of predicting it at every query."""
import numpy as np

from affordance3d.plan_hold import EVENT_STEPS, Hold, after_query, after_step, held_plan, plan_input_of, plan_to_send
from affordance3d.plan_input import no_plan_features, plan_features

DOWN = np.array([0.0, 0.0, -1.0])
CLOSE, OPEN = 1.0, -1.0  # the simulator's gripper commands
CONTACT = np.array([0.12, 0.05, 0.02])


def grasp_hold() -> Hold:
    """A hold with a grasp plan at CONTACT, predicted from a gripper position away from it."""
    start = np.array([0.30, -0.10, 0.25])
    return Hold(held_plan(plan_features(start, CONTACT, DOWN, closing=True), start))


def repeated(hold: Hold, command: float, steps: int) -> Hold:
    """The hold after `steps` steps with the same gripper command."""
    for _ in range(steps):
        hold = after_step(hold, command)
    return hold


def test_a_predicted_plan_is_kept_as_a_contact_in_the_world():
    # Arrange
    gripper = np.array([0.30, -0.10, 0.25])
    features = plan_features(gripper, CONTACT, DOWN, closing=True)

    # Act
    kept = held_plan(features, gripper)

    # Assert
    np.testing.assert_allclose(kept.contact, CONTACT, atol=1e-6)
    np.testing.assert_allclose(kept.approach, DOWN, atol=1e-6)
    assert kept.closing


def test_an_empty_plan_is_not_kept():
    assert held_plan(no_plan_features(), np.zeros(3)) is None


def test_a_kept_plan_gives_the_plan_input_for_wherever_the_gripper_is_now():
    # Arrange: the gripper has moved since the plan was predicted
    kept = grasp_hold().plan
    now = np.array([0.20, 0.00, 0.10])

    # Act
    features = plan_input_of(kept, now)

    # Assert
    np.testing.assert_allclose(features, plan_features(now, CONTACT, DOWN, closing=True), atol=1e-6)


def test_a_grasp_plan_is_kept_while_the_gripper_stays_open():
    hold = repeated(grasp_hold(), OPEN, 100)

    assert hold.plan is not None


def test_a_grasp_plan_is_dropped_once_the_gripper_has_been_told_to_close_for_a_few_steps():
    hold = repeated(grasp_hold(), CLOSE, EVENT_STEPS)

    assert hold.plan is None


def test_a_short_closing_command_does_not_drop_the_plan():
    # Arrange: almost long enough, interrupted, and almost long enough again
    hold = repeated(grasp_hold(), CLOSE, EVENT_STEPS - 1)
    hold = after_step(hold, OPEN)

    # Act
    hold = repeated(hold, CLOSE, EVENT_STEPS - 1)

    # Assert
    assert hold.plan is not None


def test_a_release_plan_is_kept_while_the_object_is_carried_and_dropped_once_the_gripper_opens():
    # Arrange
    start = np.array([0.1, 0.0, 0.3])
    hold = Hold(held_plan(plan_features(start, CONTACT, DOWN, closing=False), start))

    # Act
    carried = repeated(hold, CLOSE, 100)
    released = repeated(carried, OPEN, EVENT_STEPS)

    # Assert
    assert carried.plan is not None
    assert released.plan is None


def test_nothing_changes_while_no_plan_is_kept():
    assert after_step(Hold(), CLOSE) == Hold()


def test_a_release_plan_made_while_the_gripper_is_open_is_dropped_again():
    # Arrange: a release plan although nothing is held (a wrong prediction at the start)
    start = np.array([0.1, 0.0, 0.3])
    hold = Hold(held_plan(plan_features(start, CONTACT, DOWN, closing=False), start))

    # Act
    hold = repeated(hold, OPEN, EVENT_STEPS)

    # Assert: the open gripper counts as the release carried out, so the next query predicts again
    assert hold.plan is None


def test_a_query_sends_the_kept_plan_and_asks_for_a_prediction_when_nothing_is_kept():
    gripper = np.array([0.20, 0.00, 0.10])

    kept = plan_to_send(grasp_hold(), gripper)
    nothing = plan_to_send(Hold(), gripper)

    np.testing.assert_allclose(kept, plan_features(gripper, CONTACT, DOWN, closing=True), atol=1e-6)
    assert nothing is None


def test_over_an_episode_a_plan_is_predicted_once_per_phase():
    # Arrange: queries every 8 steps; the gripper is told to close from step 20 on
    hold, asked = Hold(), []

    # Act: the loop of the rollout client, with a predictor that says "grasp" before step 24 and "release" after
    for step in range(64):
        gripper = np.array([0.30 - 0.004 * step, 0.0, 0.25])
        if step % 8 == 0 and plan_to_send(hold, gripper) is None:
            asked.append(step)
            hold = after_query(hold, plan_features(gripper, CONTACT, DOWN, closing=step < 24), gripper)
        hold = after_step(hold, CLOSE if step >= 20 else OPEN)

    # Assert: one prediction for the grasp, and one at the first query after the gripper had closed for 5 steps
    assert asked == [0, 32]
    assert hold.plan is not None and not hold.plan.closing
