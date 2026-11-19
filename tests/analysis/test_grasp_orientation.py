"""Gripper orientation at grasp attempts (E0 of Pipeline v3)."""
import numpy as np
import pytest

from affordance_vla.analysis.grasp_orientation import (
    FLIP,
    close_onsets,
    closing_yaw_deg,
    grasp_angle_deg,
    grasp_attempts,
    nearest_demo_angle_deg,
    object_category,
    rotation_matrices,
    tilt_deg,
    yaw_difference_deg,
)

TOP_GRASP = np.diag([1.0, -1.0, -1.0])  # fingers point down, as in the demonstrations


def about_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def about_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])


def quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """A rotation matrix back to (x, y, z, w); enough for the well-conditioned rotations used here."""
    w = np.sqrt(max(1.0 + np.trace(rotation), 1e-12)) / 2.0
    if w > 1e-3:
        return np.array([(rotation[2, 1] - rotation[1, 2]) / (4 * w), (rotation[0, 2] - rotation[2, 0]) / (4 * w),
                         (rotation[1, 0] - rotation[0, 1]) / (4 * w), w])
    x = np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12)) / 2.0
    return np.array([x, (rotation[0, 1] + rotation[1, 0]) / (4 * x), (rotation[0, 2] + rotation[2, 0]) / (4 * x),
                     (rotation[2, 1] - rotation[1, 2]) / (4 * x)])


def test_object_category_drops_the_instance_number():
    assert object_category("ketchup_1") == "ketchup"
    assert object_category("alphabet_soup_12") == "alphabet_soup"
    assert object_category("basket") == "basket"


@pytest.mark.parametrize("rotation", [np.eye(3), TOP_GRASP, about_z(30) @ TOP_GRASP, about_x(20) @ TOP_GRASP])
def test_rotation_matrices_reads_xyzw_quaternions(rotation):
    assert np.allclose(rotation_matrices(quat_xyzw(rotation)), rotation, atol=1e-9)


def test_a_turn_about_the_approach_axis_by_180_degrees_is_the_same_grasp():
    assert grasp_angle_deg(TOP_GRASP, TOP_GRASP @ FLIP) == pytest.approx(0.0, abs=1e-6)
    assert grasp_angle_deg(TOP_GRASP, about_z(180) @ TOP_GRASP) == pytest.approx(0.0, abs=1e-6)


def test_grasp_angle_measures_a_turn_and_a_lean():
    assert grasp_angle_deg(TOP_GRASP, about_z(30) @ TOP_GRASP) == pytest.approx(30.0, abs=1e-6)
    assert grasp_angle_deg(TOP_GRASP, about_z(150) @ TOP_GRASP) == pytest.approx(30.0, abs=1e-6)
    assert grasp_angle_deg(TOP_GRASP, about_x(25) @ TOP_GRASP) == pytest.approx(25.0, abs=1e-6)


def test_tilt_and_closing_yaw_of_a_top_grasp():
    assert tilt_deg(TOP_GRASP) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg(about_x(25) @ TOP_GRASP) == pytest.approx(25.0, abs=1e-6)
    assert closing_yaw_deg(TOP_GRASP) == pytest.approx(90.0, abs=1e-6)  # closing axis along -y: a line at 90
    assert closing_yaw_deg(about_z(30) @ TOP_GRASP) == pytest.approx(120.0, abs=1e-6)


def test_yaw_difference_treats_directions_as_lines():
    assert yaw_difference_deg(10.0, 170.0) == pytest.approx(20.0)
    assert yaw_difference_deg(0.0, 90.0) == pytest.approx(90.0)
    assert yaw_difference_deg(45.0, 45.0) == pytest.approx(0.0)


def test_close_onsets_finds_each_switch_to_close():
    gripper = [-1, -1, 1, 1, -1, 1, 1]
    actions = np.zeros((len(gripper), 7))
    actions[:, 6] = gripper
    assert close_onsets(actions).tolist() == [2, 5]
    actions[0, 6] = 1
    assert close_onsets(actions).tolist() == [0, 2, 5]


def scene(steps: int = 30):
    """Two objects; the gripper sits on top of object b from step 8 on."""
    names = ["a_1", "b_1"]
    object_pos = np.tile(np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]), (steps + 1, 1, 1))
    ee_pos = np.tile(np.array([0.1, 0.0, 0.3]), (steps + 1, 1))
    ee_pos[8:] = [0.2, 0.0, 0.01]
    ee_quat = np.tile(quat_xyzw(about_z(20) @ TOP_GRASP), (steps + 1, 1))
    grasped = np.zeros((steps + 1, 2), dtype=bool)
    actions = -np.ones((steps, 7))
    return names, ee_pos, ee_quat, object_pos, grasped, actions


def test_a_close_command_followed_by_a_grasp_is_a_held_attempt_on_the_nearest_object():
    names, ee_pos, ee_quat, object_pos, grasped, actions = scene()
    actions[10:, 6] = 1
    grasped[14:, 1] = True

    [attempt] = grasp_attempts(names, ee_pos, ee_quat, object_pos, grasped, actions)

    assert (attempt.step, attempt.obj, attempt.held) == (10, "b_1", True)
    assert attempt.distance_m == pytest.approx(0.01)
    assert grasp_angle_deg(attempt.rotation, TOP_GRASP) == pytest.approx(20.0, abs=1e-6)


def test_a_close_command_without_a_grasp_in_time_is_not_held():
    names, ee_pos, ee_quat, object_pos, grasped, actions = scene()
    actions[10:, 6] = 1
    grasped[27:, 1] = True  # 17 steps later: too late

    [attempt] = grasp_attempts(names, ee_pos, ee_quat, object_pos, grasped, actions, hold_steps=15)

    assert not attempt.held


def test_grasp_attempts_wants_one_action_fewer_than_states():
    names, ee_pos, ee_quat, object_pos, grasped, actions = scene()
    with pytest.raises(ValueError, match="one action fewer"):
        grasp_attempts(names, ee_pos, ee_quat, object_pos, grasped, np.zeros((len(ee_pos), 7)))


def test_nearest_demo_angle_takes_the_closest_demonstration():
    demos = [about_z(40) @ TOP_GRASP, about_z(-10) @ TOP_GRASP]
    assert nearest_demo_angle_deg(TOP_GRASP, demos) == pytest.approx(10.0, abs=1e-6)
    assert nearest_demo_angle_deg(TOP_GRASP, []) == float("inf")
