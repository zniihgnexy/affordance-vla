import numpy as np
import pytest

from affordance_vla.plan import rc365_joint_labels as jl
from affordance_vla.plan.rc365_knob_diagnosis import burner_of, knob_slot, stove_type


def _event(joint_type, axis, anchor, contact, q=(0.0, -0.2, -1.0), part_pos=(0.0, 0.0, 0.0)):
    eye = list(np.eye(3).ravel())
    return {
        "kind": "part",
        "part": "stove_knob_front_left",
        "step": 3,
        "touch_steps": 5,
        "points": [list(contact)],
        "fingers": [True, True],
        "part_pos": list(part_pos),
        "part_rot": eye,
        "ee_pos": [0, 0, 0],
        "ee_rot": eye,
        "grasp_centre": list(contact),
        "base_pos": [0, 0, 0],
        "base_rot": eye,
        "ee_later": [0.0, 0.0, 0.0],
        "motion_steps": 20,
        "joint": "stove_knob_front_left_joint",
        "joint_type": joint_type,
        "joint_axis": list(axis),
        "joint_anchor": list(anchor),
        "q_start": q[0],
        "q_later": q[1],
        "q_end": q[2],
    }


def _touch(event):
    return jl.joint_touches([{"task": "T", "split": "pretrain", "episode": 0, "events": [event]}])[0]


def test_slide_direction_is_axis_times_sign():
    touch = _touch(_event(2, [0, 1, 0], [0, 0, 0], [0.0, 0.0, 0.05], q=(0.0, -0.1, -0.4)))
    assert np.allclose(touch.direction("1s"), [0, -1, 0])
    assert touch.joint_type == "slide"


def test_hinge_direction_is_axis_cross_lever_times_sign():
    # axis z, contact 0.3 m along +x from the axis: a positive turn moves the contact along +y
    plus = _touch(_event(3, [0, 0, 1], [0, 0, 0], [0.3, 0.0, 0.1], q=(0.0, 0.2, 0.5)))
    assert np.allclose(plus.direction("1s"), [0, 1, 0])
    assert plus.radius == pytest.approx(0.3)
    minus = _touch(_event(3, [0, 0, 1], [0, 0, 0], [0.3, 0.0, 0.1], q=(0.0, -0.2, -0.5)))
    assert np.allclose(minus.direction("1s"), [0, -1, 0])
    assert np.allclose(minus.turn("1s"), [0, 0, -1])


def test_unmoved_joint_leaves_the_label_unset_in_the_window_that_shows_no_change():
    touch = _touch(_event(3, [0, 0, 1], [0, 0, 0], [0.3, 0, 0], q=(0.0, 0.0005, -1.0)))
    assert touch.direction("1s") is None and touch.turn("1s") is None
    assert np.allclose(touch.turn("end"), [0, 0, -1])


def test_contact_on_the_axis_has_no_translation_direction_but_still_a_turn():
    touch = _touch(_event(3, [0, 0, 1], [0, 0, 0], [0.0, 0.0, 0.02], q=(0.0, -0.3, -1.0)))
    assert touch.direction("1s") is None
    assert np.allclose(touch.turn("1s"), [0, 0, -1])


def test_contact_is_given_from_the_anchor_and_from_the_body_origin():
    touch = _touch(_event(3, [0, 0, 1], [0.5, 0, 0], [0.52, 0, 0], part_pos=(0.0, 0.0, 0.0)))
    assert np.allclose(touch.contact_from_anchor(), [0.02, 0, 0])
    assert np.allclose(touch.contact_from_body(), [0.52, 0, 0])


def test_report_separates_kitchens_and_compares_with_the_old_label():
    events = [_event(2, [0, 1, 0], [0, 0, 0], [0, 0, 0.05], q=(0.0, -0.1, -0.4)) for _ in range(6)]
    results = [{"task": "T", "split": "pretrain", "episode": k, "events": [e]} for k, e in enumerate(events)]
    touches = jl.joint_touches(results)
    report = jl.joint_report(touches, ["a", "a", "b", "b", "c", "c"], groups=["x"] * 6)
    window = report["window_1s"]
    assert window["new_defined"] == 1.0
    assert window["new_direction"]["part"]["across_kitchens_within_tolerance"] == 1.0
    assert window["old_defined"] == 0.0  # the gripper did not move
    assert report["groups_1s"]["x"]["touches"] == 6


def test_knob_helpers_read_the_instruction_and_the_body_name():
    assert burner_of("Turn off the front left burner of the stove.") == "front_left"
    assert burner_of("Turn off the center burner of the stove.") == "center"
    assert knob_slot("stovetop_main_group_knob_rear_right") == "rear_right"
    assert stove_type("stovetop_main_group_knob_rear_right") == "stovetop"
