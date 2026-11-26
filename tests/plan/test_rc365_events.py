import numpy as np

from affordance_vla.plan.rc365_events import MIN_LIFT_M, mean_up, object_grasps, part_touches
from affordance_vla.sim.rc365_episodes import episode_objects, instance_key
from affordance_vla.sim.rc365_scan import runs

EYE = np.eye(3).ravel().tolist()


def object_event(lift, ee_rot=EYE, obj_rot=EYE):
    return {
        "kind": "object",
        "object": "obj",
        "category": "mug",
        "instance": "objaverse/mug/mug_1",
        "step": 10,
        "hold_steps": 8,
        "lift_m": lift,
        "obj_pos": [1, 1, 1],
        "obj_rot": obj_rot,
        "rest_rot": obj_rot,
        "base_pos": [1, 2, 1],
        "base_rot": EYE,
        "ee_pos": [1, 1, 1.1],
        "ee_rot": ee_rot,
        "grasp_centre": [1.02, 1, 1],
    }


def test_a_grasp_that_does_not_lift_the_object_is_not_a_pick_and_only_the_first_pick_counts():
    result = {
        "task": "T",
        "split": "pretrain",
        "episode": 3,
        "events": [object_event(0.0), object_event(0.2), object_event(0.5)],
    }
    picks = object_grasps([result])
    assert len(picks) == 1 and picks[0].lift_m == 0.2 and MIN_LIFT_M < 0.2


def test_the_grasp_is_expressed_in_the_object_frame_with_closing_along_the_site_x_axis():
    quarter_turn = [0, -1, 0, 1, 0, 0, 0, 0, 1]  # the object turned 90 degrees about z
    pick = object_grasps(
        [{"task": "T", "split": "target", "episode": 0, "events": [object_event(0.2, obj_rot=quarter_turn)]}]
    )[0]
    assert np.allclose(pick.grasp.centre, np.array(quarter_turn).reshape(3, 3).T @ [0.02, 0, 0])
    assert np.allclose(pick.grasp.closing, np.array(quarter_turn).reshape(3, 3).T @ [1, 0, 0])
    assert np.allclose(pick.base_in_object, np.array(quarter_turn).reshape(3, 3).T @ [0, 1, 0])


def test_the_mean_up_vector_reports_how_far_any_pose_is_from_it():
    up, spread = mean_up([np.array([0, 0, 1.0]), np.array([0, 0.1, 1.0]) / np.linalg.norm([0, 0.1, 1.0])])
    assert spread < 4 and up[2] > 0.99


def part_event(motion_to, q_later=0.1):
    return {
        "kind": "part",
        "part": "k_drawer_door_handle_main",
        "step": 5,
        "touch_steps": 9,
        "points": [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
        "fingers": [True, False],
        "part_pos": [0, 0, 0],
        "part_rot": EYE,
        "ee_pos": [0, 0, 0],
        "ee_rot": EYE,
        "grasp_centre": [0, 0, 0],
        "base_pos": [0, -1, 0],
        "base_rot": EYE,
        "ee_later": motion_to,
        "joint": "j",
        "joint_type": 2,
        "joint_axis": [0, 1, 0],
        "q_start": 0.0,
        "q_later": q_later,
        "q_end": 0.3,
    }


def test_a_touch_records_the_motion_the_joint_and_whether_both_fingers_touch():
    touch = part_touches(
        [
            {
                "task": "OpenDrawer",
                "split": "pretrain",
                "episode": 1,
                "events": [part_event([0, 0.1, 0]), part_event([0, 0.001, 0], q_later=0.0)],
            }
        ]
    )
    assert np.allclose(touch[0].contact, [0.01, 0, 0]) and not touch[0].both_fingers
    assert np.allclose(touch[0].motion, [0, 1, 0]) and touch[0].joint_type == "slide" and touch[0].joint_sign == 1
    assert touch[1].motion is None and touch[1].joint_sign == 0
    assert touch[0].fixture == "k_drawer_door_handle"


def test_runs_keeps_only_the_long_enough_stretches():
    assert runs(np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool), 3) == [(3, 6)]


def test_instance_keys_and_object_configs_are_read_from_ep_meta():
    path = "/root/robocasa/robocasa/models/assets/objects/objaverse/mug/mug_11/model.xml"
    assert instance_key(path) == "objaverse/mug/mug_11"
    meta = {
        "object_cfgs": [
            {"name": "obj", "info": {"cat": "mug", "mjcf_path": path}},
            {"name": "distr", "info": {"cat": "bowl", "mjcf_path": path.replace("mug", "bowl")}},
        ]
    }
    assert episode_objects(meta, {"mug"}) == {"obj": {"category": "mug", "instance": "objaverse/mug/mug_11"}}
    assert set(episode_objects(meta, None, {"distr"})) == {"distr"}
