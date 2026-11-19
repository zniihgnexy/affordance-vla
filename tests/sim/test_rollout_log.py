"""Reading the per-episode logs written by scripts/eval/rollout_libero_logged.py."""
import json

import numpy as np
import pytest

from affordance_vla.sim.rollout_log import load_rollout_log

NAMES = ["basket_1", "soup_1"]


def write_log(path, steps: int = 4, **overrides) -> None:
    """A small synthetic episode: `steps` actions, two objects, 8 × 8 images."""
    rows = steps + 1
    arrays = {
        "meta": json.dumps({"names": NAMES, "target": "soup_1", "instruction": "pick up the soup", "success": True}),
        "actions": np.zeros((steps, 7)), "ee_pos": np.zeros((rows, 3)), "object_pos": np.zeros((rows, 2, 3)),
        "grasped": np.zeros((rows, 2), dtype=bool), "touched": np.zeros((rows, 2), dtype=bool),
        "visible_px": np.zeros((rows, 2), dtype=int), "in_goal": np.zeros((rows, 2), dtype=bool),
        "chunk_steps": np.zeros(0, dtype=int), "chunks": np.zeros((0, 0, 7)), "intrinsics": np.eye(3),
        "cam_to_world": np.eye(4), "agentview": np.zeros((rows, 8, 8, 3), np.uint8),
        "wrist": np.zeros((rows, 8, 8, 3), np.uint8), "seg": np.zeros((rows, 8, 8), np.uint8),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **{key: value for key, value in arrays.items() if value is not None})


def test_log_exposes_steps_target_and_delivery(tmp_path):
    # Arrange: the soup satisfies the goal in the final state only
    in_goal = np.zeros((5, 2), dtype=bool)
    in_goal[-1, 1] = True
    write_log(tmp_path / "standard_ep0.npz", in_goal=in_goal)

    # Act
    log = load_rollout_log(tmp_path / "standard_ep0.npz")

    # Assert
    assert log.steps == 4
    assert log.names == ("basket_1", "soup_1")
    assert (log.target, log.target_index) == ("soup_1", 1)
    assert log.delivered is True
    assert log.agentview.shape == (5, 8, 8, 3)
    assert log.grasped.dtype == bool


def test_images_can_be_skipped(tmp_path):
    write_log(tmp_path / "ep.npz")

    log = load_rollout_log(tmp_path / "ep.npz", images=False)

    assert log.agentview is None and log.wrist is None and log.seg is None
    assert log.ee_pos.shape == (5, 3)


def test_missing_arrays_are_reported_by_name(tmp_path):
    write_log(tmp_path / "ep.npz", visible_px=None)

    with pytest.raises(KeyError, match="visible_px"):
        load_rollout_log(tmp_path / "ep.npz")


def test_state_arrays_need_one_more_row_than_actions(tmp_path):
    write_log(tmp_path / "ep.npz", ee_pos=np.zeros((4, 3)))

    with pytest.raises(ValueError, match="ee_pos"):
        load_rollout_log(tmp_path / "ep.npz")


def test_object_columns_must_match_the_names(tmp_path):
    write_log(tmp_path / "ep.npz", object_pos=np.zeros((5, 3, 3)))

    with pytest.raises(ValueError, match="object columns"):
        load_rollout_log(tmp_path / "ep.npz")


def test_image_arrays_need_one_more_row_than_actions_too(tmp_path):
    write_log(tmp_path / "ep.npz", agentview=np.zeros((4, 8, 8, 3), np.uint8))

    with pytest.raises(ValueError, match="agentview"):
        load_rollout_log(tmp_path / "ep.npz")


def test_the_target_must_be_one_of_the_logged_objects(tmp_path):
    meta = json.dumps({"names": NAMES, "target": "ketchup_1", "instruction": "pick up the ketchup", "success": False})
    write_log(tmp_path / "ep.npz", meta=meta)

    with pytest.raises(ValueError, match="target"):
        load_rollout_log(tmp_path / "ep.npz")


def test_a_log_without_depth_has_none(tmp_path):
    write_log(tmp_path / "ep.npz")

    assert load_rollout_log(tmp_path / "ep.npz").depth is None


def test_depth_is_read_with_the_images_when_the_log_has_it(tmp_path):
    # Arrange: one depth image per state row, in metres
    depth = np.full((5, 8, 8), 1.5, np.float32)
    write_log(tmp_path / "ep.npz", depth=depth)

    # Act
    with_images = load_rollout_log(tmp_path / "ep.npz")
    without = load_rollout_log(tmp_path / "ep.npz", images=False)

    # Assert
    np.testing.assert_array_equal(with_images.depth, depth)
    assert without.depth is None


def test_depth_needs_one_more_row_than_actions_too(tmp_path):
    write_log(tmp_path / "ep.npz", depth=np.zeros((4, 8, 8), np.float32))

    with pytest.raises(ValueError, match="depth"):
        load_rollout_log(tmp_path / "ep.npz")


def test_the_end_effector_orientation_is_read_when_logged_even_without_images(tmp_path):
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    write_log(tmp_path / "ep.npz", ee_quat=quat)

    np.testing.assert_array_equal(load_rollout_log(tmp_path / "ep.npz", images=False).ee_quat, quat)


def test_a_log_without_the_end_effector_orientation_has_none(tmp_path):
    write_log(tmp_path / "ep.npz")

    assert load_rollout_log(tmp_path / "ep.npz", images=False).ee_quat is None


def test_the_end_effector_orientation_needs_one_more_row_than_actions(tmp_path):
    write_log(tmp_path / "ep.npz", ee_quat=np.zeros((4, 4)))

    with pytest.raises(ValueError, match="ee_quat"):
        load_rollout_log(tmp_path / "ep.npz", images=False)


def test_object_orientations_are_read_when_logged(tmp_path):
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (5, 2, 1))
    write_log(tmp_path / "ep.npz", object_quat=quat)

    log = load_rollout_log(tmp_path / "ep.npz", images=False)

    np.testing.assert_array_equal(log.object_quat, quat)
    assert log.ee_quat is None

