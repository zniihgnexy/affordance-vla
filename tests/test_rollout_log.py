"""Reading the per-episode logs written by scripts/remote/rollout_libero_logged.py."""
import json

import numpy as np
import pytest

from affordance3d.rollout_log import load_rollout_log

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
