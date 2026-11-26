"""What a RoboCasa365 policy is fed and what it returns, written once for training-side checks and evaluation.

starVLA's QwenOFT is trained on three cameras (left agentview, right agentview, eye-in-hand, in that order),
each resized to 224 x 224 with `PIL.Image.resize`, plus the instruction; the proprioceptive state enters the
model only when the run's config says `datasets.vla_data.include_state`. The example evaluation bridge of the
starVLA snapshot feeds one camera (resized with OpenCV), always builds a state and passes no seed
(docs/vla/2026-11-26_VLA_branch_第1步_底座策略.md, section 2). This module is the evaluation side that follows
the training side instead. It needs only numpy and PIL, so the alignment check can run in the training
environment as well as in the simulator environment.
"""
from __future__ import annotations

import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")  # the training order
VIDEO_KEYS = tuple(f"video.{camera}" for camera in CAMERAS)
STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
LANGUAGE_KEY = "annotation.human.task_description"
# the trained 12-d action: end-effector position 3, rotation 3, gripper close 1, base motion 4, control mode 1
ACTION_SLICES = {
    "action.end_effector_position": (0, 3),
    "action.end_effector_rotation": (3, 6),
    "action.gripper_close": (6, 7),
    "action.base_motion": (7, 11),
    "action.control_mode": (11, 12),
}
ACTION_DIM = 12
IMAGE_SIZE = 224


def preprocess_image(frame: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """One camera frame (uint8, H x W x 3) as the training data loader resizes it (`Image.fromarray(..).resize`)."""
    return np.asarray(Image.fromarray(frame).resize((size, size)))


def build_example(obs: Mapping[str, Any], include_state: bool = False, size: int = IMAGE_SIZE) -> dict:
    """The policy-server example for one observation of the RoboCasa gym environment.

    Images are the three cameras in training order. A state is sent only for a model trained with one; that
    path is not implemented because no AFFO-16 run uses it (QwenOFT with `include_state: false`).
    """
    if include_state:
        raise NotImplementedError("a model trained with include_state=True needs the state fed as in training")
    missing = [key for key in (*VIDEO_KEYS, LANGUAGE_KEY) if key not in obs]
    if missing:
        raise KeyError(f"observation lacks {missing}")
    return {
        "image": [preprocess_image(np.asarray(obs[key]), size) for key in VIDEO_KEYS],
        "lang": str(obs[LANGUAGE_KEY]),
    }


def action_steps(actions: np.ndarray, n_steps: int) -> list[dict[str, np.ndarray]]:
    """The first `n_steps` rows of an un-normalised action chunk (T x 12) as the environment's per-step dicts."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected an action chunk of shape (T, {ACTION_DIM}), got {actions.shape}")
    if not 1 <= n_steps <= len(actions):
        raise ValueError(f"cannot take {n_steps} steps from a chunk of {len(actions)}")
    return [{key: row[lo:hi] for key, (lo, hi) in ACTION_SLICES.items()} for row in actions[:n_steps]]


def episode_seed(base_seed: int, task: str, condition: str, index: int) -> int:
    """The seed of evaluation episode `index` of one (task, condition) cell: stable across runs and processes."""
    return zlib.crc32(f"{base_seed}/{task}/{condition}/{index}".encode()) % (2**31)
