"""Per-step logs of LIBERO episodes written by scripts/eval/rollout_libero_logged.py (one .npz each).

Arrays that describe a state have T + 1 rows: the state before each of the T actions, plus the final
state. Images are stored as the policy sees them (the simulator frame rotated by 180°).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE_KEYS = ("ee_pos", "object_pos", "grasped", "touched", "visible_px", "in_goal")
IMAGE_KEYS = ("agentview", "wrist", "seg")
OPTIONAL_IMAGE_KEYS = ("depth",)  # only in logs written with --depth
OPTIONAL_STATE_KEYS = ("ee_quat", "object_quat")  # absent from logs written before the episode loop recorded them
OTHER_KEYS = ("meta", "actions", "chunk_steps", "chunks", "intrinsics", "cam_to_world")
ROBOT_SEG_ID = 255


@dataclass(frozen=True)
class RolloutLog:
    path: Path
    meta: dict  # task, instruction, target, episode, shifts, policy, success, delivered, …
    names: tuple[str, ...]  # movable objects, in the column order of the per-object arrays
    actions: np.ndarray  # (T, 7) executed actions; gripper command -1 = open, +1 = close
    ee_pos: np.ndarray  # (T + 1, 3)
    object_pos: np.ndarray  # (T + 1, N, 3)
    grasped: np.ndarray  # (T + 1, N) bool: both finger pads touch the object
    touched: np.ndarray  # (T + 1, N) bool: any gripper part touches the object
    visible_px: np.ndarray  # (T + 1, N) pixels of the object in the agent view
    in_goal: np.ndarray  # (T + 1, N) bool: the object satisfies the task's goal predicate
    chunk_steps: np.ndarray  # (C,) step at which each action chunk was requested (empty for a replay)
    chunks: np.ndarray  # (C, H, 7) chunks as the policy returned them (gripper 1 = open, 0 = close)
    intrinsics: np.ndarray  # (3, 3) of the agent-view camera
    cam_to_world: np.ndarray  # (4, 4) of the agent-view camera
    agentview: np.ndarray | None  # (T + 1, H, W, 3) uint8; None when read with images=False
    wrist: np.ndarray | None
    seg: np.ndarray | None  # (T + 1, H, W) uint8: 0 background, k + 1 = names[k], 255 robot
    depth: np.ndarray | None = None  # (T + 1, H, W) metres along the optical axis of the agent view, if logged
    ee_quat: np.ndarray | None = None  # (T + 1, 4) end-effector orientation, (x, y, z, w), if logged
    object_quat: np.ndarray | None = None  # (T + 1, N, 4) object orientations, (x, y, z, w), if logged

    @property
    def steps(self) -> int:
        return len(self.actions)

    @property
    def target(self) -> str:
        return self.meta["target"]

    @property
    def target_index(self) -> int:
        return self.names.index(self.target)

    @property
    def delivered(self) -> bool:
        """The instructed object satisfies the goal in the final state."""
        return bool(self.in_goal[-1, self.target_index])


def load_rollout_log(path: Path | str, images: bool = True) -> RolloutLog:
    """Read one episode log; fails if an array is missing or the row counts disagree."""
    path = Path(path)
    with np.load(path) as data:
        wanted = STATE_KEYS + OTHER_KEYS + (IMAGE_KEYS if images else ())
        missing = [key for key in wanted if key not in data.files]
        if missing:
            raise KeyError(f"{path.name} is missing {', '.join(missing)}")
        optional = tuple(key for key in OPTIONAL_IMAGE_KEYS if images and key in data.files)
        optional += tuple(key for key in OPTIONAL_STATE_KEYS if key in data.files)
        arrays = {key: data[key] for key in wanted + optional if key != "meta"}
        meta = json.loads(str(data["meta"]))
    steps = len(arrays["actions"])
    wrong = {key: len(arrays[key]) for key in STATE_KEYS + (IMAGE_KEYS if images else ()) + optional
             if len(arrays[key]) != steps + 1}
    if wrong:
        raise ValueError(f"{path.name}: expected {steps + 1} state rows for {steps} actions, got {wrong}")
    names = tuple(meta["names"])
    if arrays["object_pos"].shape[1] != len(names):
        raise ValueError(f"{path.name}: {arrays['object_pos'].shape[1]} object columns for {len(names)} names")
    if meta["target"] not in names:
        raise ValueError(f"{path.name}: target {meta['target']!r} is not one of the logged objects {list(names)}")
    return RolloutLog(path=path, meta=meta, names=names, grasped=arrays.pop("grasped").astype(bool),
                      touched=arrays.pop("touched").astype(bool), in_goal=arrays.pop("in_goal").astype(bool),
                      agentview=arrays.pop("agentview", None), wrist=arrays.pop("wrist", None),
                      seg=arrays.pop("seg", None), depth=arrays.pop("depth", None), **arrays)
