"""Gripper orientation at grasp attempts, against the demonstrations of the same object (E0 of Pipeline v3).

The question of E0: do failed grasps hold the gripper differently from held ones, and from the
demonstrations of the same object? This first version works in world coordinates, because the logs do not
yet record object orientations. The end effector's z axis is the approach axis (it points down in a top
grasp) and the fingers close along its y axis. The two fingers are alike, so the gripper turned by
180 degrees about its approach axis makes the same grasp; every angle here takes the smaller of the two.
Only numpy is used, so the simulator environment can import this.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

APPROACH_AXIS = 2  # column of the end-effector rotation the fingers point along (down in a top grasp)
CLOSING_AXIS = 1  # column the fingers close along
GRIPPER_COLUMN = 6  # of an action: -1 opens the gripper, +1 closes it
HOLD_STEPS = 15  # steps after the close command within which the grasp must register (0.75 s at 20 Hz)
DOWN = np.array([0.0, 0.0, -1.0])
FLIP = np.diag([-1.0, -1.0, 1.0])  # 180 degrees about the end effector's z axis: the fingers swap places


def object_category(name: str) -> str:
    """The object kind of a scene object: 'ketchup_1' -> 'ketchup'."""
    return re.sub(r"_\d+$", "", name)


def rotation_matrices(quats_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) unit quaternions in robosuite's (x, y, z, w) order to (..., 3, 3) rotation matrices."""
    q = np.asarray(quats_xyzw, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def grasp_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two gripper rotations, counting the 180-degree turn about the approach axis as the same."""
    return min(_angle_deg(a, b), _angle_deg(a, b @ FLIP))


def tilt_deg(rotation: np.ndarray) -> float:
    """How far the approach axis leans away from straight down."""
    return float(np.degrees(np.arccos(np.clip(rotation[:, APPROACH_AXIS] @ DOWN, -1.0, 1.0))))


def closing_yaw_deg(rotation: np.ndarray) -> float:
    """Direction of the closing axis in the horizontal plane, in [0, 180): a line, not an arrow."""
    axis = rotation[:, CLOSING_AXIS]
    return float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)


def yaw_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two line directions given in degrees (at most 90)."""
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def close_onsets(actions: np.ndarray) -> np.ndarray:
    """Steps at which the gripper command switches to close (or starts closed)."""
    closing = np.asarray(actions, dtype=float)[:, GRIPPER_COLUMN] > 0
    return np.flatnonzero(closing & ~np.concatenate([[False], closing[:-1]]))


@dataclass(frozen=True, eq=False)  # holds an array: compared by identity
class GraspAttempt:
    """One close command: the object nearest to the gripper then, and whether a grasp of it followed."""

    step: int
    obj: str
    distance_m: float
    rotation: np.ndarray  # (3, 3) end-effector rotation in world coordinates
    held: bool


def grasp_attempts(names: Sequence[str], ee_pos: np.ndarray, ee_quat: np.ndarray, object_pos: np.ndarray,
                   grasped: np.ndarray, actions: np.ndarray, hold_steps: int = HOLD_STEPS) -> list[GraspAttempt]:
    """Every close command of an episode. Action t is issued in state t, so the state arrays have one row
    more than `actions`. An attempt is held when the grasp of its object registers within `hold_steps`."""
    if not len(ee_pos) == len(ee_quat) == len(object_pos) == len(grasped):
        raise ValueError("the state arrays must have the same number of rows")
    if len(actions) >= len(ee_pos):
        raise ValueError(f"{len(actions)} actions for {len(ee_pos)} states: expected one action fewer than states")
    rotations = rotation_matrices(ee_quat)
    attempts = []
    for step in close_onsets(actions):
        distances = np.linalg.norm(np.asarray(object_pos[step]) - np.asarray(ee_pos[step]), axis=1)
        k = int(np.argmin(distances))
        held = bool(np.asarray(grasped)[step:step + hold_steps + 1, k].any())
        attempts.append(GraspAttempt(int(step), names[k], float(distances[k]), rotations[step], held))
    return attempts


def nearest_demo_angle_deg(rotation: np.ndarray, demo_rotations: Sequence[np.ndarray]) -> float:
    """Angle to the closest demonstrated grasp; inf when there is none."""
    return min((grasp_angle_deg(rotation, demo) for demo in demo_rotations), default=float("inf"))
