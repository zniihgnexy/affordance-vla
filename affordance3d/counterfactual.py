"""Move a demonstrated approach to another contact point.

A demonstration shows one way to reach one contact. To train an action head to follow a plan rather
than a remembered scene, the same observation is paired with a different contact, and the head is
asked for the path that leads there. That path is the demonstrated one with its end moved: the
shift is taken up early, and the last centimetres before the contact keep their demonstrated shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RIGID_TAIL_M = 0.08  # the final approach that is moved as a whole
MAX_STEP_M = 0.0125  # gripper motion per step for a full-scale translation action (LIBERO, measured)
_EPS = 1e-9


@dataclass(frozen=True)
class ChunkTarget:
    """What the next `horizon` steps should do on the moved path."""

    points: np.ndarray  # (H, 3) gripper position after each step
    valid: np.ndarray  # (H,) False for steps that start after the contact has been reached
    closed: np.ndarray  # (H,) True if the step starts within the gripper-closing distance of the contact


def arc_lengths(path: np.ndarray) -> np.ndarray:
    """(N,) distance travelled along the path up to each point; the first entry is 0."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"a path needs at least two (x, y, z) points, got shape {path.shape}")
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def retarget_path(path: np.ndarray, new_end: np.ndarray, rigid_tail: float = RIGID_TAIL_M) -> np.ndarray:
    """The path with its end at `new_end`.

    The shift grows with the distance travelled, from zero at the first point to the full shift
    where the last `rigid_tail` metres begin, so the final approach is the demonstrated one, moved.
    """
    path = np.asarray(path, dtype=float)
    arc = arc_lengths(path)
    if arc[-1] < _EPS:
        raise ValueError("the path has no length, so it cannot be moved to another end")
    weight = np.clip(arc / max(arc[-1] - rigid_tail, _EPS), 0.0, 1.0)
    return path + weight[:, None] * (np.asarray(new_end, dtype=float) - path[-1])


def walk(polyline: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """(K, 3) points reached after walking each distance along the polyline (clamped to its two ends)."""
    polyline = np.asarray(polyline, dtype=float)
    arc = arc_lengths(polyline)
    distances = np.clip(np.asarray(distances, dtype=float), 0.0, arc[-1])
    segment = np.clip(np.searchsorted(arc, distances, side="right") - 1, 0, len(polyline) - 2)
    length = arc[segment + 1] - arc[segment]
    fraction = np.where(length > 0, (distances - arc[segment]) / np.where(length > 0, length, 1.0), 0.0)
    return polyline[segment] + fraction[:, None] * (polyline[segment + 1] - polyline[segment])


def counterfactual_chunk(path: np.ndarray, new_end: np.ndarray, horizon: int, flip_remaining: float,
                         max_step: float = MAX_STEP_M, rigid_tail: float = RIGID_TAIL_M) -> ChunkTarget:
    """Targets for the next `horizon` steps if the contact were at `new_end`.

    `path` runs from the current gripper position to the demonstrated contact, one point per step.
    The moved path is walked at the demonstration's pace, stretched with the path's length so that
    the contact is reached after the same number of steps, but never faster than `max_step` per
    step. `flip_remaining` is how far before the contact the demonstration closed the gripper.
    """
    path = np.asarray(path, dtype=float)
    moved = retarget_path(path, new_end, rigid_tail)
    arc, total = arc_lengths(path), float(arc_lengths(moved)[-1])
    lengths = np.minimum(np.diff(arc) * (total / arc[-1]), max_step)
    lengths = np.concatenate([lengths, np.full(max(horizon - len(lengths), 0), max_step)])[:horizon]
    walked = np.cumsum(lengths)
    before = np.concatenate([[0.0], walked[:-1]])  # distance covered when each step starts
    return ChunkTarget(points=walk(moved, walked), valid=before < total - _EPS,
                       closed=total - before <= flip_remaining + _EPS)
