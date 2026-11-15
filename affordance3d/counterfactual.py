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
TAIL_SHARE = 0.5  # with scale_from_start, the moved final approach is at most this share of the straight distance
MAX_STEP_M = 0.0125  # gripper motion per step for a full-scale translation action (LIBERO, measured)
_EPS = 1e-9


@dataclass(frozen=True)
class Retarget:
    """How a demonstrated path is moved to another contact."""

    rigid_tail: float = RIGID_TAIL_M  # metres of the final approach that are moved as a whole
    # A shift longer than the path before the rigid tail is made first, in a straight line from the
    # current position (it also covers a difference in height); the demonstrated path follows, moved.
    # Without it such a shift is pressed into what is left of the path, down to a single step.
    align_first: bool = False
    # The part before the final approach is rotated and scaled about the start in the horizontal plane, so
    # that it heads for the moved final approach and never first moves away from it; its height is shifted
    # gradually. The final approach is at most TAIL_SHARE of the straight distance to the new contact.
    scale_from_start: bool = False

    def __post_init__(self) -> None:
        if self.align_first and self.scale_from_start:
            raise ValueError("choose one of align_first and scale_from_start")


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


def retarget_path(path: np.ndarray, new_end: np.ndarray, retarget: Retarget = Retarget(),
                  tail: np.ndarray | None = None) -> np.ndarray:
    """The path with its end at `new_end`.

    The shift grows with the distance travelled, from zero at the first point to the full shift
    where the last `rigid_tail` metres begin, so the final approach is the demonstrated one, moved.
    With `align_first`, a shift longer than the path before the rigid tail is made first instead: the
    result then has one point more, the start, followed by the whole demonstrated path, moved.
    With `scale_from_start`, see `scaled_path`; only that rule takes a `tail`, a final approach to use
    instead of the demonstrated one (points relative to their own contact, ending at zero).
    """
    path = np.asarray(path, dtype=float)
    arc = arc_lengths(path)
    if arc[-1] < _EPS:
        raise ValueError("the path has no length, so it cannot be moved to another end")
    if retarget.scale_from_start:
        return scaled_path(path, np.asarray(new_end, dtype=float), retarget.rigid_tail, tail)
    if tail is not None:
        raise ValueError("a given final approach needs the scale_from_start rule")
    shift = np.asarray(new_end, dtype=float) - path[-1]
    free = arc[-1] - retarget.rigid_tail
    distance = float(np.linalg.norm(shift))
    if retarget.align_first and free < distance:
        return path.copy() if distance < _EPS else np.vstack([path[:1], path + shift])
    weight = np.clip(arc / max(free, _EPS), 0.0, 1.0)
    return path + weight[:, None] * shift


def last_part(path: np.ndarray, length: float) -> tuple[int, np.ndarray]:
    """(index where the last `length` metres of the path begin, those points relative to the path's end)."""
    arc = arc_lengths(path)
    first = min(int(np.searchsorted(arc, arc[-1] - length - _EPS)), len(path) - 1)
    return first, path[first:] - path[-1]


def scaled_free_part(free: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The polyline `free`, from its start, turned and scaled in the horizontal plane about its start so that
    it ends at `end`; heights are shifted in proportion to the distance travelled."""
    start = free[0]
    weight = np.clip(arc_lengths(free) / max(arc_lengths(free)[-1], _EPS), 0.0, 1.0) if len(free) > 1 else np.ones(1)
    reach, target = free[-1, :2] - start[:2], end[:2] - start[:2]
    if np.linalg.norm(reach) < 1e-3:  # the free part ends right above its start: no direction to turn
        return free + weight[:, None] * (end - free[-1])
    scale = np.linalg.norm(target) / np.linalg.norm(reach)
    angle = np.arctan2(target[1], target[0]) - np.arctan2(reach[1], reach[0])
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    horizontal = start[:2] + scale * (free[:, :2] - start[:2]) @ turn.T
    return np.column_stack([horizontal, free[:, 2] + weight * (end[2] - free[-1, 2])])


def scaled_path(path: np.ndarray, new_end: np.ndarray, rigid_tail: float, tail: np.ndarray | None) -> np.ndarray:
    """The path moved to `new_end` by the scale_from_start rule (see Retarget).

    The final approach (the path's own, or `tail`) is cut to min(rigid_tail, TAIL_SHARE x straight distance)
    and placed at `new_end`; the part of the path before its own final approach of that length is turned and
    scaled about the start so that it ends where the placed final approach begins.
    """
    length = min(rigid_tail, TAIL_SHARE * float(np.linalg.norm(new_end - path[0])))
    first, own = last_part(path, length)
    approach = own if tail is None else last_part(np.asarray(tail, dtype=float), length)[1]
    placed = new_end + approach
    if first == 0:  # the whole path is final approach: go straight to where the placed approach begins
        return np.vstack([path[:1], placed]) if np.linalg.norm(placed[0] - path[0]) > _EPS else placed
    free = scaled_free_part(path[:first + 1], placed[0])
    return np.vstack([free[:-1], placed])


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
                         max_step: float = MAX_STEP_M, retarget: Retarget = Retarget(),
                         tail: np.ndarray | None = None) -> ChunkTarget:
    """Targets for the next `horizon` steps if the contact were at `new_end`.

    `path` runs from the current gripper position to the demonstrated contact, one point per step.
    The moved path is walked at the demonstration's pace, stretched with the path's length so that
    the contact is reached after the same number of steps, but never faster than `max_step` per
    step. `flip_remaining` is how far before the contact the demonstration closed the gripper.
    """
    path = np.asarray(path, dtype=float)
    moved = retarget_path(path, new_end, retarget, tail)
    arc, total = arc_lengths(path), float(arc_lengths(moved)[-1])
    lengths = np.minimum(np.diff(arc) * (total / arc[-1]), max_step)
    lengths = np.concatenate([lengths, np.full(max(horizon - len(lengths), 0), max_step)])[:horizon]
    walked = np.cumsum(lengths)
    before = np.concatenate([[0.0], walked[:-1]])  # distance covered when each step starts
    return ChunkTarget(points=walk(moved, walked), valid=before < total - _EPS,
                       closed=total - before <= flip_remaining + _EPS)
