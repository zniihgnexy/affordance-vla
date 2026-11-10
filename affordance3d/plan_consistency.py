"""Plan–action consistency: does an action chunk move the end effector along the CAW plan?

A chunk of end-effector translation commands, summed and scaled, is the path a policy asks for over
the next few steps. A CAW label (caw_labels.py) gives the waypoints that lead to the next gripper
event. The measures here compare the two. They use only sums, norms and dot products of the actions,
so the same formulas stay differentiable when written with tensors, which is what the planned
training loss needs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.caw_labels import CawLabel

MIN_CHUNK_MOTION_M = 1e-3  # below this net motion a chunk has no usable direction
_MIN_SEGMENT_SQ = 1e-18


@dataclass(frozen=True)
class PlanAhead:
    """The part of a CAW plan that is still ahead of one step."""

    polyline: np.ndarray  # (M, 3): the current position, then the waypoints not yet reached (ends at the contact)
    approach: np.ndarray  # (3,) unit approach direction of the plan; zeros if the label has none
    steps_to_event: int
    closing: bool  # the plan ends in a grasp (True) or a release (False)


@dataclass(frozen=True)
class ChunkConsistency:
    path_deviation: float  # mean distance (m) from the chunk's path points to the plan polyline
    direction_cosine: float  # cos of the angle between the chunk's net motion and the plan; nan if undefined
    net_motion: float  # metres between the first and the last point of the chunk's path


def chunk_path(start: np.ndarray, actions_xyz: np.ndarray, scale: float) -> np.ndarray:
    """(H + 1, 3) positions a chunk asks for: `start`, then start + scale · Σ of the first k actions."""
    start, actions_xyz = np.asarray(start, dtype=float), np.asarray(actions_xyz, dtype=float)
    if start.shape != (3,) or actions_xyz.ndim != 2 or actions_xyz.shape[1] != 3:
        raise ValueError(f"start must be (3,) and actions_xyz (H, 3), got {start.shape} and {actions_xyz.shape}; "
                         "pass actions[:, :3] for LIBERO actions")
    return start + scale * np.vstack([np.zeros(3), np.cumsum(actions_xyz, axis=0)])


def plan_ahead(labels: Sequence[CawLabel], position: np.ndarray, step: int) -> PlanAhead | None:
    """The plan in force at `step`: the label whose segment contains it. None after the last event."""
    for label in labels:
        if label.start <= step < label.event.step:
            ahead = label.waypoints[label.waypoint_steps > step]  # never empty: the last waypoint is the event
            return PlanAhead(polyline=np.vstack([np.asarray(position, dtype=float), ahead]),
                             approach=label.approach, steps_to_event=label.event.step - step,
                             closing=label.event.closing)
    return None


def point_along(polyline: np.ndarray, distance: float) -> np.ndarray:
    """The point reached by walking `distance` metres along the polyline from its first point (clamped to its end)."""
    polyline = np.asarray(polyline, dtype=float)
    remaining = max(float(distance), 0.0)
    for a, b in zip(polyline, polyline[1:]):
        length = float(np.linalg.norm(b - a))
        if length > 0 and remaining <= length:
            return a + (b - a) * (remaining / length)
        remaining -= length
    return polyline[-1].copy()


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """(N,) distance from each point to the nearest point on the polyline."""
    points, polyline = np.atleast_2d(np.asarray(points, dtype=float)), np.asarray(polyline, dtype=float)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    a, ab = polyline[:-1], np.diff(polyline, axis=0)  # (S, 3) segment starts and vectors
    along = np.einsum("nsk,sk->ns", points[:, None, :] - a[None], ab) / np.maximum(np.sum(ab ** 2, axis=1),
                                                                                   _MIN_SEGMENT_SQ)
    closest = a[None] + np.clip(along, 0.0, 1.0)[..., None] * ab[None]  # (N, S, 3)
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def chunk_consistency(path: np.ndarray, plan: PlanAhead) -> ChunkConsistency:
    """Compare a chunk's path (from chunk_path) with the plan ahead of its first point.

    The direction is judged against the point reached by walking the chunk's own net distance along
    the plan, so a chunk is not penalised for being slower or faster than the demonstration.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"path must hold at least two (x, y, z) points, got shape {path.shape}")
    net = path[-1] - path[0]
    net_motion = float(np.linalg.norm(net))
    wanted = point_along(plan.polyline, net_motion) - plan.polyline[0]
    wanted_norm = float(np.linalg.norm(wanted))
    if net_motion < MIN_CHUNK_MOTION_M or wanted_norm < MIN_CHUNK_MOTION_M:
        cosine = float("nan")
    else:
        cosine = float(np.dot(net, wanted) / (net_motion * wanted_norm))
    return ChunkConsistency(path_deviation=float(distance_to_polyline(path[1:], plan.polyline).mean()),
                            direction_cosine=cosine, net_motion=net_motion)


def consistency_along_episode(labels: Sequence[CawLabel], ee_pos: np.ndarray, query_steps: Sequence[int],
                              chunks_xyz: np.ndarray, scale: float) -> list[ChunkConsistency | None]:
    """One result per query step: the chunk requested there against the plan ahead of it.

    `chunks_xyz` is (Q, H, 3); chunk q starts at ee_pos[query_steps[q]]. A chunk that reaches past the
    next gripper event is cut at the event, because its later actions belong to the following plan.
    Steps after the last event have no plan and give None.
    """
    chunks_xyz = np.asarray(chunks_xyz, dtype=float)
    if len(chunks_xyz) != len(query_steps):
        raise ValueError(f"{len(chunks_xyz)} chunks for {len(query_steps)} query steps")
    results: list[ChunkConsistency | None] = []
    for step, chunk in zip(query_steps, chunks_xyz):
        plan = plan_ahead(labels, ee_pos[step], step)
        if plan is None:
            results.append(None)
            continue
        path = chunk_path(ee_pos[step], chunk[:plan.steps_to_event], scale)
        results.append(chunk_consistency(path, plan))
    return results
