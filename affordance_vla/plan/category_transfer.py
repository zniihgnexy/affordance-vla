"""Carry an interaction map from some meshes of an object category to another mesh of it (Pipeline v3, section 3).

Two ways, compared by the leave-one-out check of docs/vla/2026-11-25_VLA_branch_第0步_mug搬运检查.md:

- nearest point (this file): the grasps of the source meshes are summed in the category frame
  (affordance_vla/plan/category_frame.py), and every surface point of the target mesh takes the score and closing
  axis of the nearest summed point; a point farther than `max_distance` (category units) from all of them scores 0;
- NDF (affordance_vla/plan/ndf_transfer.py): the source grasps themselves are moved onto the target mesh and
  summed there with the same contact weights (`canonical_map`).

Contact weights are computed on each source mesh in metres, exactly as for a map of one mesh; only the sums
(score, d·dᵀ) travel to the category frame. Only numpy and scipy are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from affordance_vla.plan.category_frame import CategoryFrame
from affordance_vla.plan.interaction_labels import CanonicalMap, Grasp, dominant_axis, grasp_sums
from affordance_vla.sim.object_surface import Surface

MAX_DISTANCE = 0.15  # category units (the mug's horizontal radius and height are 1): farther points score 0


@dataclass(frozen=True)
class Source:
    """One source mesh: its frame, its surface (body frame) and the held grasps on it (body frame)."""

    name: str
    frame: CategoryFrame
    surface: Surface
    grasps: Sequence[Grasp]


@dataclass(frozen=True)
class CategoryMap:
    """The summed grasps of the source meshes as points of the category frame."""

    points: np.ndarray  # (M, 3) surface points of all source meshes, category frame
    total: np.ndarray  # (M,) summed contact weight
    tensor: np.ndarray  # (M, 3, 3) summed d·dᵀ of the closing axes, category frame
    grasps: int
    sources: tuple[str, ...]


def category_map(sources: Sequence[Source]) -> CategoryMap:
    """Sum the grasps of every source mesh in the category frame."""
    if not sources or not all(source.grasps for source in sources):
        raise ValueError("every source mesh needs at least one grasp")
    points, totals, tensors = [], [], []
    for source in sources:
        closings = source.frame.axis_to_category(np.stack([g.closing for g in source.grasps]))
        total, tensor = grasp_sums(source.surface, source.grasps, closings)
        points.append(source.frame.point_to_category(source.surface.points))
        totals.append(total)
        tensors.append(tensor)
    return CategoryMap(np.concatenate(points), np.concatenate(totals), np.concatenate(tensors),
                       sum(len(s.grasps) for s in sources), tuple(s.name for s in sources))


def transfer_nearest(cmap: CategoryMap, frame: CategoryFrame, surface: Surface, name: str,
                     max_distance: float = MAX_DISTANCE, tree: cKDTree | None = None) -> CanonicalMap:
    """The category map on another mesh: every surface point takes the nearest summed point of the category map.
    `tree` is the KD-tree of `cmap.points` when the caller carries one map to many meshes (the result is the same)."""
    distance, nearest = (tree or cKDTree(cmap.points)).query(frame.point_to_category(surface.points))
    reached = distance <= max_distance
    total = np.where(reached, cmap.total[nearest], 0.0)
    tensor = np.where(reached[:, None, None], cmap.tensor[nearest], 0.0)
    axis, agreement = dominant_axis(tensor)
    touched = np.linalg.norm(axis, axis=1) > 0
    axis[touched] = frame.axis_from_category(axis[touched])
    score = total / total.max() if total.max() > 0 else total
    return CanonicalMap(name, surface.points, surface.normals, score, axis, agreement, cmap.grasps)
