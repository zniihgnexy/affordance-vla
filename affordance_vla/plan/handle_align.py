"""Align the handle of a mug-like instance in the category frame before a map is carried to it (AFFO-18, part B).

The category frame (`category_frame.py`) fits height and body radius to 1 and puts the handle along +x, but a handle
that reaches farther than the others stays farther (`mug_8`: 2.35, source mugs 1.81-2.23), so the nearest-point transfer
finds no source grasp near it. Two alignments, both read from the instance's surface points only (never from its
demonstrations) and both leaving the body (x <= `HANDLE_X0`) where it is:

- `length_warp`: the handle's reach beyond the body, along +x, is rescaled to a reference reach;
- `registered_warp`: the handle points are registered to the handle points of the source instances by ICP with a
  scale and a shift on every axis (no rotation), a few iterations.

An `AlignedFrame` is a `CategoryFrame` whose points are warped after the usual map. Closing axes are not warped
(the transfer maps them out of and back into the category frame the same way for every instance). Only numpy and
scipy are used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from affordance_vla.plan.category_frame import CategoryFrame

HANDLE_X0 = 1.05  # category units: points farther than this along the handle direction are the handle
ICP_ITERATIONS = 20
ICP_MIN_SCALE, ICP_MAX_SCALE = 0.5, 2.0  # a fitted per-axis scale outside this range is clipped (a degenerate fit)

Warp = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class AlignedFrame(CategoryFrame):
    """A category frame whose points are warped after the map; the axes are those of the plain frame."""

    warp: Warp | None = None

    def point_to_category(self, points: np.ndarray) -> np.ndarray:
        mapped = super().point_to_category(points)
        return self.warp(mapped) if self.warp else mapped


def handle_mask(points: np.ndarray) -> np.ndarray:
    """Which category-frame points belong to the handle."""
    return np.asarray(points)[:, 0] > HANDLE_X0


def handle_reach(points: np.ndarray) -> float:
    """How far the handle reaches beyond the body along +x (category units); 0 for an object without one."""
    x = np.asarray(points)[:, 0]
    return float(max(x.max() - HANDLE_X0, 0.0))


def instance_points(frame: CategoryFrame, surface_points: np.ndarray) -> np.ndarray:
    """An instance's surface points in its own (unwarped) category frame."""
    return CategoryFrame.point_to_category(frame, surface_points)


def length_warp(reach: float, reference: float) -> Warp:
    """x -> HANDLE_X0 + (x - HANDLE_X0) * reference / reach on the handle; the rest unchanged."""
    factor = reference / reach if reach > 0 else 1.0

    def warp(points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=float)
        mask = handle_mask(out)
        out[mask, 0] = HANDLE_X0 + (out[mask, 0] - HANDLE_X0) * factor
        return out

    return warp


def registered_warp(handle: np.ndarray, template: np.ndarray, iterations: int = ICP_ITERATIONS) -> Warp:
    """Per-axis scale and shift that bring `handle` (N, 3) onto `template` (M, 3) by ICP (nearest template point of
    every handle point, least squares per axis); applied to the handle points only."""
    scale, shift = np.ones(3), np.zeros(3)
    if len(handle) == 0 or len(template) == 0:
        return lambda points: np.array(points, dtype=float)
    tree = cKDTree(template)
    for _ in range(iterations):
        _, nearest = tree.query(handle * scale + shift)
        pairs = template[nearest]
        for axis in range(3):
            spread = handle[:, axis] - handle[:, axis].mean()
            denominator = float(spread @ spread)
            fit = float(spread @ (pairs[:, axis] - pairs[:, axis].mean())) / denominator if denominator > 1e-12 else 1.0
            scale[axis] = float(np.clip(fit, ICP_MIN_SCALE, ICP_MAX_SCALE))
            shift[axis] = float(pairs[:, axis].mean() - scale[axis] * handle[:, axis].mean())
    scale, shift = scale.copy(), shift.copy()

    def warp(points: np.ndarray) -> np.ndarray:
        out = np.array(points, dtype=float)
        mask = handle_mask(out)
        out[mask] = out[mask] * scale + shift
        return out

    return warp


def with_warp(item, warp: Warp):
    """The item (`rc365_symmetry.Item`) with its category frame replaced by the same frame plus a warp."""
    f = item.frame
    return replace(item, frame=AlignedFrame(f.origin, f.rotation, f.scale, warp))


def align_length(items: Sequence, reference: float | None = None) -> list:
    """Every item's handle reach rescaled to `reference` (default: the median reach of the items)."""
    reaches = [handle_reach(instance_points(i.frame, i.surface.points)) for i in items]
    reference = float(np.median(reaches)) if reference is None else reference
    return [with_warp(i, length_warp(r, reference)) for i, r in zip(items, reaches)]


def median_reach(items: Sequence) -> float:
    """The median handle reach of the items (the reference of `align_length`)."""
    return float(np.median([handle_reach(instance_points(i.frame, i.surface.points)) for i in items]))


def align_registered(sources: Sequence, targets: Sequence) -> tuple[list, list]:
    """Sources unchanged; every target's handle registered to the pooled handle points of the sources."""
    pooled = np.concatenate([_handle(i) for i in sources])
    return list(sources), [with_warp(t, registered_warp(_handle(t), pooled)) for t in targets]


def _handle(item) -> np.ndarray:
    points = instance_points(item.frame, item.surface.points)
    return points[handle_mask(points)]
