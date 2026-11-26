"""The category frame of Pipeline v3 (section 3, "Category-level L2"): one shared coordinate system for every mesh
of an object category, so that a map made on one mesh can be carried to another.

Vertical axis up, origin at the centre of the base, height and horizontal radius each scaled to 1, handle along
+x. A frame is fitted to one mesh's surface points (in the mesh's own body frame) and a given up direction (for a
scene object: its up axis at rest on the table, see `up_at_rest`). Points move with the full map
p -> S R (p - o); a closing axis moves like a surface normal, as the pipeline specifies: d -> S^-1 R d, then
normalised (and back: d -> R^T S d). Only numpy is used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance_vla.analysis.grasp_orientation import rotation_matrices

BASE_BAND = 0.05  # the lowest share of the height whose points give the centre of the base
HANDLE_SHARE = 0.05  # the share of points farthest from the vertical axis whose direction is the handle's
WORLD_UP = np.array([0.0, 0.0, 1.0])


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@dataclass(frozen=True)
class CategoryFrame:
    origin: np.ndarray  # (3,) the base centre, in the body frame
    rotation: np.ndarray  # (3, 3) rows are the category x, y, z axes in the body frame (z up, x towards the handle)
    scale: np.ndarray  # (3,) 1/radius, 1/radius, 1/height

    def point_to_category(self, points: np.ndarray) -> np.ndarray:
        return ((np.asarray(points) - self.origin) @ self.rotation.T) * self.scale

    def point_from_category(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) / self.scale) @ self.rotation + self.origin

    def axis_to_category(self, axes: np.ndarray) -> np.ndarray:
        """Unit axes (or normals) of the body frame in the category frame: S^-1 R d, normalised."""
        return _unit((np.asarray(axes) @ self.rotation.T) / self.scale)

    def axis_from_category(self, axes: np.ndarray) -> np.ndarray:
        """Unit axes of the category frame in the body frame: R^T S c, normalised."""
        return _unit((np.asarray(axes) * self.scale) @ self.rotation)


def up_at_rest(object_quats: np.ndarray) -> np.ndarray:
    """The body-frame direction of world up, from object orientations (x, y, z, w) at rest on the table: the mean
    over the given poses, normalised."""
    ups = rotation_matrices(np.asarray(object_quats).reshape(-1, 4)).transpose(0, 2, 1) @ WORLD_UP
    return _unit(ups.mean(axis=0))


def scene_aligned_frame(frame: CategoryFrame, rest_quat: np.ndarray) -> CategoryFrame:
    """The same frame with its x axis turned from the handle to the scene's world +x, as the object rests (diagnostic:
    it keeps where the demonstrators stood, which for LIBERO's fixed mug orientations is not tied to the handle)."""
    up = frame.rotation[2]
    x_axis = rotation_matrices(np.asarray(rest_quat)).T @ np.array([1.0, 0.0, 0.0])
    x_axis = _unit(x_axis - (x_axis @ up) * up)
    return CategoryFrame(frame.origin, np.stack([x_axis, np.cross(up, x_axis), up]), frame.scale)


def fit_frame(points: np.ndarray, up: np.ndarray) -> CategoryFrame:
    """The category frame of one mesh from its surface points (body frame) and its up direction."""
    points, up = np.asarray(points, dtype=float), _unit(np.asarray(up, dtype=float))
    height = points @ up
    low, span = height.min(), height.max() - height.min()
    horizontal = points - np.outer(height, up)
    base = horizontal[height <= low + BASE_BAND * span].mean(axis=0)  # centre of the base disc, in the horizontal plane
    radial = horizontal - base
    distance = np.linalg.norm(radial, axis=1)
    handle = radial[distance >= np.quantile(distance, 1 - HANDLE_SHARE)]
    x_axis = _unit(_unit(handle).mean(axis=0))
    x_axis = _unit(x_axis - (x_axis @ up) * up)
    rotation = np.stack([x_axis, np.cross(up, x_axis), up])
    radius = np.median(distance)
    return CategoryFrame(base + low * up, rotation, np.array([1 / radius, 1 / radius, 1 / span]))
