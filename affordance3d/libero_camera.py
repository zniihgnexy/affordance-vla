"""Projection of world points into LIBERO camera images.

robosuite's `get_camera_extrinsic_matrix` returns a camera-to-world transform in the usual vision
convention (x right, y down the image, z forward) and `get_camera_intrinsic_matrix` a pinhole K, so
a world point projects to pixel (u, v) = (column, row) of the upright image.

The images that LIBERO VLA pipelines train on are the simulator's raw frames rotated by 180°. The
raw frames are stored bottom row first, so the rotation leaves an upright image that is mirrored
left to right. Pass `mirrored=True` to get pixel coordinates in that image.
"""
from __future__ import annotations

import numpy as np

from affordance3d.geometry import transform_points


def project_points(points_world: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray,
                   width: int | None = None, mirrored: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (N, 2) as (u, v) and depth (N,) along the optical axis; pixels are nan for points behind the camera."""
    if mirrored and width is None:
        raise ValueError("mirrored projection needs the image width")
    points_world = np.atleast_2d(np.asarray(points_world, dtype=float))
    camera = transform_points(np.linalg.inv(cam_to_world), points_world)
    depth = camera[:, 2]
    in_front = depth > 0
    safe_depth = np.where(in_front, depth, 1.0)
    uv = (camera @ np.asarray(intrinsics, dtype=float).T)[:, :2] / safe_depth[:, None]
    if mirrored:
        uv[:, 0] = (width - 1) - uv[:, 0]
    uv[~in_front] = np.nan
    return uv, depth
