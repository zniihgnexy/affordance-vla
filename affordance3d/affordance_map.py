"""An affordance map built from ground truth, and the 3D point such a map stands for.

The map is an image: each pixel says how much the surface seen there belongs to the place where the
gripper should act on an object. Here it is not predicted. It is computed from the simulator (which
pixels show the object, how far each is from where the gripper acts) and stands in for the output of
an affordance head until one is trained. The definition is one possible label, not a settled choice.
"""
from __future__ import annotations

import numpy as np


def contact_region_map(points: np.ndarray, mask: np.ndarray, contact: np.ndarray, sigma: float,
                       horizontal: bool = False) -> np.ndarray:
    """Map (H, W) in [0, 1]: a Gaussian of each object pixel's distance to the contact, 0 off the object.

    `points` (H, W, 3) are the world points seen in each pixel, `mask` (H, W) the pixels that show the
    object, `contact` (3,) where the gripper acts and `sigma` the width in metres. With `horizontal`
    only the distance in the ground plane counts, for a contact above the surface (a release point).
    """
    points, contact = np.asarray(points, dtype=float), np.asarray(contact, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or contact.shape != (3,):
        raise ValueError(f"points must be (H, W, 3) and contact (3,), got {points.shape} and {contact.shape}")
    if mask.shape != points.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match the {points.shape[:2]} image")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    offset = points - contact
    if horizontal:
        offset = offset[..., :2]
    heat = np.exp(-0.5 * (offset ** 2).sum(axis=-1) / sigma ** 2)
    return np.where(mask, heat, 0.0)


def map_point(points: np.ndarray, heat: np.ndarray) -> np.ndarray | None:
    """The 3D point (3,) a map stands for: the mean of the pixels' world points weighted by the map.

    Returns None for a map that is zero everywhere (the object is not visible) or has undefined weights.
    """
    points, heat = np.asarray(points, dtype=float), np.asarray(heat, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or heat.shape != points.shape[:2]:
        raise ValueError(f"the map {heat.shape} does not match the points {points.shape}")
    total = heat.sum()
    if not total > 0:  # also catches a sum that is not a number
        return None
    return (points * heat[..., None]).sum(axis=(0, 1)) / total
