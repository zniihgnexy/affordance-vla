"""Interaction affordance labels of Pipeline v3 (L1, L2; docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md, section 3).

A held grasp of a demonstration, expressed in the object's own frame, marks where on the object's surface the
fingers close and along which line: the closing axis, a line rather than an arrow, since turning the gripper
by 180 degrees gives the same grasp. Summing the grasps of every demonstration of one object category gives
its 3D canonical map: for each surface point (affordance_vla/sim/object_surface.py) an interaction score in
[0, 1], the dominant closing axis there, and how much the grasps agree on that axis. A map is projected into a
logged camera frame with the object's pose; points hidden by other things are dropped with the logged
segmentation, points hidden by the object itself with a coarse depth test. Object poses serve the labels and
the evaluation only, never the model. Only numpy is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from affordance_vla.analysis.grasp_orientation import APPROACH_AXIS, CLOSING_AXIS, rotation_matrices
from affordance_vla.sim.libero_camera import project_points
from affordance_vla.sim.object_surface import Surface

PAD_RADIUS_M = 0.012  # finger pad half-width: a surface point this far from the closing line still counts ~0.6
MAX_REACH_M = 0.05  # half the widest finger opening plus a margin: farther along the closing line is out of reach
MIN_FACING = 0.3  # |normal . closing axis| up to which a point does not face a finger at all
AXIS_TIP_M = 0.01  # length of the axis segment projected to find a point's closing direction in the image
DEPTH_BLOCK_PX = 4  # pixel blocks of the self-occlusion test
SELF_OCCLUSION_M = 0.015  # a point deeper than its block's nearest point by more than this is behind the object
CELLS = 8
MIN_HOLD_STEPS = 5  # a grasp counts when the simulator reports it for this many steps in a row (0.25 s at 20 Hz)


@dataclass(frozen=True)
class Grasp:
    """One held grasp in the object's frame."""

    centre: np.ndarray  # (3,) the grasp point between the fingers
    closing: np.ndarray  # (3,) unit, the line the fingers close along
    approach: np.ndarray  # (3,) unit, the direction the fingers point


@dataclass(frozen=True)
class CanonicalMap:
    """The interaction map of one object category on its surface, in the object's frame."""

    category: str
    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3)
    score: np.ndarray  # (N,) in [0, 1], 1 where the most grasp weight lands
    axis: np.ndarray  # (N, 3) unit dominant closing axis (sign arbitrary); zero where no grasp touches
    agreement: np.ndarray  # (N,) in [0, 1]: 1 when every grasp there closes along one line, 0 when spread evenly
    grasps: int


@dataclass(frozen=True)
class FrameLabel:
    """A map projected into one camera frame, per image cell of the policy's view."""

    score: np.ndarray  # (CELLS, CELLS) highest score of the object's visible map points in each cell
    direction: np.ndarray  # (CELLS, CELLS, 2) score-weighted mean of (cos 2θ, sin 2θ) of the projected axis
    visible: int  # map points that survived the visibility tests


def held_grasp_steps(grasped: np.ndarray, min_steps: int = MIN_HOLD_STEPS) -> list[tuple[int, int]]:
    """(object index, step) of every held grasp of an episode, in time order: the step a grasp starts to register,
    kept when it lasts `min_steps` steps. Unlike E0's attempts this does not tie the grasp to a close command:
    in many demonstrations the fingers close well before the grasp takes hold."""
    grasped = np.asarray(grasped, dtype=bool)
    found = []
    for k in range(grasped.shape[1]):
        column = np.concatenate([[False], grasped[:, k], [False]])
        starts = np.flatnonzero(~column[:-1] & column[1:])
        ends = np.flatnonzero(column[:-1] & ~column[1:])
        found += [(k, int(start)) for start, end in zip(starts, ends) if end - start >= min_steps]
    return sorted(found, key=lambda item: item[1])


def grasp_in_object(ee_pos: np.ndarray, ee_quat: np.ndarray, object_pos: np.ndarray,
                    object_quat: np.ndarray) -> Grasp:
    """The end-effector pose at a grasp (quaternions in robosuite's x, y, z, w order) in the object's frame."""
    ee_rotation, inverse = rotation_matrices(np.asarray(ee_quat)), rotation_matrices(np.asarray(object_quat)).T
    return Grasp(inverse @ (np.asarray(ee_pos) - np.asarray(object_pos)), inverse @ ee_rotation[:, CLOSING_AXIS],
                 inverse @ ee_rotation[:, APPROACH_AXIS])


def contact_weights(surface: Surface, grasp: Grasp) -> np.ndarray:
    """(N,) in [0, 1]: how much each surface point is touched by a grasp's fingers.

    A Gaussian of the point's distance to the closing line, cut beyond the fingers' reach along it, times how
    squarely the point faces along that line (a point on the top face next to the line is not under a finger).
    """
    offset = surface.points - grasp.centre
    along = offset @ grasp.closing
    off_line = np.linalg.norm(offset - along[:, None] * grasp.closing, axis=1)
    facing = np.clip((np.abs(surface.normals @ grasp.closing) - MIN_FACING) / (1 - MIN_FACING), 0.0, 1.0)
    return np.exp(-0.5 * (off_line / PAD_RADIUS_M) ** 2) * facing * (np.abs(along) <= MAX_REACH_M)


def canonical_map(category: str, surface: Surface, grasps: Sequence[Grasp]) -> CanonicalMap:
    """Sum the grasps of one category on its surface: score, dominant closing axis and agreement per point."""
    if not grasps:
        raise ValueError(f"no grasps of {category}")
    weights = np.stack([contact_weights(surface, grasp) for grasp in grasps])  # (G, N)
    closings = np.stack([grasp.closing for grasp in grasps])  # (G, 3)
    total = weights.sum(axis=0)
    score = total / total.max() if total.max() > 0 else total
    tensor = np.einsum("gn,gi,gj->nij", weights, closings, closings)  # axial: d and -d add up the same
    values, vectors = np.linalg.eigh(tensor)  # ascending eigenvalues
    trace = values.sum(axis=1)
    touched = trace > 1e-9
    axis = np.where(touched[:, None], vectors[:, :, -1], 0.0)
    agreement = np.where(touched, (3 * values[:, -1] / np.where(touched, trace, 1.0) - 1) / 2, 0.0)
    return CanonicalMap(category, surface.points, surface.normals, score, axis, np.clip(agreement, 0.0, 1.0),
                        len(grasps))


def line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines (directions up to sign), in [0, 90]."""
    cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def axis_groups(closings: Sequence[np.ndarray], max_angle_deg: float = 20.0) -> list[int]:
    """Sizes of the groups of closing axes (largest first): an axis joins the first group whose first axis is
    within `max_angle_deg` of it as a line. The pass check of L wants a category's grasps in a few groups."""
    leaders, sizes = [], []
    for closing in closings:
        for k, leader in enumerate(leaders):
            if line_angle_deg(closing, leader) <= max_angle_deg:
                sizes[k] += 1
                break
        else:
            leaders.append(np.asarray(closing))
            sizes.append(1)
    return sorted(sizes, reverse=True)


def _visible_points(uv: np.ndarray, depth: np.ndarray, seg: np.ndarray, seg_id: int) -> np.ndarray:
    """Points inside the image, on a pixel of this object, and not behind the object's own nearer surface."""
    size = seg.shape[0]
    inside = np.isfinite(uv).all(axis=1) & (depth > 0)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    keep = np.zeros(len(uv), dtype=bool)
    idx = np.flatnonzero(inside)
    cols, rows = uv[idx, 0].astype(int), uv[idx, 1].astype(int)
    on_object = seg[rows, cols] == seg_id
    idx, cols, rows = idx[on_object], cols[on_object], rows[on_object]
    blocks = (rows // DEPTH_BLOCK_PX) * (size // DEPTH_BLOCK_PX + 1) + cols // DEPTH_BLOCK_PX
    nearest = np.full(blocks.max() + 1 if len(blocks) else 1, np.inf)
    np.minimum.at(nearest, blocks, depth[idx])
    keep[idx[depth[idx] <= nearest[blocks] + SELF_OCCLUSION_M]] = True
    return keep


def project_map(cmap: CanonicalMap, object_pos: np.ndarray, object_quat: np.ndarray, intrinsics: np.ndarray,
                cam_to_world: np.ndarray, seg: np.ndarray, seg_id: int, cells: int = CELLS) -> FrameLabel:
    """The map of one object in one logged frame (seg in the logged image's orientation, value seg_id on it)."""
    rotation = rotation_matrices(np.asarray(object_quat))
    world = cmap.points @ rotation.T + np.asarray(object_pos)
    size = seg.shape[0]
    uv, depth = project_points(world, intrinsics, cam_to_world, width=size, mirrored=True)
    tip_uv, _ = project_points(world + AXIS_TIP_M * (cmap.axis @ rotation.T), intrinsics, cam_to_world,
                               width=size, mirrored=True)
    visible = _visible_points(uv, depth, seg, seg_id) & (cmap.score > 0)
    score = np.zeros((cells, cells))
    direction = np.zeros((cells, cells, 2))
    weight = np.zeros((cells, cells))
    if visible.any():
        rows = (uv[visible, 1] * cells / size).astype(int)
        cols = (uv[visible, 0] * cells / size).astype(int)
        step = tip_uv[visible] - uv[visible]
        theta = np.arctan2(step[:, 1], step[:, 0])
        doubled = np.stack([np.cos(2 * theta), np.sin(2 * theta)], axis=1)
        point_score = cmap.score[visible]
        np.maximum.at(score, (rows, cols), point_score)
        np.add.at(direction, (rows, cols), point_score[:, None] * doubled)
        np.add.at(weight, (rows, cols), point_score)
        direction = np.where(weight[..., None] > 0, direction / np.maximum(weight, 1e-12)[..., None], 0.0)
    return FrameLabel(score, direction, int(visible.sum()))


def save_maps(path, maps: Sequence[CanonicalMap]) -> None:
    arrays = {}
    for cmap in maps:
        for field in ("points", "normals", "score", "axis", "agreement"):
            arrays[f"{cmap.category}/{field}"] = getattr(cmap, field)
        arrays[f"{cmap.category}/grasps"] = np.array(cmap.grasps)
    np.savez_compressed(path, **arrays)


def load_maps(path) -> dict[str, CanonicalMap]:
    with np.load(path) as data:
        categories = sorted({key.split("/")[0] for key in data.files})
        return {c: CanonicalMap(c, data[f"{c}/points"], data[f"{c}/normals"], data[f"{c}/score"], data[f"{c}/axis"],
                                data[f"{c}/agreement"], int(data[f"{c}/grasps"])) for c in categories}
