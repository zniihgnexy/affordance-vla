"""Where a hand touches an object, and along which line the thumb and the other fingers oppose.

A surface point of the object counts as touched in a frame when a point of a hand lies within `threshold`
(5 mm by default) of it. The hand is a set of points on the UmeTrack mesh, each tagged with its finger (the
skin weights of the bones; the five fingers are the bone groups 2-4, 5-7, 8-10, 11-13, 14-16, palm and wrist the
rest). The closing axis of a frame is the line from the centroid of the thumb's touching points to the centroid
of the other fingers' touching points: where a parallel gripper's two pads would sit. Only numpy and scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from affordance_vla.plan.interaction_labels import Grasp

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_BONES = (slice(2, 5), slice(5, 8), slice(8, 11), slice(11, 14), slice(14, 17))
PALM = len(FINGERS)  # label of the points that belong to no finger
THUMB = 0
THRESHOLD_M = 0.005
MIN_PAD_SPAN_M = 0.02  # thumb and finger centroids closer than this do not oppose
MIN_PAD_POINTS = 3  # hand points a finger group needs within the threshold to count as touching


@dataclass(frozen=True)
class HandSamples:
    """Fixed points on the hand mesh: a face and barycentric weights each, and the finger they belong to."""

    faces: np.ndarray  # (P,) index into the mesh triangles
    bary: np.ndarray  # (P, 3)
    finger: np.ndarray  # (P,) 0-4 in FINGERS order, PALM otherwise

    def points(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """(P, 3) the samples on a posed mesh."""
        corners = vertices[triangles[self.faces]]
        return np.einsum("pc,pcd->pd", self.bary, corners)


def _finger_labels(bone_weights: np.ndarray) -> np.ndarray:
    palm = bone_weights[:, :2].sum(axis=1)  # wrist and palm bones
    groups = np.stack([bone_weights[:, bones].sum(axis=1) for bones in FINGER_BONES] + [palm], axis=1)
    return groups.argmax(axis=1)


def vertex_fingers(model) -> np.ndarray:
    """(V,) finger label of each vertex of a UmeTrack hand model's mesh."""
    return _finger_labels(model.dense_bone_weights.numpy().astype(float))


def sample_hand(model, count: int, seed: int = 0) -> tuple[HandSamples, np.ndarray]:
    """Sample points on a UmeTrack hand model's mesh; returns them and the mesh's triangles.

    `count` points are uniform over the surface (by the rest-pose face areas); count = 0 takes the mesh's own
    vertices, a few millimetres apart, which under-counts touches at a 5 mm threshold.
    """
    triangles = model.mesh_triangles.int().numpy()
    bone_weights = model.dense_bone_weights.numpy().astype(float)
    if count == 0:
        owner = np.zeros(len(bone_weights), dtype=int)
        corner = np.zeros(len(bone_weights), dtype=int)
        for c in range(3):
            owner[triangles[:, c]] = np.arange(len(triangles))
            corner[triangles[:, c]] = c
        return HandSamples(owner, np.eye(3)[corner], _finger_labels(bone_weights)), triangles
    rest = model.mesh_vertices.numpy().astype(float)[triangles]
    area = np.linalg.norm(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(triangles), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    outside = u + v > 1
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    bary = np.stack([1 - u - v, u, v], axis=1)
    weights = np.einsum("pc,pcb->pb", bary, bone_weights[triangles[faces]])
    return HandSamples(faces, bary, _finger_labels(weights)), triangles


def touched_points(surface: np.ndarray, hand: np.ndarray, threshold: float = THRESHOLD_M) -> np.ndarray:
    """(N,) bool: surface points with a hand point within `threshold`."""
    distance, _ = cKDTree(hand).query(surface, distance_upper_bound=threshold)
    return np.isfinite(distance)


def touching_hand_points(surface: np.ndarray, hand: np.ndarray, threshold: float = THRESHOLD_M) -> np.ndarray:
    """(P,) bool: hand points within `threshold` of the object surface (sampled as `surface`)."""
    distance, _ = cKDTree(surface).query(hand, distance_upper_bound=threshold)
    return np.isfinite(distance)


def to_object_frame(points_world: np.ndarray, object_pose: np.ndarray) -> np.ndarray:
    """Points of the world frame in the frame of an object with pose T_world_from_object."""
    return (points_world - object_pose[:3, 3]) @ object_pose[:3, :3]


@dataclass(frozen=True)
class Opposition:
    centre: np.ndarray  # (3,) midpoint of the two pads
    closing: np.ndarray  # (3,) unit, from the thumb pad towards the other fingers
    span: float  # distance between the pads
    fingers: tuple[str, ...]  # the fingers opposing the thumb


def opposition(hand_obj: np.ndarray, finger: np.ndarray, touching: np.ndarray) -> Opposition | None:
    """The thumb against the other touching fingers, or None when the thumb or all the others do not touch.

    hand_obj: (P, 3) hand points in the object frame; finger: (P,) labels; touching: (P,) points near the surface.
    """
    thumb = touching & (finger == THUMB)
    others = touching & (finger != THUMB) & (finger != PALM)
    if thumb.sum() < MIN_PAD_POINTS or others.sum() < MIN_PAD_POINTS:
        return None
    pad_a, pad_b = hand_obj[thumb].mean(axis=0), hand_obj[others].mean(axis=0)
    span = float(np.linalg.norm(pad_b - pad_a))
    if span < MIN_PAD_SPAN_M:
        return None
    names = tuple(FINGERS[k] for k in range(1, len(FINGERS)) if (others & (finger == k)).sum() >= MIN_PAD_POINTS)
    return Opposition((pad_a + pad_b) / 2, (pad_b - pad_a) / span, span, names)


def as_grasp(pair: Opposition, wrist_obj: np.ndarray) -> Grasp:
    """The opposition as a gripper grasp of interaction_labels: the fingers point from the wrist to the pads."""
    approach = pair.centre - wrist_obj
    approach = approach - (approach @ pair.closing) * pair.closing  # the gripper's approach is square to its closing
    norm = np.linalg.norm(approach)
    if norm < 1e-6:
        raise ValueError("the wrist is on the closing line")
    return Grasp(pair.centre, pair.closing, approach / norm)


def hold_segments(flags: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """(start, end) frame indices (end exclusive) of runs of True that last at least `min_length` frames."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_length]
