"""How alike the touches of one kind of part (drawer handle, door handle, stove knob, faucet handle) are across
kitchens: contact position, closing line and the direction the gripper moves next, in the part's own frame and, for
comparison, in the world and robot-base frames (AFFO-15, question 2). Only numpy is used.

Across kitchens means: every touch is compared with the mean of the touches in the *other* layouts (leave one layout
out), so a kitchen never helps to define the reference it is judged by.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from affordance_vla.plan.rc365_events import PartTouch, unit

CLOSING_TOLERANCE_DEG = 20.0  # the step-0 threshold for a closing line
MOTION_TOLERANCE_DEG = 30.0  # for the direction of motion (an arrow, not a line)


def axial_mean(vectors: np.ndarray) -> np.ndarray:
    """The line (unit vector, sign arbitrary) most of the given lines lie along: the top eigenvector of sum d d^T."""
    values, vectors_ = np.linalg.eigh(np.einsum("ni,nj->ij", vectors, vectors))
    return vectors_[:, -1]


def vector_mean(vectors: np.ndarray) -> np.ndarray:
    return unit(vectors.mean(axis=0))


def angle_deg(a: np.ndarray, b: np.ndarray, axial: bool) -> np.ndarray:
    cos = np.clip(a @ b, -1, 1)
    return np.degrees(np.arccos(np.abs(cos) if axial else cos))


def resultant(vectors: np.ndarray, axial: bool) -> float:
    """1 when all alike, near 0 when spread evenly (for lines: the top eigenvalue's excess, in [0, 1])."""
    if axial:
        top = np.linalg.eigvalsh(np.einsum("ni,nj->ij", vectors, vectors))[-1]
        return float((3 * top / len(vectors) - 1) / 2)
    return float(np.linalg.norm(vectors.mean(axis=0)))


def leave_one_layout_out(vectors: np.ndarray, layouts: Sequence, axial: bool) -> np.ndarray:
    """Angle (degrees) of every vector to the mean of the vectors from the other layouts; NaN if there are none."""
    layouts = np.asarray(layouts)
    out = np.full(len(vectors), np.nan)
    for layout in np.unique(layouts):
        inside = layouts == layout
        if inside.all():
            continue
        ref = (axial_mean if axial else vector_mean)(vectors[~inside])
        out[inside] = angle_deg(vectors[inside], ref, axial)
    return out


def direction_stats(vectors: np.ndarray, layouts: Sequence, axial: bool, tolerance: float) -> dict:
    """Spread of a set of directions (or lines): resultant, angle to the overall mean, and the same across kitchens."""
    if len(vectors) < 2:
        return {"n": len(vectors)}
    mean = (axial_mean if axial else vector_mean)(vectors)
    own, across = angle_deg(vectors, mean, axial), leave_one_layout_out(vectors, layouts, axial)
    ok = ~np.isnan(across)
    return {
        "n": len(vectors),
        "resultant": resultant(vectors, axial),
        "median_deg": float(np.median(own)),
        "within_tolerance": float(np.mean(own <= tolerance)),
        "across_kitchens_median_deg": float(np.median(across[ok])) if ok.any() else None,
        "across_kitchens_within_tolerance": float(np.mean(across[ok] <= tolerance)) if ok.any() else None,
    }


def position_stats(points: np.ndarray, layouts: Sequence) -> dict:
    """Contact positions in the part frame: median, scatter (median distance from the median) in cm, per-axis IQR."""
    median = np.median(points, axis=0)
    across = []
    for layout in np.unique(np.asarray(layouts)):
        inside = np.asarray(layouts) == layout
        if not inside.all():
            across += list(np.linalg.norm(points[inside] - np.median(points[~inside], axis=0), axis=1))
    return {
        "median_cm": (100 * median).round(2).tolist(),
        "scatter_cm": float(100 * np.median(np.linalg.norm(points - median, axis=1))),
        "across_kitchens_scatter_cm": float(100 * np.median(across)) if across else None,
        "iqr_cm": (100 * (np.percentile(points, 75, axis=0) - np.percentile(points, 25, axis=0))).round(2).tolist(),
    }


def part_report(touches: Sequence[PartTouch], layouts: Sequence) -> dict:
    """The consistency numbers of one task's touches (`layouts`: one layout id per touch)."""
    layouts = np.asarray(layouts)
    stack = lambda get, keep=None: np.stack([get(t) for t in touches if keep is None or keep(t)])  # noqa: E731
    moving = [k for k, t in enumerate(touches) if t.motion is not None]
    report = {
        "touches": len(touches),
        "kitchens": int(len(np.unique(layouts))),
        "parts": len({t.part for t in touches}),
        "both_fingers": float(np.mean([t.both_fingers for t in touches])),
        "joint_types": {k: sum(t.joint_type == k for t in touches) for k in ("slide", "hinge")},
        "contact_part_frame": position_stats(stack(lambda t: t.contact), layouts),
        "approach_part_frame": direction_stats(stack(lambda t: t.approach), layouts, False, MOTION_TOLERANCE_DEG),
    }
    for frame, getter in (
        ("part", lambda t: t.closing),
        ("world", lambda t: t.world["closing"]),
        ("base", lambda t: t.world["closing_base"]),
    ):
        report[f"closing_{frame}_frame"] = direction_stats(stack(getter), layouts, True, CLOSING_TOLERANCE_DEG)
    for frame, getter in (
        ("part", lambda t: t.motion),
        ("world", lambda t: t.world["motion"]),
        ("base", lambda t: t.world["motion_base"]),
    ):
        vectors = np.stack([getter(touches[k]) for k in moving]) if moving else np.zeros((0, 3))
        report[f"motion_{frame}_frame"] = direction_stats(vectors, layouts[moving], False, MOTION_TOLERANCE_DEG)
    report["motion_defined"] = len(moving) / len(touches)
    signs = defaultdict(int)
    for t in touches:
        signs[t.joint_sign] += 1
    report["joint_sign_counts"] = {str(k): v for k, v in sorted(signs.items())}
    return report
