"""Where a part moves after a touch, read from the joint it drives instead of from the gripper (AFFO-18, question 2).

The extraction (`affordance_vla/sim/rc365_scan.py`) stores, for the touched part, the joint above it: type, axis and
anchor (a point on the axis, world frame) and the joint value at the touch, one second later and at the end of the
demonstration. From these the label of "where it moves next" needs no gripper displacement:

- slide joint: direction = sign * axis;
- hinge joint: direction at the contact = sign * (axis x r) / |axis x r|, r the perpendicular vector from the axis to
  the contact point (FlowBot3D, Eq. 3, without the length). Alongside it the turn itself: sign * axis, which way
  about which axis (the more stable label when the contact is near the axis and r is small, as on a knob).

`sign` is the sign of the change of the joint value over a window (`WINDOWS`): the second after the touch (the old
label's window) or the rest of the demonstration; a change below `SIGN_THRESHOLD` leaves it, and the label, unset.
The part's contact position is also given relative to the joint anchor, in the part body's orientation: a body's own
origin is not where its joint is (the stove knobs). Directions are given in the part, world and robot-base frames
for the same cross-kitchen comparison as `rc365_parts.py`. Only numpy is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from affordance_vla.plan.rc365_events import PartTouch, part_touches, unit
from affordance_vla.plan.rc365_parts import MOTION_TOLERANCE_DEG, angle_deg, direction_stats, position_stats

SIGN_THRESHOLD = 1e-3  # a joint value (rad or m) that changed less than this did not move (as in the old label)
MIN_LEVER_M = 0.01  # a hinge contact closer than this to its axis has an unstable translation direction
WINDOWS = ("1s", "end")  # the sign is read over the second after the touch, or to the end of the demonstration
FRAMES = ("part", "world", "base")
SLIDE, HINGE = "slide", "hinge"


def _mat(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(3, 3)


def _vec(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class JointTouch:
    """One touch of a part with the joint it drives, in the world frame, and the old label of the same touch."""

    task: str
    split: str
    episode: int
    part: str
    joint: str
    joint_type: str  # 'slide' or 'hinge'
    contact: np.ndarray  # (3,) mean contact point of the fingers on the part
    anchor: np.ndarray  # (3,) a point on the joint axis
    axis: np.ndarray  # (3,) unit joint axis
    part_pos: np.ndarray  # (3,) the part body's origin
    part_rot: np.ndarray  # (3, 3)
    base_rot: np.ndarray  # (3, 3)
    dq: dict  # window -> change of the joint value
    old: PartTouch  # the same touch with the gripper-displacement label

    @property
    def lever(self) -> np.ndarray:
        """Perpendicular vector from the joint axis to the contact point (3,)."""
        r = self.contact - self.anchor
        return r - (r @ self.axis) * self.axis

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.lever))

    def sign(self, window: str) -> float:
        """+1 or -1: the way the joint value changed over the window; 0 when it moved less than SIGN_THRESHOLD."""
        change = self.dq[window]
        return float(np.sign(change)) if abs(change) > SIGN_THRESHOLD else 0.0

    def direction(self, window: str) -> np.ndarray | None:
        """Unit direction (world frame) of the part's motion at the contact, or None (no sign, or a hinge contact on
        its axis)."""
        sign = self.sign(window)
        if sign == 0.0:
            return None
        if self.joint_type == SLIDE:
            return sign * self.axis
        tangent = np.cross(self.axis, self.lever)
        return sign * unit(tangent) if np.linalg.norm(tangent) > 1e-9 else None

    def turn(self, window: str) -> np.ndarray | None:
        """sign * axis (world frame): which way about which axis, or None when the sign is unset."""
        sign = self.sign(window)
        return sign * self.axis if sign else None

    def contact_from_anchor(self) -> np.ndarray:
        """The contact point relative to the joint anchor, in the part body's orientation (3,)."""
        return self.part_rot.T @ (self.contact - self.anchor)

    def contact_from_body(self) -> np.ndarray:
        """The contact point relative to the part body's origin, in its orientation: the old record (3,)."""
        return self.part_rot.T @ (self.contact - self.part_pos)

    def in_frame(self, vector: np.ndarray, frame: str) -> np.ndarray:
        """A world-frame direction in the part, world or robot-base frame."""
        return {"part": self.part_rot.T, "world": np.eye(3), "base": self.base_rot.T}[frame] @ vector


def joint_touches(results: Sequence[dict]) -> list[JointTouch]:
    """One JointTouch per part event of the extraction results that records a joint (all of AFFO-15's four tasks do)."""
    found = []
    for result in results:
        for e in result["events"]:
            if e["kind"] != "part" or e.get("joint_type") is None:
                continue
            old = part_touches([{**result, "events": [e]}])[0]
            points = np.array(e["points"]) if e["points"] else _vec(e["grasp_centre"])[None]
            found.append(
                JointTouch(
                    result["task"],
                    result["split"],
                    result["episode"],
                    e["part"],
                    e["joint"],
                    {2: SLIDE, 3: HINGE}[e["joint_type"]],
                    points.mean(axis=0),
                    _vec(e["joint_anchor"]),
                    unit(_vec(e["joint_axis"])),
                    _vec(e["part_pos"]),
                    _mat(e["part_rot"]),
                    _mat(e["base_rot"]),
                    {"1s": e["q_later"] - e["q_start"], "end": e["q_end"] - e["q_start"]},
                    old,
                )
            )
    return found


def _stats(vectors: list[np.ndarray], layouts: Sequence) -> dict:
    return direction_stats(np.stack(vectors) if vectors else np.zeros((0, 3)), layouts, False, MOTION_TOLERANCE_DEG)


def _frame_stats(touches: Sequence[JointTouch], layouts: Sequence, get) -> dict:
    """direction_stats of `get(touch)` (a world-frame vector or None) in each frame, over the touches that have one."""
    keep = [k for k, t in enumerate(touches) if get(t) is not None]
    lay = np.asarray(layouts)[keep]
    return {f: _stats([t.in_frame(get(t), f) for t in (touches[k] for k in keep)], lay) for f in FRAMES}


def contact_report(touches: Sequence[JointTouch], layouts: Sequence) -> dict:
    """Scatter of the contact position with the part body's origin and with the joint anchor as the frame's origin."""
    stack = lambda get: np.stack([get(t) for t in touches])  # noqa: E731
    return {
        "from_body_origin": position_stats(stack(JointTouch.contact_from_body), layouts),
        "from_joint_anchor": position_stats(stack(JointTouch.contact_from_anchor), layouts),
        "contact_to_body_origin_cm": float(100 * np.median([np.linalg.norm(t.contact - t.part_pos) for t in touches])),
        "contact_to_joint_anchor_cm": float(100 * np.median([np.linalg.norm(t.contact - t.anchor) for t in touches])),
    }


def window_report(touches: Sequence[JointTouch], layouts: Sequence, window: str) -> dict:
    """The new label next to the old one on the same touches, for one sign window."""
    n = len(touches)
    new_dir = lambda t: t.direction(window)  # noqa: E731
    turn = lambda t: t.turn(window)  # noqa: E731
    both = [k for k, t in enumerate(touches) if new_dir(t) is not None and t.old.motion is not None]
    common = [touches[k] for k in both]
    angles = np.array([angle_deg(t.in_frame(new_dir(t), "part"), t.old.motion, False) for t in common])
    lay_both = np.asarray(layouts)[both]
    old_moving = [k for k, t in enumerate(touches) if t.old.motion is not None]
    lay = np.asarray(layouts)
    return {
        "window": window,
        "new_defined": float(np.mean([new_dir(t) is not None for t in touches])),
        "new_direction": _frame_stats(touches, layouts, new_dir),
        "turn_axis_sign": _frame_stats(touches, layouts, turn),
        "old_defined": len(old_moving) / n,
        "old_direction_part_frame": _stats([touches[k].old.motion for k in old_moving], lay[old_moving]),
        "both_defined": len(both),
        "on_both_defined": {
            "new_direction_part_frame": _stats([t.in_frame(new_dir(t), "part") for t in common], lay_both),
            "old_direction_part_frame": _stats([t.old.motion for t in common], lay_both),
        },
        "new_vs_old_angle_deg": {
            "n": len(both),
            "median": float(np.median(angles)) if len(angles) else None,
            "within_30": float(np.mean(angles <= 30)) if len(angles) else None,
            "within_45": float(np.mean(angles <= 45)) if len(angles) else None,
        },
        "sign_counts": {str(s): int(sum(t.sign(window) == s for t in touches)) for s in (-1.0, 0.0, 1.0)},
    }


def grouped_report(touches: Sequence[JointTouch], layouts: Sequence, groups: Sequence[str], window: str) -> dict:
    """Part-frame consistency of the new direction, the turn (axis and sign) and the old direction inside each group
    of touches (parts of one kind in different designs, e.g. stoves with knobs on the front or on the top)."""
    layouts, groups = np.asarray(layouts), np.asarray(groups)
    out = {}
    for name in sorted(set(groups)):
        keep = np.flatnonzero(groups == name)
        sub = [touches[k] for k in keep]
        has_old = np.array([t.old.motion is not None for t in sub])
        moving = [t.old.motion for t in sub if t.old.motion is not None]
        direction, turn = (lambda t: t.direction(window)), (lambda t: t.turn(window))
        out[str(name)] = {
            "touches": len(sub),
            "new_direction": _frame_stats(sub, layouts[keep], direction)["part"],
            "turn_axis_sign": _frame_stats(sub, layouts[keep], turn)["part"],
            "old_direction": _stats(moving, layouts[keep][has_old]),
            "sign_counts": {str(v): int(sum(t.sign(window) == v for t in sub)) for v in (-1.0, 0.0, 1.0)},
        }
    return out


def joint_report(touches: Sequence[JointTouch], layouts: Sequence, groups: Sequence[str] | None = None) -> dict:
    """Everything the comparison of the two labels needs for one set of touches of one task; with `groups` (one name
    per touch) also the consistency inside each group."""
    radius = np.array([t.radius for t in touches])
    return {
        "touches": len(touches),
        "kitchens": int(len(np.unique(np.asarray(layouts)))),
        "joint_types": {k: sum(t.joint_type == k for t in touches) for k in (SLIDE, HINGE)},
        "lever_radius_cm": {
            "median": float(100 * np.median(radius)),
            "below_min_lever": float(np.mean(radius < MIN_LEVER_M)),
        },
        "contact": contact_report(touches, layouts),
        **{f"window_{w}": window_report(touches, layouts, w) for w in WINDOWS},
        **({f"groups_{w}": grouped_report(touches, layouts, groups, w) for w in WINDOWS} if groups is not None else {}),
    }
