"""Interaction maps of a category whose parts are symmetric about the vertical axis (mug body, cup, bowl), and the
two ways of handling a demonstration set that covers only the side facing the robot (AFFO-15, question 3).

Every treatment is built from the grasps of source instances and gives, for a held-out grasp of a target instance,
a map on the target's surface that the same check as in step 0 (`transfer_check.py`) judges:

- `PlainTreatment`: the step-0 transfer, in the category frame fitted to the handle; the robot is not used;
- `RingTreatment` (1): a grasp on the body (more than HANDLE_ZONE_DEG from the handle) is repeated around the vertical
  axis, so the map holds the whole ring; at use the map is cut to the half of the body facing the robot (WINDOW_DEG)
  and the handle zone; a category without a handle has no handle zone;
- `RobotFrameTreatment` (2): every grasp is recorded in a frame whose x axis points from the object to the robot's
  base, and the map is carried into the same frame of the held-out grasp's robot.
Only numpy and scipy are used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import numpy as np
from scipy.spatial import cKDTree

from affordance_vla.plan.category_frame import CategoryFrame, fit_frame
from affordance_vla.plan.category_transfer import Source, category_map, transfer_nearest
from affordance_vla.plan.interaction_labels import CanonicalMap, Grasp
from affordance_vla.plan.rc365_events import ObjectGrasp, mean_up
from affordance_vla.plan.transfer_check import Outcomes, grasp_outcomes, summarise
from affordance_vla.sim.object_surface import Surface

HANDLE_ZONE_DEG = 50.0  # a grasp centre this close (azimuth) to the handle direction is a handle grasp
WINDOW_DEG = 90.0  # at use, the body ring is kept this far either side of the robot's azimuth
RING_COPIES = 12
MAX_UP_SPREAD_DEG = 5.0


@dataclass(frozen=True)
class Item:
    """One object instance: its surface, category frame and held grasps (body frame) with where the robot stood."""

    name: str
    surface: Surface
    frame: CategoryFrame
    grasps: tuple[Grasp, ...]
    robot: tuple[np.ndarray, ...]  # per grasp: the robot base in the body frame


def build_items(grasps: Sequence[ObjectGrasp], surfaces: dict[str, Surface]) -> list[Item]:
    """One item per instance; an instance whose up direction at rest varies by over MAX_UP_SPREAD_DEG is refused."""
    items = []
    for name in sorted({g.instance for g in grasps}):
        own = [g for g in grasps if g.instance == name]
        up, spread = mean_up([g.up for g in own])
        if spread > MAX_UP_SPREAD_DEG:
            raise ValueError(f"{name} does not rest upright: {spread:.1f} degrees")
        items.append(
            Item(
                name,
                surfaces[name],
                fit_frame(surfaces[name].points, up),
                tuple(g.grasp for g in own),
                tuple(g.base_in_object for g in own),
            )
        )
    return items


def azimuth(frame: CategoryFrame, points: np.ndarray) -> np.ndarray:
    """Angle (radians, (-pi, pi]) of body-frame points around the frame's vertical axis, from its x axis."""
    local = frame.point_to_category(np.atleast_2d(points))
    return np.arctan2(local[:, 1], local[:, 0])


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def turned(frame: CategoryFrame, angle: float) -> CategoryFrame:
    """The frame with its x axis turned by `angle` (radians) about its vertical axis."""
    x, y, z = frame.rotation
    c, s = np.cos(angle), np.sin(angle)
    return CategoryFrame(frame.origin, np.stack([c * x + s * y, -s * x + c * y, z]), frame.scale)


def robot_frame(frame: CategoryFrame, robot: np.ndarray) -> CategoryFrame:
    """The frame with its x axis pointing from the object to the robot's base."""
    return turned(frame, float(azimuth(frame, robot)[0]))


def rotate_grasp(frame: CategoryFrame, grasp: Grasp, angle: float) -> Grasp:
    """The grasp turned by `angle` about the frame's vertical axis (through the base centre)."""
    c, s = np.cos(angle), np.sin(angle)
    turn = frame.rotation.T @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ frame.rotation
    return Grasp(frame.origin + turn @ (grasp.centre - frame.origin), turn @ grasp.closing, turn @ grasp.approach)


def circular_stats(angles: np.ndarray) -> dict:
    """Mean direction (degrees), resultant length (1 = all alike, 0 = spread evenly) and a 12-bin histogram."""
    z = np.exp(1j * np.asarray(angles)).mean()
    counts, _ = np.histogram(np.degrees(angles), bins=12, range=(-180, 180))
    return {"mean_deg": float(np.degrees(np.angle(z))), "resultant": float(abs(z)), "histogram_30deg": counts.tolist()}


def azimuth_distributions(items: Sequence[Item]) -> dict:
    """The azimuth of the grasp centres in the handle-aligned frame and relative to the robot."""
    own = np.concatenate([azimuth(i.frame, np.stack([g.centre for g in i.grasps])) for i in items])
    robot = np.concatenate([azimuth(i.frame, np.stack(i.robot)) for i in items])
    return {
        "object_frame": circular_stats(own),
        "robot_facing_frame": circular_stats(wrap(own - robot)),
        "robot_in_object_frame": circular_stats(robot),
        "grasps": len(own),
    }


class PlainTreatment:
    """Base of the treatments: `maps(target, which)` gives the map carried to the target for each held-out grasp."""

    name = "plain"

    def __init__(self, sources: Sequence[Item], handle: bool = True):
        self.cmap = category_map([Source(s.name, s.frame, s.surface, s.grasps) for s in sources])
        self.tree, self.handle = cKDTree(self.cmap.points), handle

    def carried(self, target: Item, frame: CategoryFrame | None = None) -> CanonicalMap:
        return transfer_nearest(self.cmap, frame or target.frame, target.surface, target.name, tree=self.tree)

    def maps(self, target: Item, which: Sequence[int]) -> Iterator[tuple[int, CanonicalMap]]:
        cmap = self.carried(target)
        for k in which:
            yield k, cmap

    def outcomes(self, target: Item, which: Sequence[int]) -> Outcomes:
        total = None
        for k, cmap in self.maps(target, which):
            out = grasp_outcomes(cmap, target.surface, [target.grasps[k]])
            total = out if total is None else total + out
        return total


class RingTreatment(PlainTreatment):
    name = "ring"

    def __init__(self, sources: Sequence[Item], handle: bool = True, copies: int = RING_COPIES):
        filled = []
        for item in sources:
            grasps = []
            for g in item.grasps:
                home = float(azimuth(item.frame, g.centre)[0])
                if handle and abs(np.degrees(home)) <= HANDLE_ZONE_DEG:
                    grasps.append(g)  # a handle grasp stays where it is
                    continue
                for k in range(copies):
                    angle = 2 * np.pi * k / copies
                    if handle and abs(np.degrees(wrap(home + angle))) <= HANDLE_ZONE_DEG:
                        continue  # the copy would land on the handle
                    grasps.append(rotate_grasp(item.frame, g, angle))
            filled.append(replace(item, grasps=tuple(grasps)))
        super().__init__(filled, handle)

    def window(self, cmap: CanonicalMap, target: Item, robot: np.ndarray) -> CanonicalMap:
        az = azimuth(target.frame, target.surface.points)
        keep = np.abs(wrap(az - azimuth(target.frame, robot)[0])) <= np.radians(WINDOW_DEG)
        if self.handle:
            keep |= np.abs(np.degrees(az)) <= HANDLE_ZONE_DEG
        return replace(cmap, score=np.where(keep, cmap.score, 0.0))

    def maps(self, target: Item, which: Sequence[int]) -> Iterator[tuple[int, CanonicalMap]]:
        full = self.carried(target)
        for k in which:
            yield k, self.window(full, target, target.robot[k])


class RobotFrameTreatment(PlainTreatment):
    name = "robot"

    def __init__(self, sources: Sequence[Item], handle: bool = True):
        self.handle = handle
        parts = [
            Source(f"{s.name}#{k}", robot_frame(s.frame, s.robot[k]), s.surface, (s.grasps[k],))
            for s in sources
            for k in range(len(s.grasps))
        ]
        self.cmap = category_map(parts)
        self.tree = cKDTree(self.cmap.points)

    def maps(self, target: Item, which: Sequence[int]) -> Iterator[tuple[int, CanonicalMap]]:
        for k in which:
            yield k, self.carried(target, robot_frame(target.frame, target.robot[k]))


TREATMENTS = {"plain": PlainTreatment, "ring": RingTreatment, "robot": RobotFrameTreatment}


def own_halves(item: Item) -> tuple[Item, Item]:
    """The instance's grasps split into two alternating halves (the check's upper reference: its own demonstrations)."""
    parts = []
    for first in (0, 1):
        idx = range(first, len(item.grasps), 2)
        parts.append(replace(item, grasps=tuple(item.grasps[k] for k in idx), robot=tuple(item.robot[k] for k in idx)))
    return parts[0], parts[1]


def evaluate(kind: str, sources: Sequence[Item], targets: Sequence[Item], handle: bool) -> dict:
    """Per target instance and pooled: the check result of treatment `kind` (sources -> targets)."""
    model = TREATMENTS[kind](sources, handle)
    rows, pooled = {}, None
    for target in targets:
        out = model.outcomes(target, range(len(target.grasps)))
        rows[target.name] = summarise(out)
        pooled = out if pooled is None else pooled + out
    return {
        "per_instance": rows,
        "pooled": summarise(pooled),
        "grasps": len(pooled.shares),
        "source_grasps": sum(len(s.grasps) for s in sources),
    }


def evaluate_own(kind: str, targets: Sequence[Item], handle: bool) -> dict:
    """The instance's own demonstrations: the map of one half against the grasps of the other half, both ways."""
    rows, pooled = {}, None
    for target in targets:
        if len(target.grasps) < 4:
            continue
        a, b = own_halves(target)
        out = None
        for source, held in ((a, b), (b, a)):
            part = TREATMENTS[kind]([source], handle).outcomes(held, range(len(held.grasps)))
            out = part if out is None else out + part
        rows[target.name] = summarise(out)
        pooled = out if pooled is None else pooled + out
    return {"per_instance": rows, "pooled": summarise(pooled) if pooled else None}
