"""The leave-one-out check of a transferred map (Pipeline v3, section 3): do the held-out mesh's own grasps agree
with the map carried to it?

Per held-out grasp, on the held-out mesh's surface, with the grasp's contact weights (`contact_weights`):
- landing: the share of the grasp's contact weight that falls in the high-score region of the map (score in the
  top `HIGH_SHARE` of the surface, and above 0); the grasp lands when that share is at least `LAND_SHARE`;
- closing-axis error: the angle (a line, so at most 90 degrees) between the grasp's closing axis and the map's
  axis at the point of the grasp's contact set (weight at least `CONTACT_FRACTION` of the grasp's largest) where
  the map scores highest; 90 when the grasp touches no map point or nothing at all.
The pass line of Pipeline v3 is a landing rate of at least 0.8 and a median closing-axis error below 20 degrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from affordance_vla.plan.interaction_labels import CanonicalMap, Grasp, contact_weights, line_angle_deg
from affordance_vla.sim.object_surface import Surface

HIGH_SHARE = 0.2  # the high-score region: this share of the surface, the best-scoring points
LAND_SHARE = 0.5  # a grasp lands when at least this share of its contact weight is in the region
CONTACT_FRACTION = 0.5  # the contact set of a grasp: points with at least this fraction of its largest weight
PASS_LANDING = 0.8
PASS_AXIS_DEG = 20.0
NO_MATCH_DEG = 90.0


@dataclass(frozen=True)
class Outcomes:
    """Per held-out grasp: the share of its contact weight in the high-score region, and its closing-axis error."""

    shares: np.ndarray
    errors: np.ndarray
    untouched: int  # grasps with no contact on the surface at all
    no_axis: int  # grasps with contact whose contact set holds no point the map scores

    def __add__(self, other: Outcomes) -> Outcomes:
        return Outcomes(np.concatenate([self.shares, other.shares]), np.concatenate([self.errors, other.errors]),
                        self.untouched + other.untouched, self.no_axis + other.no_axis)


@dataclass(frozen=True)
class CheckResult:
    grasps: int
    landing_rate: float  # share of grasps that land in the high-score region
    mean_region_weight: float  # mean over grasps of the share of contact weight in the region
    axis_error_median_deg: float
    untouched: int
    no_axis: int

    @property
    def landing_pass(self) -> bool:
        return self.landing_rate >= PASS_LANDING

    @property
    def axis_pass(self) -> bool:
        return self.axis_error_median_deg < PASS_AXIS_DEG

    @property
    def passed(self) -> bool:
        return self.landing_pass and self.axis_pass


def high_region(cmap: CanonicalMap, share: float = HIGH_SHARE) -> np.ndarray:
    """Boolean mask of the best-scoring `share` of the surface points; a point that scores 0 is never in it."""
    if not (cmap.score > 0).any():
        return np.zeros(len(cmap.score), dtype=bool)
    return (cmap.score > 0) & (cmap.score >= np.quantile(cmap.score, 1 - share))


def grasp_outcomes(cmap: CanonicalMap, surface: Surface, grasps: Sequence[Grasp]) -> Outcomes:
    """Landing share and closing-axis error of every held-out grasp against a map carried to their mesh."""
    if not grasps:
        raise ValueError("no held-out grasps")
    region = high_region(cmap)
    shares, errors, untouched, no_axis = [], [], 0, 0
    for grasp in grasps:
        weights = contact_weights(surface, grasp)
        if weights.max() <= 0:
            untouched += 1
            shares.append(0.0)
            errors.append(NO_MATCH_DEG)
            continue
        shares.append(float(weights[region].sum() / weights.sum()))
        contact = np.flatnonzero((weights >= CONTACT_FRACTION * weights.max()) & (cmap.score > 0))
        if contact.size == 0:
            no_axis += 1
            errors.append(NO_MATCH_DEG)
            continue
        errors.append(line_angle_deg(grasp.closing, cmap.axis[contact[np.argmax(cmap.score[contact])]]))
    return Outcomes(np.array(shares), np.array(errors), untouched, no_axis)


def summarise(outcomes: Outcomes) -> CheckResult:
    return CheckResult(len(outcomes.shares), float(np.mean(outcomes.shares >= LAND_SHARE)),
                       float(outcomes.shares.mean()), float(np.median(outcomes.errors)), outcomes.untouched,
                       outcomes.no_axis)


def check_transfer(cmap: CanonicalMap, surface: Surface, grasps: Sequence[Grasp]) -> CheckResult:
    """Landing rate and closing-axis error of the held-out grasps against a map carried to their mesh."""
    return summarise(grasp_outcomes(cmap, surface, grasps))
