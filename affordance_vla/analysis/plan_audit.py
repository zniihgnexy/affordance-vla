"""Where the plans given to a rollout pointed.

Each plan input is turned back into the contact it names (affordance_vla/plan/plan_input.py) and compared with the
oracle grasp points of the scene's objects: the object's position plus the average grasp offset of the
demonstrations. Only grasp plans made before anything is held are judged; a release plan or an empty plan
made that early points at no object. Only numpy is used.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from affordance_vla.analysis.rollout_outcome import DEFAULT_MIN_GRASP_STEPS, first_grasp_steps
from affordance_vla.plan.plan_input import EVENT_COLUMN, plan_contact

NO_PLAN, RELEASE_PLAN, ELSEWHERE = "no plan", "release plan", "elsewhere"
NOT_AN_OBJECT = (NO_PLAN, RELEASE_PLAN, ELSEWHERE)
POINTING_RANGE_M = 0.10  # a grasp plan farther than this from every grasp point points at no object


@dataclass(frozen=True)
class PlanPoint:
    """What one plan pointed at."""

    pointed: str  # the object whose grasp point is nearest to the plan's contact, or one of NOT_AN_OBJECT
    error_m: float  # distance from the plan's contact to the named object's grasp point; nan unless a grasp plan


def plan_point(plan: np.ndarray, ee_pos: np.ndarray, grasp_points: Mapping[str, np.ndarray], target: str) -> PlanPoint:
    """Judge one plan input made for gripper position `ee_pos` against the grasp points of the scene."""
    if target not in grasp_points:
        raise KeyError(f"no grasp point for the named object {target!r}; known: {sorted(grasp_points)}")
    contact = plan_contact(plan, ee_pos)
    if contact is None:
        return PlanPoint(NO_PLAN, float("nan"))
    if np.asarray(plan, dtype=float).reshape(-1)[EVENT_COLUMN] < 0:
        return PlanPoint(RELEASE_PLAN, float("nan"))
    gaps = {name: float(np.linalg.norm(np.asarray(point, dtype=float) - contact))
            for name, point in grasp_points.items()}
    nearest = min(gaps, key=gaps.get)
    return PlanPoint(nearest if gaps[nearest] <= POINTING_RANGE_M else ELSEWHERE, gaps[target])


def plans_before_grasp(plans: np.ndarray, chunk_steps: Sequence[int], ee_pos: np.ndarray, object_pos: np.ndarray,
                       names: Sequence[str], grasp_offsets: Mapping[str, np.ndarray], target: str,
                       grasped: np.ndarray) -> tuple[PlanPoint, ...]:
    """One PlanPoint per query made before the first grasp of any object (all queries if nothing is ever held).

    `plans` (C, PLAN_DIM) are the plan inputs of the queries at `chunk_steps`; `ee_pos` (T + 1, 3),
    `object_pos` (T + 1, N, 3) and `grasped` (T + 1, N) are per step, with one column per name. Objects
    without a grasp offset (the container) are not candidates.
    """
    if len(plans) != len(chunk_steps):
        raise ValueError(f"{len(plans)} plans for {len(chunk_steps)} queries")
    starts = [step for step in first_grasp_steps(np.asarray(grasped, dtype=bool), DEFAULT_MIN_GRASP_STEPS)
              if step is not None]
    until = min(starts) if starts else len(ee_pos)
    known = [(k, name) for k, name in enumerate(names) if name in grasp_offsets]
    points = []
    for plan, step in zip(plans, chunk_steps):
        if step >= until:
            break
        grasp_points = {name: object_pos[step, k] + np.asarray(grasp_offsets[name], dtype=float) for k, name in known}
        points.append(plan_point(plan, ee_pos[step], grasp_points, target))
    return tuple(points)
