"""How far a closed-loop rollout that was given a plan got, and how jerky its commands were.

A rollout can fail although the policy followed its plan: the gripper reaches the planned grasp point
and then knocks the object over. The stages separate "went where the plan said" from "completed the
task"; the roughness measures how much the translation command jumps from one step to the next.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance_vla.analysis.rollout_outcome import DEFAULT_MIN_GRASP_STEPS, first_grasp_steps

REACH_TOLERANCE_M = 0.02  # about twice the spread of the demonstrated grasp point relative to the object
TOPPLE_DROP_M = 0.03  # a standing bottle or carton that falls over ends more than this below its start height


@dataclass(frozen=True)
class PlanStages:
    reached: bool  # at some moment the gripper was within the tolerance of the planned grasp contact
    held: bool  # the target was held: two-finger contact for the minimum number of consecutive steps
    toppled: bool  # the target sank below its start height before it was first held
    delivered: bool  # the target satisfied the goal at the end (whatever was grasped first)
    holding_at_end: bool  # the target was still in the gripper through the last steps


def reach_along_rollout(ee_pos: np.ndarray, query_steps: np.ndarray,
                        to_contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the gripper to the planned contact at every step, and the query in force at that step.

    `ee_pos` is (T + 1, 3); `query_steps` (C,) are the steps at which the policy was queried and `to_contact`
    (C, 3) the vector in metres from the gripper to the planned contact at each query. Between two queries
    the contact of the earlier one is kept. Returns `reach` (T + 1,), infinite before the first query, and
    `query` (T + 1,), the index of the query in force (-1 before the first).
    """
    ee_pos, to_contact = np.asarray(ee_pos, dtype=float), np.asarray(to_contact, dtype=float)
    query_steps = np.asarray(query_steps, dtype=int)
    if to_contact.shape != (len(query_steps), 3) or ee_pos.ndim != 2 or ee_pos.shape[1] != 3:
        raise ValueError(f"need ee_pos (T + 1, 3) and one (3,) vector per query, got {ee_pos.shape}, "
                         f"{to_contact.shape} for {len(query_steps)} query steps")
    if len(query_steps) and (np.diff(query_steps) <= 0).any():
        raise ValueError("query steps must increase")
    if len(query_steps) and not 0 <= query_steps[0] <= query_steps[-1] < len(ee_pos):
        raise ValueError(f"query steps must lie inside the {len(ee_pos)} logged steps")
    query = np.searchsorted(query_steps, np.arange(len(ee_pos)), side="right") - 1
    reach = np.full(len(ee_pos), np.inf)
    known = query >= 0
    contacts = ee_pos[query_steps] + to_contact
    reach[known] = np.linalg.norm(contacts[query[known]] - ee_pos[known], axis=1)
    return reach, query


def plan_stages(reach: np.ndarray, closing: np.ndarray, grasped: np.ndarray, height: np.ndarray, delivered: bool,
                reach_tolerance: float = REACH_TOLERANCE_M, topple_drop: float = TOPPLE_DROP_M,
                min_grasp_steps: int = DEFAULT_MIN_GRASP_STEPS) -> PlanStages:
    """Stages of one rollout.

    `reach` (M,) is the distance in metres from the gripper to the planned contact at M moments (every
    query, or every step: see reach_along_rollout) and `closing` (M,) says whether the plan in force at
    that moment was a grasp; `grasped` and `height` (T + 1,) are the target's two-finger contact flag and
    z coordinate at every step.
    """
    reach, closing = np.asarray(reach, dtype=float), np.asarray(closing, dtype=bool)
    grasped, height = np.asarray(grasped, dtype=bool), np.asarray(height, dtype=float)
    if reach.shape != closing.shape or reach.ndim != 1:
        raise ValueError(f"reach {reach.shape} and closing {closing.shape} must be 1-D: "
                         "one entry for each of the moments (queries or steps)")
    if grasped.shape != height.shape or grasped.ndim != 1 or len(grasped) == 0:
        raise ValueError(f"grasped {grasped.shape} and height {height.shape} must be 1-D: "
                         "one entry for each of the steps")
    first_held = first_grasp_steps(grasped[:, None], min_grasp_steps)[0]
    before_held = height if first_held is None else height[:first_held]
    toppled = bool(len(before_held)) and bool(before_held.min() < height[0] - topple_drop)
    return PlanStages(reached=bool((reach[closing] <= reach_tolerance).any()), held=first_held is not None,
                      toppled=toppled, delivered=bool(delivered),
                      holding_at_end=len(grasped) >= min_grasp_steps and bool(grasped[-min_grasp_steps:].all()))


def command_roughness(chunks: np.ndarray) -> float:
    """Mean absolute change of the translation command between consecutive steps of a chunk (action units).

    `chunks` is (C, H, >=3) with the translation in the first three columns. Demonstrated commands change
    slowly within 8 steps, so a large value means the policy jumps back and forth around its path.
    """
    chunks = np.asarray(chunks, dtype=float)
    if chunks.ndim != 3 or chunks.shape[0] < 1 or chunks.shape[1] < 2 or chunks.shape[2] < 3:
        raise ValueError(f"chunks must be (C >= 1, H >= 2, >= 3), got {chunks.shape}")
    return float(np.abs(np.diff(chunks[:, :, :3], axis=1)).mean())
