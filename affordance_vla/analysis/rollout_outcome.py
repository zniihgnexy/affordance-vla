"""Label what happened in one closed-loop rollout from simulator logs.

First version of the error types in Pipeline v2 (section 5.2). Only what the grasp flags and the final
goal check can decide is covered: wrong object, no grasp, not delivered. Approach-direction and
collision errors need contact geometry and are not labelled here.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SUCCESS = "success"
WRONG_OBJECT = "wrong object"  # the first object grasped is not the target
NO_GRASP = "no grasp"
NOT_DELIVERED = "not delivered"  # the target was grasped first but did not end at the goal
DEFAULT_MIN_GRASP_STEPS = 5  # consecutive steps of two-finger contact that count as a grasp (0.25 s at 20 Hz)


@dataclass(frozen=True)
class Outcome:
    label: str
    first_grasped: str | None
    first_grasp_step: int | None
    grasped: tuple[str, ...]  # every object that was grasped, in order of its first grasp
    delivered: bool  # the target satisfied the goal at the end


def first_grasp_steps(grasped: np.ndarray, min_steps: int = DEFAULT_MIN_GRASP_STEPS) -> list[int | None]:
    """Per object (column): the first step of a run of at least `min_steps` grasped steps, or None."""
    grasped = np.asarray(grasped, dtype=bool)
    starts: list[int | None] = []
    for column in grasped.T:
        run, found = 0, None
        for step, flag in enumerate(column):
            run = run + 1 if flag else 0
            if run >= min_steps:
                found = step - min_steps + 1
                break
        starts.append(found)
    return starts


def classify(names: Sequence[str], grasped: np.ndarray, target: str, delivered: bool,
             min_grasp_steps: int = DEFAULT_MIN_GRASP_STEPS) -> Outcome:
    """Outcome of a rollout. `grasped` is (T, N) with one column per name; `delivered` is the final goal check."""
    grasped = np.asarray(grasped, dtype=bool)
    if target not in names:
        raise ValueError(f"target {target!r} is not one of the logged objects {list(names)}")
    if grasped.ndim != 2 or grasped.shape[1] != len(names):
        raise ValueError(f"grasped has shape {grasped.shape}; it needs one of {len(names)} columns per object")
    starts = first_grasp_steps(grasped, min_grasp_steps)
    order = sorted((step, name) for step, name in zip(starts, names) if step is not None)
    if not order:
        return Outcome(label=NO_GRASP, first_grasped=None, first_grasp_step=None, grasped=(), delivered=delivered)
    first_step, first_name = order[0]
    if first_name != target:
        label = WRONG_OBJECT
    else:
        label = SUCCESS if delivered else NOT_DELIVERED
    return Outcome(label=label, first_grasped=first_name, first_grasp_step=first_step,
                   grasped=tuple(name for _, name in order), delivered=delivered)
