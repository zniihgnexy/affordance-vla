"""Keep a predicted plan for a whole phase of the task instead of predicting it at every query.

A plan names the contact of the next gripper event in world coordinates. Until that event the contact
stands still: nothing is held before a grasp, and the container does not move before a release. So the
plan can be predicted once, when a phase begins, and kept; only the plan input changes, because it is
relative to the gripper. The phase ends when the gripper has carried out the event: its command was
"close" after a grasp plan, or "open" after a release plan, for a few steps in a row. The next query then
predicts a new plan. Only numpy is used, so the simulator environment can import this.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from affordance3d.plan_input import EVENT_COLUMN, plan_contact, plan_features

EVENT_STEPS = 5  # steps in a row with the event's gripper command after which the event counts as carried out
APPROACH_COLUMNS = slice(6, 9)  # of the plan input


@dataclass(frozen=True, eq=False)  # holds arrays: compared by identity
class HeldPlan:
    """A plan in world coordinates, which stay valid while the gripper moves."""

    contact: np.ndarray  # (3,) metres
    approach: np.ndarray  # (3,) unit direction of the gripper's motion before the event
    closing: bool  # True: the event is a grasp; False: a release


@dataclass(frozen=True)
class Hold:
    """What is kept between queries."""

    plan: HeldPlan | None = None
    event_steps: int = 0  # steps in a row in which the gripper command carried out the plan's event


def held_plan(features: np.ndarray, ee_pos: np.ndarray) -> HeldPlan | None:
    """The plan behind a plan input that was made for gripper position `ee_pos`; None for an empty plan."""
    contact = plan_contact(features, ee_pos)
    if contact is None:
        return None
    features = np.asarray(features, dtype=float).reshape(-1)
    return HeldPlan(contact, features[APPROACH_COLUMNS].copy(), closing=bool(features[EVENT_COLUMN] > 0))


def plan_input_of(plan: HeldPlan, ee_pos: np.ndarray) -> np.ndarray:
    """The plan input of a kept plan for where the gripper is now."""
    return plan_features(ee_pos, plan.contact, plan.approach, plan.closing)


def after_step(hold: Hold, gripper_command: float) -> Hold:
    """The hold after one executed step with the simulator's gripper command (+1 close, -1 open)."""
    if hold.plan is None:
        return hold
    carried_out = (gripper_command > 0) == hold.plan.closing
    steps = hold.event_steps + 1 if carried_out else 0
    return Hold() if steps >= EVENT_STEPS else replace(hold, event_steps=steps)


def plan_to_send(hold: Hold, ee_pos: np.ndarray) -> np.ndarray | None:
    """The plan input a query should carry: the kept plan, or None when a new plan must be predicted."""
    return None if hold.plan is None else plan_input_of(hold.plan, ee_pos)


def after_query(hold: Hold, predicted: np.ndarray | None, ee_pos: np.ndarray) -> Hold:
    """The hold after a query made from `ee_pos`: a plan input the predictor just returned is kept from now on."""
    return hold if predicted is None else Hold(held_plan(predicted, ee_pos))
