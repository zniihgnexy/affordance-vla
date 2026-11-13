"""The plan as the action head reads it: a short vector about the next gripper event.

The vector holds where the gripper should be when it next closes or opens (relative to where it is
now), the direction it should arrive from, and whether the event is a grasp or a release. In
training it is built from a demonstration's CAW label; `oracle_plan` builds it from simulator
ground truth during a rollout. Only numpy is used, so the simulator environment can import this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

PLAN_DIM = 10
FAR_SCALE_M = 0.10  # coarse part: 10 cm counts as 1
NEAR_SCALE_M = 0.02  # fine part: tanh(reach / 2 cm) resolves the last centimetres and saturates beyond
GRASP_CODE, RELEASE_CODE = 1.0, -1.0
EVENT_COLUMN = PLAN_DIM - 1  # the event code: +1 grasp, -1 release, 0 when there is no plan


@dataclass(frozen=True)
class ContactOffsets:
    """Where demonstrations put the gripper at an event, relative to an object's position."""

    grasp: dict[str, np.ndarray]  # object name -> gripper position at the grasp minus the object's position
    grasp_approach: dict[str, np.ndarray]  # object name -> unit direction of the gripper's motion before the grasp
    release: np.ndarray  # gripper position at the release minus the container's position
    release_approach: np.ndarray  # unit direction of the gripper's motion before the release


def plan_features(ee_pos: np.ndarray, contact: np.ndarray, approach: np.ndarray, closing: bool) -> np.ndarray:
    """(PLAN_DIM,) reach / 10 cm, tanh(reach / 2 cm), approach direction, event code (+1 grasp, -1 release)."""
    reach = np.asarray(contact, dtype=float) - np.asarray(ee_pos, dtype=float)
    approach = np.asarray(approach, dtype=float)
    if reach.shape != (3,) or approach.shape != (3,):
        raise ValueError(f"positions and the approach direction must be (3,), got {reach.shape} and {approach.shape}")
    code = GRASP_CODE if closing else RELEASE_CODE
    return np.concatenate([reach / FAR_SCALE_M, np.tanh(reach / NEAR_SCALE_M), approach, [code]]).astype(np.float32)


def no_plan_features() -> np.ndarray:
    """The input when no gripper event is ahead (after the last release)."""
    return np.zeros(PLAN_DIM, dtype=np.float32)


def plan_contact(features: np.ndarray, ee_pos: np.ndarray) -> np.ndarray | None:
    """The contact a plan input points at from the gripper position it was built for; None when it is no plan.

    The inverse of plan_features: the coarse part of the vector is the reach in units of FAR_SCALE_M.
    """
    features = np.asarray(features, dtype=float).reshape(-1)
    if features.shape != (PLAN_DIM,):
        raise ValueError(f"a plan input has {PLAN_DIM} numbers, got {features.shape[0]}")
    if features[EVENT_COLUMN] == 0.0:
        return None
    return np.asarray(ee_pos, dtype=float) + FAR_SCALE_M * features[:3]


def oracle_plan(offsets: ContactOffsets, ee_pos: np.ndarray, target_pos: np.ndarray, container_pos: np.ndarray,
                target: str, grasped: bool, delivered: bool) -> np.ndarray:
    """Plan input from simulator ground truth: grasp the target, then release it over the container.

    While the target is not held the plan is its grasp; while it is held, the release; once it lies
    in the container and is no longer held there is no plan.
    """
    if delivered and not grasped:
        return no_plan_features()
    if grasped:
        return plan_features(ee_pos, np.asarray(container_pos, dtype=float) + offsets.release,
                             offsets.release_approach, closing=False)
    if target not in offsets.grasp:
        raise KeyError(f"no grasp offset for {target!r}; known objects: {sorted(offsets.grasp)}")
    return plan_features(ee_pos, np.asarray(target_pos, dtype=float) + offsets.grasp[target],
                         offsets.grasp_approach[target], closing=True)


def offsets_to_json(offsets: ContactOffsets) -> str:
    return json.dumps({
        "grasp": {name: value.tolist() for name, value in offsets.grasp.items()},
        "grasp_approach": {name: value.tolist() for name, value in offsets.grasp_approach.items()},
        "release": offsets.release.tolist(), "release_approach": offsets.release_approach.tolist()}, indent=2)


def offsets_from_json(text: str) -> ContactOffsets:
    data = json.loads(text)
    return ContactOffsets(
        grasp={name: np.asarray(value, dtype=float) for name, value in data["grasp"].items()},
        grasp_approach={name: np.asarray(value, dtype=float) for name, value in data["grasp_approach"].items()},
        release=np.asarray(data["release"], dtype=float),
        release_approach=np.asarray(data["release_approach"], dtype=float))
