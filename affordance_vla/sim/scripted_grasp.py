"""A scripted pick-and-place that goes to the oracle contact points, step by step.

The grasp and release points are the ones the oracle plan uses (affordance_vla/plan/plan_input.py: offsets
measured on the training demonstrations), so the motion shows where a plan tells the gripper to go. The
gripper comes in from straight above: the demonstrated "approach" direction is measured over the steps
around the event and already contains the start of the lift, so it is not used here. This is a controller
for looking at the simulator, not a policy: no images, only simulator state.

Phases: above (`above_m` over the grasp point) -> descend -> close -> lift, and with `place`: carry (over the
release point) -> lower -> release -> retreat -> done. A phase ends when the gripper is within `tolerance_m`
of its target (closing and releasing last `close_steps`) or after `max_phase_steps`.
Actions are LIBERO's OSC_POSE deltas: (dx, dy, dz) in [-1, 1] with 1 = 5 cm, no rotation, gripper +1 closes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

STEP_M = 0.05  # OSC_POSE: an action of 1 asks for 5 cm
OPEN, CLOSE = -1.0, 1.0
TIMED = {"close", "release"}  # phases that last a fixed number of steps


@dataclass(frozen=True)
class GraspSettings:
    above_m: float = 0.10
    lift_m: float = 0.15
    tolerance_m: float = 0.008
    gain: float = 1.0
    close_steps: int = 15
    max_phase_steps: int = 120
    place: bool = True


@dataclass(frozen=True)
class Targets:
    """Where the gripper closes on the object and where it opens over the goal region (world, metres)."""

    grasp: np.ndarray
    release: np.ndarray


@dataclass(frozen=True)
class GraspState:
    phase: str = "above"
    steps: int = 0  # steps spent in this phase


def phase_order(place: bool) -> tuple[str, ...]:
    pick = ("above", "descend", "close", "lift")
    return pick + (("carry", "lower", "release", "retreat") if place else ()) + ("done",)


def phase_goal(phase: str, targets: Targets, settings: GraspSettings) -> tuple[np.ndarray, float]:
    """(position target, gripper command) of a phase."""
    up = np.array([0.0, 0.0, 1.0])
    goals = {
        "above": (targets.grasp + settings.above_m * up, OPEN),
        "descend": (targets.grasp, OPEN),
        "close": (targets.grasp, CLOSE),
        "lift": (targets.grasp + settings.lift_m * up, CLOSE),
        "carry": (targets.release + settings.above_m * up, CLOSE),
        "lower": (targets.release, CLOSE),
        "release": (targets.release, OPEN),
        "retreat": (targets.release + settings.above_m * up, OPEN),
    }
    return goals[phase]


def position_action(ee_pos: np.ndarray, target: np.ndarray, gain: float) -> np.ndarray:
    """(3,) OSC_POSE translation that moves the gripper towards `target`, clipped to [-1, 1]."""
    return np.clip(gain * (np.asarray(target, dtype=float) - np.asarray(ee_pos, dtype=float)) / STEP_M, -1.0, 1.0)


def next_action(state: GraspState, ee_pos: np.ndarray, targets: Targets,
                settings: GraspSettings) -> tuple[np.ndarray, GraspState]:
    """The action for this step and the state for the next one."""
    order = phase_order(settings.place)
    if state.phase == "done":  # stay put, the gripper as the last phase left it
        return np.r_[np.zeros(6), phase_goal(order[-2], targets, settings)[1]], state
    target, gripper = phase_goal(state.phase, targets, settings)
    action = np.r_[position_action(ee_pos, target, settings.gain), np.zeros(3), gripper]
    reached = (state.steps + 1 >= settings.close_steps if state.phase in TIMED
               else float(np.linalg.norm(np.asarray(target) - ee_pos)) < settings.tolerance_m)
    if reached or state.steps + 1 >= settings.max_phase_steps:
        return action, GraspState(order[order.index(state.phase) + 1], 0)
    return action, replace(state, steps=state.steps + 1)
