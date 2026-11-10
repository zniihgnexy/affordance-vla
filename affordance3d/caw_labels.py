"""Contact–approach–waypoint (CAW) labels from a demonstration's end-effector path.

A CAW label summarises how the end effector reaches one gripper event (a grasp or a release):
the position where the gripper command changes (a proxy for the contact point), the unit
direction of motion just before it, and a few waypoints along the path since the previous event.
Events come either from the gripper command (`gripper_events`: a proxy that needs proprioception
only) or from simulator contact flags of a replayed episode (`contact_events`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_WAYPOINTS = 4
DEFAULT_APPROACH_STEPS = 10
DEFAULT_MIN_CONTACT_STEPS = 5  # two-finger contact must last this long to count as a grasp (0.25 s at 20 Hz)
MIN_MOTION_M = 1e-6  # below this the end effector is treated as stationary


@dataclass(frozen=True)
class GripperEvent:
    step: int  # first step carrying the new gripper command
    closing: bool  # True: open → close (grasp); False: close → open (release)


@dataclass(frozen=True)
class CawLabel:
    event: GripperEvent
    contact: np.ndarray  # (3,) end-effector position at the event step
    approach: np.ndarray  # (3,) unit motion direction before the event; zeros if it did not move
    waypoints: np.ndarray  # (K, 3) evenly spaced in time from the segment start, ending at `contact`
    start: int  # first step of the segment: the previous event's step, or 0
    waypoint_steps: np.ndarray  # (K,) step of each waypoint; the last one is the event step


@dataclass(frozen=True)
class ActionPathFit:
    scale: float  # metres of end-effector motion per unit of translation action
    r2: float  # 1 − SSE/SST of the displacement; can be negative because the fit passes through the origin
    predicted: np.ndarray  # (T, 3) path predicted from the integrated actions


def gripper_events(command: np.ndarray) -> list[GripperEvent]:
    """Steps where the sign of the gripper command flips; zero commands keep the previous sign."""
    signs = np.sign(np.asarray(command, dtype=float))
    events: list[GripperEvent] = []
    previous = 0.0
    for step, sign in enumerate(signs):
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            events.append(GripperEvent(step=step, closing=bool(sign > 0)))
        previous = sign
    return events


def contact_events(grasped: np.ndarray, min_steps: int = DEFAULT_MIN_CONTACT_STEPS) -> list[GripperEvent]:
    """Grasp and release events from per-step grasp flags (simulator contact), ignoring short touches.

    A grasp starts at the first step of a run of at least `min_steps` grasped steps and the release
    is the first step after that run. A run that lasts until the end has no release.
    """
    grasped = np.asarray(grasped, dtype=bool)
    events: list[GripperEvent] = []
    start = None
    for step, flag in enumerate(np.append(grasped, False)):
        if flag and start is None:
            start = step
        elif not flag and start is not None:
            if step - start >= min_steps:
                events.append(GripperEvent(step=start, closing=True))
                if step < len(grasped):
                    events.append(GripperEvent(step=step, closing=False))
            start = None
    return events


def _approach(ee_pos: np.ndarray, start: int, step: int, approach_steps: int) -> np.ndarray:
    motion = ee_pos[step] - ee_pos[max(start, step - approach_steps)]
    norm = float(np.linalg.norm(motion))
    return motion / norm if norm > MIN_MOTION_M else np.zeros(3)


def caw_labels(ee_pos: np.ndarray, events: list[GripperEvent], n_waypoints: int = DEFAULT_WAYPOINTS,
               approach_steps: int = DEFAULT_APPROACH_STEPS) -> list[CawLabel]:
    """One CAW label per gripper event; each segment runs from the previous event (or step 0)."""
    ee_pos = np.asarray(ee_pos, dtype=float)
    if any(later.step < earlier.step for earlier, later in zip(events, events[1:])):
        raise ValueError("gripper events must be sorted by step")
    labels: list[CawLabel] = []
    start = 0
    for event in events:
        if not 0 <= event.step < len(ee_pos):
            raise ValueError(f"event step {event.step} is outside the path of {len(ee_pos)} steps")
        steps = np.rint(np.linspace(start, event.step, n_waypoints + 1)[1:]).astype(int)
        labels.append(CawLabel(event=event, contact=ee_pos[event.step].copy(),
                               approach=_approach(ee_pos, start, event.step, approach_steps),
                               waypoints=ee_pos[steps].copy(), start=start, waypoint_steps=steps))
        start = event.step
    return labels


def _paired_xyz(actions_xyz: np.ndarray, ee_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actions_xyz, ee_pos = np.asarray(actions_xyz, dtype=float), np.asarray(ee_pos, dtype=float)
    for name, array in (("actions_xyz", actions_xyz), ("ee_pos", ee_pos)):
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"{name} must be (T, 3), got {array.shape}; pass actions[:, :3] for LIBERO actions")
    if len(actions_xyz) != len(ee_pos):
        raise ValueError(f"actions and positions need the same number of steps "
                         f"({len(actions_xyz)} vs {len(ee_pos)})")
    if len(ee_pos) == 0:
        raise ValueError("actions and positions are empty")
    return actions_xyz, ee_pos


def fit_action_path(actions_xyz: np.ndarray, ee_pos: np.ndarray) -> ActionPathFit:
    """Least-squares scale s such that ee_pos[t] - ee_pos[0] ≈ s · Σ_{i<t} actions[i]."""
    actions_xyz, ee_pos = _paired_xyz(actions_xyz, ee_pos)
    integrated = np.vstack([np.zeros(3), np.cumsum(actions_xyz, axis=0)[:-1]])
    displacement = ee_pos - ee_pos[0]
    scale = float(np.sum(integrated * displacement) / max(np.sum(integrated ** 2), 1e-12))
    residual = displacement - scale * integrated
    total = np.sum((displacement - displacement.mean(axis=0)) ** 2)
    r2 = 1.0 - float(np.sum(residual ** 2) / total) if total > 0 else 1.0
    return ActionPathFit(scale=scale, r2=r2, predicted=ee_pos[0] + scale * integrated)


def chunk_endpoint_errors(actions_xyz: np.ndarray, ee_pos: np.ndarray, scale: float, horizon: int,
                          stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Per action chunk: |s·Σ actions − recorded displacement| and the recorded travel (both in metres).

    Chunk k covers steps [t, t + horizon) with t = k·stride, so it measures how well summed actions
    predict where the end effector is `horizon` steps later — the quantity a chunk-level
    plan–action consistency loss would compare.
    """
    actions_xyz, ee_pos = _paired_xyz(actions_xyz, ee_pos)
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if not 1 <= horizon < len(ee_pos):
        raise ValueError(f"horizon {horizon} must be between 1 and the path length ({len(ee_pos)} steps) − 1")
    starts = np.arange(0, len(ee_pos) - horizon, stride)
    predicted = np.array([scale * actions_xyz[t:t + horizon].sum(axis=0) for t in starts])
    recorded = ee_pos[starts + horizon] - ee_pos[starts]
    return np.linalg.norm(predicted - recorded, axis=1), np.linalg.norm(recorded, axis=1)
