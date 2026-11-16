"""Episode affordance state (EAS): one slot per object, carried through a single episode.

First, rule-based version (Pipeline v2, section 3.1). A slot stores where an object is believed to
be, its status and how long ago it was seen. A seen object is refreshed; an object that is not seen
keeps its last position and is marked occluded; a grasped object follows the end effector. Nothing
is kept after the episode ends. The functions are pure: `update` returns a new state.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

MOVED_M = 0.03  # a belief this far from where the object was first seen counts as "moved"


class Status(str, Enum):
    UNSEEN = "unseen"  # not observed yet in this episode
    VISIBLE = "visible"
    OCCLUDED = "occluded"  # seen before, not visible now, not in the gripper
    GRASPED = "grasped"


@dataclass(frozen=True)
class Slot:
    name: str
    status: Status = Status.UNSEEN
    position: np.ndarray | None = None  # believed position (3,); None until the object is first seen
    first_position: np.ndarray | None = None  # where the object was first seen
    steps_since_seen: int = 0
    grasp_offset: np.ndarray | None = None  # believed position − end-effector position, while grasped

    @property
    def moved(self) -> bool:
        if self.position is None or self.first_position is None:
            return False
        return bool(np.linalg.norm(self.position - self.first_position) > MOVED_M)


@dataclass(frozen=True)
class StepObservation:
    """What one step gives the state: the end effector, the objects seen now, and the objects held."""

    ee_pos: np.ndarray
    seen: Mapping[str, np.ndarray]  # object name → observed position, only for objects visible at this step
    grasped: frozenset[str] = field(default_factory=frozenset)


def initial_state(names: Sequence[str]) -> tuple[Slot, ...]:
    return tuple(Slot(name=name) for name in names)


def _update_slot(slot: Slot, observation: StepObservation) -> Slot:
    ee_pos = np.asarray(observation.ee_pos, dtype=float)
    seen = observation.seen.get(slot.name)
    held = slot.name in observation.grasped
    if seen is not None:
        position, age = np.asarray(seen, dtype=float).copy(), 0
    elif held and slot.position is not None:
        offset = slot.grasp_offset if slot.grasp_offset is not None else slot.position - ee_pos
        position, age = ee_pos + offset, slot.steps_since_seen + 1
    else:
        position, age = slot.position, slot.steps_since_seen + 1
    if held:
        status = Status.GRASPED
    elif seen is not None:
        status = Status.VISIBLE
    else:
        status = Status.OCCLUDED if position is not None else Status.UNSEEN
    return Slot(name=slot.name, status=status, position=position,
                first_position=slot.first_position if slot.first_position is not None else position,
                steps_since_seen=age, grasp_offset=position - ee_pos if held and position is not None else None)


def update(state: Sequence[Slot], observation: StepObservation) -> tuple[Slot, ...]:
    """The state after one step; the input state is left unchanged."""
    return tuple(_update_slot(slot, observation) for slot in state)


def run_episode(names: Sequence[str], observations: Sequence[StepObservation]) -> list[tuple[Slot, ...]]:
    """The state after each step of one episode, starting from empty slots."""
    history: list[tuple[Slot, ...]] = []
    state = initial_state(names)
    for observation in observations:
        state = update(state, observation)
        history.append(state)
    return history


def observations_from_log(names: Sequence[str], ee_pos: np.ndarray, object_pos: np.ndarray, visible_px: np.ndarray,
                          grasped: np.ndarray, min_pixels: int) -> list[StepObservation]:
    """Per-step observations from logged arrays: an object counts as seen when it covers `min_pixels` pixels.

    ee_pos (T, 3), object_pos (T, N, 3), visible_px (T, N), grasped (T, N) bool; N = len(names).
    """
    ee_pos, object_pos = np.asarray(ee_pos, dtype=float), np.asarray(object_pos, dtype=float)
    visible_px, grasped = np.asarray(visible_px), np.asarray(grasped, dtype=bool)
    steps = len(ee_pos)
    expected = {"object_pos": (steps, len(names), 3), "visible_px": (steps, len(names)),
                "grasped": (steps, len(names))}
    for key, array in (("object_pos", object_pos), ("visible_px", visible_px), ("grasped", grasped)):
        if array.shape != expected[key]:
            raise ValueError(f"{key} has shape {array.shape}, expected {expected[key]} "
                             f"({steps} steps, {len(names)} objects)")
    return [StepObservation(ee_pos=ee_pos[t],
                            seen={name: object_pos[t, n] for n, name in enumerate(names)
                                  if visible_px[t, n] >= min_pixels},
                            grasped=frozenset(name for n, name in enumerate(names) if grasped[t, n]))
            for t in range(steps)]
