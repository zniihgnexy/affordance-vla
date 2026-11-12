"""What a plan predictor is trained to output at one step of a demonstration.

The target is the plan the oracle would give in that state: the grasp point of the instructed object
while it is not held, the release point over the container while it is, nothing after the release.
Before the grasp the same frame can also be paired with an instruction that names another object of
the scene; its target is that object's grasp point. No extra demonstrations are needed for this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance3d.plan_consistency import plan_ahead
from affordance3d.plan_input import ContactOffsets
from affordance3d.plan_samples import DemoTrack

NONE, GRASP, RELEASE = 0, 1, 2  # classes of the next gripper event


@dataclass(frozen=True)
class PlanTarget:
    event: int  # NONE, GRASP or RELEASE
    contact: np.ndarray  # (3,) where the gripper should be at the event (world frame); zeros for NONE
    approach: np.ndarray  # (3,) unit direction the gripper should arrive from; zeros for NONE


def oracle_target(track: DemoTrack, step: int, offsets: ContactOffsets, container: str,
                  name: str | None = None) -> PlanTarget:
    """The oracle plan at `step` of a demonstration, for its own target or for another object `name`."""
    ahead = plan_ahead(track.labels, track.ee_pos[step], step)
    if name is not None and name != track.target and (ahead is None or not ahead.closing):
        raise ValueError(f"step {step}: another object can only be asked for before the grasp of the demonstration")
    if ahead is None:
        return PlanTarget(event=NONE, contact=np.zeros(3), approach=np.zeros(3))
    if not ahead.closing:
        position = track.object_pos[step, track.names.index(container)]
        return PlanTarget(event=RELEASE, contact=position + offsets.release,
                          approach=np.asarray(offsets.release_approach, dtype=float))
    asked = track.target if name is None else name
    if asked not in offsets.grasp:
        raise KeyError(f"no grasp offset for {asked!r}; known objects: {sorted(offsets.grasp)}")
    position = track.object_pos[step, track.names.index(asked)]
    return PlanTarget(event=GRASP, contact=position + offsets.grasp[asked],
                      approach=np.asarray(offsets.grasp_approach[asked], dtype=float))
