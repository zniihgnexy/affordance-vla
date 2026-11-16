"""Demonstration tracks shared by the plan-sample tests (tests/training/test_plan_samples*.py)."""
import numpy as np

from affordance_vla.training.plan_samples import (
    track_from_arrays,
)

SCALE = 0.0125  # metres of gripper motion per unit of translation action in these tests
HORIZON = 8
GRASP, RELEASE, STEPS = 40, 70, 90
BOTTLE, BASKET = np.array([0.2, 0.0, 0.0]), np.array([0.2, 0.3, 0.0])


def make_track(target: str = "bottle_1", lift: float = 0.05):
    """Down onto the bottle (grasp at step 40), over to the basket (release at step 70), then back up."""
    start, grasp, drop = np.array([0.0, 0.0, 0.3]), BOTTLE + [0.0, 0.0, lift], BASKET + [0.0, 0.0, 0.15]
    ee = np.vstack([np.linspace(start, grasp, GRASP + 1), np.linspace(grasp, drop, RELEASE - GRASP + 1)[1:],
                    np.linspace(drop, drop + [0.0, 0.0, 0.1], STEPS - RELEASE)[1:]])
    actions = np.zeros((STEPS - 1, 7))
    actions[:, :3] = np.diff(ee, axis=0) / SCALE
    actions[:, 3] = 0.1
    actions[:, 6] = -1.0  # simulator convention: -1 open, +1 close
    actions[GRASP - 4:RELEASE - 2, 6] = 1.0  # the command flips a few steps before the contacts change
    grasped = np.zeros(STEPS, dtype=bool)
    grasped[GRASP:RELEASE] = True
    object_pos = np.tile(np.stack([BASKET, BOTTLE, np.array([0.1, -0.2, 0.0])]), (STEPS, 1, 1))
    return track_from_arrays(ee_pos=ee, actions=actions, grasped=grasped, object_pos=object_pos,
                             names=("basket_1", "bottle_1", "can_1"), target=target,
                             instruction="pick up the bottle and place it in the basket")


def make_lagging_track():
    """As make_track, but the commands are 1.25 times the motion they produce (a controller that lags)."""
    track = make_track()
    actions = track.actions.copy()
    actions[:, :3] *= 1.25
    return track_from_arrays(track.ee_pos, actions, track.grasped, track.object_pos, track.names, track.target,
                             track.instruction)


def track_with_commands(closed: list[tuple[int, int]]):
    """make_track with the gripper commanded closed over the given step ranges and open elsewhere."""
    track = make_track()
    actions = track.actions.copy()
    actions[:, 6] = -1.0
    for start, stop in closed:
        actions[start:stop, 6] = 1.0
    return track_from_arrays(track.ee_pos, actions, track.grasped, track.object_pos, track.names, track.target,
                             track.instruction)


def own_approach(track) -> np.ndarray:
    """The demonstrated way to the grasp, relative to the grasp point."""
    return track.ee_pos[:GRASP + 1] - track.ee_pos[GRASP]
