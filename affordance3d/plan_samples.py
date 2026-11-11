"""Training samples for a plan-conditioned action head, from one replayed demonstration.

A sample is an action chunk at one step together with the plan input for that step. `demo_sample`
is the demonstration as it happened. `shifted_sample` pairs the same step with a contact somewhere
else: the translation is then supervised through the path the chunk asks for (the plan–action
consistency loss), because no demonstrated action exists for it.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from affordance3d.caw_labels import CawLabel, GripperEvent, caw_labels, contact_events, gripper_events
from affordance3d.counterfactual import arc_lengths, counterfactual_chunk
from affordance3d.plan_consistency import PlanAhead, plan_ahead
from affordance3d.plan_input import ContactOffsets, no_plan_features, plan_features

MIN_REDIRECT_M = 0.03  # closer to the contact than this, a step is not paired with another contact
_MIN_PATH_M = 1e-6


@dataclass(frozen=True)
class DemoTrack:
    """What the plan labels need from one replayed demonstration of T actions."""

    ee_pos: np.ndarray  # (T + 1, 3) gripper position before each action, and after the last
    actions: np.ndarray  # (T, 7) simulator convention: gripper command -1 open, +1 close
    object_pos: np.ndarray  # (T + 1, N, 3)
    grasped: np.ndarray  # (T + 1,) the target is held between both fingers (simulator contacts)
    names: tuple[str, ...]
    target: str
    instruction: str
    labels: tuple[CawLabel, ...]  # from the target's contact flags
    command_flips: tuple[GripperEvent, ...]


@dataclass(frozen=True)
class PlanSample:
    features: np.ndarray  # (PLAN_DIM,) input of the plan encoder
    actions: np.ndarray  # (H, 7) clean chunk, policy convention (gripper 1 open, 0 close), not normalised
    path_target: np.ndarray  # (H, 3) where the gripper should be after each step (metres)
    path_valid: np.ndarray  # (H,) steps supervised by the path loss
    action_valid: np.ndarray  # (H, 7) entries supervised by the flow-matching loss
    start: np.ndarray  # (3,) gripper position at the sample's step


def track_from_arrays(ee_pos: np.ndarray, actions: np.ndarray, grasped: np.ndarray, object_pos: np.ndarray,
                      names: Sequence[str], target: str, instruction: str) -> DemoTrack:
    """Build a track; `grasped` is the target's two-finger contact flag for each of the T + 1 states."""
    ee_pos, actions = np.asarray(ee_pos, dtype=float), np.asarray(actions, dtype=float)
    object_pos, names = np.asarray(object_pos, dtype=float), tuple(names)
    rows = len(actions) + 1
    if actions.ndim != 2 or actions.shape[1] != 7 or ee_pos.shape != (rows, 3):
        raise ValueError(f"need actions (T, 7) and ee_pos (T + 1, 3), got {actions.shape} and {ee_pos.shape}")
    if object_pos.shape != (rows, len(names), 3) or len(grasped) != rows:
        raise ValueError(f"object_pos must be ({rows}, {len(names)}, 3) and grasped ({rows},), got "
                         f"{object_pos.shape} and {np.shape(grasped)}")
    if target not in names:
        raise ValueError(f"target {target!r} is not one of the objects {names}")
    grasped = np.asarray(grasped, dtype=bool)
    labels = caw_labels(ee_pos, contact_events(grasped))
    return DemoTrack(ee_pos=ee_pos, actions=actions, object_pos=object_pos, grasped=grasped, names=names,
                     target=target, instruction=instruction, labels=tuple(labels),
                     command_flips=tuple(gripper_events(actions[:, 6])))


def single_pick_and_place(track: DemoTrack) -> bool:
    """True if the target's contact flags show exactly one grasp followed by one release.

    A demonstration that loses and regains the object, or never lets go before it ends, would add
    events that are not a plan (a mid-air "release") to the labels and to the average offsets.
    """
    return [label.event.closing for label in track.labels] == [True, False]


def _policy_chunk(track: DemoTrack, step: int, horizon: int) -> np.ndarray:
    if step < 0 or step + horizon > len(track.actions):
        raise ValueError(f"step {step} leaves no full chunk of {horizon} steps "
                         f"(the demonstration has {len(track.actions)} actions)")
    chunk = track.actions[step:step + horizon].copy()
    chunk[:, 6] = (1.0 - chunk[:, 6]) / 2.0
    return chunk


def _noisy(contact: np.ndarray, noise: np.ndarray | None) -> np.ndarray:
    return contact if noise is None else contact + np.asarray(noise, dtype=float)


def demo_sample(track: DemoTrack, step: int, horizon: int, contact_noise: np.ndarray | None = None) -> PlanSample:
    """The demonstrated chunk at `step` with the plan of the demonstration (none after the last event)."""
    actions = _policy_chunk(track, step, horizon)
    plan = plan_ahead(track.labels, track.ee_pos[step], step)
    features = no_plan_features() if plan is None else plan_features(
        track.ee_pos[step], _noisy(plan.polyline[-1], contact_noise), plan.approach, plan.closing)
    return PlanSample(features=features, actions=actions, path_target=np.zeros((horizon, 3)),
                      path_valid=np.zeros(horizon, dtype=bool), action_valid=np.ones((horizon, 7), dtype=bool),
                      start=track.ee_pos[step].copy())


def _flip_remaining(track: DemoTrack, step: int, event_step: int, arc: np.ndarray) -> float:
    """Distance before the contact at which the demonstration commanded the gripper to close."""
    flips = [flip.step for flip in track.command_flips if flip.closing and flip.step <= event_step]
    if not flips:
        return 0.0
    if flips[-1] <= step:
        return float("inf")  # the command has already been given
    return float(arc[-1] - arc[flips[-1] - step])


def shifted_sample(track: DemoTrack, step: int, horizon: int, new_contact: np.ndarray, motion_scale: float,
                   approach: np.ndarray | None = None, contact_noise: np.ndarray | None = None,
                   min_remaining: float = MIN_REDIRECT_M) -> PlanSample | None:
    """The step paired with a grasp at `new_contact` instead of the demonstrated one.

    Returns None unless the step is on the way to a grasp with at least `min_remaining` metres to
    go. `motion_scale` is the gripper motion per unit of translation action. The clean chunk's
    translation is the demonstrated command plus what the plan change adds: the displacement
    between walking the moved path and walking the demonstrated one, divided by the scale and
    clipped to the action range. An unchanged contact therefore gives the demonstrated commands,
    lag compensation included. The path target is the path those commands ask for. Rotation is
    the demonstration's (not supervised after the arrival), the gripper closes at the
    demonstrated distance from the contact.
    """
    actions = _policy_chunk(track, step, horizon)
    plan: PlanAhead | None = plan_ahead(track.labels, track.ee_pos[step], step)
    if plan is None or not plan.closing:
        return None
    event_step = step + plan.steps_to_event
    path = track.ee_pos[step:event_step + 1]
    arc = arc_lengths(path)
    if arc[-1] <= max(min_remaining, _MIN_PATH_M):
        return None
    new_contact = np.asarray(new_contact, dtype=float)
    flip = _flip_remaining(track, step, event_step, arc)
    target = counterfactual_chunk(path, new_contact, horizon, flip, max_step=motion_scale)
    unmoved = counterfactual_chunk(path, path[-1], horizon, flip, max_step=motion_scale)
    added = np.diff(np.vstack([np.zeros(3), target.points - unmoved.points]), axis=0) / motion_scale
    actions[:, :3] = np.clip(actions[:, :3] + added, -1.0, 1.0)
    actions[:, 6] = np.where(target.closed, 0.0, 1.0)
    action_valid = np.zeros((horizon, 7), dtype=bool)
    action_valid[:, 3:6] = target.valid[:, None]
    action_valid[:, 6] = True
    features = plan_features(track.ee_pos[step], _noisy(new_contact, contact_noise),
                             plan.approach if approach is None else approach, closing=True)
    path_target = track.ee_pos[step] + motion_scale * np.cumsum(actions[:, :3], axis=0)
    return PlanSample(features=features, actions=actions, path_target=path_target, path_valid=target.valid,
                      action_valid=action_valid, start=track.ee_pos[step].copy())


def _unit_mean(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else np.zeros(3)


def contact_offsets(tracks: Sequence[DemoTrack], container: str) -> ContactOffsets:
    """Average gripper position at grasps (per target object) and at releases, relative to the object
    grasped and to the container."""
    grasps, grasp_approaches = defaultdict(list), defaultdict(list)
    releases, release_approaches = [], []
    for track in tracks:
        if container not in track.names:
            raise ValueError(f"{container!r} is not an object of the demonstration {track.instruction!r}")
        target_index, container_index = track.names.index(track.target), track.names.index(container)
        for label in track.labels:
            if label.event.closing:
                grasps[track.target].append(label.contact - track.object_pos[label.event.step, target_index])
                grasp_approaches[track.target].append(label.approach)
            else:
                releases.append(label.contact - track.object_pos[label.event.step, container_index])
                release_approaches.append(label.approach)
    if not grasps or not releases:
        raise ValueError("the demonstrations contain no grasp or no release, so there is nothing to average")
    return ContactOffsets(
        grasp={name: np.mean(values, axis=0) for name, values in grasps.items()},
        grasp_approach={name: _unit_mean(values) for name, values in grasp_approaches.items()},
        release=np.mean(releases, axis=0), release_approach=_unit_mean(release_approaches))
