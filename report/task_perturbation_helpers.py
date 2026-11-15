"""Tables and figures for report/2026-11-15_task_perturbation_failures.ipynb.

Why does the action head fail three tasks of LIBERO-PRO's task perturbation even with the oracle plan? Inputs:
the rollouts of runs/libero_pro/full_oracle_s9_seed<seed> (the papers' protocol, 14 Nov) and the demonstration
tracks of runs/plan_following/dataset10, from which the training labels for "another object named" are made.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance3d.counterfactual import Retarget, arc_lengths, retarget_path
from affordance3d.plan_consistency import plan_ahead
from affordance3d.plan_input import ContactOffsets
from affordance3d.plan_samples import DemoTrack
from affordance3d.rollout_log import RolloutLog, load_rollout_log
from report.libero_pro_helpers import EPISODE_FILE
from report.vla_pipeline_demo_helpers import pretty

REACH_M = 0.02  # the gripper has reached the grasp point when it is this close
STILL_M = 0.002  # per step: the gripper hardly moves
FAR_M = 0.05  # ... while it is still this far from the grasp point
STALL_WINDOW = 150  # steps at the start of a rollout in which stalling is counted
LOW_M = 0.04  # sideways travel below this height above the contact counts as low
STEP_LIMIT = 280  # LIBERO-Object's step limit


def grasp_points(log: RolloutLog, offsets: ContactOffsets) -> np.ndarray:
    """(T + 1, 3) the named object's grasp point where the object is at each step."""
    return np.asarray(log.object_pos[:, log.target_index], dtype=float) + offsets.grasp[log.target]


def first_reach_step(log: RolloutLog, offsets: ContactOffsets) -> int | None:
    """First step at which the gripper is within REACH_M of the named object's grasp point; None if never."""
    gap = np.linalg.norm(np.asarray(log.ee_pos, dtype=float) - grasp_points(log, offsets), axis=1)
    inside = np.flatnonzero(gap <= REACH_M)
    return int(inside[0]) if len(inside) else None


def stall_steps(log: RolloutLog, offsets: ContactOffsets, window: int = STALL_WINDOW) -> int:
    """Steps among the first `window` in which the gripper hardly moves while it is far from the grasp point."""
    ee_pos = np.asarray(log.ee_pos, dtype=float)
    steps = min(window, len(ee_pos) - 1)
    speed = np.linalg.norm(np.diff(ee_pos[:steps + 1], axis=0), axis=1)
    gap = np.linalg.norm(ee_pos[:steps] - grasp_points(log, offsets)[:steps], axis=1)
    return int(np.sum((speed < STILL_M) & (gap > FAR_M)))


def label_path_metrics(path: np.ndarray, new: np.ndarray, rule: Retarget) -> dict[str, float]:
    """How the demonstrated `path` (start to the original contact) looks once moved to `new` under `rule`:
    its length over the straight distance from the start (`detour`), how far it first moves away from `new`
    horizontally (`away`, m), and its lowest height above `new` while more than LOW_M aside (`low_aside`, m)."""
    path, new = np.asarray(path, dtype=float), np.asarray(new, dtype=float)
    moved = retarget_path(path, new, rule)
    horizontal = np.hypot(*(moved - new)[:, :2].T)
    aside = horizontal > LOW_M
    return {"detour": float(arc_lengths(moved)[-1]) / max(float(np.linalg.norm(new - path[0])), 1e-6),
            "away": float(horizontal.max() - horizontal[0]),
            "low_aside": float((moved - new)[aside, 2].min()) if aside.any() else float("nan")}


def tail_shape(path: np.ndarray, rigid_tail: float = 0.18) -> dict[str, float]:
    """Where the last `rigid_tail` metres of an approach start relative to the contact (`start_height`,
    `start_aside`, m) and how far the approach travels sideways while lower than LOW_M (`low_aside`, m)."""
    path = np.asarray(path, dtype=float)
    arc = arc_lengths(path)
    tail = path[int(np.searchsorted(arc, arc[-1] - rigid_tail - 1e-9)):] - path[-1]
    horizontal = np.hypot(tail[:, 0], tail[:, 1])
    low = horizontal[tail[:, 2] < LOW_M]
    return {"start_height": float(tail[0, 2]), "start_aside": float(horizontal[0]),
            "low_aside": float(low.max()) if len(low) else 0.0}


def approach_path(track: DemoTrack) -> np.ndarray | None:
    """The demonstrated path from the first step to the grasp; None if the track does not start with one."""
    plan = plan_ahead(track.labels, track.ee_pos[0], 0)
    if plan is None or not plan.closing:
        return None
    return np.asarray(track.ee_pos[:plan.steps_to_event + 1], dtype=float)


@dataclass(frozen=True)
class TaskRow:
    """One rollout of the task perturbation, seen through the scene it is set in."""

    task: int
    asked: str
    scene_of: str  # the object LIBERO-Object's own task with this index asks for
    distance: float  # m between that object and the one asked for, at the start
    start_gap: float  # m from the gripper's start to the grasp point of the object asked for
    solved: bool
    reach: int | None
    ran_to_limit: bool
    stalled: int


def task_rows(paths: list[Path], scene_targets: dict[int, str], offsets: ContactOffsets) -> list[TaskRow]:
    """TaskRow for every log; `scene_targets` maps a task index to the target of LIBERO-Object's own task."""
    rows = []
    for path in paths:
        task = int(EPISODE_FILE.search(path.name).group(1))
        log = load_rollout_log(path, images=False)
        names = list(log.names)
        origin = scene_targets[task]
        start = log.object_pos[0]
        distance = (float(np.linalg.norm(start[names.index(origin), :2] - start[log.target_index, :2]))
                    if origin in names else float("nan"))
        rows.append(TaskRow(task, log.target, origin, distance,
                            float(np.linalg.norm(log.ee_pos[0] - grasp_points(log, offsets)[0])),
                            bool(log.meta["success"]), first_reach_step(log, offsets), log.steps >= STEP_LIMIT,
                            stall_steps(log, offsets)))
    return rows


def task_table(rows: list[TaskRow]) -> pd.DataFrame:
    """Per task: what is asked for, in whose scene, and how the rollouts went (medians over rollouts)."""
    table = {}
    for task in sorted({row.task for row in rows}):
        group = [row for row in rows if row.task == task]
        reached = [row.reach for row in group if row.reach is not None]
        table[task] = {"asked for": pretty(group[0].asked), "scene of LIBERO-Object task": pretty(group[0].scene_of),
                       "gripper start to grasp point (cm)": round(100 * np.median([r.start_gap for r in group]), 1),
                       "original target to the object asked for (cm)":
                           round(100 * np.nanmedian([r.distance for r in group]), 1),
                       "solved": f"{sum(r.solved for r in group)} / {len(group)}",
                       "step of first reach (median)": float(np.median(reached)) if reached else np.nan,
                       "never reached": sum(r.reach is None for r in group),
                       "ran to the step limit": sum(r.ran_to_limit for r in group),
                       "steps stalled far away (median)": float(np.median([r.stalled for r in group]))}
    frame = pd.DataFrame.from_dict(table, orient="index")
    frame.index.name = "task"
    return frame


def label_table(tracks: list[DemoTrack], pairs: dict[int, tuple[str, str]], offsets: ContactOffsets,
                rules: dict[str, Retarget]) -> pd.DataFrame:
    """For each task of the task perturbation, (object asked for, original target of its scene): the training
    labels made from the scene's demonstrations, moved to the object asked for (medians over demonstrations)."""
    table = {}
    for task, (asked, origin) in pairs.items():
        rows = []
        for track in tracks:
            path = approach_path(track) if track.target == origin and asked in track.names else None
            if path is None:
                continue
            new = track.object_pos[0, track.names.index(asked)] + offsets.grasp[asked]
            rows.append({name: label_path_metrics(path, new, rule) for name, rule in rules.items()})
        entry = {"asked for": pretty(asked), "scene of": pretty(origin), "demonstrations": len(rows)}
        for name in rules:
            for key, scale in (("detour", 1.0), ("away", 100.0), ("low_aside", 100.0)):
                values = [row[name][key] for row in rows]
                entry[(name, key if key == "detour" else f"{key} (cm)")] = (
                    round(scale * float(np.nanmedian(values)), 2) if values else np.nan)
        table[task] = entry
    frame = pd.DataFrame.from_dict(table, orient="index")
    frame.index.name = "task"
    return frame


def tail_table(tracks: list[DemoTrack], rigid_tail: float = 0.18) -> pd.DataFrame:
    """Per object: the shape of the last `rigid_tail` metres of its demonstrated approaches (medians, cm)."""
    shapes: dict[str, list[dict[str, float]]] = {}
    for track in tracks:
        path = approach_path(track)
        if path is not None:
            shapes.setdefault(track.target, []).append(tail_shape(path, rigid_tail))
    def median_cm(items: list[dict[str, float]], key: str) -> float:
        return round(100 * float(np.median([item[key] for item in items])), 1)

    table = {pretty(name): {"demonstrations": len(items),
                            "start of the last part: height (cm)": median_cm(items, "start_height"),
                            "start of the last part: aside (cm)": median_cm(items, "start_aside"),
                            "sideways while lower than 4 cm (cm)": median_cm(items, "low_aside")}
             for name, items in sorted(shapes.items())}
    return pd.DataFrame.from_dict(table, orient="index")


def course_figure(logs: dict[str, RolloutLog], offsets: ContactOffsets) -> plt.Figure:
    """Height above and horizontal distance to the named object's grasp point at every step, one line per log."""
    fig, (side, high) = plt.subplots(1, 2, figsize=(12.5, 3.8), sharex=True)
    for name, log in logs.items():
        relative = np.asarray(log.ee_pos, dtype=float) - grasp_points(log, offsets)
        side.plot(100 * np.hypot(relative[:, 0], relative[:, 1]), label=name)
        high.plot(100 * relative[:, 2], label=name)
    side.set(xlabel="step", ylabel="horizontal distance to the grasp point (cm)")
    high.set(xlabel="step", ylabel="height above the grasp point (cm)")
    for axis in (side, high):
        axis.axvline(STEP_LIMIT, color="grey", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="grey", linewidth=0.5)
    side.legend(fontsize=8)
    fig.tight_layout()
    return fig


def frames_figure(logs: dict[str, RolloutLog], steps: tuple[int, ...] = (0, 40, 100, 160, 220, 270)) -> plt.Figure:
    """Camera frames of each log at the given steps (a blank where the rollout had already ended)."""
    fig, axes = plt.subplots(len(logs), len(steps), figsize=(2.6 * len(steps), 2.9 * len(logs)), squeeze=False)
    for row, (name, log) in enumerate(logs.items()):
        for col, step in enumerate(steps):
            axis = axes[row, col]
            axis.set_axis_off()
            if log.agentview is not None and step < len(log.agentview):
                axis.imshow(log.agentview[step])
            axis.set_title(f"{name}\nstep {step}" if col == 0 else f"step {step}", fontsize=8)
    fig.tight_layout()
    return fig
