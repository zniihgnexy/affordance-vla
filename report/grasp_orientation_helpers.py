"""Tables and figures for report/2026-11-20_grasp_orientation.ipynb (E0 of Pipeline v3).

Do failed grasps hold the gripper differently from held ones and from the demonstrations of the same object?
Inputs: the demonstration replays (runs/plan_following/demos) and closed-loop rollouts of LIBERO-PRO
(runs/libero_pro/<run>). Everything is in world coordinates; the logs do not record object orientations yet.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance_vla.analysis.grasp_orientation import (
    closing_yaw_deg,
    grasp_attempts,
    nearest_demo_angle_deg,
    object_category,
    tilt_deg,
    yaw_difference_deg,
)
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log
from report.libero_pro_helpers import episode_paths, task_of

PASS_GAP_DEG = 10.0  # E0 passes when failed grasps sit at least this much further from the demonstrations
GROUPS = ("held, task solved", "held, task failed", "not held")


def _attempts(log: RolloutLog):
    if log.ee_quat is None:
        raise ValueError(f"{log.path.name} has no end-effector orientation")
    return grasp_attempts(log.names, log.ee_pos, log.ee_quat, log.object_pos, log.grasped, log.actions)


def demo_grasps(paths: Iterable[Path]) -> pd.DataFrame:
    """The first held grasp of its own target in each demonstration."""
    rows = []
    for path in paths:
        log = load_rollout_log(path, images=False)
        held = [a for a in _attempts(log) if a.obj == log.target and a.held]
        if held:
            grasp = held[0]
            rows.append({"file": path.name, "category": object_category(log.target), "step": grasp.step,
                         "rotation": grasp.rotation, "tilt_deg": tilt_deg(grasp.rotation),
                         "closing_yaw_deg": closing_yaw_deg(grasp.rotation)})
    return pd.DataFrame(rows)


def demo_rotations(demos: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    return {category: list(group["rotation"]) for category, group in demos.groupby("category")}


def demo_spread(demos: pd.DataFrame) -> pd.DataFrame:
    """Per object kind: how much the demonstrated grasps differ from each other."""
    rows = []
    for category, group in demos.groupby("category"):
        yaws = group["closing_yaw_deg"].to_numpy()
        centre = yaws[0]
        spread = [yaw_difference_deg(y, centre) for y in yaws]
        rows.append({"object": category, "demos": len(group), "tilt median (deg)": group["tilt_deg"].median(),
                     "tilt max (deg)": group["tilt_deg"].max(),
                     "closing yaw spread, 90th pct (deg)": float(np.percentile(spread, 90))})
    return pd.DataFrame(rows).set_index("object").round(1)


def rollout_attempts(run: Path, suites: Iterable[str], demos: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """One row per episode: its first close command near the named object, against the demonstrations."""
    rows = []
    for suite in suites:
        for path in episode_paths(run, suite):
            log = load_rollout_log(path, images=False)
            on_target = [a for a in _attempts(log) if a.obj == log.target]
            row = {"run": run.name, "suite": suite, "task": task_of(log), "file": path.name, "target": log.target,
                   "category": object_category(log.target), "success": bool(log.meta["success"]),
                   "attempted": bool(on_target)}
            if on_target:
                attempt = on_target[0]
                reference = demos.get(row["category"], [])
                yaws = [closing_yaw_deg(r) for r in reference]
                row.update({"step": attempt.step, "held": attempt.held, "distance_m": attempt.distance_m,
                            "angle_to_demo_deg": nearest_demo_angle_deg(attempt.rotation, reference),
                            "yaw_to_demo_deg": min((yaw_difference_deg(closing_yaw_deg(attempt.rotation), y)
                                                    for y in yaws), default=np.nan),
                            "tilt_deg": tilt_deg(attempt.rotation)})
            rows.append(row)
    table = pd.DataFrame(rows)
    held = table["held"].astype("boolean").fillna(False).astype(bool) if "held" in table else False
    table["group"] = np.where(~table["attempted"], "no close near the target",
                              np.where(held, np.where(table["success"], GROUPS[0], GROUPS[1]), GROUPS[2]))
    return table


def distance_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """How far the gripper is from the named object's centre when it starts to close, by outcome (cm)."""
    tried = attempts[attempts["attempted"]]
    table = tried.groupby("group")["distance_m"].describe()[["count", "25%", "50%", "75%"]]
    table[["25%", "50%", "75%"]] *= 100.0
    return table.reindex(list(GROUPS)).rename(columns=lambda c: c if c == "count" else f"{c} (cm)").round(1)


def distance_figure(attempts: pd.DataFrame) -> plt.Figure:
    tried = attempts[attempts["attempted"]]
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    for group in GROUPS:
        values = np.sort(100.0 * tried.loc[tried["group"] == group, "distance_m"].to_numpy())
        if len(values):
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                      label=f"{group} ({len(values)})")
    axis.set_xlabel("gripper to object centre at the close command (cm)")
    axis.set_ylabel("share of first attempts")
    axis.set_xlim(0, 15)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    return fig


def group_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """Angles to the demonstrations by outcome of the first attempt on the named object."""
    tried = attempts[attempts["attempted"]]
    table = tried.groupby("group").agg(
        episodes=("file", "size"),
        angle_median=("angle_to_demo_deg", "median"),
        angle_p75=("angle_to_demo_deg", lambda s: s.quantile(0.75)),
        yaw_median=("yaw_to_demo_deg", "median"),
        tilt_median=("tilt_deg", "median"))
    untried = int((~attempts["attempted"]).sum())
    table.loc["no close near the target", "episodes"] = untried
    return table.reindex([*GROUPS, "no close near the target"]).round(1)


def verdict(attempts: pd.DataFrame) -> tuple[float, bool]:
    """Median angle to the demonstrations of failed grasps minus that of held ones, and whether E0 passes."""
    tried = attempts[attempts["attempted"]]
    gap = float(tried.loc[tried["group"] == GROUPS[2], "angle_to_demo_deg"].median()
                - tried.loc[tried["group"] != GROUPS[2], "angle_to_demo_deg"].median())
    return gap, gap >= PASS_GAP_DEG


def per_object_table(attempts: pd.DataFrame) -> pd.DataFrame:
    tried = attempts[attempts["attempted"]]
    table = tried.pivot_table(index="category", columns="group", values="angle_to_demo_deg",
                              aggfunc="median").reindex(columns=list(GROUPS))
    counts = tried.pivot_table(index="category", columns="group", values="file", aggfunc="size")
    return table.round(1).astype(str) + " (" + counts.reindex_like(table).fillna(0).astype(int).astype(str) + ")"


def angle_figure(attempts: pd.DataFrame) -> plt.Figure:
    """Cumulative share of first attempts by angle to the nearest demonstration, per outcome."""
    tried = attempts[attempts["attempted"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for group in GROUPS:
        for axis, column in zip(axes, ("angle_to_demo_deg", "yaw_to_demo_deg")):
            values = np.sort(tried.loc[tried["group"] == group, column].dropna().to_numpy())
            if len(values):
                axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                          label=f"{group} ({len(values)})")
    axes[0].set_xlabel("angle to the nearest demonstrated grasp (deg)")
    axes[1].set_xlabel("closing-direction difference about the vertical (deg)")
    for axis in axes:
        axis.set_ylabel("share of first attempts")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def with_images(rows: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    """The rows whose logs kept their camera images (the papers' protocol keeps them for some episodes only)."""
    def kept(row: pd.Series) -> bool:
        with np.load(run_root / row["run"] / row["suite"] / row["file"]) as data:
            return "agentview" in data.files
    return rows[rows.apply(kept, axis=1)] if len(rows) else rows


def grasp_frames(rows: pd.DataFrame, run_root: Path, views: tuple[str, ...] = ("agentview", "wrist")) -> plt.Figure:
    """The camera images at the close command of some attempts, one row of views per attempt."""
    fig, axes = plt.subplots(len(rows), len(views), figsize=(2.6 * len(views), 2.6 * len(rows)), squeeze=False)
    for axis_row, (_, row) in zip(axes, rows.iterrows()):
        log = load_rollout_log(run_root / row["run"] / row["suite"] / row["file"])
        for axis, view in zip(axis_row, views):
            axis.imshow(getattr(log, view)[int(row["step"])])
            axis.set_axis_off()
        axis_row[0].set_title(f"{row['suite'].split('_')[-1]} {row['file']}: {row['category']}, "
                              f"{row['angle_to_demo_deg']:.0f} deg", fontsize=8, loc="left")
    fig.tight_layout()
    return fig
