"""Figures and tables for report/experiments/2026-11-11_plan_following_experiment.ipynb.

Inputs: the training set of scripts/data/build_plan_dataset.py, the run folders written by
scripts/vla/run_plan_following_gpu.sh (train.jsonl, metrics.json, rollouts_<plan>/), and the
rollouts of the released checkpoint from the pipeline demo of 10 Nov 2026.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance_vla.analysis.plan_stages import (
    REACH_TOLERANCE_M,
    PlanStages,
    command_roughness,
    plan_stages,
    reach_along_rollout,
)
from affordance_vla.analysis.rollout_outcome import SUCCESS, classify
from affordance_vla.plan.plan_consistency import plan_ahead
from affordance_vla.plan.plan_input import FAR_SCALE_M
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log
from affordance_vla.training.counterfactual import arc_lengths, counterfactual_chunk, retarget_path
from affordance_vla.training.plan_samples import DemoTrack
from report.vla_pipeline_demo_helpers import draw_top_view, outcome_table, pretty

SETS = {  # the 21 rollouts of the pipeline demo, grouped
    "standard": ("standard",),
    "target moved 4–16 cm": ("shift_4cm", "shift_9cm", "shift_16cm"),
    "another object named": ("ask_cream_cheese", "ask_milk", "ask_alphabet_soup", "ask_ketchup", "ask_tomato_sauce"),
}
DEMO_COLOUR, MOVED_COLOURS = "black", ("#d81b60", "#1e88e5", "#43a047")
EVENT_COLUMN = 9  # of the plan input: +1 grasp, -1 release, 0 no plan
PERTURBED_SETS = ("target moved 4–16 cm", "another object named")


def load_run(folder: Path, images: bool = False) -> dict[str, list[RolloutLog]]:
    """{tag: rollouts in start order} for every tag of SETS found in the folder."""
    run: dict[str, list[RolloutLog]] = {}
    for tag in (tag for tags in SETS.values() for tag in tags):
        paths = sorted(Path(folder).glob(f"{tag}_ep*.npz"), key=lambda p: int(p.stem.rsplit("_ep", 1)[1]))
        if paths:
            run[tag] = [load_rollout_log(path, images=images) for path in paths]
    if not run:
        raise FileNotFoundError(f"no rollout logs in {folder}")
    return run


def merge_runs(*runs: dict[str, list[RolloutLog]]) -> dict[str, list[RolloutLog]]:
    """The rollouts of several folders of the same policy, tag by tag; a rollout may appear only once."""
    tags = [tag for run in runs for tag in run]
    merged = {tag: [log for run in runs for log in run.get(tag, [])] for tag in dict.fromkeys(tags)}
    for tag, logs in merged.items():
        stems = [log.path.stem for log in logs]
        if len(set(stems)) != len(stems):
            raise ValueError(f"{tag}: the same rollout is in more than one folder ({sorted(stems)})")
    return merged


def is_success(log: RolloutLog) -> bool:
    return classify(log.names, log.grasped, log.target, log.delivered).label == SUCCESS


def successes(run: dict[str, list[RolloutLog]], set_name: str) -> tuple[int, int]:
    """(successes, rollouts) of a set; a success is the named object grasped first and left in the basket."""
    logs = [log for tag in SETS[set_name] for log in run.get(tag, [])]
    return sum(is_success(log) for log in logs), len(logs)


def success_table(runs: dict[str, dict[str, list[RolloutLog]]]) -> pd.DataFrame:
    """Successes per set and run: the named object is the first one grasped and ends in the basket."""
    rows = {}
    for name in SETS:
        row = {}
        for run_name, run in runs.items():
            done, total = successes(run, name)
            row[run_name] = f"{done} / {total}" if total else "not run"
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def chance_table(runs: dict[str, dict[str, list[RolloutLog]]], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Fisher's exact test (two-sided p) for the success counts of two policies, per perturbed set.

    The test takes the rollouts as independent. They share start states across policies and the pairs are
    not corrected for being several, so the p-values are a rough guide.
    """
    from scipy.stats import fisher_exact

    rows = {}
    for first, second in pairs:
        row = {}
        for name in PERTURBED_SETS:
            (a, n), (b, m) = successes(runs[first], name), successes(runs[second], name)
            row[name] = (f"{a}/{n} vs {b}/{m}: p = {fisher_exact([[a, n - a], [b, m - b]])[1]:.3f}"
                         if n and m else "not run")
        rows[f"{first}  vs  {second}"] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def outcome_details(runs: dict[str, dict[str, list[RolloutLog]]], set_name: str) -> pd.DataFrame:
    """One row per rollout of a set, with the outcome label and step count of every run side by side."""
    frames = []
    for run_name, run in runs.items():
        logs = [log for tag in SETS[set_name] for log in run.get(tag, [])]
        if not logs:
            continue
        table = outcome_table(logs).set_index(["condition", "start"])
        frames.append(table[["outcome", "steps"]].rename(columns=lambda c: f"{run_name}: {c}"))
    return pd.concat(frames, axis=1)


def training_figure(folders: dict[str, Path]) -> plt.Figure:
    """Flow-matching loss and path loss over the training steps of each run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name, folder in folders.items():
        records = [json.loads(line) for line in (Path(folder) / "train.jsonl").read_text().splitlines()]
        steps = [record["step"] for record in records]
        axes[0].plot(steps, [record["flow"] for record in records], label=name)
        axes[1].plot(steps, [record["path_cm"] for record in records], label=name)
    axes[0].set(xlabel="training step", ylabel="flow-matching loss", yscale="log",
                title="demonstrated actions (all runs)")
    axes[1].set(xlabel="training step", ylabel="path loss (cm, Huber)",
                title="plans moved elsewhere: distance of the asked-for path")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def open_loop_table(folders: dict[str, Path]) -> pd.DataFrame:
    """Held-out open-loop numbers before and after training, from each run's metrics.json."""
    rows = {}
    for name, folder in folders.items():
        facts = json.loads((Path(folder) / "metrics.json").read_text())
        for moment in ("before", "after"):
            rows[(name, moment)] = facts[f"open_loop_{moment}"]
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["run", "training"]
    return table


def facts_table(folders: dict[str, Path]) -> pd.DataFrame:
    keys = ("demonstrations", "train_frames", "train_approach_frames", "lora_layers", "trained_parameters_millions",
            "max_abs_difference_from_released_head", "gpu_memory_peak_gib")
    rows = {}
    for name, folder in folders.items():
        facts = json.loads((Path(folder) / "metrics.json").read_text())
        rows[name] = {key: facts.get(key) for key in keys} | {
            "counterfactual": facts["args"]["counterfactual"], "steps": facts["args"]["steps"]}
    return pd.DataFrame.from_dict(rows, orient="index").T


def moved_paths_figure(track: DemoTrack, step: int, contacts: dict[str, np.ndarray], horizon: int = 8,
                       max_step: float = 0.0125) -> plt.Figure:
    """A demonstrated approach and the same approach moved to other contacts: from above and from the side.

    Dots are the targets of the next `horizon` steps, i.e. what the path loss asks the chunk to reach.
    """
    plan = plan_ahead(track.labels, track.ee_pos[step], step)
    if plan is None or not plan.closing:
        raise ValueError(f"step {step} of this demonstration is not on the way to a grasp")
    path = track.ee_pos[step:step + plan.steps_to_event + 1]
    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 4.6))
    curves = [("demonstration", path, DEMO_COLOUR)] + [
        (label, retarget_path(path, contact), colour) for (label, contact), colour in zip(contacts.items(),
                                                                                           MOVED_COLOURS)]
    for label, curve, colour in curves:
        chunk = counterfactual_chunk(path, curve[-1], horizon, flip_remaining=0.0, max_step=max_step)
        travelled = arc_lengths(curve)
        top.plot(curve[:, 1], curve[:, 0], color=colour, linewidth=2, label=label)
        top.scatter(chunk.points[:, 1], chunk.points[:, 0], color=colour, s=18, zorder=3)
        top.scatter(curve[-1, 1], curve[-1, 0], marker="*", s=200, color=colour, edgecolor="black", zorder=4)
        side.plot(travelled, curve[:, 2], color=colour, linewidth=2, label=label)
    top.scatter(path[0, 1], path[0, 0], marker="x", color="black", s=70, zorder=5)
    top.set(xlabel="world y (m)", ylabel="world x (m)",
            title="from above (× gripper now, ★ contact, dots: next 8 steps)")
    top.set_aspect("equal", adjustable="datalim")
    top.invert_xaxis()
    top.invert_yaxis()
    top.legend(fontsize=8)
    side.set(xlabel="distance travelled along the path (m)", ylabel="gripper height z (m)",
             title="from the side: the last 8 cm keep the demonstrated shape")
    fig.tight_layout()
    return fig


def plan_reach(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One rollout that carried plans: query steps, distance in cm from the gripper to the planned contact
    at each query, and whether the plan at that query was a grasp."""
    with np.load(path) as log:
        plans, steps = log["plans"], log["chunk_steps"]
    if plans.size == 0:
        raise ValueError(f"{path.name} carries no plans")
    return steps, 100.0 * FAR_SCALE_M * np.linalg.norm(plans[:, :3], axis=1), plans[:, EVENT_COLUMN] > 0


def set_paths(folders: tuple[Path, ...], set_name: str) -> list[Path]:
    """Log files of one set of rollouts in the folders of one policy; a rollout may appear only once."""
    missing = [str(folder) for folder in folders if not Path(folder).is_dir()]
    if missing:
        raise FileNotFoundError(f"no such rollout folder: {', '.join(missing)}")
    paths = [path for folder in folders for tag in SETS[set_name]
             for path in sorted(Path(folder).glob(f"{tag}_ep*.npz"))]
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError(f"{set_name}: the same rollout is in more than one of {[str(f) for f in folders]}")
    return paths


def rollout_stages(path: Path) -> PlanStages:
    """How far one rollout with a plan got. The distance to the planned grasp point is taken at every step,
    against the contact of the latest plan; `delivered` is the goal check on the named object alone."""
    log = load_rollout_log(path, images=False)
    with np.load(path) as data:
        plans = data["plans"]
    if len(plans) != len(log.chunk_steps) or plans.size == 0:
        raise ValueError(f"{path.name} carries {len(plans)} plans for {len(log.chunk_steps)} queries")
    reach, query = reach_along_rollout(log.ee_pos, log.chunk_steps, FAR_SCALE_M * plans[:, :3])
    closing = plans[np.clip(query, 0, None), EVENT_COLUMN] > 0
    target = log.target_index
    return plan_stages(reach, closing, log.grasped[:, target], log.object_pos[:, target, 2],
                       delivered=log.delivered)


def stage_table(folders: dict[str, tuple[Path, ...]], set_name: str) -> pd.DataFrame:
    """Per policy: rollouts of a set that reached the planned grasp point, held the named object and delivered
    it, and how the rollouts that held it without delivering it ended."""
    rows = {}
    for name, group in folders.items():
        stages = [rollout_stages(path) for path in set_paths(group, set_name)]
        rows[name] = {"rollouts": len(stages),
                      f"gripper came within {100 * REACH_TOLERANCE_M:.0f} cm of the planned grasp point":
                          sum(stage.reached for stage in stages),
                      "object knocked over before it was held": sum(stage.toppled for stage in stages),
                      "object held": sum(stage.held for stage in stages),
                      "object delivered": sum(stage.delivered for stage in stages),
                      "held, then let go outside the basket":
                          sum(stage.held and not stage.delivered and not stage.holding_at_end for stage in stages),
                      "held, still in the gripper at the end":
                          sum(stage.held and not stage.delivered and stage.holding_at_end for stage in stages)}
    return pd.DataFrame.from_dict(rows, orient="index")


def approach_roughness(log: RolloutLog) -> float | None:
    """Roughness of the chunks requested before the gripper first touches any object; None without one."""
    touching = log.touched.any(axis=1)
    first_touch = int(np.argmax(touching)) if touching.any() else len(touching)
    chunks = log.chunks[log.chunk_steps < first_touch]
    return command_roughness(chunks) if len(chunks) else None


def roughness_table(folders: dict[str, tuple[Path, ...]]) -> pd.DataFrame:
    """Per policy and set: mean change of the translation command between consecutive steps of a chunk,
    over the chunks requested before the gripper first touches an object (action units)."""
    rows = {}
    for name, group in folders.items():
        row = {}
        for set_name in SETS:
            values = [approach_roughness(load_rollout_log(path, images=False)) for path in set_paths(group, set_name)]
            values = [value for value in values if value is not None]
            row[set_name] = round(float(np.mean(values)), 3) if values else None
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def reach_figure(folders: dict[str, Path], tags: tuple[str, ...]) -> plt.Figure:
    """Distance from the gripper to the planned contact at every query, one panel per run."""
    fig, axes = plt.subplots(1, len(folders), figsize=(5.6 * len(folders), 3.6), squeeze=False, sharey=True)
    for axis, (name, folder) in zip(axes[0], folders.items()):
        for tag in tags:
            for path in sorted(Path(folder).glob(f"{tag}_ep*.npz")):
                steps, reach, _ = plan_reach(path)
                log = load_rollout_log(path, images=False)
                axis.plot(steps, reach, color="#2e7d32" if is_success(log) else "#c62828", linewidth=1.4, alpha=0.85)
        axis.set(xlabel="step", title=name)
        axis.axhline(0.0, color="grey", linewidth=0.6)
    axes[0][0].set_ylabel("gripper to planned contact (cm)")
    fig.tight_layout()
    return fig


def rollout_paths_figure(runs: dict[str, dict[str, list[RolloutLog]]], tags: tuple[str, ...],
                         layout: RolloutLog) -> plt.Figure:
    """Gripper paths from above, one panel per run; ★ marks where the object named by the instruction starts."""
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.0), squeeze=False)
    for axis, (name, run) in zip(axes[0], runs.items()):
        draw_top_view(axis, layout, 0)
        for log in (log for tag in tags for log in run.get(tag, [])):
            colour = "#2e7d32" if is_success(log) else "#c62828"
            axis.plot(log.ee_pos[:, 1], log.ee_pos[:, 0], color=colour, linewidth=1.6, alpha=0.8, zorder=3)
            target = log.object_pos[0, log.target_index, :2]
            axis.scatter(target[1], target[0], marker="*", s=260, color="gold", edgecolor="black", zorder=5)
        axis.set_title(name, fontsize=10)
    fig.tight_layout()
    return fig


def offsets_table(dataset: Path) -> pd.DataFrame:
    """Per target object: demonstrations kept and how much the grasp point varies relative to the object."""
    stats = json.loads((Path(dataset) / "stats.json").read_text())
    rows = {pretty(name): {"demonstrations": entry["demos"],
                           "grasp point std x, y, z (mm)": entry["grasp_offset_std_mm"],
                           "release point std x, y, z (mm)": entry["release_offset_std_mm"],
                           "steps from close command to contact (median)":
                               entry["steps_from_close_command_to_contact_median"]}
            for name, entry in stats["per_target"].items()}
    return pd.DataFrame.from_dict(rows, orient="index")
