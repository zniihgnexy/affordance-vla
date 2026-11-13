"""Tables and figures for report/2026-11-13_plan_predictor.ipynb.

Inputs: runs/plan_predictor/<run>/{metrics.json, predictions.npz} written by
scripts/remote/train_plan_predictor.py, and the closed-loop folders runs/libero_pro/<name>/ written by
scripts/remote/run_libero_pro_gpu.sh with PLAN=predicted, whose logs keep the plan inputs the action
head was given at every query.
"""
from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance3d.counterfactual import MAX_STEP_M, Retarget, counterfactual_chunk, retarget_path
from affordance3d.plan_audit import NOT_AN_OBJECT, PlanPoint, plans_before_grasp
from affordance3d.plan_grid import CELLS, IMAGE_SIZE, Camera, contact_pixel
from affordance3d.plan_input import ContactOffsets
from affordance3d.rollout_log import RolloutLog
from report.libero_pro_helpers import SUITES, task_of
from report.vla_pipeline_demo_helpers import pretty

TEST_SETS = {  # key in metrics.json -> (part of metrics.json, row label)
    "own: grasp": ("held_out", "held-out demonstrations, own instruction: next grasp"),
    "own: release": ("held_out", "held-out demonstrations, own instruction: next release"),
    "other: grasp": ("held_out", "held-out demonstrations, another object named"),
    "pro libero_object_task: grasp": ("benchmark", "LIBERO-PRO start frames: another object asked for"),
    "pro libero_object_swap: grasp": ("benchmark", "LIBERO-PRO start frames: target swapped with another object"),
}
MEASURES = {"contact_cm_median": "median error (cm)", "within_2cm": "within 2 cm",
            "another_object_nearer": "another object nearer", "event_accuracy": "event right"}
ORACLE_COLOUR = "#39ff14"
HEAD_STYLES = (("#ff2d95", "X"), ("#00b7ff", "o"), ("#ffb000", "s"))  # one per head, in the order given
NEAR_CM = 2.0
MIN_ROLLOUTS = 20  # rollouts a point of the approach figure must rest on
MAP_ALPHA = 0.8  # opacity of a cell that holds the whole score


@dataclass(frozen=True)
class Predictions:
    """The test items a predictor was run on after training, in the order they were stored."""

    kind: np.ndarray  # (N,) "own", "other", "pro <suite>"
    instruction: np.ndarray  # (N,)
    agentview: np.ndarray  # (N, IMAGE_SIZE, IMAGE_SIZE, 3) the agent view the VLM was given
    contact: np.ndarray  # (N, 3) predicted contact, world frame, metres
    target_contact: np.ndarray  # (N, 3) oracle contact
    target_event: np.ndarray  # (N,)
    cell_scores: np.ndarray | None  # (N, CELLS²) of the grid head


def load_metrics(run: Path) -> dict:
    """metrics.json of a predictor run."""
    return json.loads((Path(run) / "metrics.json").read_text())


def load_predictions(run: Path) -> Predictions:
    with np.load(Path(run) / "predictions.npz") as data:
        scores = data["cell_scores"] if "cell_scores" in data.files else None
        return Predictions(data["kind"], data["instruction"], data["agentview"], data["contact"],
                           data["target_contact"], data["target_event"], scores)


def check_same_items(predictions: dict[str, Predictions]) -> None:
    """Raise unless every head was tested on the same items in the same order."""
    first, *rest = predictions.values()
    for other in rest:
        if not (np.array_equal(first.instruction, other.instruction) and np.array_equal(first.kind, other.kind)
                and np.allclose(first.target_contact, other.target_contact)):
            raise ValueError("the predictors were tested on different items; their pictures cannot share a panel")


def offline_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Rows: the test sets; columns: per head, the contact error of the predicted next gripper event."""
    columns: dict[tuple[str, str], dict[str, float]] = {}
    for head, facts in metrics.items():
        for key, (part, label) in TEST_SETS.items():
            row = facts[part].get(key)
            if row is None:
                continue
            columns.setdefault(("", "items"), {})[label] = row["items"]
            for measure, name in MEASURES.items():
                columns.setdefault((head, name), {})[label] = row[measure]
    table = pd.DataFrame(columns).reindex([label for _, label in TEST_SETS.values()]).dropna(how="all")
    table[("", "items")] = table[("", "items")].astype(int)
    return table


def contact_errors_cm(predictions: Predictions) -> np.ndarray:
    """(N,) distance between the predicted and the oracle contact of every item."""
    return 100.0 * np.linalg.norm(predictions.contact - predictions.target_contact, axis=1)


def items_of(predictions: Predictions, kind: str, every: int = 1) -> list[int]:
    """Indices of the items of one kind, every n-th of them."""
    return [int(i) for i in np.flatnonzero(predictions.kind == kind)[::every]]


def _panel(axis: plt.Axes, image: np.ndarray, instruction: str, note: str) -> None:
    """Show an agent view in image pixels with its instruction and a note as the title."""
    axis.imshow(image, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0))
    axis.set_title("\n".join(textwrap.wrap(f"“{instruction}”", 34)) + f"\n{note}", fontsize=8)
    axis.set_xlim(0, IMAGE_SIZE), axis.set_ylim(IMAGE_SIZE, 0)
    axis.set_axis_off()


def _mark_oracle(axis: plt.Axes, contact: np.ndarray, camera: Camera) -> None:
    """Ring around the pixel of the oracle contact."""
    u, v, _ = contact_pixel(contact, camera)
    axis.scatter([u], [v], s=260, facecolors="none", edgecolors=ORACLE_COLOUR, linewidths=2.0, label="oracle contact")


def prediction_figure(predictions: dict[str, Predictions], camera: Camera, picks: list[int],
                      columns: int = 5) -> plt.Figure:
    """The agent view of the picked items with the oracle contact (ring) and each head's predicted contact."""
    check_same_items(predictions)
    shown = next(iter(predictions.values()))
    rows = int(np.ceil(len(picks) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 4.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, index in zip(axes.ravel(), picks):
        notes = [f"{head}: {contact_errors_cm(found)[index]:.1f} cm" for head, found in predictions.items()]
        _panel(axis, shown.agentview[index], str(shown.instruction[index]), "\n".join(notes))
        _mark_oracle(axis, shown.target_contact[index], camera)
        for (head, found), (colour, marker) in zip(predictions.items(), HEAD_STYLES):
            u, v, _ = contact_pixel(found.contact[index], camera)
            axis.scatter([u], [v], s=70, c=colour, marker=marker, edgecolors="white", linewidths=0.8, label=head)
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    return fig


def map_figure(found: Predictions, camera: Camera, picks: list[int], columns: int = 5) -> plt.Figure:
    """The grid head's score of every image cell, laid over the agent view; the ring is the oracle contact."""
    if found.cell_scores is None:
        raise ValueError("these predictions carry no cell scores: they are not from a grid head")
    rows = int(np.ceil(len(picks) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.9 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, index in zip(axes.ravel(), picks):
        scores = found.cell_scores[index].reshape(CELLS, CELLS)
        _panel(axis, found.agentview[index], str(found.instruction[index]),
               f"best cell holds {scores.max():.0%} of the score")
        # a cell is tinted in proportion to its score, so cells without score leave the image as it is
        axis.imshow(scores, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), cmap="autumn_r", vmin=0.0, vmax=1.0,
                    alpha=MAP_ALPHA * np.sqrt(np.clip(scores, 0.0, 1.0)), interpolation="nearest")
        edges = np.linspace(0, IMAGE_SIZE, CELLS + 1)
        axis.vlines(edges, 0, IMAGE_SIZE, colors="white", linewidths=0.3, alpha=0.5)
        axis.hlines(edges, 0, IMAGE_SIZE, colors="white", linewidths=0.3, alpha=0.5)
        _mark_oracle(axis, found.target_contact[index], camera)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class GraspPlans:
    """Where the plans of one closed-loop rollout pointed before anything was held."""

    task: int
    solved: bool
    target: str
    points: tuple[PlanPoint, ...]  # one per query before the first grasp of any object

    @property
    def first(self) -> PlanPoint | None:
        """The first plan of the rollout if it is a grasp plan, else None."""
        return self.points[0] if self.points and not np.isnan(self.points[0].error_m) else None


def grasp_plans(log: RolloutLog, offsets: ContactOffsets) -> GraspPlans:
    """Judge the plans a rollout was given (affordance3d/plan_audit.py); they are read from the log file."""
    with np.load(log.path) as data:
        plans = data["plans"]
    points = plans_before_grasp(plans, log.chunk_steps, log.ee_pos, log.object_pos, log.names, offsets.grasp,
                                log.target, log.grasped)
    return GraspPlans(task_of(log), bool(log.meta["success"]), log.target, points)


def _share(count: int, total: int) -> str:
    return f"{count} / {total}" if total else "–"


def pointing_table(audits: dict[str, list[GraspPlans]]) -> pd.DataFrame:
    """Per policy: where the first plan of each rollout pointed, and what became of the rollout."""
    rows = {}
    for name, group in audits.items():
        first = [(audit, audit.first) for audit in group if audit.first is not None]
        right = [audit for audit, point in first if point.pointed == audit.target]
        other = [audit for audit, point in first if point.pointed not in (audit.target, *NOT_AN_OBJECT)]
        errors = 100.0 * np.array([point.error_m for _, point in first])
        hits = {id(audit) for audit in right}
        missed = [audit for audit in group if id(audit) not in hits]
        rows[name] = {
            "rollouts": len(group),
            "first plan is a grasp plan": _share(len(first), len(group)),
            "first plan points at the named object": _share(len(right), len(first)),
            "first plan points at another object": _share(len(other), len(first)),
            "first plan: median error (cm)": round(float(np.median(errors)), 1) if len(first) else np.nan,
            f"first plan within {NEAR_CM:.0f} cm": _share(int(np.sum(errors <= NEAR_CM)), len(first)),
            "solved": _share(sum(audit.solved for audit in group), len(group)),
            "solved when the first plan pointed at the named object": _share(sum(a.solved for a in right), len(right)),
            "solved when it did not": _share(sum(a.solved for a in missed), len(missed)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def per_task_pointing(audits: dict[str, list[GraspPlans]]) -> pd.DataFrame:
    """Per task and policy: rollouts whose first plan pointed at the named object, and the solved ones.

    The groups must come from one suite: tasks are told apart by their index alone.
    """
    counts: dict[int, dict[str, tuple[int, int, int]]] = {}
    asked: dict[int, str] = {}
    for name, group in audits.items():
        for audit in group:
            asked.setdefault(audit.task, pretty(audit.target))
            right, done, total = counts.setdefault(audit.task, {}).get(name, (0, 0, 0))
            hit = audit.first is not None and audit.first.pointed == audit.target
            counts[audit.task][name] = (right + hit, done + audit.solved, total + 1)
    rows = {}
    for task in sorted(counts):
        rows[task] = {"asked for": asked[task]}
        for name in audits:
            cell = counts[task].get(name)
            rows[task][name] = "–" if cell is None else f"plan right {cell[0]} · solved {cell[1]} / {cell[2]}"
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "task"
    return table


def approach_figure(audits: dict[str, dict[str, list[GraspPlans]]], queries: int = 12) -> plt.Figure:
    """Median error of the predicted grasp point at the n-th query before anything is held, per suite.

    Query n only counts the rollouts that had held nothing by then, so later points come from fewer,
    slower rollouts; a point is drawn when at least MIN_ROLLOUTS rollouts had a grasp plan at that query.
    Rollouts in which nothing is ever held contribute all their queries.
    """
    labels = {suite: label.split(":")[0] for label, suite in SUITES.items()}
    suites = [suite for suite in SUITES.values() if any(suite in run for run in audits.values())]
    colours = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(suites), figsize=(5.6 * len(suites), 3.8), squeeze=False, sharey=True)
    for axis, suite in zip(axes[0], suites):
        for k, (name, run) in enumerate(audits.items()):
            columns = [np.array([100.0 * audit.points[q].error_m for audit in run.get(suite, [])
                                 if len(audit.points) > q]) for q in range(queries)]
            kept = [(q, column[np.isfinite(column)]) for q, column in enumerate(columns)]
            kept = [(q, column) for q, column in kept if len(column) >= MIN_ROLLOUTS]
            if not kept:
                continue
            marks = [q for q, _ in kept]
            axis.plot(marks, [np.median(column) for _, column in kept], color=colours[k % len(colours)], marker="o",
                      markersize=4, label=name)
            axis.fill_between(marks, [np.percentile(column, 25) for _, column in kept],
                              [np.percentile(column, 75) for _, column in kept], color=colours[k % len(colours)],
                              alpha=0.15)
        axis.axhline(NEAR_CM, color="grey", linewidth=0.8, linestyle="--")
        axis.set_title(f"{labels[suite]} perturbation", fontsize=10)
        axis.set_xlabel("query number, while nothing is held yet")
        axis.set_yscale("log")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    axes[0, 0].set_ylabel("error of the planned grasp point (cm)\nmedian and quartiles over rollouts")
    fig.tight_layout()
    return fig


def facts_table(metrics: dict[str, dict]) -> pd.DataFrame:
    """Size and training of each predictor, from its metrics.json."""
    rows = {}
    for head, facts in metrics.items():
        args = facts["args"]
        rows[head] = {"trained parameters (millions)": facts["parameters_millions"],
                      "training steps": int(args["steps"]), "frames per step": int(args["frames"]),
                      "training frames to draw from": facts["train_frames"],
                      "test items from held-out demonstrations": facts["held_out_items"],
                      "LIBERO-PRO start frames": facts["benchmark_items"],
                      "peak GPU memory (GiB)": facts["gpu_memory_peak_gib"]}
    return pd.DataFrame.from_dict(rows, orient="index").T


TALL_OBJECTS = ("bbq_sauce_1", "ketchup_1", "milk_1", "orange_juice_1", "salad_dressing_1")
HEIGHT_MARKS_M = (0.12, 0.09, 0.06, 0.04, 0.02)


def heights_when_near(ee_pos: np.ndarray, point: np.ndarray, marks: Sequence[float] = HEIGHT_MARKS_M) -> np.ndarray:
    """Height (m) of the gripper above `point` at the first step it is within each horizontal distance of it;
    nan for a distance it never comes within. `point` is (3,), or (T, 3) for a point that moves."""
    relative = np.asarray(ee_pos, dtype=float) - np.asarray(point, dtype=float)
    horizontal = np.hypot(relative[:, 0], relative[:, 1])
    heights = []
    for mark in marks:
        inside = np.flatnonzero(horizontal <= mark)
        heights.append(relative[inside[0], 2] if len(inside) else np.nan)
    return np.array(heights)


def approach_heights(log: RolloutLog, offsets: ContactOffsets) -> np.ndarray:
    """heights_when_near for the named object's grasp point where the object is at each step, over the steps
    before the gripper first touches it (all steps if it never does); all nan if it touches it at once."""
    k = log.target_index
    touched = np.asarray(log.touched)[:, k]
    if touched[0]:
        return np.full(len(HEIGHT_MARKS_M), np.nan)
    end = int(np.argmax(touched)) if touched.any() else len(log.ee_pos) - 1
    points = np.asarray(log.object_pos[:end + 1, k], dtype=float) + offsets.grasp[log.target]
    return heights_when_near(log.ee_pos[:end + 1], points, HEIGHT_MARKS_M)


def approach_height_table(groups: dict[str, list[RolloutLog]], offsets: ContactOffsets) -> pd.DataFrame:
    """Per group of rollouts: median height (cm) above the grasp point when the gripper first comes within
    each horizontal distance of it, over the rollouts that came that close; and how many came within the
    smallest distance (a median over few rollouts is a median over the approaches that got there)."""
    rows = {}
    for name, logs in groups.items():
        heights = 100.0 * np.array([approach_heights(log, offsets) for log in logs])
        heights = heights.reshape(-1, len(HEIGHT_MARKS_M))
        reached = np.isfinite(heights)
        medians = [round(float(np.median(heights[reached[:, k], k])), 1) if reached[:, k].any() else np.nan
                   for k in range(len(HEIGHT_MARKS_M))]
        closest = f"came within {100 * HEIGHT_MARKS_M[-1]:.0f} cm"
        rows[name] = {"rollouts": len(logs), closest: int(reached[:, -1].sum())}
        rows[name] |= {f"within {100 * mark:.0f} cm": value for mark, value in zip(HEIGHT_MARKS_M, medians)}
    return pd.DataFrame.from_dict(rows, orient="index")


def retarget_figure(path: np.ndarray, new_contact: np.ndarray, rules: dict[str, Retarget], horizon: int = 8,
                    max_step: float = MAX_STEP_M) -> plt.Figure:
    """The demonstrated approach and the same approach moved to `new_contact` under each rule, from above and
    from the side (height against horizontal distance to the contact it heads for). Dots: the next
    `horizon` steps the rewritten action labels ask for."""
    path = np.asarray(path, dtype=float)
    colours = plt.get_cmap("tab10").colors
    fig, (top, side) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    curves = [("demonstration, to its own contact", path, path[-1], None, "black")] + [
        (f"moved: {name}", retarget_path(path, new_contact, rule), np.asarray(new_contact, dtype=float), rule,
         colours[k % len(colours)]) for k, (name, rule) in enumerate(rules.items())]
    for label, curve, end, rule, colour in curves:
        relative = curve - end
        top.plot(curve[:, 1], curve[:, 0], color=colour, linewidth=2, label=label)
        side.plot(np.hypot(relative[:, 0], relative[:, 1]), relative[:, 2], color=colour, linewidth=2, label=label)
        if rule is not None:
            chunk = counterfactual_chunk(path, end, horizon, flip_remaining=0.0, max_step=max_step, retarget=rule)
            near = chunk.points - end
            top.scatter(chunk.points[:, 1], chunk.points[:, 0], color=colour, s=18, zorder=3)
            side.scatter(np.hypot(near[:, 0], near[:, 1]), near[:, 2], color=colour, s=18, zorder=3)
        top.scatter(end[1], end[0], marker="*", s=200, color=colour, edgecolor="black", zorder=4)
    top.scatter(path[0, 1], path[0, 0], marker="x", color="black", s=70, zorder=5)
    top.set(xlabel="world y (m)", ylabel="world x (m)",
            title="from above, axes flipped as in the camera image (× gripper now, ★ contact)")
    top.set_aspect("equal", adjustable="datalim")
    top.invert_xaxis()
    top.invert_yaxis()
    side.set(xlabel="horizontal distance to the contact it heads for (m), far to near", ylabel="height above it (m)",
             title="from the side: where the gripper comes down")
    side.invert_xaxis()
    side.legend(fontsize=8)
    fig.tight_layout()
    return fig
