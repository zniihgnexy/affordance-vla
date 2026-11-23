"""Videos, figures and status tables for report/2026-11-16_weekly_review.ipynb (week 2: 16-22 Nov 2026).

The demos are real closed-loop rollouts of LIBERO-PRO that kept their camera images (episodes 0 and 1 of each
task). The plan drawn into a frame is the one the policy was sent for that step: the grid plan head's
contact, held until the gripper has carried out its event.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance_vla.analysis.grasp_orientation import close_onsets, grasp_attempts
from affordance_vla.analysis.semantic_probe import MIN_COVER, object_at_peak, object_means
from affordance_vla.heads.plan_grid import CELL_PX, CELLS, IMAGE_SIZE, Camera, camera_of_log, contact_pixel
from affordance_vla.plan.plan_input import plan_contact
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log
from report.vla_pipeline_demo_helpers import pretty

PANEL_PX = 320  # side of one camera panel in the videos
BAR_PX = 64  # caption bar under the panels
ORANGE = (237, 125, 49)
STEPS_AFTER_GRASP = 12
# the released checkpoint on LIBERO-PRO LIBERO-10; the second run redid tasks 2 and 3 of libero_10_task, whose
# first object of interest is a fixture (fixed in rollout_libero_logged.py on 20 Nov)
LIBERO10_RUNS = ("libero10_released", "libero10_released_task23")


def plan_contacts(log: RolloutLog) -> np.ndarray:
    """(T + 1, 3) world contact of the plan in force at every step; NaN where no grasp or release plan is held."""
    contacts = np.full((log.steps + 1, 3), np.nan)
    with np.load(log.path) as data:
        plans = data["plans"]
    for k, step in enumerate(log.chunk_steps):
        end = int(log.chunk_steps[k + 1]) if k + 1 < len(log.chunk_steps) else log.steps + 1
        contact = plan_contact(plans[k], log.ee_pos[int(step)])
        if contact is not None:
            contacts[int(step):end] = contact
    return contacts


def _panel(frame: np.ndarray, contact: np.ndarray | None, camera: Camera | None) -> np.ndarray:
    """A camera frame at PANEL_PX with the 8 x 8 token cells, the planned cell shaded and the contact crossed."""
    import cv2

    image = cv2.resize(frame, (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_LINEAR)
    if camera is None:
        return image
    scale = PANEL_PX / IMAGE_SIZE
    grid = image.copy()
    for k in range(1, int(IMAGE_SIZE / CELL_PX)):
        at = int(round(k * CELL_PX * scale))
        cv2.line(grid, (at, 0), (at, PANEL_PX), (255, 255, 255), 1)
        cv2.line(grid, (0, at), (PANEL_PX, at), (255, 255, 255), 1)
    image = cv2.addWeighted(grid, 0.35, image, 0.65, 0)
    if contact is not None and np.isfinite(contact).all():
        u, v, _ = contact_pixel(contact, camera)
        if 0 <= u < IMAGE_SIZE and 0 <= v < IMAGE_SIZE:
            col, row = int(u // CELL_PX), int(v // CELL_PX)
            top_left = (int(col * CELL_PX * scale), int(row * CELL_PX * scale))
            bottom_right = (int((col + 1) * CELL_PX * scale), int((row + 1) * CELL_PX * scale))
            shade = image.copy()
            cv2.rectangle(shade, top_left, bottom_right, ORANGE, -1)
            image = cv2.addWeighted(shade, 0.35, image, 0.65, 0)
            cv2.drawMarker(image, (int(u * scale), int(v * scale)), ORANGE, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return image


def _holding(log: RolloutLog, step: int) -> str:
    held = [pretty(name) for n, name in enumerate(log.names) if log.grasped[step, n]]
    return held[0] if held else "-"


def plan_video(log: RolloutLog, out_path: Path, fps: int = 20) -> Path:
    """Agent view with the held plan drawn in, and the wrist view, for a whole rollout."""
    import cv2
    import imageio_ffmpeg

    camera = camera_of_log(log)
    contacts = plan_contacts(log)
    size = (2 * PANEL_PX, PANEL_PX + BAR_PX)
    writer = imageio_ffmpeg.write_frames(str(out_path), size, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    writer.send(None)
    try:
        for step in range(log.steps + 1):
            canvas = np.full((size[1], size[0], 3), 255, np.uint8)
            canvas[:PANEL_PX, :PANEL_PX] = _panel(log.agentview[step], contacts[step], camera)
            canvas[:PANEL_PX, PANEL_PX:] = _panel(log.wrist[step], None, None)
            lines = (log.meta["instruction"], f"step {step:3d}/{log.steps}   holding: {_holding(log, step)}   "
                     f"result: {'success' if log.meta['success'] else 'failure'}")
            for text, row in zip(lines, (22, 46)):
                cv2.putText(canvas, text, (8, PANEL_PX + row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1,
                            cv2.LINE_AA)
            writer.send(np.ascontiguousarray(canvas))
    finally:  # do not leave the encoder running if a frame fails
        writer.close()
    return out_path


def key_steps(log: RolloutLog) -> dict[str, int]:
    """First query, first close command, a little after it, and the last step."""
    steps = {"first query": int(log.chunk_steps[0]) if len(log.chunk_steps) else 0}
    onsets = close_onsets(log.actions)
    if len(onsets):
        steps["gripper closes"] = int(onsets[0])
        steps["0.6 s later"] = min(int(onsets[0]) + STEPS_AFTER_GRASP, log.steps)
    steps["last step"] = log.steps
    return steps


def plan_strip(log: RolloutLog) -> plt.Figure:
    """Agent view (plan drawn in) over wrist view at the key steps of a rollout."""
    camera = camera_of_log(log)
    contacts = plan_contacts(log)
    steps = key_steps(log)
    fig, axes = plt.subplots(2, len(steps), figsize=(3.2 * len(steps), 6.6), squeeze=False)
    for k, (label, step) in enumerate(steps.items()):
        axes[0, k].imshow(_panel(log.agentview[step], contacts[step], camera))
        axes[1, k].imshow(_panel(log.wrist[step], None, None))
        axes[0, k].set_title(f"{label} (step {step})", fontsize=9)
        for axis in axes[:, k]:
            axis.set_axis_off()
    fig.suptitle(f"{log.meta['instruction']}  ·  {'success' if log.meta['success'] else 'failure'}", fontsize=10)
    fig.tight_layout()
    return fig


def with_images(paths: list[Path]) -> list[Path]:
    keep = []
    for path in paths:
        with np.load(path) as data:
            if {"agentview", "wrist"} <= set(data.files):
                keep.append(path)
    return keep


def first_failure(run: Path, suites: list[str]) -> RolloutLog | None:
    """The first failed rollout with images whose first close near the named object did not hold it."""
    for suite in suites:
        for path in with_images(sorted((run / suite).glob("task*_ep[01].npz"))):
            log = load_rollout_log(path)
            if log.ee_quat is None:
                continue
            attempts = grasp_attempts(log.names, log.ee_pos, log.ee_quat, log.object_pos, log.grasped, log.actions)
            on_target = [a for a in attempts if a.obj == log.target]
            if not log.meta["success"] and on_target and not on_target[0].held:
                return log
    return None


def task_results(runs: list[Path], suite: str) -> pd.DataFrame:
    """Successes per task of one suite from the runs' task summaries; a later run only fills tasks the earlier lack."""
    rows = {}
    for run in runs:
        for path in sorted((run / suite).glob("task*_summary.json")):
            task = int(path.stem.split("_")[0].removeprefix("task"))
            if task not in rows:
                episodes = json.loads(path.read_text())
                rows[task] = {"instruction": episodes[0]["instruction"],
                              "successes": sum(bool(e["success"]) for e in episodes), "episodes": len(episodes)}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["instruction", "successes", "episodes"]).sort_index()
    table.index.name = "task"
    return table


LABEL_COLUMNS = {"oracle plan, position": ("oracle_plan_{}", "libero_object_swap"),
                 "oracle plan, task": ("oracle_plan_{}", "libero_object_task"),
                 "g1 plan kept, position": ("predicted_g1_hold_{}", "libero_object_swap"),
                 "g1 plan kept, task": ("predicted_g1_hold_{}", "libero_object_task")}


def label_runs_table(pro: Path, heads: dict[str, str]) -> pd.DataFrame:
    """Successes of action heads (run suffix s9, s10, ...) with oracle plans and with kept g1 map plans."""
    rows = {}
    for label, head in heads.items():
        rows[label] = {}
        for column, (pattern, suite) in LABEL_COLUMNS.items():
            table = task_results([pro / pattern.format(head)], suite)
            rows[label][column] = f"{table['successes'].sum()}/{table['episodes'].sum()}"
    return pd.DataFrame(rows).T


def label_summary(summary_path: Path) -> pd.DataFrame:
    """L: grasps, groups of closing axes and agreement per object category (build_interaction_labels.py)."""
    found = json.loads(summary_path.read_text())["categories"]
    rows = {category: {"grasps": entry["grasps"], "axis groups (20°)": entry["axis_groups"],
                       "largest group": round(entry["axis_groups"][0] / entry["grasps"], 2),
                       "contact share of surface": entry["contact_share"],
                       "axis agreement in contact": entry["agreement_in_contact"]}
            for category, entry in found.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "category"
    return table


def probe_table(summary_path: Path) -> pd.DataFrame | None:
    """E1: share of first frames whose similarity map peaks on the named object, best layer per variant."""
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = {variant: {"best layer": found["best_layer"], "peak on the named object": found["best_share"]}
            for variant, found in summary["maps"].items()}
    return pd.DataFrame(rows).T.astype({"best layer": int})


SUITE_GROUPS = {"LIBERO-Object (g1 trained)": "libero_object", "LIBERO-Spatial (g1 not trained)": "libero_spatial"}


def _head_outcome(probe: dict, i: int) -> str:
    k = object_at_peak(probe["head_scores"][i], probe["coverage"][i])
    if k is None:
        return "background"
    if k != probe["target"][i]:
        return "another object"
    error = probe["head_error_m"][i]
    return "named, no grasp point" if np.isnan(error) else "named, within 2 cm" if error < 0.02 else "named, over 2 cm"


def probe_head_table(probe_path: Path) -> pd.DataFrame:
    """E1 (1): where the g1 grid head's best first-frame cell lands, per LIBERO-PRO suite."""
    probe = dict(np.load(probe_path, allow_pickle=True))
    outcomes = [(str(suite), _head_outcome(probe, i)) for i, suite in enumerate(probe["suite"])]
    return pd.crosstab([s for s, _ in outcomes], [o for _, o in outcomes]).rename_axis(index="suite", columns=None)


def probe_object_table(probe_path: Path) -> pd.DataFrame:
    """E1 (2): how often the named object has the highest mean similarity among the objects in view.

    Best layer per variant and suite group, next to chance (1 / objects in view); 40 and 34 frames, so one
    standard deviation is about 6-8 points.
    """
    probe = np.load(probe_path, allow_pickle=True)
    maps, coverage, target = probe["maps"].astype(np.float32), probe["coverage"], probe["target"]
    rows = {}
    for group, prefix in SUITE_GROUPS.items():
        frames = np.flatnonzero(np.char.startswith(probe["suite"].astype(str), prefix))
        in_view = (coverage[frames].sum(axis=2) > 0).sum(axis=1)
        row = {"frames": len(frames), "chance": round(float((1 / in_view).mean()), 2)}
        for t, tokens in enumerate(probe["token_sets"]):
            for m, merge in enumerate(probe["merges"]):
                hits = [np.mean([np.argmax(object_means(maps[i, t, m, layer], coverage[i])) == target[i]
                                 for i in frames]) for layer in range(maps.shape[3])]
                best = int(np.argmax(hits))
                row[f"{tokens}/{merge}"] = f"{hits[best]:.2f} (layer {best})"
        rows[group] = row
    return pd.DataFrame(rows).T


def map_panels(probe_path: Path, picks: list[tuple[str, str]]) -> plt.Figure:
    """The g1 map on LIBERO-PRO first frames (E1 probe): softmax over the 8 x 8 cells, best cell, named object.

    `picks` are (suite, log file) pairs of runs/semantic_probe/e1/probe.npz. The frame is the agent view as the
    policy receives it, so cell (row, column) of the map covers the same part of the image.
    """
    import textwrap

    probe = dict(np.load(probe_path, allow_pickle=True))
    suites, files = probe["suite"].astype(str), probe["log_file"].astype(str)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 4.9), squeeze=False)
    for axis, (suite, name) in zip(axes[0], picks):
        found = np.flatnonzero((suites == suite) & (files == name))
        if len(found) == 0:
            raise KeyError(f"no first frame of {suite}/{name} in {probe_path}")
        i = int(found[0])
        side = probe["agentview"][i].shape[0]
        cell = side / CELLS
        scores = probe["head_scores"][i].astype(float)
        heat = np.exp(scores - scores.max())
        heat = (heat / heat.sum()).reshape(CELLS, CELLS)
        overlay = plt.get_cmap("plasma")(heat / heat.max())
        overlay[..., 3] = 0.7 * heat / heat.max()  # opacity follows the score: the scene stays visible
        axis.imshow(probe["agentview"][i])
        axis.imshow(overlay, interpolation="nearest", extent=(0, side, side, 0))
        named = probe["coverage"][i, probe["target"][i]].reshape(CELLS, CELLS) >= MIN_COVER
        for row, column in zip(*np.nonzero(named)):
            axis.add_patch(plt.Rectangle((column * cell, row * cell), cell, cell, fill=False,
                                         edgecolor=np.array(ORANGE) / 255, linewidth=1.2))
        best = int(np.argmax(scores))
        axis.add_patch(plt.Rectangle(((best % CELLS) * cell, (best // CELLS) * cell), cell, cell, fill=False,
                                     edgecolor="white", linewidth=2.5))
        instruction = textwrap.fill(str(probe["instruction"][i]), 40)
        axis.set_title(f"{suite} {name[:-4]}\n{instruction}\n"
                       f"g1: {_head_outcome(probe, i)}; best cell {heat.max():.2f}", fontsize=8.5)
        axis.set_axis_off()
    fig.tight_layout()
    return fig
