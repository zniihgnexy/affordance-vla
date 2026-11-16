"""Figures and tables for report/2026-11-10_vla_pipeline_demo.ipynb: CAW plan, consistency, episode state.

Everything here is computed from logs (simulator ground truth) or from labels derived from them. No
model predicts these quantities yet; the notebook marks them ORACLE.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from affordance_vla.plan.caw_labels import CawLabel, caw_labels, contact_events, fit_action_path, gripper_events
from affordance_vla.plan.episode_state import Slot, Status, observations_from_log, run_episode
from affordance_vla.plan.plan_consistency import ChunkConsistency, chunk_path, consistency_along_episode, plan_ahead
from affordance_vla.sim.libero_camera import project_points
from affordance_vla.sim.rollout_log import RolloutLog
from report.vla_pipeline_demo_helpers import (
    CONTROL_HZ,
    GRASP_COLOUR,
    RELEASE_COLOUR,
    Probe,
    draw_top_view,
    named_object,
    object_colour,
    pretty,
    query_row,
)

STEP_SECONDS = 1.0 / CONTROL_HZ
APPROACH_ARROW_M = 0.10
MIN_VISIBLE_PIXELS = 40  # of 256 × 256; below this an object counts as not seen
STATUS_ORDER = (Status.UNSEEN, Status.VISIBLE, Status.OCCLUDED, Status.GRASPED)
STATUS_COLOURS = {Status.UNSEEN: "#bdbdbd", Status.VISIBLE: "#66bb6a", Status.OCCLUDED: "#ffa726",
                  Status.GRASPED: "#e53935"}


# --------------------------------------------------------------------------- 5 · CAW plan labels


def demo_labels(log: RolloutLog) -> tuple[list[CawLabel], list[CawLabel], float]:
    """(labels from simulator contact, labels from the gripper command, metres per unit of translation action)."""
    contact = caw_labels(log.ee_pos, contact_events(log.grasped[:, log.target_index]))
    command = caw_labels(log.ee_pos, gripper_events(log.actions[:, 6]))
    return contact, command, fit_action_path(log.actions[:, :3], log.ee_pos[:-1]).scale


def event_timing_table(log: RolloutLog, contact: list[CawLabel], command: list[CawLabel]) -> pd.DataFrame:
    """Gripper-command flip (the proxy) against the simulator's grasp contact, per event."""
    if [c.event.closing for c in contact] != [c.event.closing for c in command]:
        raise ValueError("the command and the contact give different event sequences; compare them by hand")
    rows = []
    for by_contact, by_command in zip(contact, command):
        step, flip = by_contact.event.step, by_command.event.step
        to_object = np.linalg.norm(log.object_pos[step, log.target_index] - by_contact.contact)
        rows.append({
            "event": "grasp" if by_contact.event.closing else "release", "command flips at step": flip,
            "contact changes at step": step, "delay": f"{step - flip} steps = {(step - flip) * STEP_SECONDS:.2f} s",
            "gripper moved in between": f"{np.linalg.norm(by_contact.contact - by_command.contact) * 1000:.0f} mm",
            "gripper → object centre at contact": f"{to_object * 1000:.0f} mm",
            "approach elevation": f"{np.degrees(np.arcsin(np.clip(by_contact.approach[2], -1, 1))):.0f}°",
        })
    return pd.DataFrame(rows).set_index("event")


def _pixels(log: RolloutLog, points: np.ndarray) -> np.ndarray:
    """Pixel coordinates of world points in the image the policy sees."""
    return project_points(points, log.intrinsics, log.cam_to_world, width=log.agentview.shape[2], mirrored=True)[0]


def caw_overlay_figure(log: RolloutLog, labels: list[CawLabel], steps: list[int]) -> plt.Figure:
    """The plan still ahead at each step, drawn into the agent view: waypoints, contact point, approach direction."""
    fig, axes = plt.subplots(1, len(steps), figsize=(4.0 * len(steps), 4.3), squeeze=False)
    for axis, step in zip(axes[0], steps):
        axis.imshow(log.agentview[step])
        axis.set_axis_off()
        plan = plan_ahead(labels, log.ee_pos[step], step)
        if plan is None:
            axis.set_title(f"step {step}: no event ahead", fontsize=10)
            continue
        colour = GRASP_COLOUR if plan.closing else RELEASE_COLOUR
        line = _pixels(log, plan.polyline)
        axis.plot(line[:, 0], line[:, 1], color="white", linewidth=3.5)
        axis.plot(line[:, 0], line[:, 1], color=colour, linewidth=2, marker="o", markersize=5, markeredgecolor="white")
        contact = plan.polyline[-1]
        tail, tip = _pixels(log, np.stack([contact - APPROACH_ARROW_M * plan.approach, contact]))
        axis.annotate("", xy=tip, xytext=tail, arrowprops={"arrowstyle": "-|>", "color": "cyan", "lw": 3.0,
                                                           "mutation_scale": 18})
        axis.scatter(*line[-1], marker="X", s=140, color="red", edgecolor="white", zorder=5)
        axis.scatter(*line[0], marker="s", s=40, color="black", edgecolor="white", zorder=5)
        axis.set_title(f"step {step}: {'grasp' if plan.closing else 'release'} in {plan.steps_to_event} steps",
                       fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 6 · plan–action consistency


def _summarise(results: list[ChunkConsistency | None]) -> dict:
    """Means over the chunks that have a plan; the direction columns use only chunks that move enough to have one."""
    valid = [r for r in results if r is not None]
    if not valid:
        raise ValueError("no chunk has a plan ahead of it; check the query steps against the labels")
    cosines = np.array([r.direction_cosine for r in valid])
    cosines = cosines[~np.isnan(cosines)]
    return {"chunks with a plan": len(valid),
            "path deviation (mm, mean)": np.mean([r.path_deviation for r in valid]) * 1000,
            "chunks with a direction": len(cosines),
            "direction cosine (mean)": cosines.mean() if len(cosines) else np.nan,
            "of those, share with cosine > 0.9": np.mean(cosines > 0.9) if len(cosines) else np.nan}


def consistency_sources(probe: Probe, log: RolloutLog, labels: list[CawLabel],
                        scale: float) -> dict[str, list[ChunkConsistency | None]]:
    """Per source of action chunks, the consistency of each chunk with the plan ahead of its query step."""
    sources = {"demonstrated actions": probe.true_chunks[..., :3]}
    for k, instruction in enumerate(probe.instructions):
        name = pretty(named_object(instruction, log.names))
        label = "model · trained instruction" if k == 0 else f"model · asked for the {name}"
        sources[label] = probe.predicted[k, ..., :3]
    return {source: consistency_along_episode(labels, log.ee_pos, probe.query_steps, chunks, scale)
            for source, chunks in sources.items()}


def consistency_table(sources: dict[str, list[ChunkConsistency | None]]) -> pd.DataFrame:
    table = pd.DataFrame({source: _summarise(results) for source, results in sources.items()}).T
    return table.astype({"chunks with a plan": int, "chunks with a direction": int}).round(
        {"path deviation (mm, mean)": 1, "direction cosine (mean)": 3, "of those, share with cosine > 0.9": 2})


def consistency_figure(probe: Probe, log: RolloutLog, labels: list[CawLabel], scale: float,
                       sources: dict[str, list[ChunkConsistency | None]], step: int) -> plt.Figure:
    """Left: one query step from above (plan ahead, each source's chunk). Right: direction cosine over the episode."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1, 1.5]})
    q = query_row(probe, step)
    plan = plan_ahead(labels, log.ee_pos[step], step)
    if plan is None:
        raise ValueError(f"step {step} has no plan ahead of it (it lies after the last gripper event)")
    draw_top_view(axes[0], log, step)
    axes[0].plot(plan.polyline[:, 1], plan.polyline[:, 0], color="black", linestyle="--", marker="o", markersize=5,
                 linewidth=1.4, zorder=3, label="plan ahead (waypoints → contact)")
    chunks = {"demonstrated actions": (probe.true_chunks[q, :, :3], "black")}
    for k, instruction in enumerate(probe.instructions):
        name = named_object(instruction, log.names)
        label = "model · trained instruction" if k == 0 else f"model · asked for the {pretty(name)}"
        chunks[label] = (probe.predicted[k, q, :, :3], object_colour(log.names, name))
    for label, (chunk, colour) in chunks.items():
        path = chunk_path(log.ee_pos[step], chunk, scale)
        axes[0].plot(path[:, 1], path[:, 0], color=colour, linewidth=2.6, zorder=4, label=label,
                     linestyle=":" if colour == "black" else "-")
    axes[0].set_title(f"chunk paths at step {step} (0.4 s each) · dashed: plan ahead · dotted: demonstrated · "
                      "colours as on the right", fontsize=9)
    for label, results in sources.items():
        cosine = [np.nan if r is None else r.direction_cosine for r in results]
        colour = chunks[label][1]
        axes[1].plot(probe.query_steps, cosine, color=colour, marker="o", markersize=3,
                     linewidth=2.6 if colour == "black" else 1.5, label=label)
    for lab in labels:
        axes[1].axvline(lab.event.step, color=GRASP_COLOUR if lab.event.closing else RELEASE_COLOUR, linewidth=1.2)
    axes[1].set(title="Direction cosine between each chunk and the plan ahead (vertical lines: grasp, release)",
                xlabel="step at which the chunk is requested", ylabel="cosine (1 = along the plan)", ylim=(-1.05, 1.05))
    axes[1].legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 7 · episode affordance state


def episode_state_history(log: RolloutLog, min_pixels: int = MIN_VISIBLE_PIXELS) -> list[tuple[Slot, ...]]:
    """The rule-based state after every logged step, fed with simulator positions, visibility and grasp contact."""
    return run_episode(log.names, observations_from_log(log.names, log.ee_pos, log.object_pos, log.visible_px,
                                                        log.grasped, min_pixels))


def state_key_steps(log: RolloutLog, before_grasp: int = 6) -> list[int]:
    """Four steps worth showing: the start, shortly before the target is first held, mid-hold, and the end."""
    held = np.flatnonzero(log.grasped[:, log.target_index])
    if len(held) == 0:  # the target was never held: spread the steps evenly
        return [int(step) for step in np.linspace(0, log.steps, 4)]
    return [0, max(int(held[0]) - before_grasp, 0), int(held[len(held) // 2]), log.steps]


def belief_errors(log: RolloutLog, history: list[tuple[Slot, ...]]) -> np.ndarray:
    """(T + 1, N) distance in metres between each slot's believed position and the true one; nan before first sight."""
    errors = np.full(log.object_pos.shape[:2], np.nan)
    for t, state in enumerate(history):
        for n, slot in enumerate(state):
            if slot.position is not None:
                errors[t, n] = np.linalg.norm(slot.position - log.object_pos[t, n])
    return errors


def state_timeline_figure(log: RolloutLog, history: list[tuple[Slot, ...]],
                          min_pixels: int = MIN_VISIBLE_PIXELS) -> plt.Figure:
    """Status of every slot over the episode, and the visible pixels that decide between seen and occluded."""
    codes = np.array([[STATUS_ORDER.index(slot.status) for slot in state] for state in history]).T
    fig, axes = plt.subplots(2, 1, figsize=(15, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    axes[0].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=len(STATUS_ORDER) - 1,
                   cmap=ListedColormap([STATUS_COLOURS[s] for s in STATUS_ORDER]))
    axes[0].set_yticks(range(len(log.names)), [pretty(n) + (" ★" if n == log.target else "") for n in log.names],
                       fontsize=9)
    axes[0].legend(handles=[Patch(color=STATUS_COLOURS[s], label=s.value) for s in STATUS_ORDER], ncol=4, fontsize=8,
                   loc="upper center", bbox_to_anchor=(0.5, 1.22))
    for n, name in enumerate(log.names):
        if not name.startswith("basket"):
            axes[1].plot(np.maximum(log.visible_px[:, n], 1), color=object_colour(log.names, name),
                         linewidth=2.4 if name == log.target else 1.2, label=pretty(name))
    axes[1].axhline(min_pixels, color="black", linestyle="--", linewidth=1, label=f"seen if ≥ {min_pixels} px")
    axes[1].set(xlabel="step", ylabel="visible pixels (agent view)", yscale="log")
    axes[1].legend(fontsize=7, ncol=4, loc="lower left")
    fig.tight_layout()
    return fig


def state_overlay_figure(log: RolloutLog, history: list[tuple[Slot, ...]], steps: list[int]) -> plt.Figure:
    """Believed positions drawn into the agent view: filled = seen now, hollow = remembered, red = in the gripper."""
    fig, axes = plt.subplots(1, len(steps), figsize=(4.0 * len(steps), 4.3), squeeze=False)
    for axis, step in zip(axes[0], steps):
        axis.imshow(log.agentview[step])
        axis.set_axis_off()
        for slot in history[step]:
            if slot.position is None:
                continue
            (u, v), = _pixels(log, slot.position[None])
            colour = STATUS_COLOURS[slot.status]
            axis.scatter(u, v, s=150, marker="o", facecolor=colour if slot.status is Status.VISIBLE else "none",
                         edgecolor="white" if slot.status is Status.VISIBLE else colour, linewidth=2.4, zorder=4)
            if slot.status is not Status.VISIBLE:
                axis.annotate(f"{pretty(slot.name)}: {slot.status.value}, {slot.steps_since_seen} steps", (u, v),
                              xytext=(8, -10), textcoords="offset points", fontsize=7, color="white",
                              bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5, "edgecolor": "none"})
        axis.set_title(f"step {step}", fontsize=10)
    fig.tight_layout()
    return fig


def state_summary_table(log: RolloutLog, history: list[tuple[Slot, ...]]) -> pd.DataFrame:
    """Per object: share of steps in each status, the longest time unseen, the largest belief error."""
    errors = belief_errors(log, history) * 1000
    rows = {}
    for n, name in enumerate(log.names):
        statuses = [state[n].status for state in history]
        rows[pretty(name) + (" ★" if name == log.target else "")] = {
            **{f"{status.value} (% of steps)": 100 * np.mean([s is status for s in statuses])
               for status in (Status.VISIBLE, Status.OCCLUDED, Status.GRASPED)},
            "longest unseen (steps)": max(state[n].steps_since_seen for state in history),
            "largest belief error (mm)": np.nanmax(errors[:, n]) if not np.isnan(errors[:, n]).all() else np.nan,
            "moved at the end": history[-1][n].moved,
        }
    return pd.DataFrame(rows).T.infer_objects().round(1)


def _longest_run(flags: np.ndarray) -> int:
    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def hidden_share_table(logs: list[RolloutLog], min_pixels: int = MIN_VISIBLE_PIXELS) -> pd.DataFrame:
    """Over several rollouts of one scene: how often each object is hidden from the agent view, and for how long."""
    rows = {}
    for n, name in enumerate(logs[0].names):
        hidden = [log.visible_px[:, n] < min_pixels for log in logs]
        rows[pretty(name)] = {"hidden (% of steps)": 100 * np.concatenate(hidden).mean(),
                              "longest hidden stretch (steps)": max(_longest_run(flags) for flags in hidden),
                              "rollouts with a hidden step": f"{sum(bool(f.any()) for f in hidden)} of {len(logs)}"}
    return pd.DataFrame(rows).T.infer_objects().round(1)
