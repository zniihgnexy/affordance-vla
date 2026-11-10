"""Figures and tables for report/2026-11-10_vla_pipeline_demo.ipynb: data, model, open loop, closed loop.

The label side of the demo (CAW plan, plan–action consistency, episode state) is in
vla_pipeline_demo_labels.py. Inputs come from scripts/remote/probe_vla_demo.py (probe.npz / probe.json)
and scripts/remote/rollout_libero_logged.py (one .npz per episode).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance3d.caw_labels import GripperEvent
from affordance3d.rollout_log import RolloutLog, load_rollout_log
from affordance3d.rollout_outcome import NO_GRASP, NOT_DELIVERED, SUCCESS, WRONG_OBJECT, classify

ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
XYZ_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c")
GRASP_COLOUR, RELEASE_COLOUR = "#2e7d32", "#c62828"
_TAB10 = plt.get_cmap("tab10").colors
VIDEO_PANEL_PX, VIDEO_BAR_PX = 384, 48  # both multiples of 8, as the H.264 writer needs
CONTROL_HZ = 20  # LIBERO's control rate
OUTCOME_STYLE = {SUCCESS: "background-color:#c8e6c9", WRONG_OBJECT: "background-color:#ffcdd2",
                 NO_GRASP: "background-color:#ffe0b2", NOT_DELIVERED: "background-color:#fff9c4"}


def pretty(name: str) -> str:
    """'salad_dressing_1' → 'salad dressing'."""
    return name.rsplit("_", 1)[0].replace("_", " ") if name.rsplit("_", 1)[-1].isdigit() else name.replace("_", " ")


def object_colour(names: tuple[str, ...], name: str) -> tuple[float, float, float]:
    return _TAB10[sorted(names).index(name) % len(_TAB10)]


@dataclass(frozen=True)
class Probe:
    """Results of probe_vla_demo.py for one demonstration."""

    facts: dict
    query_steps: np.ndarray  # (Q,) steps of the demonstration at which a chunk was requested
    instructions: tuple[str, ...]  # the true instruction first, then instructions naming other objects
    true_chunks: np.ndarray  # (Q, H, 7) demonstrated actions, gripper 1 = open
    predicted: np.ndarray  # (I, Q, H, 7) model chunks per instruction
    trace: np.ndarray  # (S, steps + 1, H, 7) one chunk after each Euler step, for S noise seeds
    trace_step: int
    flow_loss: np.ndarray  # (L, Q) flow-matching loss at fixed noise levels
    one_step: np.ndarray  # (L, Q, H, 7) one-step action estimates at those levels


def load_probe(folder: Path) -> Probe:
    facts = json.loads((Path(folder) / "probe.json").read_text())
    with np.load(Path(folder) / "probe.npz") as data:
        return Probe(facts=facts, query_steps=data["query_steps"],
                     instructions=tuple(str(i) for i in data["instructions"]), true_chunks=data["true_chunks"],
                     predicted=data["predicted"], trace=data["trace"], trace_step=int(data["trace_step"]),
                     flow_loss=data["flow_loss"], one_step=data["one_step"])


def query_row(probe: Probe, step: int) -> int:
    """Row of the probe's arrays that holds the chunk requested at `step` of the demonstration."""
    rows = np.flatnonzero(probe.query_steps == step)
    if len(rows) != 1:
        raise ValueError(f"no chunk was requested at step {step}; the query steps are {probe.query_steps.tolist()}")
    return int(rows[0])


def load_logs(folder: Path, tag: str, images: bool = True) -> list[RolloutLog]:
    """All episodes of one condition, in episode order."""
    paths = sorted(Path(folder).glob(f"{tag}_ep*.npz"), key=lambda p: int(p.stem.rsplit("_ep", 1)[1]))
    if not paths:
        raise FileNotFoundError(f"no {tag}_ep*.npz in {folder}; run scripts/remote/run_vla_demo_gpu.sh first")
    return [load_rollout_log(path, images=images) for path in paths]


def _mark_events(axis, events: list[GripperEvent], style: str, label: str) -> None:
    for k, event in enumerate(events):
        axis.axvline(event.step, color=GRASP_COLOUR if event.closing else RELEASE_COLOUR, linestyle=style,
                     linewidth=1.3, label=f"{label}: {'grasp' if event.closing else 'release'}" if k < 2 else None)


# --------------------------------------------------------------------------- 1 · the training sample


def sample_frames(log: RolloutLog, steps: list[int]) -> list[tuple[np.ndarray, str]]:
    """Agent view above wrist view at each step: the two images the policy gets."""
    return [(np.vstack([log.agentview[step], log.wrist[step]]), f"step {step}") for step in steps]


def actions_figure(log: RolloutLog, command: list[GripperEvent], contact: list[GripperEvent]) -> plt.Figure:
    """The 7 action dimensions of one episode, with gripper-command flips and simulator grasp contacts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), gridspec_kw={"width_ratios": [2, 2, 1.4]})
    for k in range(3):
        axes[0].plot(log.actions[:, k], color=XYZ_COLOURS[k], label=ACTION_NAMES[k])
        axes[1].plot(log.actions[:, 3 + k], color=XYZ_COLOURS[k], label=ACTION_NAMES[3 + k])
    axes[2].step(np.arange(log.steps), log.actions[:, 6], where="post", color="#333", label="command")
    axes[0].set(title="Translation command (−1 … 1 per step)", xlabel="step", ylim=(-1.05, 1.05))
    axes[1].set(title="Rotation command", xlabel="step")
    axes[2].set(title="Gripper command (−1 open, +1 close)", xlabel="step", ylim=(-1.3, 1.3))
    for axis in axes:
        _mark_events(axis, command, ":", "command flips")
        _mark_events(axis, contact, "-", "contact")
        axis.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig


def contract_table(probe: Probe, log: RolloutLog) -> pd.DataFrame:
    facts = probe.facts
    rows = [
        ("input", "images", f"2 views (agent, wrist), {log.agentview.shape[1]} × {log.agentview.shape[2]} RGB, "
                            f"resized to 224 × 224 → {facts['tokens_image'] // 2} tokens each"),
        ("input", "instruction", f"“{log.meta['instruction']}” inside a fixed prompt → {facts['tokens_text']} "
                                 "text tokens"),
        ("input", "robot state", "not used by this checkpoint (images and text only)"),
        ("output", "action chunk", f"{facts['action_horizon']} steps × {facts['action_dim']} numbers: "
                                   "dx, dy, dz, droll, dpitch, dyaw, gripper"),
        ("output", "units", "end-effector deltas in controller units (−1 … 1), min–max normalised for training; "
                            "gripper 1 = open, 0 = close"),
        ("control", "rate", f"{CONTROL_HZ} Hz; one chunk = {facts['action_horizon'] / CONTROL_HZ:.1f} s, executed "
                            "fully before the next query"),
    ]
    return pd.DataFrame(rows, columns=["", "item", "in this checkpoint"]).set_index(["", "item"])


# --------------------------------------------------------------------------- 2 · inside the VLA


def model_facts_table(probe: Probe) -> pd.DataFrame:
    facts, params = probe.facts, probe.facts["parameters_millions"]
    layers, seq, width = facts["hidden_layers_used"], facts["hidden_state_shape"][1], facts["hidden_state_shape"][2]
    rows = [
        ("VLM (Qwen3-VL-4B)", f"{params['qwen_vl_interface'] / 1000:.2f} B parameters"),
        ("action head (layer-wise cross-DiT)", f"{params['action_model'] / 1000:.2f} B parameters · "
                                                f"{facts['dit_layers']} blocks × width {facts['dit_width']}"),
        ("total", f"{params['total'] / 1000:.2f} B parameters · {facts['weight_dtype'].replace('torch.', '')} · "
                  f"{facts['gpu_memory_after_load_gib']:.1f} GiB on the card "
                  f"({facts['gpu_memory_peak_gib']:.1f} GiB peak)"),
        ("tokens per observation", f"{facts['tokens_total']} = {facts['tokens_image']} image + "
                                   f"{facts['tokens_text']} text"),
        ("what the head reads", f"hidden states of {layers} VLM layers, each {seq} × {width}; block k attends to "
                                "layer k"),
        ("what the head denoises", f"{facts['future_tokens']} learned tokens + {facts['action_horizon']} action "
                                   "tokens"),
        ("sampling", f"{facts['inference_steps']} Euler steps from Gaussian noise"),
        ("time per policy call", f"{facts['ms_policy_call'] / 1000:.1f} s end to end; the model part is VLM "
                                 f"{facts['ms_vlm']:.0f} ms + {facts['inference_steps']} × "
                                 f"{facts['ms_dit_step']:.0f} ms (card shared with two other jobs)"),
        ("load time", f"{facts['load_seconds']:.0f} s from the NAS"),
    ]
    return pd.DataFrame(rows, columns=["", "measured"]).set_index("")


def sampling_figure(probe: Probe) -> plt.Figure:
    """Noise → action chunk over the Euler steps, for several noise seeds, at one frame."""
    trace = probe.trace
    seeds, n_steps = trace.shape[0], trace.shape[1] - 1
    # the demonstrated chunk is only known if the trace step is also one of the open-loop query steps
    row = query_row(probe, probe.trace_step) if probe.trace_step in probe.query_steps else None
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for axis, dim in zip(axes[:3], (0, 2, 6)):
        for seed in range(seeds):
            axis.plot(range(n_steps + 1), trace[seed, :, 0, dim], color="#888", alpha=0.7, marker="o", markersize=3)
        if row is not None:
            axis.axhline(probe.true_chunks[row, 0, dim], color="black", linestyle="--", label="demonstrated")
        axis.set(title=f"first action of the chunk: {ACTION_NAMES[dim]}", xlabel="Euler step (0 = noise)")
        axis.set_xticks(range(n_steps + 1))
        axis.legend(fontsize=8)
    for k in range(3):
        for seed in range(seeds):
            axes[3].plot(trace[seed, -1, :, k], color=XYZ_COLOURS[k], alpha=0.5, linewidth=1)
        if row is not None:
            axes[3].plot(probe.true_chunks[row, :, k], color=XYZ_COLOURS[k], linestyle="--", linewidth=2,
                         label=f"{ACTION_NAMES[k]} demonstrated")
    axes[3].set(title=f"final chunk, {seeds} seeds (thin) vs demonstration", xlabel="step within the chunk")
    axes[3].legend(fontsize=7)
    fig.suptitle(f"Flow sampling at step {probe.trace_step} of the demonstration ({seeds} noise seeds)", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 3 · open loop


def open_loop_figure(probe: Probe, events: list[GripperEvent]) -> plt.Figure:
    """Demonstrated actions (black) and the chunks the model predicts from the same frames (colour)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    horizon = probe.true_chunks.shape[1]
    demo = {step + k: probe.true_chunks[q, k] for q, step in enumerate(probe.query_steps) for k in range(horizon)}
    steps = sorted(demo)
    for axis, dim in zip(axes, (0, 1, 2, 6)):
        axis.plot(steps, [demo[s][dim] for s in steps], color="black", linewidth=1.6, label="demonstrated")
        for q, step in enumerate(probe.query_steps):
            axis.plot(np.arange(step, step + horizon), probe.predicted[0, q, :, dim], color="#d62728", alpha=0.55,
                      linewidth=1, label="predicted chunks" if q == 0 else None)
        _mark_events(axis, events, "-", "contact")
        axis.set(title=ACTION_NAMES[dim] + (" (1 = open)" if dim == 6 else ""), xlabel="step")
        axis.legend(fontsize=7)
    fig.tight_layout()
    return fig


def open_loop_errors(probe: Probe) -> pd.DataFrame:
    """Mean absolute error of the predicted chunks against the demonstration, per action dimension."""
    error = np.abs(probe.predicted[0] - probe.true_chunks).mean(axis=(0, 1))
    spread = probe.trace[:, -1].std(axis=0).mean(axis=0)
    table = pd.DataFrame({"mean |predicted − demonstrated|": error, "std over noise seeds": spread},
                         index=ACTION_NAMES)
    return table.T.round(4)


def flow_objective_figure(probe: Probe) -> plt.Figure:
    """The training loss and the one-step action estimate at fixed noise levels."""
    levels = np.asarray(probe.facts["noise_levels"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
    axes[0].errorbar(levels, probe.flow_loss.mean(axis=1), yerr=probe.flow_loss.std(axis=1), marker="o", color="#333",
                     capsize=3)
    axes[0].set(title="Flow-matching loss  mean (v_pred − (action − noise))²",
                xlabel="flow time t (0 = noise, 1 = action)", ylabel="loss (normalised units)", yscale="log")
    one_step = np.abs(probe.one_step[..., :3] - probe.true_chunks[None, ..., :3]).mean(axis=(1, 2, 3))
    sampled = np.abs(probe.predicted[0, ..., :3] - probe.true_chunks[..., :3]).mean()
    axes[1].plot(levels, one_step, marker="o", color="#1f77b4", label="one-step estimate  x_t + (1 − t)·v_pred")
    axes[1].axhline(sampled, color="#d62728", linestyle="--", label=f"{probe.facts['inference_steps']}-step sample")
    axes[1].set(title="Translation error against the demonstration", xlabel="flow time t",
                ylabel="mean |error| (action units)", ylim=(0, max(one_step.max(), sampled) * 1.6))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def named_object(instruction: str, names: tuple[str, ...]) -> str:
    """The object a 'pick up the X and place it in the basket' instruction names."""
    matches = [name for name in names if f"pick up the {pretty(name)} " in instruction.lower()]
    if len(matches) != 1:
        raise ValueError(f"cannot tell which of {names} the instruction {instruction!r} names")
    return matches[0]


def draw_top_view(axis, log: RolloutLog, step: int = 0, label: bool = True) -> None:
    """Objects at one step seen from above, oriented like the policy's agent view (camera at the bottom)."""
    for n, name in enumerate(log.names):
        x, y = log.object_pos[step, n, :2]
        basket = name.startswith("basket")
        axis.scatter(y, x, s=420 if basket else 130, marker="s" if basket else "o",
                     color=object_colour(log.names, name), alpha=0.35 if basket else 0.9, edgecolor="black",
                     linewidth=0.6, zorder=2)
        if label:
            axis.annotate(pretty(name), (y, x), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    axis.set(xlabel="world y (m)", ylabel="world x (m)")
    axis.set_aspect("equal", adjustable="datalim")
    if not axis.xaxis_inverted():
        axis.invert_xaxis()
        axis.invert_yaxis()


def language_figure(probe: Probe, log: RolloutLog, scale: float, steps: tuple[int, ...] = (0, 8, 16)) -> plt.Figure:
    """Where the first chunks point under each instruction: same frames, only the object name changes."""
    fig, axes = plt.subplots(1, len(steps), figsize=(5.4 * len(steps), 5.0), squeeze=False)
    for axis, step in zip(axes[0], steps):
        q = query_row(probe, step)
        draw_top_view(axis, log, step)
        start = log.ee_pos[step, :2]
        axis.scatter(start[1], start[0], marker="x", color="black", s=60, zorder=4)
        for k, instruction in enumerate(probe.instructions):
            name = named_object(instruction, log.names)
            path = start + scale * np.cumsum(probe.predicted[k, q, :, :2], axis=0)
            axis.plot(np.r_[start[1], path[:, 1]], np.r_[start[0], path[:, 0]], color=object_colour(log.names, name),
                      linewidth=3 if k == 0 else 2, zorder=3,
                      label=f"“… the {pretty(name)} …”" + (" (trained)" if k == 0 else ""))
        axis.set_title(f"chunk requested at step {step} (× = gripper)", fontsize=10)
    axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), title="instruction",
                       title_fontsize=8)
    fig.tight_layout()
    return fig


def language_table(probe: Probe, log: RolloutLog, before_step: int) -> pd.DataFrame:
    """How far each instruction's chunks are from the trained instruction's, on frames before `before_step`."""
    rows = probe.query_steps < before_step
    table = []
    for k, instruction in enumerate(probe.instructions):
        difference = np.abs(probe.predicted[k, rows, :, :3] - probe.predicted[0, rows, :, :3]).mean()
        demo = np.abs(probe.predicted[k, rows, :, :3] - probe.true_chunks[rows, :, :3]).mean()
        name = pretty(named_object(instruction, log.names))
        table.append({"instruction names": name + (" (trained)" if k == 0 else ""),
                      "|Δ| vs trained instruction": difference, "|Δ| vs demonstration": demo})
    return pd.DataFrame(table).set_index("instruction names").round(3)


# --------------------------------------------------------------------------- 4 · closed loop


def delivered_objects(log: RolloutLog) -> list[str]:
    return [name for n, name in enumerate(log.names) if log.in_goal[-1, n]]


def condition_of(log: RolloutLog) -> str:
    if log.meta["shifts"]:
        (dx, dy), = log.meta["shifts"].values()
        return f"target shifted {np.hypot(dx, dy) * 100:.0f} cm"
    if log.meta["instruction"] != log.meta["task"]:
        return f"asked for the {pretty(log.target)}"
    return "standard"


def outcome_table(logs: list[RolloutLog]) -> pd.DataFrame:
    """One row per rollout: what was asked, what the simulator says, and the rule-based outcome label."""
    rows = []
    for log in logs:
        outcome = classify(log.names, log.grasped, log.target, log.delivered)
        rows.append({"condition": condition_of(log), "start": log.meta["episode"], "asked for": pretty(log.target),
                     "steps": log.steps,
                     "first grasped": pretty(outcome.first_grasped) if outcome.first_grasped else "–",
                     "in basket at the end": ", ".join(pretty(n) for n in delivered_objects(log)) or "–",
                     "outcome": outcome.label})
    return pd.DataFrame(rows)


def outcome_summary(logs: list[RolloutLog]) -> pd.DataFrame:
    """Counts of each outcome label per condition."""
    table = outcome_table(logs)
    order = list(dict.fromkeys(table["condition"]))
    return table.groupby("condition")["outcome"].value_counts().unstack(fill_value=0).reindex(order)


def style_outcomes(table: pd.DataFrame):
    return table.style.map(lambda value: OUTCOME_STYLE.get(value, ""), subset=["outcome"]).hide(axis="index")


def rollout_frames(log: RolloutLog, n: int = 6) -> list[tuple[np.ndarray, str]]:
    steps = np.linspace(0, log.steps, n).astype(int)
    return [(log.agentview[step], f"step {step}") for step in steps]


def render_rollout_grid(logs: list[RolloutLog], captions: list[str], out_path: Path, fps: int = 20) -> Path:
    """Several rollouts playing side by side (agent view); shorter ones hold their last frame."""
    import cv2
    import imageio_ffmpeg

    size = (VIDEO_PANEL_PX * len(logs), VIDEO_PANEL_PX + VIDEO_BAR_PX)
    writer = imageio_ffmpeg.write_frames(str(out_path), size, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    writer.send(None)
    for t in range(max(log.steps for log in logs) + 1):
        canvas = np.full((size[1], size[0], 3), 255, np.uint8)
        for k, (log, caption) in enumerate(zip(logs, captions)):
            step = min(t, log.steps)
            left = k * VIDEO_PANEL_PX
            canvas[:VIDEO_PANEL_PX, left:left + VIDEO_PANEL_PX] = cv2.resize(
                log.agentview[step], (VIDEO_PANEL_PX, VIDEO_PANEL_PX), interpolation=cv2.INTER_LINEAR)
            held = [pretty(name) for n, name in enumerate(log.names) if log.grasped[step, n]]
            status = f"step {step:3d}/{log.steps}  holding: {held[0] if held else '-'}"
            for text, row, grey in ((caption, 18, 20), (status, 40, 90)):
                cv2.putText(canvas, text, (left + 6, VIDEO_PANEL_PX + row), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (grey, grey, grey), 1, cv2.LINE_AA)
        writer.send(np.ascontiguousarray(canvas))
    writer.close()
    return out_path


def paths_figure(groups: dict[str, list[RolloutLog]], layout: RolloutLog) -> plt.Figure:
    """End-effector paths from above, one panel per condition; ★ marks where the object named by the instruction is.

    The objects are drawn at their positions in `layout` (an unperturbed start), so a shifted target shows
    as a ★ away from its usual place.
    """
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 5.0), squeeze=False)
    for axis, (title, logs) in zip(axes[0], groups.items()):
        draw_top_view(axis, layout, 0)
        for log in logs:
            outcome = classify(log.names, log.grasped, log.target, log.delivered)
            colour = GRASP_COLOUR if outcome.label == SUCCESS else RELEASE_COLOUR
            axis.plot(log.ee_pos[:, 1], log.ee_pos[:, 0], color=colour, linewidth=1.6, alpha=0.8, zorder=3)
            target = log.object_pos[0, log.target_index, :2]
            axis.scatter(target[1], target[0], marker="*", s=260, color="gold", edgecolor="black", zorder=5)
        axis.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
