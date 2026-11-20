"""Tables and figures for report/experiments/2026-11-10_vla_datasets_sim_report.ipynb (VLA branch)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAPABILITIES = ("Robot demos + actions", "Closed-loop eval", "Replay → depth / seg", "Clutter / distractors",
                "Occluded / moved target", "Headroom")
MARKS = {"Y": "✓", "~": "~", "-": "–"}  # provided / partial, unverified or proxy / not provided
MARK_STYLE = {"✓": "background-color:#2e7d32;color:white", "~": "background-color:#f9c74f;color:#333",
              "–": "background-color:#e0e0e0;color:#777"}


@dataclass(frozen=True)
class Benchmark:
    name: str
    kind: str
    marks: str  # one of Y ~ - per capability, in CAPABILITIES order
    role: str
    status: str


BENCHMARKS = (
    Benchmark("LIBERO-Plus", "sim · Franka · MuJoCo", "YY~YY~",
              "main claim (perturbation splits)", "not downloaded"),
    Benchmark("LIBERO-Occ", "sim · LIBERO + occluders", "~YY~YY", "diagnostic: occlusion → episode state",
              "not downloaded"),
    Benchmark("LIBERO-PRO", "sim · LIBERO + swaps", "~YY~~Y", "diagnostic: object / instruction swaps",
              "not downloaded"),
    Benchmark("LIBERO", "sim · Franka · MuJoCo", "YYY~--", "sanity check; label pipeline dev",
              "1 task file (50 demos) on Mac"),
    Benchmark("RoboTwin 2.0 (Hard)", "sim · dual-arm · SAPIEN", "YYYY~Y", "second benchmark", "not downloaded"),
    Benchmark("RoboCasa365", "sim · mobile kitchen · MuJoCo", "YY~Y~Y", "fallback second benchmark",
              "not downloaded"),
    Benchmark("SimplerEnv", "real-to-sim eval · SAPIEN", "-Y-~--", "not used (no sim demos)", "–"),
    Benchmark("CALVIN", "sim · PyBullet", "YY~---", "not used (saturated)", "–"),
    Benchmark("VLABench", "sim · MuJoCo", "YY~Y-~", "not used (repo idle)", "–"),
    Benchmark("ManiSkill3", "sim toolkit · SAPIEN", "~~Y~-~", "label tool / custom variants", "–"),
    Benchmark("DROID · BridgeData V2 · OXE", "real robot data", "Y-~Y--", "optional offline labels (S7)", "–"),
)


def capability_table(benchmarks: tuple[Benchmark, ...] = BENCHMARKS) -> pd.io.formats.style.Styler:
    rows = {b.name: {"type": b.kind, **{cap: MARKS[m] for cap, m in zip(CAPABILITIES, b.marks, strict=True)},
                     "role in plan": b.role, "local": b.status} for b in benchmarks}
    table = pd.DataFrame.from_dict(rows, orient="index")
    return (table.style.map(lambda v: MARK_STYLE.get(v, ""), subset=list(CAPABILITIES))
            .set_properties(subset=list(CAPABILITIES), **{"text-align": "center", "font-weight": "bold"})
            .set_properties(subset=["role in plan", "local"], **{"text-align": "left"}))


def first_frame(path: Path) -> np.ndarray:
    """RGB array of an image file; animated or WebP files give their first frame."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200_000_000  # official teasers (e.g. DROID, ~108 MP) exceed Pillow's default guard
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# --------------------------------------------------------------------------- LIBERO demo figures


def file_overview(path: Path) -> pd.DataFrame:
    """Shape and dtype of every array stored for the first demo, plus the demo-level attributes."""
    import h5py

    rows = []
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        group = f["data"][demos[0]]
        group.visititems(lambda name, item: rows.append(
            {"key": name, "shape (demo 0)": str(item.shape), "dtype": str(item.dtype)})
            if isinstance(item, h5py.Dataset) else None)
        rows += [{"key": f"@{name}", "shape (demo 0)": "attribute", "dtype": type(value).__name__}
                 for name, value in group.attrs.items()]
        lengths = [f["data"][d]["actions"].shape[0] for d in demos]
    table = pd.DataFrame(rows).set_index("key")
    table.attrs["summary"] = (f"{len(demos)} demos · {sum(lengths):,} steps · "
                              f"{min(lengths)}–{max(lengths)} steps per demo (mean {np.mean(lengths):.0f})")
    return table


def event_frames(demo, labels, before: int = 15) -> list[tuple[np.ndarray, str]]:
    """Agent-view frames at the start, just before / at each gripper event, and at the end."""
    picks = [(0, "start")]
    for label in labels:
        verb = "grasp" if label.event.closing else "release"
        picks += [(max(label.event.step - before, 0), f"{before} steps before {verb}"),
                  (label.event.step, f"{verb} (step {label.event.step})")]
    picks.append((len(demo.actions) - 1, "end"))
    return [(demo.agentview[step], caption) for step, caption in picks]


def render_demo_video(demo, labels, out_path: Path, fps: int = 20, scale: int = 3) -> Path:
    """Agent view and wrist view side by side, with the gripper command and the next event."""
    import cv2
    import imageio_ffmpeg

    steps = len(demo.actions)
    h, w = demo.agentview.shape[1:3]
    size = (2 * w * scale, h * scale + 40)
    writer = imageio_ffmpeg.write_frames(str(out_path), size, fps=fps, codec="libx264", quality=7,
                                         macro_block_size=8)
    writer.send(None)
    for t in range(steps):
        canvas = np.full((size[1], size[0], 3), 255, np.uint8)
        for k, view in enumerate((demo.agentview[t], demo.wrist[t])):
            canvas[:h * scale, k * w * scale:(k + 1) * w * scale] = cv2.resize(
                view, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        upcoming = next((lab for lab in labels if lab.event.step >= t), None)
        phase = ("→ grasp" if upcoming.event.closing else "→ release") if upcoming else "done"
        state = "closed" if demo.actions[t, -1] > 0 else "open"
        text = f"step {t:3d}/{steps - 1}   gripper command: {state}   phase: {phase}"
        cv2.putText(canvas, text, (10, h * scale + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1,
                    cv2.LINE_AA)
        for lab in labels:  # timeline ticks for the events
            x = int(lab.event.step / max(steps - 1, 1) * (size[0] - 1))
            cv2.line(canvas, (x, h * scale), (x, h * scale + 6), (0, 120, 0) if lab.event.closing else (200, 0, 0), 2)
        cv2.line(canvas, (int(t / max(steps - 1, 1) * (size[0] - 1)), h * scale),
                 (int(t / max(steps - 1, 1) * (size[0] - 1)), h * scale + 6), (0, 0, 0), 1)
        writer.send(np.ascontiguousarray(canvas))
    writer.close()
    return out_path


def path_3d_figure(demos, labels_per_demo, highlight: int = 0):
    """End-effector paths of all demos (grey) with one demo's CAW labels highlighted."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, demo in enumerate(demos):
        if i != highlight:
            fig.add_trace(go.Scatter3d(x=demo.ee_pos[:, 0], y=demo.ee_pos[:, 1], z=demo.ee_pos[:, 2], mode="lines",
                                       line={"color": "rgba(150,150,150,0.25)", "width": 2}, hoverinfo="skip",
                                       showlegend=i == (1 if highlight == 0 else 0), name="other demos"))
    demo = demos[highlight]
    fig.add_trace(go.Scatter3d(x=demo.ee_pos[:, 0], y=demo.ee_pos[:, 1], z=demo.ee_pos[:, 2], mode="lines",
                               line={"color": np.arange(len(demo.ee_pos)), "colorscale": "Viridis", "width": 6},
                               name=f"{demo.name} path (colour = time)"))
    for lab in labels_per_demo[highlight]:
        colour, verb = ("#2e7d32", "grasp") if lab.event.closing else ("#c62828", "release")
        fig.add_trace(go.Scatter3d(x=lab.waypoints[:, 0], y=lab.waypoints[:, 1], z=lab.waypoints[:, 2],
                                   mode="markers", marker={"size": 4, "color": colour, "symbol": "diamond"},
                                   name=f"{verb}: waypoints"))
        fig.add_trace(go.Scatter3d(x=[lab.contact[0]], y=[lab.contact[1]], z=[lab.contact[2]], mode="markers",
                                   marker={"size": 8, "color": colour},
                                   name=f"{verb}: contact (step {lab.event.step})"))
        fig.add_trace(go.Cone(x=[lab.contact[0]], y=[lab.contact[1]], z=[lab.contact[2]], u=[lab.approach[0]],
                              v=[lab.approach[1]], w=[lab.approach[2]], sizemode="absolute", sizeref=0.04,
                              anchor="tip", colorscale=[[0, colour], [1, colour]], showscale=False,
                              name=f"{verb}: approach direction", showlegend=True))
    fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 30, "b": 0}, legend={"x": 0.0, "y": 1.0},
                      scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "y (m)",
                             "zaxis_title": "z (m)"})
    return fig


def caw_spread_figure(labels_per_demo):
    """Where grasps and releases happen across demos, and from which direction the gripper arrives."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for closing, colour, verb in [(True, "#2e7d32", "grasp"), (False, "#c62828", "release")]:
        chosen = [lab for labels in labels_per_demo for lab in labels if lab.event.closing == closing]
        if not chosen:
            continue
        contacts = np.array([lab.contact for lab in chosen])
        approach = np.array([lab.approach for lab in chosen])
        axes[0].scatter(contacts[:, 0], contacts[:, 1], s=14, c=colour, label=f"{verb} ({len(chosen)})")
        elevation = np.degrees(np.arcsin(np.clip(approach[:, 2], -1, 1)))
        axes[1].hist(elevation, bins=18, range=(-90, 90), color=colour, alpha=0.6, label=verb)
        axes[2].hist([lab.event.step for lab in chosen], bins=15, color=colour, alpha=0.6, label=verb)
    axes[0].set(title="Contact proxies, top view", xlabel="x (m)", ylabel="y (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(title="Approach elevation (−90° = straight down)", xlabel="degrees", ylabel="events")
    axes[2].set(title="Event step within the demo", xlabel="step", ylabel="events")
    for axis in axes:
        axis.legend(fontsize=9)
    fig.tight_layout()
    return fig


def action_fit_figure(demo, fit, fits):
    """Does summing the translation actions reproduce the end-effector path?"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.0), gridspec_kw={"width_ratios": [2.2, 1]})
    for k, (axis_name, colour) in enumerate(zip("xyz", ("#1f77b4", "#ff7f0e", "#2ca02c"))):
        axes[0].plot(demo.ee_pos[:, k], color=colour, label=f"{axis_name}: recorded")
        axes[0].plot(fit.predicted[:, k], color=colour, linestyle="--", label=f"{axis_name}: Σ actions × s")
    axes[0].set(title=f"{demo.name}: recorded vs integrated actions (s = {fit.scale * 1000:.1f} mm per unit, "
                      f"R² = {fit.r2:.3f})", xlabel="step", ylabel="position (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].hist([f.r2 for f in fits], bins=20, color="#555")
    axes[1].set(title=f"R² over {len(fits)} demos (median {np.median([f.r2 for f in fits]):.3f})",
                xlabel="R²", ylabel="demos")
    fig.tight_layout()
    return fig
