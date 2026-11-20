"""Figures and tables for report/experiments/2026-11-11_affordance_pipeline_walkthrough.ipynb.

One closed-loop rollout that was logged with depth (run_plan_following_gpu.sh with DEPTH=1) is followed
step by step. What no model predicts yet is computed from the simulator and labelled ORACLE in the
notebook: the affordance map (affordance_vla/plan/affordance_map.py), the planned contact, the approach
direction and the episode state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from affordance_vla.analysis.rollout_outcome import SUCCESS, classify
from affordance_vla.plan.affordance_map import contact_region_map, map_point
from affordance_vla.plan.episode_state import Slot
from affordance_vla.plan.plan_input import FAR_SCALE_M, plan_features
from affordance_vla.sim.libero_camera import project_points, unproject_depth
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log
from report.vla_pipeline_demo_helpers import ACTION_NAMES, CONTROL_HZ, Probe, draw_top_view, pretty

GRASP_SIGMA_M = 0.03  # width of the map around a grasp contact (distance in 3D)
RELEASE_SIGMA_M = 0.04  # width of the map under a release point (distance in the ground plane)
MIN_OBJECT_PIXELS = 20  # an object showing fewer pixels gives no map point
MIN_MAP_PEAK = 0.05  # nor does a map that is weak everywhere: what is visible of the object is far from the contact
EVENT_COLUMN, APPROACH_COLUMNS = 9, slice(6, 9)
PLAN_NAMES = ("to contact x / 10 cm", "to contact y / 10 cm", "to contact z / 10 cm", "tanh(x / 2 cm)",
              "tanh(y / 2 cm)", "tanh(z / 2 cm)", "approach x", "approach y", "approach z",
              "event (+1 grasp, −1 release)")
ARROW_M = 0.10  # length of the approach arrow drawn into the image
HEAT_RGB, HEAT_ALPHA = np.array([255.0, 70.0, 0.0]), 0.75
PATH_RGB, CONTACT_RGB, POINT_RGB, GRIPPER_RGB = (0, 229, 255), (255, 255, 255), (255, 235, 59), (233, 30, 99)
ZOOM = 2  # overlays are drawn at twice the logged resolution
VIDEO_BAR_PX = 40  # keeps the frame height a multiple of 8


@dataclass(frozen=True)
class StepView:
    """One step of a rollout: the plan in force, and the ORACLE stand-ins for what is not predicted yet."""

    step: int
    query: int  # index of the query whose plan and chunk are in force at this step
    query_step: int  # step at which that query was made
    plan: np.ndarray  # (10,) plan input sent with that query
    event: str  # "grasp", "release" or "none"
    acts_on: str | None  # object the event acts on: the target for a grasp, the container for a release
    contact: np.ndarray | None  # (3,) where the gripper should be at the event
    approach: np.ndarray | None  # (3,) direction the gripper should arrive from
    points: np.ndarray  # (H, W, 3) world point seen in every pixel at this step
    heat: np.ndarray  # (H, W) affordance map at this step
    map_point: np.ndarray | None  # (3,) the point the map stands for; None if the object is hidden
    chunk: np.ndarray  # (H, 7) chunk returned for that query


def load_plans(log: RolloutLog) -> np.ndarray:
    """(C, 10) plan inputs sent with the queries of a rollout."""
    with np.load(log.path) as data:
        plans = data["plans"]
    if len(plans) != len(log.chunk_steps) or plans.size == 0:
        raise ValueError(f"{log.path.name} carries {len(plans)} plans for {len(log.chunk_steps)} queries")
    if log.depth is None:
        raise ValueError(f"{log.path.name} has no depth; log it with --depth")
    return plans


def view_at(log: RolloutLog, plans: np.ndarray, step: int, container: str) -> StepView:
    """Everything shown about one step. The plan and chunk are those of the latest query.

    Between two queries the contact stays where that plan put it, while the map is recomputed from the
    current image; at a query step (the tables and curves use only those) both belong to the same moment.
    """
    query = int(np.searchsorted(log.chunk_steps, step, side="right") - 1)
    if query < 0:
        raise ValueError(f"step {step} lies before the first query")
    plan, query_step = plans[query], int(log.chunk_steps[query])
    points = unproject_depth(log.depth[step], log.intrinsics, log.cam_to_world, mirrored=True)
    common = dict(step=step, query=query, query_step=query_step, plan=plan, points=points, chunk=log.chunks[query])
    if plan[EVENT_COLUMN] == 0:
        return StepView(event="none", acts_on=None, contact=None, approach=None, heat=np.zeros(points.shape[:2]),
                        map_point=None, **common)
    closing = bool(plan[EVENT_COLUMN] > 0)
    acts_on = log.target if closing else container
    contact = log.ee_pos[query_step] + FAR_SCALE_M * plan[:3]
    mask = log.seg[step] == log.names.index(acts_on) + 1
    heat = contact_region_map(points, mask, contact, GRASP_SIGMA_M if closing else RELEASE_SIGMA_M,
                              horizontal=not closing)
    visible = mask.sum() >= MIN_OBJECT_PIXELS and heat.max() >= MIN_MAP_PEAK
    point = map_point(points, heat) if visible else None
    return StepView(event="grasp" if closing else "release", acts_on=acts_on, contact=contact,
                    approach=plan[APPROACH_COLUMNS], heat=heat, map_point=point, **common)


def pixels(log: RolloutLog, points: np.ndarray) -> np.ndarray:
    """(N, 2) pixel coordinates (u, v) of world points in the agent view as logged."""
    return project_points(points, log.intrinsics, log.cam_to_world, width=log.agentview.shape[2], mirrored=True)[0]


def chunk_path(log: RolloutLog, view: StepView, scale: float) -> np.ndarray:
    """(H + 1, 3) gripper positions the chunk asks for: its translations summed from where the query was made."""
    steps = np.cumsum(view.chunk[:, :3], axis=0) * scale
    return log.ee_pos[view.query_step] + np.vstack([np.zeros(3), steps])


def describe(view: StepView) -> str:
    return "no plan" if view.acts_on is None else f"{view.event} → {pretty(view.acts_on)}"


def need_plan(view: StepView, map_point: bool = False) -> None:
    """Fail with a clear message when a figure or table is asked for a step that cannot show it."""
    if view.contact is None:
        raise ValueError(f"step {view.step} has no plan (it lies after the last release): choose another step")
    if map_point and view.map_point is None:
        raise ValueError(f"at step {view.step} the {pretty(view.acts_on)} is hidden or far from its contact, "
                         "so the map stands for no point: choose another step")


# --------------------------------------------------------------------------- overlays


def overlay(log: RolloutLog, view: StepView, scale: float, markers: bool = True) -> np.ndarray:
    """The agent view at twice its size with the map blended in; optionally the contact, the map's point,
    the path the current chunk asks for and the gripper."""
    import cv2

    image = cv2.resize(log.agentview[view.step], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_LINEAR)
    heat = cv2.resize(view.heat, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_LINEAR)[..., None] * HEAT_ALPHA
    canvas = np.ascontiguousarray((image * (1.0 - heat) + HEAT_RGB * heat).clip(0, 255).astype(np.uint8))
    if not markers:
        return canvas

    def at(point: np.ndarray) -> tuple[int, int]:
        u, v = pixels(log, point[None])[0] * ZOOM
        return int(round(u)), int(round(v))

    path = np.round(pixels(log, chunk_path(log, view, scale)) * ZOOM).astype(np.int32)
    cv2.polylines(canvas, [path.reshape(-1, 1, 2)], False, PATH_RGB, 2, cv2.LINE_AA)
    for u, v in path[1:]:
        cv2.circle(canvas, (int(u), int(v)), 3, PATH_RGB, -1, cv2.LINE_AA)
    cv2.circle(canvas, at(log.ee_pos[view.step]), 5, GRIPPER_RGB, -1, cv2.LINE_AA)
    if view.contact is not None:
        cv2.drawMarker(canvas, at(view.contact), CONTACT_RGB, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    if view.map_point is not None:
        cv2.circle(canvas, at(view.map_point), 5, POINT_RGB, -1, cv2.LINE_AA)
        cv2.circle(canvas, at(view.map_point), 5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def crop_box(log: RolloutLog, view: StepView, half: int = 56) -> tuple[slice, slice]:
    """Rows and columns of a square window around the planned contact in the image (logged resolution)."""
    need_plan(view)
    height, width = view.heat.shape
    column, row = pixels(log, view.contact[None])[0]
    row, column = int(np.clip(round(row), half, height - half)), int(np.clip(round(column), half, width - half))
    return slice(row - half, row + half), slice(column - half, column + half)


LEGEND = ("orange: affordance map · white cross: planned contact · yellow dot: the map's point · "
          "cyan: path the chunk asks for · pink: gripper")


def film_strip(log: RolloutLog, views: list[StepView], scale: float) -> plt.Figure:
    fig, axes = plt.subplots(1, len(views), figsize=(3.6 * len(views), 4.1), squeeze=False)
    for axis, view in zip(axes[0], views):
        axis.imshow(overlay(log, view, scale))
        axis.set_title(f"step {view.step} · {describe(view)}", fontsize=9)
        axis.set_axis_off()
    fig.suptitle(LEGEND, fontsize=9, y=0.02)
    fig.tight_layout()
    return fig


def render_overlay_video(log: RolloutLog, plans: np.ndarray, container: str, scale: float, out_path: Path,
                         fps: int = CONTROL_HZ) -> Path:
    """The whole rollout with the map, the planned contact and the current chunk drawn into every frame."""
    import cv2
    import imageio_ffmpeg

    side = log.agentview.shape[1] * ZOOM
    writer = imageio_ffmpeg.write_frames(str(out_path), (side, side + VIDEO_BAR_PX), fps=fps, codec="libx264",
                                         quality=7, macro_block_size=8)
    writer.send(None)
    try:
        for step in range(log.steps + 1):
            view = view_at(log, plans, step, container)
            canvas = np.full((side + VIDEO_BAR_PX, side, 3), 255, np.uint8)
            canvas[:side] = overlay(log, view, scale)
            cv2.putText(canvas, f"step {step:3d}/{log.steps}   plan: {describe(view)}".replace("→", "->"),
                        (8, side + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
            writer.send(np.ascontiguousarray(canvas))
    finally:  # do not leave the encoder running if a frame fails
        writer.close()
    return out_path


# --------------------------------------------------------------------------- 1 · observation, 2 · VLM


def observation_figure(log: RolloutLog, view: StepView) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(log.agentview[view.step])
    axes[0].set_title("agent view (RGB)")
    axes[1].imshow(log.wrist[view.step])
    axes[1].set_title("wrist view (RGB)")
    shown = axes[2].imshow(log.depth[view.step], cmap="viridis")
    axes[2].set_title("agent view: depth (m)")
    fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.02)
    for axis in axes[:3]:
        axis.set_axis_off()
    draw_top_view(axes[3], log, view.step)
    axes[3].scatter(log.ee_pos[view.step, 1], log.ee_pos[view.step, 0], marker="x", s=90, color="black", zorder=5)
    axes[3].set_title("simulator positions from above (× gripper)")
    fig.tight_layout()
    return fig


def observation_table(log: RolloutLog, view: StepView) -> pd.DataFrame:
    shape = "{} × {}".format(*log.agentview.shape[1:3])
    rows = [
        ("instruction", f"“{log.meta['instruction']}”", "VLM"),
        ("agent view, wrist view", f"RGB, {shape}; resized to 224 × 224 for the VLM", "VLM"),
        ("depth of the agent view", f"{shape}, metres; here {log.depth[view.step].min():.2f}–"
                                    f"{log.depth[view.step].max():.2f} m", "step 4 (map → 3D point), step 9"),
        ("camera matrices", "intrinsics 3 × 3, camera-to-world 4 × 4", "step 4, step 9"),
        ("gripper position", np.round(log.ee_pos[view.step], 3).tolist(), "step 6 (contact relative to the gripper)"),
    ]
    return pd.DataFrame(rows, columns=["input", "value at this step", "used by"]).set_index("input")


def vlm_table(probe: Probe, log: RolloutLog) -> pd.DataFrame:
    facts = probe.facts
    layers, tokens, width = facts["hidden_layers_used"], facts["hidden_state_shape"][1], facts["hidden_state_shape"][2]
    rows = [
        ("in", "two images", f"224 × 224 each → {facts['tokens_image'] // 2} tokens per image"),
        ("in", "instruction", f"“{log.meta['instruction']}” inside a fixed prompt → about {facts['tokens_text']} "
                              "text tokens"),
        ("in", "robot state", "not used by this checkpoint"),
        ("in", "episode state tokens", "not connected yet (step 9)"),
        ("out", "hidden states", f"{layers} layers × {tokens} tokens × {width} numbers; the affordance head, the "
                                 "plan head and the action head all read these"),
    ]
    return pd.DataFrame(rows, columns=["", "item", "measured on 10 Nov (same frozen VLM)"]).set_index(["", "item"])


def token_grid_figure(log: RolloutLog, view: StepView, tokens_per_image: int) -> plt.Figure:
    """The two images with one cell per image token, if the tokens form a square grid."""
    side = int(round(np.sqrt(tokens_per_image)))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for axis, image, name in zip(axes, (log.agentview[view.step], log.wrist[view.step]), ("agent", "wrist")):
        axis.imshow(image)
        if side * side == tokens_per_image:
            for line in np.linspace(0, image.shape[0], side + 1)[1:-1]:
                axis.axhline(line - 0.5, color="white", linewidth=0.5, alpha=0.7)
                axis.axvline(line - 0.5, color="white", linewidth=0.5, alpha=0.7)
        axis.set_title(f"{name} view: {tokens_per_image} tokens" + (f" ({side} × {side} cells)"
                                                                     if side * side == tokens_per_image else ""))
        axis.set_axis_off()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 3 · map, 4 · contact, 5 · plan head


def map_figure(log: RolloutLog, view: StepView) -> plt.Figure:
    need_plan(view)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    axes[0].imshow(log.agentview[view.step])
    axes[0].set_title("agent view")
    axes[1].imshow(log.seg[view.step] == log.names.index(view.acts_on) + 1, cmap="gray")
    axes[1].set_title(f"pixels that show the {pretty(view.acts_on)} (simulator)")
    shown = axes[2].imshow(view.heat, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[2].set_title("affordance map (ORACLE)")
    fig.colorbar(shown, ax=axes[2], fraction=0.046, pad=0.02)
    rows, columns = crop_box(log, view)
    zoomed = overlay(log, view, scale=0.0, markers=False)
    axes[3].imshow(zoomed[rows.start * ZOOM:rows.stop * ZOOM, columns.start * ZOOM:columns.stop * ZOOM])
    axes[3].set_title("map on the image, zoomed in")
    for axis in axes:
        axis.set_axis_off()
    fig.tight_layout()
    return fig


def contact_figure(log: RolloutLog, view: StepView) -> plt.Figure:
    """The map lifted to 3D: the surface points it covers, the point it stands for, and the planned contact."""
    need_plan(view, map_point=True)
    weights = view.heat.reshape(-1)
    keep = weights > 0.02
    surface, weights = view.points.reshape(-1, 3)[keep], weights[keep]
    fig, (image, top, side) = plt.subplots(1, 3, figsize=(16, 4.6))
    rows, columns = crop_box(log, view)
    image.imshow(log.agentview[view.step][rows, columns], extent=(columns.start, columns.stop, rows.stop, rows.start))
    for point, colour, marker, label in ((view.map_point, "#ffeb3b", "o", "the map's point"),
                                         (view.contact, "white", "P", "planned contact (ORACLE)")):
        u, v = pixels(log, point[None])[0]
        image.scatter(u, v, s=110, color=colour, marker=marker, edgecolor="black", label=label, zorder=4)
    image.set(xlim=(columns.start, columns.stop), ylim=(rows.stop, rows.start), title="in the image")
    image.legend(fontsize=8, loc="lower right")
    image.set_axis_off()
    for axis, (a, b), names in ((top, (1, 0), ("world y (m)", "world x (m)")), (side, (0, 2), ("world x (m)",
                                                                                                "height z (m)"))):
        axis.scatter(surface[:, a], surface[:, b], c=weights, cmap="inferno", vmin=0, vmax=1, s=6)
        axis.scatter(view.map_point[a], view.map_point[b], s=130, color="#ffeb3b", edgecolor="black", zorder=4)
        axis.scatter(view.contact[a], view.contact[b], s=130, color="white", marker="P", edgecolor="black", zorder=4)
        axis.set(xlabel=names[0], ylabel=names[1])
        axis.set_aspect("equal", adjustable="datalim")
    top.invert_xaxis()
    top.invert_yaxis()
    top.set_title("from above: surface points under the map")
    side.set_title("from the side")
    fig.tight_layout()
    return fig


def gap(view: StepView) -> tuple[float, float]:
    """(horizontal, vertical) distance in cm from the map's point to the planned contact; vertical > 0 = above.

    Both are computed from the simulator, so this is how far the two definitions lie apart (a visible
    surface against the centre of the gripper), not the error of a model.
    """
    need_plan(view, map_point=True)
    offset = view.contact - view.map_point
    return 100.0 * float(np.linalg.norm(offset[:2])), 100.0 * float(offset[2])


def gap_table(views: list[StepView]) -> pd.DataFrame:
    """Per kind of event: how far the planned contact is from the point the map stands for, over the queries."""
    rows = {}
    for event in ("grasp", "release"):
        gaps = np.array([gap(view) for view in views if view.event == event and view.map_point is not None])
        if len(gaps):
            rows[event] = {"queries": len(gaps), "horizontal gap, median (cm)": round(float(np.median(gaps[:, 0])), 1),
                           "horizontal gap, range (cm)": f"{gaps[:, 0].min():.1f}–{gaps[:, 0].max():.1f}",
                           "contact above the map's point, median (cm)": round(float(np.median(gaps[:, 1])), 1)}
    return pd.DataFrame.from_dict(rows, orient="index")


def approach_figure(log: RolloutLog, view: StepView) -> plt.Figure:
    """The approach direction as an arrow that ends at the planned contact."""
    need_plan(view)
    fig, axis = plt.subplots(figsize=(4.6, 4.6))
    rows, columns = crop_box(log, view)
    axis.imshow(log.agentview[view.step][rows, columns], extent=(columns.start, columns.stop, rows.stop, rows.start))
    (u0, v0), (u1, v1) = pixels(log, np.stack([view.contact - ARROW_M * view.approach, view.contact]))
    axis.annotate("", xy=(u1, v1), xytext=(u0, v0), arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 3})
    axis.scatter(u1, v1, s=90, color="white", marker="P", edgecolor="black", zorder=4)
    axis.set(xlim=(columns.start, columns.stop), ylim=(rows.stop, rows.start))
    axis.set_title(f"{view.event}: arrive along the arrow ({100 * ARROW_M:.0f} cm drawn)", fontsize=10)
    axis.set_axis_off()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 6 · plan, 7 · chunk, 8 · execution


def plan_table(log: RolloutLog, view: StepView) -> pd.DataFrame:
    """The ten numbers sent to the action head, and what they would be with the map's point as the contact."""
    need_plan(view, map_point=True)
    from_map = plan_features(log.ee_pos[view.query_step], view.map_point, view.approach, view.event == "grasp")
    table = pd.DataFrame({"sent in this rollout (contact: ORACLE)": view.plan,
                          "if the contact were the map's point": from_map}, index=list(PLAN_NAMES))
    return table.round(3)


def chunk_table(view: StepView) -> pd.DataFrame:
    table = pd.DataFrame(view.chunk, columns=list(ACTION_NAMES)).round(3)
    table.index = [f"step {view.query_step + k}" for k in range(len(table))]
    return table


def chunk_figure(log: RolloutLog, view: StepView, scale: float) -> plt.Figure:
    """Where the chunk would take the gripper, against the planned contact: from above and as heights."""
    need_plan(view)
    path = chunk_path(log, view, scale)
    fig, (top, height) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    draw_top_view(top, log, view.query_step)
    top.plot(path[:, 1], path[:, 0], "-o", color="#00acc1", markersize=4, linewidth=2, zorder=5, label="chunk path")
    top.scatter(path[0, 1], path[0, 0], marker="x", color="black", s=80, zorder=6, label="gripper now")
    top.scatter(view.contact[1], view.contact[0], marker="P", s=150, color="white", edgecolor="black", zorder=6,
                label="planned contact")
    top.legend(fontsize=8, loc="upper left")
    top.set_title("from above")
    height.plot(np.arange(len(path)), path[:, 2], "-o", color="#00acc1", label="chunk path")
    height.axhline(view.contact[2], color="black", linestyle="--", linewidth=1, label="height of the planned contact")
    height.set(xlabel="step of the chunk", ylabel="gripper height z (m)", title="height")
    height.legend(fontsize=8)
    fig.tight_layout()
    return fig


def execution_figure(log: RolloutLog, view: StepView, scale: float) -> plt.Figure:
    """The chunk's commanded path against the path the gripper took, and the view before and after."""
    last = min(view.query_step + len(view.chunk), log.steps)
    asked, done = chunk_path(log, view, scale), log.ee_pos[view.query_step:last + 1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for axis, step, name in ((axes[0], view.query_step, "before"), (axes[1], last, "after")):
        axis.imshow(log.agentview[step])
        axis.set_title(f"{name}: step {step}")
        axis.set_axis_off()
    axes[2].plot(asked[:, 1], asked[:, 0], "-o", color="#00acc1", markersize=4, label="asked for by the chunk")
    axes[2].plot(done[:, 1], done[:, 0], "-o", color="#e91e63", markersize=4, label="gripper path (simulator)")
    axes[2].set(xlabel="world y (m)", ylabel="world x (m)", title="from above")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].invert_xaxis()
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)
    axes[3].plot(asked[:, 2], "-o", color="#00acc1", markersize=4)
    axes[3].plot(done[:, 2], "-o", color="#e91e63", markersize=4)
    axes[3].set(xlabel="step of the chunk", ylabel="gripper height z (m)", title="height")
    fig.tight_layout()
    return fig


def execution_table(log: RolloutLog, view: StepView, scale: float) -> pd.DataFrame:
    last = min(view.query_step + len(view.chunk), log.steps)
    asked, done = chunk_path(log, view, scale), log.ee_pos[view.query_step:last + 1]
    before = np.linalg.norm(view.contact - done[0]) if view.contact is not None else np.nan
    after = np.linalg.norm(view.contact - done[-1]) if view.contact is not None else np.nan
    end = asked[len(done) - 1]
    rows = {"gripper moved (cm)": 100 * np.linalg.norm(done[-1] - done[0]),
            "chunk asked for (cm)": 100 * np.linalg.norm(end - asked[0]),
            "end of the chunk path to where the gripper ended (cm)": 100 * np.linalg.norm(end - done[-1]),
            "gripper to planned contact before (cm)": 100 * before,
            "gripper to planned contact after (cm)": 100 * after}
    return pd.DataFrame({"value": rows}).round(1)


# --------------------------------------------------------------------------- 9 · episode state, whole episode


def slot_table(history: list[tuple[Slot, ...]], step: int) -> pd.DataFrame:
    rows = {pretty(slot.name): {"status": slot.status.value,
                                "believed position (m)": None if slot.position is None
                                else np.round(slot.position, 3).tolist(),
                                "steps since seen": slot.steps_since_seen} for slot in history[step]}
    return pd.DataFrame.from_dict(rows, orient="index")


def timeline_figure(log: RolloutLog, views: list[StepView]) -> plt.Figure:
    """Over the queries: distance from the gripper to the planned contact, and from the map's point to it."""
    fig, (reach, gaps) = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    steps = np.array([view.step for view in views], dtype=float)
    for event, colour in (("grasp", "#1e88e5"), ("release", "#43a047")):
        # nan where another plan was in force, so a line never bridges a change of plan
        chosen = np.array([view.event == event for view in views])
        seen = np.array([view.event == event and view.map_point is not None for view in views])
        distance = [100 * np.linalg.norm(view.contact - log.ee_pos[view.step]) if use else np.nan
                    for view, use in zip(views, chosen)]
        across = [gap(view)[0] if use else np.nan for view, use in zip(views, seen)]
        above = [gap(view)[1] if use else np.nan for view, use in zip(views, seen)]
        reach.plot(steps, distance, "-o", color=colour, markersize=4, label=f"plan: {event}")
        gaps.plot(steps, across, "-o", color=colour, markersize=4, label=f"{event}: horizontal")
        gaps.plot(steps, above, "--s", color=colour, markersize=4, alpha=0.7,
                  label=f"{event}: contact above the map's point")
    reach.set(xlabel="step", ylabel="cm", title="gripper to planned contact")
    gaps.set(xlabel="step", ylabel="cm", title="planned contact relative to the map's point")
    reach.legend(fontsize=8)
    gaps.legend(fontsize=8)
    fig.tight_layout()
    return fig


def object_size_text(log: RolloutLog, view: StepView, tokens_per_image: int) -> str:
    """How much of the image the object the map lies on takes up, against the size of one VLM token."""
    need_plan(view)
    mask = log.seg[view.step] == log.names.index(view.acts_on) + 1
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    cell = log.agentview.shape[1] / np.sqrt(tokens_per_image)
    return (f"the {pretty(view.acts_on)} shows {int(mask.sum())} pixels, {len(columns)} wide × {len(rows)} high; "
            f"one image token of the VLM covers {cell:.0f} × {cell:.0f} pixels of this image")


def gap_overview(folder: Path, container: str) -> pd.DataFrame:
    """Every rollout of a folder (all must carry depth and plans): its outcome, and the planned contact relative
    to the map's point (median over the queries; horizontal distance and height, cm)."""
    rows = {}
    for path in sorted(Path(folder).glob("*_ep*.npz")):
        if path.name.startswith("."):  # a log that was being written when its run stopped
            continue
        log = load_rollout_log(path)
        plans = load_plans(log)
        table = gap_table([view_at(log, plans, int(step), container) for step in log.chunk_steps])
        outcome = classify(log.names, log.grasped, log.target, log.delivered).label
        row = {"asked for": pretty(log.target), "outcome": "delivered" if outcome == SUCCESS else outcome}
        for event in ("grasp", "release"):
            known = event in table.index
            row[f"{event}: horizontal"] = table.loc[event, "horizontal gap, median (cm)"] if known else None
            row[f"{event}: contact above"] = (table.loc[event, "contact above the map's point, median (cm)"]
                                              if known else None)
        rows[path.stem] = row
    return pd.DataFrame.from_dict(rows, orient="index")
