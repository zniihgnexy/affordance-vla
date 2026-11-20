"""Videos, figures and status tables for report/2026-11-20_week2_review.ipynb (week 2: 16-22 Nov 2026).

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
from affordance_vla.heads.plan_grid import CELL_PX, IMAGE_SIZE, Camera, camera_of_log, contact_pixel
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


def _count(folder: Path, pattern: str) -> int:
    return len(list(folder.glob(pattern))) if folder.is_dir() else 0


def progress_table(work: Path) -> pd.DataFrame:
    """Where the jobs started this week stand, read from the server's folders."""
    runs = work / "runs"
    demos = runs / "plan_following"
    pro = runs / "libero_pro"
    probed = (runs / "semantic_probe" / "e1" / "summary.json").exists()
    rows = [
        ("L：LIBERO-Object 示范重新回放，补记物体朝向", f"{_count(demos / 'demos_pose', '*_demo*.npz')} / 500"),
        ("LIBERO-10：下载示范（状态和动作）",
         f"{_count(work / 'data' / 'libero_states' / 'libero_10', '*_demo.hdf5')} / 10"),
        ("LIBERO-10：示范重新回放，补记物体朝向（11-20 修好后重跑）",
         f"{_count(demos / 'demos_libero10', '*_demo*.npz')} / 约 500"),
        ("LIBERO-10：官方权重在 LIBERO-PRO 上（位置扰动、任务扰动）",
         f"{sum(_count(pro / run, '*/task*_ep*.npz') for run in LIBERO10_RUNS)} / 400 个回合"),
        ("L：每类物体的 3D 标准 map（Object、LIBERO-10）",
         " / ".join("已完成" if (runs / "interaction_labels" / name / "maps.npz").exists() else "未完成"
                    for name in ("object", "libero10"))),
        ("E1：语义诊断（冻结 VLM，不训练）", "已完成" if probed else "排队或运行中（前两次运行失败，已修复）"),
    ]
    return pd.DataFrame(rows, columns=["任务", "进度"]).set_index("任务")


def task_results(runs: list[Path], suite: str) -> pd.DataFrame:
    """Successes per task of one suite from the runs' task summaries; a later run only fills tasks the earlier lack."""
    rows = {}
    for run in runs:
        for path in sorted((run / suite).glob("task*_summary.json")):
            task = int(path.stem.split("_")[0].removeprefix("task"))
            if task not in rows:
                episodes = json.loads(path.read_text())
                rows[task] = {"指令": episodes[0]["instruction"], "成功": sum(bool(e["success"]) for e in episodes),
                              "回合": len(episodes)}
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["指令", "成功", "回合"]).sort_index()
    table.index.name = "任务"
    return table


def label_summary(summary_path: Path) -> pd.DataFrame:
    """L: grasps, groups of closing axes and agreement per object category (build_interaction_labels.py)."""
    found = json.loads(summary_path.read_text())["categories"]
    rows = {category: {"抓取数": entry["grasps"], "闭合轴分组（20° 内）": entry["axis_groups"],
                       "最大组占比": round(entry["axis_groups"][0] / entry["grasps"], 2),
                       "接触区域占表面": entry["contact_share"], "接触区域内方向一致度": entry["agreement_in_contact"]}
            for category, entry in found.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "类别"
    return table


def probe_table(summary_path: Path) -> pd.DataFrame | None:
    """E1: share of first frames whose similarity map peaks on the named object, best layer per variant."""
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = {variant: {"best layer": found["best_layer"], "on the named object": found["best_share"]}
            for variant, found in summary["maps"].items()}
    return pd.DataFrame(rows).T
