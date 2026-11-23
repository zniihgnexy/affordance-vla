"""Crops of the affordance map for the v3 pipeline figures (report/media/pipeline_overview/); square, no text.

  map_now.png    what the grid head g1 outputs on the first frame of a LIBERO-PRO rollout: its 8 x 8 map, a
                 softmax over the cells (transparent near 0), with the best cell outlined. Read from the E1 probe,
                 which kept g1's cell scores and the frame
  map_label.png  g1's training label on a demonstration's first frame: the one cell that holds the ORACLE contact
                 (the object's position + its mean demonstrated grasp offset), the contact dotted
  map_v3.png     the planned v3 label on the same frame: every object's category-level interaction map (L2)
                 projected onto it; per cell the highest score over the objects, and the closing direction as a dash

The figures say what the crops are. Usage on the server (starvla env, no GPU), from the repository root:
    W=/mnt/nas/xinyizheng/affordance_vla
    D=$W/runs/plan_following/demos_pose/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz
    python scripts/report/map_figure_crops.py --probe runs/semantic_probe/e1/probe.npz \\
        --probe-frame libero_object_swap/task2_ep0.npz --demo $D --data $W/runs/plan_following/dataset10 \\
        --maps runs/interaction_labels/object/maps.npz --out report/media/pipeline_overview
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pipeline_figure_crops import CELL_PX, CELLS, IMAGE_SIZE, _canvas, _grid, _save, plan_crop

from affordance_vla.analysis.grasp_orientation import object_category
from affordance_vla.heads.plan_grid import camera_of_log
from affordance_vla.plan.interaction_labels import load_maps, project_map
from affordance_vla.plan.plan_input import grasp_contact, offsets_from_json
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log

HEAT = "plasma"
MAX_ALPHA = 0.7  # opacity of the hottest cell; the scene stays visible elsewhere
DASH_MIN_SCORE = 0.3  # cells whose map score is below this get no direction dash
DASH_HALF_PX = 0.36 * CELL_PX
DASH_COLOUR = "#2E75B6"


def _heat(axis: plt.Axes, cells: np.ndarray) -> None:
    """An (CELLS, CELLS) map in [0, 1] over the image, opacity following the value."""
    peak = float(cells.max()) if cells.max() > 0 else 1.0
    overlay = plt.get_cmap(HEAT)(cells / peak)
    overlay[..., 3] = MAX_ALPHA * cells / peak
    axis.imshow(overlay, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), interpolation="nearest", zorder=2)


def _outline(axis: plt.Axes, cell: int, colour: str) -> None:
    axis.add_patch(plt.Rectangle(((cell % CELLS) * CELL_PX, (cell // CELLS) * CELL_PX), CELL_PX, CELL_PX,
                                 fill=False, edgecolor=colour, linewidth=1.2, zorder=3))


def map_now(probe_path: Path, frame: str, out: Path) -> None:
    """g1's map on the first frame named `frame` (suite/log file) of the E1 probe."""
    probe = np.load(probe_path, allow_pickle=True)
    suite, name = frame.split("/")
    found = np.flatnonzero((probe["suite"].astype(str) == suite) & (probe["log_file"].astype(str) == name))
    if len(found) == 0:
        raise SystemExit(f"{frame} is not among the probe's first frames")
    i = int(found[0])
    scores = probe["head_scores"][i].astype(float)
    heat = np.exp(scores - scores.max())
    fig, axis = _canvas(probe["agentview"][i])
    _grid(axis)
    _heat(axis, (heat / heat.sum()).reshape(CELLS, CELLS))
    _outline(axis, int(np.argmax(scores)), "white")
    _save(fig, out)


def map_label(demo: RolloutLog, offsets_path: Path, out: Path) -> None:
    """The one-cell ORACLE label g1 is trained on, for the demonstration's first frame."""
    offsets = offsets_from_json(offsets_path.read_text())
    target = demo.names[demo.target_index]
    contact = grasp_contact(offsets, target, demo.object_pos[0, demo.target_index])
    plan_crop(demo.agentview[0], contact, camera_of_log(demo), out)


def map_v3(demo: RolloutLog, maps_path: Path, out: Path) -> None:
    """Every object's L2 map projected onto the demonstration's first frame, with its closing directions."""
    maps = load_maps(maps_path)
    score = np.zeros((CELLS, CELLS))
    direction = np.zeros((CELLS, CELLS, 2))
    for k, name in enumerate(demo.names):
        cmap = maps.get(object_category(name))
        if cmap is None:  # the basket: a container, not grasped
            continue
        label = project_map(cmap, demo.object_pos[0, k], demo.object_quat[0, k], demo.intrinsics, demo.cam_to_world,
                            demo.seg[0], k + 1)
        better = label.score > score
        score = np.where(better, label.score, score)
        direction = np.where(better[..., None], label.direction, direction)
    fig, axis = _canvas(demo.agentview[0])
    _grid(axis)
    _heat(axis, score)
    for row, col in zip(*np.nonzero(score >= DASH_MIN_SCORE)):
        theta = 0.5 * np.arctan2(direction[row, col, 1], direction[row, col, 0])  # the axis is 180° symmetric
        centre = np.array([(col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX])
        ends = centre + np.outer([-DASH_HALF_PX, DASH_HALF_PX], [np.cos(theta), np.sin(theta)])
        axis.plot(ends[:, 0], ends[:, 1], color=DASH_COLOUR, linewidth=1.4, solid_capstyle="round", zorder=3)
    _save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True, type=Path, help="probe.npz of the E1 semantic probe")
    parser.add_argument("--probe-frame", required=True, help="suite/log file of the probe frame, e.g. "
                                                             "libero_object_swap/task2_ep0.npz")
    parser.add_argument("--demo", required=True, type=Path, help="state-replay log with object poses (demos_pose)")
    parser.add_argument("--data", required=True, type=Path, help="plan dataset whose offsets.json g1 trained on")
    parser.add_argument("--maps", required=True, type=Path, help="maps.npz of build_interaction_labels.py")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demo = load_rollout_log(args.demo)
    if demo.agentview is None or demo.object_quat is None or demo.seg is None:
        raise SystemExit("the demonstration log needs images, object orientations and segmentation (demos_pose)")
    map_now(args.probe, args.probe_frame, args.out / "map_now.png")
    map_label(demo, args.data / "offsets.json", args.out / "map_label.png")
    map_v3(demo, args.maps, args.out / "map_v3.png")


if __name__ == "__main__":
    main()
