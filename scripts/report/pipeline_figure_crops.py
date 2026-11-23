"""Real image crops for the pipeline figure of the weekly review (report/media/pipeline_overview/).

Square crops, no text (the figure's labels say what they are):
  input.png           camera frame at the first query of a LIBERO-PRO rollout
  wrist.png           the wrist camera at the same step
  plan.png            the same frame with the 8 x 8 cells of the grid plan head, the cell of the predicted
                      grasp contact shaded and the contact marked (the plan the policy was sent)
  counterfactual.png  a demonstration's first frame: its own way to the grasp (dark) and the same way
                      rewritten to another object's grasp (orange), as the 15 Nov labels do
  outcome.png         the rollout shortly after the gripper has closed on the target
  <name>.png          (one per --scene name=log) the log's first frame with the object its instruction
                      names outlined, for the generalisation figure

Usage on the server (starvla env, no GPU):
    R=$W/runs/libero_pro/predicted_g1_hold_s10
    D=$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz
    python scripts/report/pipeline_figure_crops.py --rollout $R/libero_object_swap/task2_ep0.npz --demo $D \\
        --data $W/runs/plan_following/dataset10 --other ketchup_1 --out report/media/pipeline_overview \\
        --scene scene_train=$D --scene scene_swap=$R/libero_object_swap/task2_ep0.npz \\
        --scene scene_task=$R/libero_object_task/task2_ep0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2]))
from affordance_vla.heads.plan_grid import (  # noqa: E402
    CELL_PX,
    CELLS,
    IMAGE_SIZE,
    Camera,
    camera_of_log,
    contact_pixel,
)
from affordance_vla.plan.plan_consistency import plan_ahead  # noqa: E402
from affordance_vla.plan.plan_input import offsets_from_json, plan_contact  # noqa: E402
from affordance_vla.sim.rollout_log import RolloutLog, load_rollout_log  # noqa: E402
from affordance_vla.training.counterfactual import Retarget, retarget_path  # noqa: E402
from affordance_vla.training.plan_dataset import load_tracks  # noqa: E402
from affordance_vla.training.plan_samples import approaches_by_target  # noqa: E402

CROP_PX = 600  # side of each saved crop
ORANGE, INK, PALE = "#ED7D31", "#1a1a1a", "#ffffff"
STEPS_AFTER_GRASP = 12  # the outcome frame: the object is lifted a little


def _canvas(image: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(1, 1), dpi=CROP_PX)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.imshow(image, extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0), interpolation="lanczos")
    axis.set_xlim(0, IMAGE_SIZE), axis.set_ylim(IMAGE_SIZE, 0)
    axis.set_axis_off()
    return fig, axis


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=CROP_PX)
    plt.close(fig)
    print("wrote", path)


def _pixels(points: np.ndarray, camera: Camera) -> np.ndarray:
    return np.array([contact_pixel(point, camera)[:2] for point in points])


def first_plan_contact(path: Path, log: RolloutLog) -> tuple[int, np.ndarray]:
    """(step, world contact) of the first plan the rollout sent."""
    with np.load(path) as data:
        plan = data["plans"][0]
    step = int(log.chunk_steps[0])
    contact = plan_contact(plan, log.ee_pos[step])
    if contact is None:
        raise SystemExit(f"the first plan of {path.name} has no contact")
    return step, contact


def plan_crop(image: np.ndarray, contact: np.ndarray, camera: Camera, out: Path) -> None:
    fig, axis = _canvas(image)
    for k in range(1, CELLS):
        axis.axhline(k * CELL_PX, color=PALE, linewidth=0.35, alpha=0.8)
        axis.axvline(k * CELL_PX, color=PALE, linewidth=0.35, alpha=0.8)
    u, v = contact_pixel(contact, camera)[:2]
    col, row = int(u // CELL_PX), int(v // CELL_PX)
    axis.add_patch(plt.Rectangle((col * CELL_PX, row * CELL_PX), CELL_PX, CELL_PX, facecolor=ORANGE, alpha=0.45,
                                 edgecolor=ORANGE, linewidth=0.8))
    axis.scatter([u], [v], s=10, color=ORANGE, edgecolor=INK, linewidth=0.3, zorder=3)
    _save(fig, out)


def counterfactual_crop(demo_path: Path, image: np.ndarray, data: Path, other: str, camera: Camera,
                        out: Path) -> None:
    """The demonstration's way to its grasp event and the same way moved to `other`, both cut where the
    training samples cut them (plan_ahead from the first step, as approaches_by_target and shifted_sample do)."""
    offsets = offsets_from_json((data / "offsets.json").read_text())
    files = json.loads((data / "stats.json").read_text())["files"]
    if demo_path.name not in files:
        raise SystemExit(f"{demo_path.name} is not in the dataset {data}")
    tracks = load_tracks(data / "tracks.npz")
    track = tracks[files.index(demo_path.name)]
    plan = plan_ahead(track.labels, track.ee_pos[0], 0)
    if plan is None or not plan.closing:
        raise SystemExit("the demonstration does not start on its way to a grasp")
    path = np.asarray(track.ee_pos[:plan.steps_to_event + 1], dtype=float)
    new_contact = track.object_pos[0, track.names.index(other)] + offsets.grasp[other]
    tails = approaches_by_target(tracks).get(other)
    if not tails:
        raise SystemExit(f"no demonstrated approach to {other}")
    moved = retarget_path(path, new_contact, Retarget(scale_from_start=True, rigid_tail=0.18), tail=tails[0])
    fig, axis = _canvas(image)
    for points, colour in ((path, INK), (moved, ORANGE)):
        pixels = _pixels(points, camera)
        axis.plot(pixels[:, 0], pixels[:, 1], color=colour, linewidth=0.9, solid_capstyle="round")
        axis.scatter(*pixels[-1], marker="*", s=22, color=colour, edgecolor=PALE, linewidth=0.3, zorder=3)
    _save(fig, out)


def scene_crop(log: RolloutLog, out: Path) -> None:
    """First frame of a log with the outline of the object its instruction names (from the segmentation)."""
    if log.seg is None:
        raise SystemExit(f"{log.path.name} has no segmentation to outline its target with")
    mask = (log.seg[0] == log.target_index + 1).astype(float)
    if not mask.any():
        raise SystemExit(f"the target {log.target} is not visible in the first frame of {log.path.name}")
    fig, axis = _canvas(log.agentview[0])
    rows, cols = mask.shape  # pixel centres in the IMAGE_SIZE frame the image is drawn in, row 0 at the top
    xs, ys = (np.arange(cols) + 0.5) * IMAGE_SIZE / cols, (np.arange(rows) + 0.5) * IMAGE_SIZE / rows
    axis.contour(xs, ys, mask, levels=[0.5], colors=[ORANGE], linewidths=1.0)
    _save(fig, out)


def _scene(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--scene takes name=log.npz, got {text!r}")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollout", required=True, type=Path, help="rollout log with images and plans")
    parser.add_argument("--demo", required=True, type=Path, help="state-replay log of a demonstration")
    parser.add_argument("--data", required=True, type=Path, help="plan dataset (tracks.npz, offsets.json)")
    parser.add_argument("--other", required=True, help="object the counterfactual sample names")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scene", action="append", default=[], type=_scene, metavar="NAME=LOG",
                        help="also save NAME.png: the log's first frame with its target outlined (repeatable)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rollout, demo = load_rollout_log(args.rollout), load_rollout_log(args.demo)
    if rollout.agentview is None or demo.agentview is None:
        raise SystemExit("both logs need their camera images")
    if args.other not in demo.names:
        raise SystemExit(f"{args.other} is not in the demonstration's scene {demo.names}")
    camera = camera_of_log(demo)  # the agent-view camera does not move between LIBERO scenes
    step, contact = first_plan_contact(args.rollout, rollout)
    _save(_canvas(rollout.agentview[step])[0], args.out / "input.png")
    _save(_canvas(rollout.wrist[step])[0], args.out / "wrist.png")
    plan_crop(rollout.agentview[step], contact, camera, args.out / "plan.png")
    counterfactual_crop(args.demo, demo.agentview[0], args.data, args.other, camera, args.out / "counterfactual.png")
    held = np.flatnonzero(rollout.grasped[:, rollout.target_index])
    if held.size == 0:
        raise SystemExit("the rollout never grasps its target")
    last = min(int(held[0]) + STEPS_AFTER_GRASP, len(rollout.agentview) - 1)
    _save(_canvas(rollout.agentview[last])[0], args.out / "outcome.png")
    for name, path in args.scene:
        scene_crop(load_rollout_log(path), args.out / f"{name}.png")


if __name__ == "__main__":
    main()
