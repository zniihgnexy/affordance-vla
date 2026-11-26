"""Figures of the RoboCasa365 set-up check (AFFO-10) for the weekly review, from files that exist on the server.

  three_cameras.png              recorded frames of the three cameras, one demonstration of each of the 3 tasks
  replay_video_vs_rerender.png   recorded frame | re-rendered from the saved state | |difference| x4 | depth | instances
  category_surface_points.png    surface points of mug / cup / bowl instances, exported from the object meshes

Usage (repo root, server, robocasa365 env):
  "$RC365_PYTHON" scripts/report/make_rc365_figures.py --replay $W/runs/robocasa365_replay/<run> \
      --out report/media/weekly_2026_11_23/robocasa365
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import robocasa365 as rc  # noqa: E402

TASKS = ("CoffeeSetupMug", "OpenDrawer", "TurnOffStove")
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (250, 250, 250)


def label(img: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    bar = np.full((height, img.shape[1], 3), BG, np.uint8)
    cv2.putText(bar, text, (6, height - 9), FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def side_label(img: np.ndarray, text: str, width: int = 130) -> np.ndarray:
    bar = np.full((img.shape[0], width, 3), BG, np.uint8)
    for k, part in enumerate(text.split("|")):
        cv2.putText(bar, part, (6, 22 + 20 * k), FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=1)


def three_cameras(root: str, out: Path) -> None:
    rows = []
    for task in TASKS:
        ds = rc.dataset_dir(root, task)
        frames = []
        for cam in rc.CAMERAS:
            video = rc.read_video(ds, cam, 0)
            frames.append(video[len(video) // 2])
        rows.append(side_label(np.concatenate(frames, axis=1), f"{task}|episode 0|frame {len(video) // 2}"))
    header = np.concatenate(
        [np.full((28, 130, 3), BG, np.uint8)] + [label(np.zeros((0, 256, 3), np.uint8), c)[:28] for c in rc.CAMERAS],
        axis=1,
    )
    cv2.imwrite(str(out), np.concatenate([header] + rows, axis=0)[..., ::-1])


def replay_figure(replay_dir: Path, out: Path) -> None:
    rows = []
    for task in TASKS:
        full = cv2.imread(str(replay_dir / f"{task}_ep0_video_replay_diff_depth_seg.png"))[..., ::-1]
        rows.append(side_label(full[256:512], f"{task}|episode 0|middle frame"))  # the middle of the three saved frames
    titles = [
        "recorded video",
        "re-rendered from state",
        "|difference| x4",
        "depth (yellow = near)",
        "instance segmentation",
    ]
    header = np.concatenate(
        [np.full((28, 130, 3), BG, np.uint8)] + [label(np.zeros((0, 256, 3), np.uint8), t)[:28] for t in titles], axis=1
    )
    cv2.imwrite(str(out), np.concatenate([header] + rows, axis=0)[..., ::-1])


def project(points: np.ndarray, axes: tuple[int, int], depth_axis: int, size: int = 150) -> np.ndarray:
    """Orthographic scatter of points (m), colour = height (z), common scale 0.16 m across the tile."""
    img = np.full((size, size, 3), 255, np.uint8)
    c = points.mean(0)
    scale = size / 0.16
    u = (points[:, axes[0]] - c[axes[0]]) * scale + size / 2
    v = size / 2 - (points[:, axes[1]] - c[axes[1]]) * scale
    z = points[:, 2]
    colour = cv2.applyColorMap(
        ((z - z.min()) / max(np.ptp(z), 1e-6) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_VIRIDIS
    )[:, 0, ::-1]
    for i in np.argsort(-points[:, depth_axis]):  # far to near
        cv2.circle(img, (int(u[i]), int(v[i])), 1, colour[i].tolist(), -1)
    return img


def surface_figure(out: Path, per_split: int = 2) -> None:
    rows = []
    for category in ("mug", "cup", "bowl"):
        inst = rc.category_instances(category)
        reg = max(inst, key=lambda r: len(inst[r]["all"]))  # the registry with most instances
        picks = [("train", p) for p in inst[reg]["pretrain"][:per_split]] + [
            ("held-out", p) for p in inst[reg]["target"][:per_split]
        ]
        tiles = []
        for split, path in picks:
            pts = rc.surface_points(path, 3000)
            pair = np.concatenate([project(pts, (0, 2), 1), project(pts, (0, 1), 2)], axis=1)
            tiles.append(label(pair, f"{Path(path).parent.name} ({split})", 22))
        n_all = sum(len(d["all"]) for d in inst.values())
        n_target = sum(len(d["target"]) for d in inst.values())
        rows.append(
            side_label(
                np.concatenate(tiles, axis=1),
                f"{category}|{n_all} instances|{n_target} held-out",
                150,
            )
        )
    cv2.imwrite(str(out), np.concatenate(rows, axis=0)[..., ::-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", required=True, help="output folder of replay_robocasa365.py")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--root", default=str(Path(os.environ.get("W", "/mnt/nas/xinyizheng/affordance_vla")) / "data/robocasa365")
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    three_cameras(args.root, out / "three_cameras.png")
    replay_figure(Path(args.replay), out / "replay_video_vs_rerender.png")
    surface_figure(out / "category_surface_points.png")


if __name__ == "__main__":
    main()
