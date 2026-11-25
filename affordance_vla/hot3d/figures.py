"""The two figures of the HOT3D demo (matplotlib around the painter in render.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from affordance_vla.hot3d import contact as C  # noqa: E402
from affordance_vla.hot3d import render  # noqa: E402
from affordance_vla.hot3d.clips import Clip  # noqa: E402
from affordance_vla.plan.interaction_labels import Grasp  # noqa: E402

HAND_COLOURS = {"left": (0.85, 0.25, 0.25), "right": (0.25, 0.45, 0.9)}
THUMB_COLOUR = (1.0, 0.65, 0.1)
OBJECT_COLOUR = (0.35, 0.8, 0.45)
VIEWS = [(0, 20), (90, 20), (180, 20), (270, 20)]
AXIS_HALF_LENGTH_M = 0.05  # drawn length of a closing axis: the gripper's widest opening


def _face_colours(faces: np.ndarray, finger: np.ndarray, base: tuple) -> np.ndarray:
    is_thumb = (finger[faces] == C.THUMB).any(axis=1)
    return np.where(is_thumb[:, None], THUMB_COLOUR, base)


def crop_around(image: np.ndarray, points_uv: np.ndarray, factor: float = 3.0, out: int = 520) -> np.ndarray:
    centre = np.median(points_uv, axis=0)
    half = max(factor * np.ptp(points_uv, axis=0).max() / 2, 150)
    x0, y0 = np.round(centre - half).astype(int)
    x1, y1 = np.round(centre + half).astype(int)
    pad = int(half) + 1
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def overlay_figure(shots: list[tuple[Clip, int, str, np.ndarray, np.ndarray, str]], vertex_finger: np.ndarray,
                   triangles: np.ndarray, path: Path) -> None:
    """Rows of (camera frame, the same frame with the hand and object meshes drawn in): one row per shot
    (clip, frame, object id, mesh vertices, faces, title)."""
    fig, axes = plt.subplots(len(shots), 2, figsize=(8.4, 4.3 * len(shots)), squeeze=False)
    for row, (clip, frame, object_id, vertices, faces, title) in zip(axes, shots):
        camera, image = clip.camera(frame), clip.image(frame)
        pose = clip.object_pose(frame, object_id)
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        meshes = [(world, faces, OBJECT_COLOUR)]
        for hand in clip.hands(frame):
            colours = _face_colours(triangles, vertex_finger, HAND_COLOURS[hand.side])
            meshes.append((hand.vertices, triangles, colours))
        uv = camera.world_to_window3(world)[:, :2]
        drawn = render.overlay_meshes(image, camera, meshes)
        panels = ((row[0], image, f"{title}: RGB camera"), (row[1], drawn, "annotated hands and object"))
        for axis, picture, label in panels:
            axis.imshow(crop_around(picture, uv))
            axis.set_title(label, fontsize=9)
            axis.axis("off")
    fig.suptitle("HOT3D-Clips, Aria RGB camera (fisheye, cropped); meshes at the ground-truth poses\n"
                 "green = mug, orange = thumb, red / blue = rest of the left / right hand", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bar_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([render.UNTOUCHED, *render.colormap(np.linspace(0.01, 1, 255))])


def _vertex_values(vertices: np.ndarray, samples: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values[cKDTree(samples).query(vertices)[1]]


def contact_figure(vertices: np.ndarray, faces: np.ndarray, rows: list[tuple[str, np.ndarray, np.ndarray, float]],
                   grasps: list[Grasp], path: Path, title: str) -> None:
    """One row per map (label, sample points, values, value of the top of the colour bar): the mug painted with the
    values from four sides; the closing axes of the held grasps as cyan lines in the first row."""
    fig, axes = plt.subplots(len(rows), len(VIEWS), figsize=(11, 2.9 * len(rows)), squeeze=False)
    lines = [(g.centre - AXIS_HALF_LENGTH_M * g.closing, g.centre + AXIS_HALF_LENGTH_M * g.closing) for g in grasps]
    for r, (label, points, values, top) in enumerate(rows):
        shown = np.clip(_vertex_values(vertices, points, values) / top, 0, 1)
        for c, (azimuth, elevation) in enumerate(VIEWS):
            view = render.view_rotation(azimuth, elevation)
            axes[r, c].imshow(render.paint_object(vertices, faces, shown, view, lines=lines if r == 0 else None))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"view {azimuth}°", fontsize=9)
        axes[r, 0].text(-0.02, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=9)
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=_bar_cmap(), norm=plt.Normalize(0, top)), ax=axes[r].tolist(),
                           fraction=0.015, pad=0.01)
        bar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
