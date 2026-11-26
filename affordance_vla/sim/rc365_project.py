"""Project 3D labels (object-surface maps, contact points, axes, motion) onto a rendered RoboCasa365 camera frame
and draw them (AFFO-15, question 3 of the figures). Rendering: `robocasa365.render_view` (depth, upright images);
drawing: OpenCV. Visibility follows the rendered depth: a point is drawn when it is within `DEPTH_TOLERANCE_M` of the
depth map at its pixel, so what the robot's own arm or another object hides is not drawn."""

from __future__ import annotations

import cv2
import numpy as np

DEPTH_TOLERANCE_M = 0.02
SCORE_FLOOR = 0.05  # map points below this score are not drawn


def camera_from(env, name: str, size: int) -> dict:
    """Pose and focal length (pixels) of a named camera in the current simulator state (MuJoCo: looks along -z)."""
    model, data = env.sim.model, env.sim.data
    cam = model.camera_name2id(name)
    return {
        "pos": data.cam_xpos[cam].copy(),
        "rot": data.cam_xmat[cam].reshape(3, 3).copy(),
        "size": size,
        "focal": 0.5 * size / np.tan(np.radians(model.cam_fovy[cam]) / 2),
    }


def project(cam: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (N, 2; x right, y down in the upright image) and depth along the optical axis (N,)."""
    local = (np.atleast_2d(points) - cam["pos"]) @ cam["rot"]
    depth = -local[:, 2]
    safe = np.where(depth > 1e-6, depth, np.nan)
    half = cam["size"] / 2
    return np.stack([half + cam["focal"] * local[:, 0] / safe, half - cam["focal"] * local[:, 1] / safe], axis=1), depth


def visible(uv: np.ndarray, depth: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    size = depth_map.shape[0]
    ok = (
        np.isfinite(uv).all(axis=1)
        & (depth > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < size)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < size)
    )
    near = np.zeros(len(uv), dtype=bool)
    idx = np.flatnonzero(ok)
    near[idx] = np.abs(depth_map[uv[idx, 1].astype(int), uv[idx, 0].astype(int)] - depth[idx]) <= DEPTH_TOLERANCE_M
    return near


def draw_map(image: np.ndarray, uv: np.ndarray, score: np.ndarray, radius: int = 2, alpha: float = 0.8) -> np.ndarray:
    """Dots coloured by score (yellow high, purple low) on a copy of the image; low scores are left out."""
    layer = image.copy()
    keep = score > SCORE_FLOOR
    colours = cv2.applyColorMap((255 * np.clip(score[keep], 0, 1)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
    for (x, y), colour in zip(uv[keep], colours):
        cv2.circle(layer, (int(x), int(y)), radius, tuple(int(c) for c in colour), -1, cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, image, 1 - alpha, 0)


def draw_segment(
    image: np.ndarray, cam: dict, start: np.ndarray, end: np.ndarray, colour, width: int = 2, arrow: bool = False
) -> None:
    uv, depth = project(cam, np.stack([start, end]))
    if not (np.isfinite(uv).all() and (depth > 0).all()):
        return
    a, b = (tuple(int(v) for v in p) for p in uv)
    (
        cv2.arrowedLine(image, a, b, colour, width, cv2.LINE_AA, tipLength=0.25)
        if arrow
        else cv2.line(image, a, b, colour, width, cv2.LINE_AA)
    )


def draw_point(image: np.ndarray, cam: dict, point: np.ndarray, colour, radius: int = 4) -> None:
    uv, depth = project(cam, point)
    if np.isfinite(uv).all() and depth[0] > 0:
        cv2.circle(image, (int(uv[0, 0]), int(uv[0, 1])), radius, colour, -1, cv2.LINE_AA)
        cv2.circle(image, (int(uv[0, 0]), int(uv[0, 1])), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def caption(image: np.ndarray, text: str) -> np.ndarray:
    """The image with a white strip of text above it."""
    strip = np.full((22, image.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([strip, image], axis=0)
