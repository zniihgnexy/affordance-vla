"""Drawing for the HOT3D demo: meshes on a camera frame, and an object mesh painted with a per-point value.

A small painter's-algorithm renderer with OpenCV (faces sorted far to near, flat shading); the meshes here have
a few thousand faces, so no GPU or rasteriser library is needed.
"""
from __future__ import annotations

import cv2
import numpy as np

LIGHT = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])  # towards the viewer, from above-right
AMBIENT = 0.35


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)


def shade(colours: np.ndarray, normals_view: np.ndarray) -> np.ndarray:
    """Flat shading of (F, 3) float colours in [0, 1] with a light near the viewing direction (two-sided)."""
    lit = AMBIENT + (1 - AMBIENT) * np.abs(normals_view @ LIGHT)
    return np.clip(colours * lit[:, None], 0, 1)


def paint(canvas: np.ndarray, uv: np.ndarray, depth: np.ndarray, faces: np.ndarray, colours: np.ndarray,
          alpha: float = 1.0) -> np.ndarray:
    """Fill the faces (screen points uv, viewing depth, (F, 3) colours in [0, 1]) far to near on a uint8 image."""
    layer, covered = canvas.copy(), np.zeros(canvas.shape[:2], np.uint8)
    visible = np.isfinite(uv[faces]).all(axis=(1, 2)) & (depth[faces] > 0).all(axis=1)
    order = np.argsort(-depth[faces].mean(axis=1))
    for index in order[visible[order]]:
        polygon = np.round(uv[faces[index]] * 16).astype(np.int32)  # 4 bits of sub-pixel precision
        colour = tuple(int(c) for c in colours[index] * 255)
        cv2.fillConvexPoly(layer, polygon, colour, lineType=cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(covered, polygon, 1, shift=4)
    blend = (alpha * covered[..., None]).astype(np.float32)
    return (canvas * (1 - blend) + layer * blend).astype(np.uint8)


def overlay_meshes(image: np.ndarray, camera, meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                   alpha: float = 0.75) -> np.ndarray:
    """Meshes (world vertices, faces, colour) drawn into a camera frame, hands and objects together. A colour is an
    RGB triple in [0, 1] for the whole mesh or an (F, 3) array with one per face."""
    all_uv, all_depth, all_faces, all_colours, offset = [], [], [], [], 0
    for vertices, faces, colour in meshes:
        window = camera.world_to_window3(vertices)
        eye = camera.world_to_eye(vertices)
        normals = face_normals(eye, faces)  # in the camera's frame, so the light is at the camera
        all_uv.append(window[:, :2])
        all_depth.append(window[:, 2])
        all_faces.append(faces + offset)
        all_colours.append(shade(np.broadcast_to(colour, (len(faces), 3)), normals))
        offset += len(vertices)
    return paint(image, np.concatenate(all_uv), np.concatenate(all_depth), np.concatenate(all_faces),
                 np.concatenate(all_colours), alpha)


def view_rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rows are the view's right, up and towards-the-viewer axes in the object frame (y up, as HOT3D's models)."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    toward = np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(toward, right), toward])


def project_orthographic(points: np.ndarray, rotation: np.ndarray, centre: np.ndarray, scale: float,
                         size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel positions (y down) and depth (larger = nearer to the viewer) of object-frame points."""
    local = (points - centre) @ rotation.T
    uv = np.stack([size / 2 + scale * local[:, 0], size / 2 - scale * local[:, 1]], axis=1)
    return uv, local[:, 2] + 10.0  # +10 keeps all depths positive for `paint`; nearer points have larger local z


UNTOUCHED = np.array([0.88, 0.88, 0.88])


def colormap(values: np.ndarray, name: str = "YlOrRd") -> np.ndarray:
    """Light grey for 0, then the colour map from pale yellow (small) to dark red (1)."""
    from matplotlib import colormaps
    hot = colormaps[name](0.12 + 0.88 * np.clip(values, 0, 1))[:, :3]
    return np.where((values <= 0.005)[:, None], UNTOUCHED, hot)


def paint_object(vertices: np.ndarray, faces: np.ndarray, vertex_values: np.ndarray, view: np.ndarray,
                 size: int = 480, lines: list[tuple[np.ndarray, np.ndarray]] | None = None,
                 background: float = 1.0) -> np.ndarray:
    """The object painted with a value in [0, 1] per vertex (grey = 0, then YlOrRd) from one view; `lines` are
    object-frame segments (start, end) drawn on top in cyan."""
    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = 0.8 * size / np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    uv, depth = project_orthographic(vertices, view, centre, scale, size)
    face_values = vertex_values[faces].mean(axis=1)
    normals = face_normals(vertices @ view.T, faces)
    colours = shade(colormap(face_values), normals)
    canvas = np.full((size, size, 3), int(background * 255), np.uint8)
    canvas = paint(canvas, uv, 20.0 - depth, faces, colours)  # `paint` draws large depth first: far to near
    for start, end in lines or []:
        (a, _), (b, _) = (project_orthographic(p[None], view, centre, scale, size) for p in (start, end))
        cv2.line(canvas, tuple(np.round(a[0]).astype(int)), tuple(np.round(b[0]).astype(int)), (0, 200, 255), 3,
                 cv2.LINE_AA)
    return canvas
