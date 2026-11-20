"""Points on the surface of a scene object, in the object's own (body) frame, from its visual meshes.

The interaction labels of Pipeline v3 (affordance_vla/plan/interaction_labels.py) live on these points. The
logged object pose (`object_pos`, `object_quat`) is the pose of this body, so a point here moves into the world
with it. Points are sampled on the mesh faces in proportion to their area, so dense and sparse parts of a mesh
get the same density; each point carries the normal of its face (outward when the mesh is wound the usual
way; the labels only use the normal's line, not its sign).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MESH_GEOM = 7  # mujoco.mjtGeom.mjGEOM_MESH
SURFACE_POINTS = 3000


@dataclass(frozen=True)
class Surface:
    points: np.ndarray  # (N, 3) metres, in the object's frame
    normals: np.ndarray  # (N, 3) unit


def quat_wxyz_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a MuJoCo quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def sample_faces(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> Surface:
    """`count` points on a triangle mesh, uniform over its area, each with the unit normal of its face."""
    triangles = np.asarray(vertices, dtype=float)[np.asarray(faces)]  # (F, 3, 3)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    if area.sum() <= 0:
        raise ValueError("the mesh has no area")
    pick = rng.choice(len(triangles), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    outside = u + v > 1  # fold the far half of the parallelogram back into the triangle
    u[outside], v[outside] = 1 - u[outside], 1 - v[outside]
    corner = triangles[pick]
    points = corner[:, 0] + u[:, None] * (corner[:, 1] - corner[:, 0]) + v[:, None] * (corner[:, 2] - corner[:, 0])
    normals = cross[pick] / np.linalg.norm(cross[pick], axis=1, keepdims=True)
    return Surface(points, normals)


def body_meshes(model, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (in the body frame) and faces of the visual meshes of one body: mesh geoms that do not collide."""
    vertices, faces = [], []
    for geom in range(model.ngeom):
        visual = not (model.geom_contype[geom] or model.geom_conaffinity[geom])
        if model.geom_bodyid[geom] != body_id or model.geom_type[geom] != MESH_GEOM or not visual:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        first, nfaces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        rotation = quat_wxyz_matrix(model.geom_quat[geom])
        faces.append(np.asarray(model.mesh_face[first:first + nfaces]) + sum(len(v) for v in vertices))
        vertices.append(np.asarray(model.mesh_vert[start:start + count]) @ rotation.T + model.geom_pos[geom])
    if not vertices:
        raise ValueError(f"body {body_id} has no visual mesh")
    return np.concatenate(vertices), np.concatenate(faces)


def body_surface(model, body_id: int, count: int = SURFACE_POINTS, seed: int = 0) -> Surface:
    """Surface points of one body's visual meshes, in the body frame (reproducible for a seed)."""
    vertices, faces = body_meshes(model, body_id)
    return sample_faces(vertices, faces, count, np.random.default_rng(seed))
