"""The inputs of the category-level transfer check: per mesh its surface, up direction and held grasps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from affordance_vla.plan.category_frame import CategoryFrame, fit_frame, up_at_rest
from affordance_vla.plan.demo_grasps import scan_demos
from affordance_vla.plan.interaction_labels import Grasp
from affordance_vla.sim.object_surface import Surface

MAX_UP_SPREAD_DEG = 5.0  # the objects' up directions at rest, over all demonstrations, must agree to this


@dataclass(frozen=True)
class MeshData:
    name: str
    surface: Surface  # body frame, metres
    up: np.ndarray  # (3,) the body-frame direction of world up when the object rests on the table
    grasps: tuple[Grasp, ...]  # body frame
    logs: tuple[str, ...]  # per grasp: the demonstration log it came from (file name)
    objects: tuple[int, ...]  # per grasp: the object's column in that log
    steps: tuple[int, ...]  # per grasp: the step it registers

    @property
    def frame(self) -> CategoryFrame:
        return fit_frame(self.surface.points, self.up)


def collect_mesh_data(paths: Iterable[Path], surfaces: dict[str, Surface], names: Sequence[str]) -> list[MeshData]:
    """The held grasps of every named mesh in the replayed demonstrations, with its surface and up direction."""
    scan = scan_demos(paths, set(names))
    meshes = []
    for name in names:
        found = [g for g in scan.grasps if g.category == name]
        if not found or name not in scan.rest_quats:
            raise SystemExit(f"no held grasp of {name} in the demonstrations")
        up = up_at_rest(scan.rest_quats[name])
        spread = max(np.degrees(np.arccos(np.clip(up_at_rest(q[None]) @ up, -1, 1))) for q in scan.rest_quats[name])
        if spread > MAX_UP_SPREAD_DEG:
            raise SystemExit(f"{name} does not rest upright: up directions differ by {spread:.1f} degrees")
        meshes.append(MeshData(name, surfaces[name], up, tuple(g.grasp for g in found),
                               tuple(g.log.name for g in found), tuple(g.object_index for g in found),
                               tuple(g.step for g in found)))
    return meshes


def save_mesh_data(path: Path, meshes: Sequence[MeshData]) -> None:
    arrays = {"names": np.array([m.name for m in meshes])}
    for m in meshes:
        arrays |= {f"{m.name}/points": m.surface.points, f"{m.name}/normals": m.surface.normals, f"{m.name}/up": m.up,
                   f"{m.name}/centre": np.stack([g.centre for g in m.grasps]),
                   f"{m.name}/closing": np.stack([g.closing for g in m.grasps]),
                   f"{m.name}/approach": np.stack([g.approach for g in m.grasps]),
                   f"{m.name}/objects": np.array(m.objects), f"{m.name}/steps": np.array(m.steps),
                   f"{m.name}/logs": np.array(json.dumps(m.logs))}
    np.savez_compressed(path, **arrays)


def load_mesh_data(path: Path) -> list[MeshData]:
    with np.load(path) as data:
        meshes = []
        for name in map(str, data["names"]):
            grasps = tuple(Grasp(*row) for row in zip(data[f"{name}/centre"], data[f"{name}/closing"],
                                                      data[f"{name}/approach"]))
            meshes.append(MeshData(name, Surface(data[f"{name}/points"], data[f"{name}/normals"]), data[f"{name}/up"],
                                   grasps, tuple(json.loads(str(data[f"{name}/logs"]))),
                                   tuple(int(k) for k in data[f"{name}/objects"]),
                                   tuple(int(s) for s in data[f"{name}/steps"])))
    return meshes
