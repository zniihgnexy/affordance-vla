"""The picture of the transfer check: for one held-out mesh, each map carried to it and the mesh's own demonstrated
grasps, on one demonstration frame (the object's pose at step 0 of the demonstration where it is most visible)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from affordance_vla.analysis.grasp_orientation import rotation_matrices
from affordance_vla.plan.category_data import MeshData
from affordance_vla.plan.interaction_labels import CELLS, CanonicalMap, project_map
from affordance_vla.sim.libero_camera import project_points
from affordance_vla.sim.rollout_log import load_rollout_log


def best_frame(mesh: MeshData, demo_dirs: list[Path]) -> tuple[Path, int]:
    """(log path, object column) of the demonstration of this mesh whose first frame shows the most of it."""
    best, found = -1, None
    for name, k in sorted(set(zip(mesh.logs, mesh.objects))):
        path = next(folder / name for folder in demo_dirs if (folder / name).exists())
        pixels = int(load_rollout_log(path, images=False).visible_px[0, k])
        if pixels > best:
            best, found = pixels, (path, k)
    return found


def transfer_figure(mesh: MeshData, maps: Mapping[str, CanonicalMap], demo_dirs: list[Path], out_path: Path,
                    title: str) -> None:
    """One panel per map: its projected scores on the frame (8 x 8 cells), the mesh's grasp points as orange x."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path, k = best_frame(mesh, demo_dirs)
    log = load_rollout_log(path)
    size = log.agentview.shape[1]
    pos, quat = log.object_pos[0, k], log.object_quat[0, k]
    centres = np.stack([g.centre for g in mesh.grasps]) @ rotation_matrices(quat).T + pos
    uv, _ = project_points(centres, log.intrinsics, log.cam_to_world, width=size, mirrored=True)
    fig, axes = plt.subplots(1, len(maps), figsize=(4.4 * len(maps), 4.6))
    for axis, (label, cmap) in zip(np.atleast_1d(axes), maps.items()):
        frame_label = project_map(cmap, pos, quat, log.intrinsics, log.cam_to_world, log.seg[0], k + 1)
        axis.imshow(log.agentview[0])
        axis.imshow(np.kron(frame_label.score, np.ones((size // CELLS, size // CELLS))), alpha=0.5, cmap="magma",
                    vmin=0, vmax=1)
        axis.plot(uv[:, 0], uv[:, 1], "x", color="orange", markersize=5, markeredgewidth=1.2, alpha=0.8)
        axis.set_title(f"{label} ({frame_label.visible} points visible)", fontsize=9)
        axis.set_axis_off()
    fig.suptitle(f"{title}   [{path.name}]", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
