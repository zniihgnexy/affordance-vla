"""L of Pipeline v3: build the 3D canonical interaction map of every object category from replayed demonstrations.

For every object category held in the demonstrations, its surface is sampled from the simulator's visual
meshes, every held grasp is put into the object's frame, and the grasps are summed
(affordance_vla/plan/interaction_labels.py). The demonstrations must have been replayed with object
orientations (runs/plan_following/demos_pose, demos_libero10).

Outputs in --out:
- maps.npz              one canonical map per category (interaction_labels.load_maps)
- summary.json          per category: grasps, groups of closing axes in the object frame, agreement; the inputs
- check/<category>.png  the map from above and from the side, and projected onto the first frame of one
                        demonstration next to the frame where that demonstration's grasp registers

Usage on the server (CPU only, libero env, from the repository root):
    source envs/server/libero_env.sh
    PYTHONPATH=.:$PYTHONPATH $LIBERO_PYTHON scripts/data/build_interaction_labels.py --suite libero_object \\
        --demos $W/runs/plan_following/demos_pose --out $W/runs/interaction_labels/object
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from affordance_vla.analysis.grasp_orientation import object_category
from affordance_vla.plan.interaction_labels import (
    axis_groups,
    canonical_map,
    grasp_in_object,
    held_grasp_steps,
    project_map,
    save_maps,
)
from affordance_vla.sim.object_surface import SURFACE_POINTS, body_surface
from affordance_vla.sim.rollout_log import load_rollout_log

HIGH_SCORE = 0.5  # map points counted as the contact region in the summary


def collect_grasps(paths: list[Path]) -> tuple[dict[str, list], dict[str, tuple[Path, int, int]]]:
    """Held grasps per category in the object frame, and one (log, object index, grasp step) per category."""
    grasps, examples = defaultdict(list), {}
    for path in paths:
        log = load_rollout_log(path, images=False)
        if log.object_quat is None:
            raise SystemExit(f"{path.name} has no object orientations: replay the demonstrations again")
        for k, step in held_grasp_steps(log.grasped):
            category = object_category(log.names[k])
            grasps[category].append(grasp_in_object(log.ee_pos[step], log.ee_quat[step], log.object_pos[step, k],
                                                    log.object_quat[step, k]))
            examples.setdefault(category, (path, k, step))
    return grasps, examples


def sample_surfaces(suite: str, categories: set[str], count: int, seed: int) -> dict:
    """The surface of one object of each category, read from the simulator of the suite's tasks."""
    from affordance_vla.sim.episode import make_env

    surfaces, task, n_tasks = {}, 0, 1
    while task < n_tasks and not categories <= set(surfaces):
        benchmark, _, env = make_env(suite, task, seed, cameras=False)
        n_tasks, inner = benchmark.n_tasks, env.env
        for name in sorted(inner.objects_dict):
            category = object_category(name)
            if category in categories and category not in surfaces:
                surfaces[category] = body_surface(inner.sim.model, inner.obj_body_id[name], count, seed)
        env.close()
        task += 1
    missing = categories - set(surfaces)
    if missing:
        raise SystemExit(f"no simulator body found for {sorted(missing)}")
    return surfaces


def check_figure(cmap, example: tuple[Path, int, int], out_path: Path) -> None:
    """Top and side views of the map (closing axes on the best points), and the map projected onto a frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from affordance_vla.heads.plan_grid import camera_of_log, contact_pixel

    path, k, step = example
    log = load_rollout_log(path)
    label = project_map(cmap, log.object_pos[0, k], log.object_quat[0, k], log.intrinsics, log.cam_to_world,
                        log.seg[0], k + 1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    best = np.argsort(cmap.score)[-25:]
    for axis, (i, j, name) in zip(axes[:2], [(0, 1, "from above (x, y)"), (0, 2, "from the side (x, z)")]):
        axis.scatter(cmap.points[:, i], cmap.points[:, j], c=cmap.score, s=3, cmap="viridis", vmin=0, vmax=1)
        for p in best:
            ends = cmap.points[p] + np.outer([-0.01, 0.01], cmap.axis[p])
            axis.plot(ends[:, i], ends[:, j], color="red", linewidth=0.8)
        axis.set_aspect("equal")
        axis.set_title(f"{cmap.category}: {name}", fontsize=9)
    size = log.agentview.shape[1]
    axes[2].imshow(log.agentview[0])
    axes[2].imshow(np.kron(label.score, np.ones((size // 8, size // 8))), alpha=0.45, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(f"map projected on frame 0 ({label.visible} points visible)", fontsize=9)
    camera = camera_of_log(log)
    u, v, _ = contact_pixel(log.ee_pos[step], camera)
    scale = size / 224
    axes[3].imshow(log.agentview[step])
    axes[3].plot(u * scale, v * scale, "x", color="orange", markersize=12, markeredgewidth=2)
    axes[3].set_title(f"grasp registers at step {step} (x = gripper)", fontsize=9)
    for axis in axes[2:]:
        axis.set_axis_off()
    fig.suptitle(path.name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, help="LIBERO suite whose simulator holds the objects")
    parser.add_argument("--demos", required=True, type=Path, nargs="+", help="folders of replayed demonstrations")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--points", type=int, default=SURFACE_POINTS, help="surface points per category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-deg", type=float, default=20.0, help="angle within which closing axes group")
    args = parser.parse_args()
    paths = sorted(p for folder in args.demos for p in folder.glob("*_demo*.npz"))
    if not paths:
        raise SystemExit("no demonstration logs found")
    grasps, examples = collect_grasps(paths)
    surfaces = sample_surfaces(args.suite, set(grasps), args.points, args.seed)
    maps = [canonical_map(category, surfaces[category], grasps[category]) for category in sorted(grasps)]
    (args.out / "check").mkdir(parents=True, exist_ok=True)
    save_maps(args.out / "maps.npz", maps)
    summary = {"logs": len(paths), "categories": {}}
    for cmap in maps:
        region = cmap.score >= HIGH_SCORE
        summary["categories"][cmap.category] = {
            "grasps": cmap.grasps,
            "axis_groups": axis_groups([g.closing for g in grasps[cmap.category]], args.group_deg),
            "contact_share": round(float(region.mean()), 4),
            "agreement_in_contact": round(float(cmap.agreement[region].mean()), 3) if region.any() else None,
        }
        check_figure(cmap, examples[cmap.category], args.out / "check" / f"{cmap.category}.png")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary["args"] = {key: str(value) for key, value in vars(args).items()} | {"commit": commit}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["categories"], indent=2))


if __name__ == "__main__":
    main()
