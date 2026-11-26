"""AFFO-11: can HOT3D's hand and object annotations give an interaction map like ours (interaction_labels.py)?

Reads HOT3D-Clips tars with a grasped object, and for the object:
1. draws the UmeTrack hand mesh and the object mesh on a camera frame (overlay.png);
2. counts, over the frames, how often each surface point has a hand point within the threshold, and finds the
   frames where the thumb and the other fingers oppose and the closing axis of each (a "held grasp" is a run of
   at least --min-hold such frames);
3. turns the held grasps into one `CanonicalMap` of interaction_labels.py (same file format as the maps of the
   LIBERO demonstrations) and draws it next to the contact frequency (contact.png).

Outputs in --out: overlay.png, contact.png, maps.npz (interaction_labels.load_maps), summary.json.

Usage on the server (CPU only, hot3d env, from the repository root):
    source envs/server/hot3d_env.sh
    PYTHONPATH=. python scripts/data/hot3d_contact_demo.py --out $W/runs/hot3d_demo/mug_white
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from affordance_vla.hot3d import contact as C
from affordance_vla.hot3d.analysis import analyse_clip, grasps_of, held_grasps
from affordance_vla.hot3d.clips import Clip, load_object_mesh, object_name
from affordance_vla.hot3d.figures import contact_figure, overlay_figure
from affordance_vla.plan.interaction_labels import axis_groups, canonical_map, save_maps
from affordance_vla.sim.object_surface import SURFACE_POINTS, sample_faces

HOT3D_DIR = Path(os.environ.get("HOT3D_DIR", "/mnt/nas/xinyizheng/affordance_vla/data/hot3d"))
DEFAULT_CLIPS = ("clip-003345", "clip-003346")  # train_aria clips of sequence P0015_dbe00981 with the white mug held
SENSITIVITY_MM = (3.0, 5.0, 10.0)
CONTACT_SURFACE_POINTS = 20000  # dense enough for a 5 mm threshold on a 10 cm mug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="+", default=list(DEFAULT_CLIPS), help="clip names in <hot3d>/clips")
    parser.add_argument("--object-id", default="9", help="HOT3D object id (9 = mug_white, 8 = mug_patterned)")
    parser.add_argument("--threshold-mm", type=float, default=5.0, help="a hand point this close touches the surface")
    parser.add_argument("--hand-points", type=int, default=4000, help="hand mesh points; 0 = the mesh's vertices")
    parser.add_argument("--min-hold", type=int, default=5, help="frames an opposition must last to count as a grasp")
    parser.add_argument("--max-drift-deg", type=float, default=20.0,
                        help="a grasp run is cut where the closing axis turns this far from its first")
    parser.add_argument("--stride", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hot3d-dir", type=Path, default=HOT3D_DIR)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.hot3d_dir / "object_models"
    name = object_name(models, args.object_id)
    mesh = load_object_mesh(models, args.object_id)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    surface = sample_faces(vertices, faces, CONTACT_SURFACE_POINTS, np.random.default_rng(args.seed))
    main_t = args.threshold_mm / 1000
    # the first threshold also decides the opposition; the others only feed the sensitivity numbers
    thresholds = (main_t, *sorted({t / 1000 for t in SENSITIVITY_MM} - {main_t}))

    clips = [Clip(args.hot3d_dir / "clips" / f"{c}.tar") for c in args.clips]
    samples, triangles = C.sample_hand(clips[0].hand_model, args.hand_points, args.seed)
    results, grasp_rows = [], []
    for clip in clips:
        result = analyse_clip(clip, args.object_id, surface.points, samples, triangles, thresholds, args.stride)
        results.append(result)
        grasp_rows += held_grasps(result, args.min_hold, args.max_drift_deg)

    args.out.mkdir(parents=True, exist_ok=True)
    frames = {t: np.concatenate([r.touched[t] for r in results]) for t in thresholds}
    frequency = frames[main_t].mean(axis=0)
    held_frames = np.concatenate([r.opposed() for r in results])
    grasps = grasps_of(grasp_rows)
    low_res = sample_faces(vertices, faces, SURFACE_POINTS, np.random.default_rng(0))  # our maps' surface
    cmap = canonical_map(name, low_res, grasps) if grasps else None
    if cmap is not None:
        save_maps(args.out / "maps.npz", [cmap])

    shots = []  # per clip, the middle frame of its longest held grasp
    for clip in clips:
        rows_of_clip = [row for row in grasp_rows if row["clip"] == clip.path.name]
        if rows_of_clip:
            row = max(rows_of_clip, key=lambda r: r["last_frame"] - r["first_frame"])
            title = f"{clip.path.stem} frame {row['frame']}"
            shots.append((clip, row["frame"], args.object_id, vertices, faces, title))
    if not shots:  # no grasp found: show the first frame of the first clip
        shots = [(clips[0], clips[0].frames[0], args.object_id, vertices, faces, f"{clips[0].path.stem} frame 0")]
    overlay_figure(shots, C.vertex_fingers(clips[0].hand_model), triangles, args.out / "overlay.png")
    rows = [(f"touch frequency ({args.threshold_mm:g} mm)", surface.points, frequency, max(frequency.max(), 1e-9))]
    if cmap is not None:
        rows.append(("map of interaction_labels.py\n(score from the held grasps)", cmap.points, cmap.score, 1.0))
    contact_figure(vertices, faces, rows, grasps, args.out / "contact.png",
                   f"{name}: {len(held_frames)} frames, {int(held_frames.sum())} with thumb-finger opposition, "
                   f"{len(grasps)} held grasps; cyan = closing axes")

    summary = {
        "object": name, "object_id": args.object_id, "clips": [c.path.name for c in clips],
        "sequences": sorted({c.sequence for c in clips}), "frames": int(len(held_frames)),
        "hand_points": args.hand_points, "threshold_mm": args.threshold_mm,
        "touched_surface_fraction": {f"{t * 1000:g}mm": float(frames[t].any(axis=0).mean()) for t in thresholds},
        "frames_touching": {f"{t * 1000:g}mm": int(frames[t].any(axis=1).sum()) for t in thresholds},
        "frames_opposed": int(held_frames.sum()), "max_drift_deg": args.max_drift_deg,
        "opposed_runs_before_drift_cut": [[r.clip, a, b] for r in results
                                          for a, b in C.hold_segments(r.opposed(), args.min_hold)],
        "max_frequency": float(frequency.max()),
        "held_grasps": [{k: (v if k != "grasp" else {"centre": v.centre.tolist(), "closing": v.closing.tolist(),
                                                     "approach": v.approach.tolist()}) for k, v in row.items()}
                        for row in grasp_rows],
        "closing_axis_groups_20deg": axis_groups([g.closing for g in grasps]) if grasps else [],
        "map": None if cmap is None else {"points": len(cmap.points), "grasps": cmap.grasps,
                                          "score_gt_0.5": float((cmap.score > 0.5).mean()),
                                          "mean_agreement_on_touched": float(cmap.agreement[cmap.score > 0].mean())},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("object", "frames", "frames_touching", "frames_opposed", "map",
                                               "closing_axis_groups_20deg")}, indent=2, default=float))


if __name__ == "__main__":
    main()
