"""The evaluate stage of the mug transfer check (scripts/data/mug_transfer_check.py): maps, checks, table, figures."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from affordance_vla.analysis.transfer_figure import transfer_figure
from affordance_vla.plan.category_data import MeshData, load_mesh_data
from affordance_vla.plan.category_frame import CategoryFrame, scene_aligned_frame
from affordance_vla.plan.category_transfer import Source, category_map, transfer_nearest
from affordance_vla.plan.interaction_labels import CanonicalMap, Grasp, canonical_map
from affordance_vla.plan.transfer_check import CheckResult, grasp_outcomes, summarise
from affordance_vla.sim.rollout_log import load_rollout_log

BODY_FRAME = CategoryFrame(np.zeros(3), np.eye(3), np.ones(3))  # the meshes' own frames, no normalisation
BODY_REACH_M = 0.01  # nearest point within 1 cm in the unnormalised body frame (reference without a category frame)


def load_ndf_grasps(path: Path) -> tuple[list[Grasp], np.ndarray, np.ndarray]:
    with np.load(path) as data:
        grasps = [Grasp(*row) for row in zip(data["centre"], data["closing"], data["approach"])]
        return grasps, data["loss"], data["source"]


def self_split(mesh: MeshData) -> tuple[CheckResult, int, int]:
    """Reference: the mesh's own demonstrations split by log (even / odd), each half's map tested on the other half."""
    logs = sorted(set(mesh.logs))
    half = {name: k % 2 for k, name in enumerate(logs)}
    parts = [[g for g, name in zip(mesh.grasps, mesh.logs) if half[name] == h] for h in (0, 1)]
    outcomes = grasp_outcomes(canonical_map(mesh.name, mesh.surface, parts[0]), mesh.surface, parts[1]) + \
        grasp_outcomes(canonical_map(mesh.name, mesh.surface, parts[1]), mesh.surface, parts[0])
    return summarise(outcomes), len(parts[0]), len(parts[1])


def rest_quat(mesh: MeshData, demo_dirs: list[Path]) -> np.ndarray:
    """The object's orientation (x, y, z, w) at step 0 of the first demonstration it is held in."""
    path = next(folder / mesh.logs[0] for folder in demo_dirs if (folder / mesh.logs[0]).exists())
    return load_rollout_log(path, images=False).object_quat[0, mesh.objects[0]]


def evaluate_mesh(held: MeshData, others: list[MeshData], cfg: dict, out: Path,
                  demo_dirs: list[Path]) -> tuple[dict, dict[str, CanonicalMap]]:
    sources = [Source(m.name, m.frame, m.surface, m.grasps) for m in others]
    maps, rows = {}, {}
    cmap = category_map(sources)
    maps["nearest point (category frame)"] = transfer_nearest(cmap, held.frame, held.surface, held.name,
                                                              cfg["nearest"]["max_distance"])
    body = category_map([Source(m.name, BODY_FRAME, m.surface, m.grasps) for m in others])
    reference = transfer_nearest(body, BODY_FRAME, held.surface, held.name, BODY_REACH_M)
    aligned = [Source(m.name, scene_aligned_frame(m.frame, rest_quat(m, demo_dirs)), m.surface, m.grasps)
               for m in others]
    scene_frame = scene_aligned_frame(held.frame, rest_quat(held, demo_dirs))
    scene = transfer_nearest(category_map(aligned), scene_frame, held.surface, held.name,
                             cfg["nearest"]["max_distance"])
    ndf_path = out / f"ndf_{held.name}.npz"
    if ndf_path.exists():
        moved, loss, _ = load_ndf_grasps(ndf_path)
        maps["NDF"] = canonical_map(held.name, held.surface, moved)
        rows["NDF"] = {"source_grasps": len(moved), "median_descriptor_loss": float(np.median(loss))}
    keys = {"nearest point (category frame)": "nearest", "NDF": "ndf"}
    results = {"nearest": {"source_grasps": cmap.grasps}}
    for label, name in keys.items():
        if label in maps:
            results.setdefault(name, {}).update(rows.get(label, {}))
            results[name] |= asdict(summarise(grasp_outcomes(maps[label], held.surface, held.grasps)))
    results["reference_body_frame"] = asdict(summarise(grasp_outcomes(reference, held.surface, held.grasps)))
    own, n_a, n_b = self_split(held)
    results["diagnostic_scene_aligned"] = asdict(summarise(grasp_outcomes(scene, held.surface, held.grasps)))
    results["reference_self_split"] = asdict(own) | {"halves": [n_a, n_b]}
    maps["own demonstrations"] = canonical_map(held.name, held.surface, held.grasps)
    return results, maps


def _row(held: str, sources: str, method: str, r: dict) -> str:
    landing, axis = r["landing_rate"] >= 0.8, r["axis_error_median_deg"] < 20
    return (f"| {held} | {sources} | {method} | {r['grasps']} | {r['landing_rate']:.2f} {'✓' if landing else '✗'} | "
            f"{r['axis_error_median_deg']:.1f}° {'✓' if axis else '✗'} | {'pass' if landing and axis else 'fail'} | "
            f"{r['mean_region_weight']:.2f} | {r['untouched']} / {r['no_axis']} |")


def markdown(results: dict) -> str:
    lines = ["| held-out | sources (grasps) | method | held-out grasps | landing rate (≥ 0.8) "
             "| median axis error (< 20°) | both | mean weight in region | untouched / no axis |",
             "|---|---|---|---|---|---|---|---|---|"]
    names = {"nearest": "nearest point", "ndf": "NDF", "reference_body_frame": "reference: body frames, no transfer",
             "diagnostic_scene_aligned": "diagnostic: nearest point, frame turned to the scene",
             "reference_self_split": "reference: own demos, half / half"}
    for held, row in results.items():
        sources = ", ".join(f"{k} ({v})" for k, v in row["sources"].items())
        for key, label in names.items():
            if key in row:
                lines.append(_row(held, sources, label, row[key]))
    return "\n".join(lines)


def stage_evaluate(cfg: dict, work: Path, out: Path) -> None:
    meshes = load_mesh_data(out / "mugs.npz")
    (out / "figures").mkdir(exist_ok=True)
    demo_dirs = [work / folder for folder in cfg["demos"]]
    results = {}
    for held in meshes:
        others = [m for m in meshes if m is not held]
        result, maps = evaluate_mesh(held, others, cfg, out, demo_dirs)
        result["sources"] = {m.name: len(m.grasps) for m in others}
        results[held.name] = result
        transfer_figure(held, maps, demo_dirs, out / "figures" / f"{held.name}.png",
                        f"held out: {held.name} ({len(held.grasps)} grasps in {len(set(held.logs))} demonstrations)")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "results.md").write_text(markdown(results) + "\n")
    print(markdown(results))
