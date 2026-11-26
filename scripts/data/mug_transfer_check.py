"""Step 0 (1) of Pipeline v4: leave one mug mesh out, carry the interaction map of the other two to it, compare with
the held-out mesh's own demonstrated grasps. Two ways of carrying: nearest point in the category frame, and NDF.

Stages (each reads the outputs of the earlier ones in --out):
  collect   [libero env, CPU]   held grasps, surfaces and up directions of the mugs      -> mugs.npz, collect.json
  ndf       [starvla env, GPU]  every source grasp moved onto each held-out mesh by NDF   -> ndf_<mug>.npz
  evaluate  [any env, CPU]      maps, checks, results table and one figure per held-out mug -> results.json,
                                results.md, figures/<mug>.png
The three stages run from one command by scripts/data/run_mug_transfer_check.sh. Settings: configs/mug_transfer/.

    PYTHONPATH=. python scripts/data/mug_transfer_check.py --config configs/mug_transfer/mug_mesh.yaml --stage collect
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from affordance_vla.config import load_config, resolved
from affordance_vla.plan.category_data import collect_mesh_data, load_mesh_data, save_mesh_data

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"


def stage_collect(cfg: dict, work: Path, out: Path) -> None:
    from affordance_vla.sim.category_surfaces import sample_surfaces

    paths = sorted(p for folder in cfg["demos"] for p in (work / folder).glob("*_demo*.npz"))
    surfaces = sample_surfaces(cfg["suites"], set(cfg["categories"]), cfg["surface_points"], cfg["seed"])
    missing = set(cfg["categories"]) - set(surfaces)
    if missing:
        raise SystemExit(f"no simulator body found for {sorted(missing)}")
    meshes = collect_mesh_data(paths, surfaces, cfg["categories"])
    save_mesh_data(out / "mugs.npz", meshes)
    summary = {"logs": len(paths), "meshes": {
        m.name: {"grasps": len(m.grasps), "demonstrations": len(set(m.logs)),
                 "up_in_body_frame": np.round(m.up, 4).tolist(), "frame_scale": np.round(m.frame.scale, 3).tolist(),
                 "frame_origin_m": np.round(m.frame.origin, 4).tolist(),
                 "handle_direction_in_body_frame": np.round(m.frame.rotation[0], 3).tolist()} for m in meshes}}
    (out / "collect.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def stage_ndf(cfg: dict, work: Path, out: Path, only: list[str]) -> None:
    from affordance_vla.plan.ndf_transfer import load_model, transfer_grasps

    options = cfg["ndf"]
    meshes = load_mesh_data(out / "mugs.npz")
    rng = np.random.default_rng(options["seed"])
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(work / options["weights"], work / options["src"], device)
    for held in meshes:
        if only and held.name not in only:
            continue
        grasps, source_points, from_mesh = [], [], []
        for source in meshes:
            if source.name == held.name:
                continue
            chosen = range(len(source.grasps))
            cap = options["max_grasps_per_source"]
            if cap and len(source.grasps) > cap:
                chosen = sorted(rng.choice(len(source.grasps), cap, replace=False))
            grasps += [source.grasps[i] for i in chosen]
            source_points += [source.surface.points] * len(chosen)
            from_mesh += [source.name] * len(chosen)
        start = time.time()
        placed = transfer_grasps(model, grasps, source_points, held.surface.points, rng, batch=options["batch"],
                                 starts=options["starts"], iterations=options["iterations"])
        print(f"{held.name}: {len(placed)} grasps transferred on {device} in {time.time() - start:.0f} s", flush=True)
        np.savez_compressed(out / f"ndf_{held.name}.npz",
                            centre=np.stack([g.centre for g, _ in placed]),
                            closing=np.stack([g.closing for g, _ in placed]),
                            approach=np.stack([g.approach for g, _ in placed]),
                            loss=np.array([loss for _, loss in placed]),
                            source=np.array(from_mesh))
        if device == "cuda":
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["collect", "ndf", "evaluate"], required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    parser.add_argument("--only", nargs="*", default=[], help="ndf stage: held-out mugs to run (default all)")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("W", DEFAULT_WORK_DIR)))
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    out = args.work_dir / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"config_{args.stage}.json").write_text(json.dumps(resolved(cfg, args.config, args.set, REPO), indent=2))
    if args.stage == "collect":
        stage_collect(cfg, args.work_dir, out)
    elif args.stage == "ndf":
        stage_ndf(cfg, args.work_dir, out, args.only)
    else:
        from affordance_vla.analysis.transfer_report import stage_evaluate

        stage_evaluate(cfg, args.work_dir, out)


if __name__ == "__main__":
    main()
