"""Build the training set of the plan-following experiment from state-replay logs (CPU only, no GPU).

Input: the `<task>_demo<k>.npz` logs written by rollout_libero_logged.py --replay-states.
Output directory:
  tracks.npz      gripper path, actions, contact flags and object positions of every demonstration
                  (affordance3d/plan_dataset.py)
  agentview.npy   (rows, 224, 224, 3) uint8, one row per recorded state, in the order of tracks.npz
  wrist.npy       the same for the wrist camera
  offsets.json    average gripper position at grasps and releases relative to the object (the oracle plan)
  stats.json      what was kept and dropped, the spread of the offsets, the action-to-motion scale

A demonstration is kept if the target's contact flags show exactly one grasp followed by one release;
stats.json counts how many gripper events the others have.

Usage (starvla env, repo root on PYTHONPATH):
    python build_plan_dataset.py --logs $W/runs/plan_following/demos --out $W/runs/plan_following/dataset
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from affordance3d.caw_labels import fit_action_path
from affordance3d.plan_dataset import frame_offsets, save_tracks
from affordance3d.plan_input import offsets_to_json
from affordance3d.plan_samples import DemoTrack, contact_offsets, single_pick_and_place, track_from_arrays

IMAGE_SIZE = 224  # what starVLA's LIBERO client feeds the policy (bilinear resize of the 256 x 256 frame)


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def resized(frames: np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.stack([np.asarray(Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
                     for frame in frames])


def read_log(path: Path) -> dict:
    """Track arrays and resized images of one log."""
    with np.load(path) as log:
        meta = json.loads(str(log["meta"]))
        target = meta["names"].index(meta["target"])
        if not len(log["agentview"]) == len(log["wrist"]) == len(log["ee_pos"]):
            raise ValueError(f"{path.name}: images and states have different numbers of rows")
        return {"file": path.name, "names": meta["names"], "target": meta["target"],
                "instruction": meta["instruction"], "ee_pos": log["ee_pos"], "actions": log["actions"],
                "object_pos": log["object_pos"], "grasped": log["grasped"][:, target],
                "agentview": resized(log["agentview"]), "wrist": resized(log["wrist"])}


def to_track(record: dict) -> DemoTrack:
    return track_from_arrays(record["ee_pos"], record["actions"], record["grasped"], record["object_pos"],
                             record["names"], record["target"], record["instruction"])


def spread_mm(tracks: list[DemoTrack], container: str) -> dict:
    """Per target object: number of demonstrations and the std (mm, per axis) of the grasp and release offsets."""
    table: dict[str, dict] = {}
    for name in sorted({track.target for track in tracks}):
        group = [track for track in tracks if track.target == name]
        grasps = [t.labels[0].contact - t.object_pos[t.labels[0].event.step, t.names.index(name)] for t in group]
        releases = [t.labels[1].contact - t.object_pos[t.labels[1].event.step, t.names.index(container)]
                    for t in group]
        commands = [[f.step for f in t.command_flips if f.closing and f.step <= t.labels[0].event.step]
                    for t in group]
        delays = [t.labels[0].event.step - steps[-1] for t, steps in zip(group, commands) if steps]
        table[name] = {"demos": len(group), "grasp_offset_std_mm": (np.std(grasps, axis=0) * 1000).round(1).tolist(),
                       "release_offset_std_mm": (np.std(releases, axis=0) * 1000).round(1).tolist(),
                       "steps_from_close_command_to_contact_median": float(np.median(delays)) if delays else None}
    return table


def motion_scale(tracks: list[DemoTrack]) -> float:
    """Median over demonstrations of the metres of gripper motion per unit of translation action."""
    return float(np.median([fit_action_path(track.actions[:, :3], track.ee_pos[:-1]).scale for track in tracks]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--container", default="basket_1", help="the object the target is released into")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    files = sorted((path for path in args.logs.glob("*_demo*.npz") if not path.name.startswith(".")),
                   key=natural_key)  # dot files are logs still being written
    if not files:
        raise SystemExit(f"no *_demo*.npz logs in {args.logs}")
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(read_log, files))
    paired = [(record, to_track(record)) for record in records]
    kept = [(record, track) for record, track in paired if single_pick_and_place(track)]
    if not kept:
        raise SystemExit("no demonstration shows one grasp followed by one release")
    event_counts = Counter(len(track.labels) for _, track in paired)
    tracks = [track for _, track in kept]
    args.out.mkdir(parents=True, exist_ok=True)
    save_tracks(args.out / "tracks.npz", tracks)
    for camera in ("agentview", "wrist"):
        np.save(args.out / f"{camera}.npy", np.concatenate([record[camera] for record, _ in kept]))
    (args.out / "offsets.json").write_text(offsets_to_json(contact_offsets(tracks, args.container)))
    stats = {"logs": len(files), "kept": len(kept), "rows": int(frame_offsets(tracks)[-1]),
             "dropped": sorted({record["file"] for record in records} - {record["file"] for record, _ in kept}),
             "gripper_events_per_demonstration": {str(k): v for k, v in sorted(event_counts.items())},
             "files": [record["file"] for record, _ in kept], "container": args.container,
             "motion_scale_m_per_unit": motion_scale(tracks), "per_target": spread_mm(tracks, args.container)}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({key: value for key, value in stats.items() if key != "files"}, indent=2))
    print("DATASET-OK", flush=True)


if __name__ == "__main__":
    main()
