"""Replay RoboCasa365 demonstrations and record the gripper's held grasps (objects) and touches (parts).

One JSON per episode under <out>/episodes/, one surface file per object instance under <out>/surfaces/ (body
frame, the frame of the grasp records). Episodes already done are skipped, so a run can be resumed. CPU only.
Settings: configs/rc365_labels/extract.yaml (tasks, splits, categories, workers).

  source envs/server/robocasa365_env.sh
  "$RC365_PYTHON" scripts/data/rc365_extract.py --config configs/rc365_labels/extract.yaml [--set workers=8]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

from affordance_vla.config import load_config, resolved
from affordance_vla.sim import rc365_scan as scan
from affordance_vla.sim import robocasa365 as rc
from affordance_vla.sim.rc365_episodes import episode_objects, list_episodes

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def process(env, ds: Path, spec: dict, split: str, task: str, ep: int, out: Path) -> dict:
    meta = json.loads((ds / f"extras/episode_{ep:06d}/ep_meta.json").read_text())
    episode = rc.load_episode(ds, ep)
    rc.reset_to_episode(env, episode)
    result = {"task": task, "split": split, "episode": ep, "lang": meta.get("lang"), "events": []}
    if spec["kind"] == "object":
        tracked = episode_objects(meta, set(spec["categories"]), set(spec["names"]) if spec.get("names") else None)
        result["objects"] = {n: o for n, o in tracked.items()}
        for event in scan.scan_object_grasps(env, episode, {n: o["category"] for n, o in tracked.items()}):
            event["instance"] = tracked[event["object"]]["instance"]
            result["events"].append(event)
        for name, info in tracked.items():
            scan.save_surface(
                out / "surfaces" / f"{info['instance'].replace('/', '__')}.npz", scan.object_surface(env, name)
            )
    else:
        event = scan.scan_part_touch(env, episode)
        result["events"] = [event] if event else []
    result["steps"] = len(episode["states"])
    return result


def worker(jobs: list[tuple[str, str, list[int]]], cfg: dict, root: str, out: str) -> None:
    out_dir = Path(out)
    envs: dict[tuple[str, str], object] = {}
    for split, task, episodes in jobs:
        ds = rc.dataset_dir(root, task, split)
        env = envs.setdefault((split, task), rc.make_env(ds))
        for ep in episodes:
            target = out_dir / "episodes" / f"{split}_{task}_{ep:04d}.json"
            if target.exists():
                continue
            began = time.time()
            try:
                result = process(env, ds, cfg["tasks"][task], split, task, ep, out_dir)
                result["seconds"] = time.time() - began
                target.write_text(json.dumps(_jsonable(result)))
                print(f"{split} {task} {ep}: {len(result['events'])} events {result['seconds']:.0f}s", flush=True)
            except Exception:  # recorded, never skipped silently: the summary counts these files
                (out_dir / "errors").mkdir(exist_ok=True)
                (out_dir / "errors" / target.name).write_text(traceback.format_exc())
                print(f"{split} {task} {ep}: FAILED", flush=True)
        envs.pop((split, task)).close()


def split_jobs(items: list[tuple[str, str, int]], workers: int) -> list[list[tuple[str, str, list[int]]]]:
    """Episodes dealt round-robin to workers, grouped per (split, task) so a worker builds few environments."""
    buckets: list[dict[tuple[str, str], list[int]]] = [{} for _ in range(workers)]
    for k, (split, task, ep) in enumerate(items):
        buckets[k % workers].setdefault((split, task), []).append(ep)
    return [[(s, t, eps) for (s, t), eps in b.items()] for b in buckets if b]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config, args.set)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    root, out = str(work / cfg["data_root"]), work / cfg["out"]
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    (out / "extract_config.json").write_text(json.dumps(resolved(cfg, args.config, args.set, REPO), indent=2))
    items = [
        (split, task, ep)
        for task, spec in cfg["tasks"].items()
        for split in cfg["splits"]
        for ep in list_episodes(
            rc.dataset_dir(root, task, split), spec.get("categories"), cfg.get("max_episodes"), spec.get("names")
        )
    ]
    print(f"{len(items)} episodes, {cfg['workers']} workers -> {out}", flush=True)
    jobs = split_jobs(items, min(cfg["workers"], len(items)))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        pool.starmap(worker, [(j, cfg, root, str(out)) for j in jobs])


if __name__ == "__main__":
    main()
