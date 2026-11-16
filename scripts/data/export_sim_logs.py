"""Copy closed-loop rollout logs without their camera images, so they can be replayed on another machine.

A log of scripts/eval/rollout_libero_logged.py keeps every step's camera images, which makes a run of 400
episodes about 1.5 GB. A replay needs only the task, the start, the executed actions and what the policy
returned at each query; this keeps every array except the images (a few kB per episode) and the per-task
summaries, under the same relative paths (<run>/<suite>/<file>).

Usage on the server (numpy only):
    python scripts/data/export_sim_logs.py --root $W/runs/libero_pro --out $W/cache/sim_logs/libero_pro \\
        --runs released predicted_g1_hold_s10 oracle_plan_s10 spatial_released_seed7
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim.rollout_log import IMAGE_KEYS, OPTIONAL_IMAGE_KEYS  # noqa: E402

IMAGE_ARRAYS = set(IMAGE_KEYS) | set(OPTIONAL_IMAGE_KEYS)


def export_log(source: Path, target: Path) -> bool:
    """Write `source` without its image arrays to `target`; False if `target` is already there."""
    if target.exists():
        return False
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files if key not in IMAGE_ARRAYS}
    if "actions" not in arrays or "meta" not in arrays:
        raise ValueError(f"{source} is not a rollout log (no actions or meta)")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.stem}.partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(target)
    return True


def export_run(root: Path, run: str, out: Path) -> tuple[int, int]:
    """(logs written, logs already present) for one run folder."""
    folder = root / run
    if not folder.is_dir():
        raise SystemExit(f"no run folder {folder}")
    written = present = 0
    for source in sorted(folder.glob("*/*.npz")):
        relative = source.relative_to(root)
        if export_log(source, out / relative):
            written += 1
        else:
            present += 1
    for summary in sorted(folder.glob("*/*_summary*.json")):
        target = out / summary.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, target)
    return written, present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="folder that holds the run folders")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", required=True, nargs="+", help="run folder names under --root")
    args = parser.parse_args()
    for run in args.runs:
        written, present = export_run(args.root, run, args.out)
        print(f"{run}: {written} logs written, {present} already there", flush=True)


if __name__ == "__main__":
    main()
