"""Which objects an evaluation episode holds, without running a policy (CPU): reset every (task, condition, index)
with its evaluation seed and write the scene's objects. For the episodes already evaluated in AFFO-16 (same seeds,
same scenes), so the moved object's category is known there too. One process per (task, condition) cell.

  source envs/server/robocasa365_env.sh && PYTHONPATH=$W/repos/starVLA_rc365:. \
      "$RC365_PYTHON" scripts/rc365/record_objects.py \
      --out $W/runs/robocasa365_eval/affo19_objects_0_19 --tasks CoffeeSetupMug ... --conditions A B --stop 20
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import rc365_eval  # noqa: E402
from affordance_vla.sim.rc365_policy_io import episode_seed  # noqa: E402
from affordance_vla.sim.rc365_probe import ObjectProbe  # noqa: E402


def read_cell(job: tuple[str, str, int, int, int]) -> list[dict]:
    task, condition, start, stop, base_seed = job
    env = rc365_eval.make_env(task, condition)
    probe = ObjectProbe(env.unwrapped, task)
    rows = []
    try:
        for index in range(start, stop):
            env.reset(seed=episode_seed(base_seed, task, index))
            probe.reset()
            rows.append({"task": task, "condition": condition, "index": index, "objects": probe.result()})
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["A", "B"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, c, args.start, args.stop, args.base_seed) for t in args.tasks for c in args.conditions]
    with Pool(min(args.processes, len(jobs))) as pool:
        rows = [row for cell in pool.map(read_cell, jobs) for row in cell]
    (args.out / "objects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} episodes written to {args.out / 'objects.jsonl'}")


if __name__ == "__main__":
    main()
