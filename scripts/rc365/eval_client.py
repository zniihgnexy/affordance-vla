"""One evaluation client: episodes of one (task, condition) cell against a running starVLA policy server.

Run in the robocasa365 environment (see scripts/rc365/run_eval_base_gpu.sh, which starts the server and many of
these). Episodes already in the output file are skipped, so a restarted job continues where it stopped.

  PYTHONPATH=<starVLA snapshot>:. "$RC365_PYTHON" scripts/rc365/eval_client.py --task OpenDrawer --condition A \
      --start 0 --stop 5 --port 5600 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import rc365_eval  # noqa: E402


def done_indices(path: Path, task: str, condition: str) -> set[int]:
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["index"] for r in rows if r["task"] == task and r["condition"] == condition}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(rc365_eval.CONDITIONS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--n-act", type=int, default=8, help="actions executed per policy query")
    parser.add_argument("--videos-per-cell", type=int, default=3)
    parser.add_argument("--probes", default="", help="comma-separated recorders (affordance_vla.sim.rc365_probe)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "episodes.jsonl"
    todo = [i for i in range(args.start, args.stop) if i not in done_indices(results, args.task, args.condition)]
    if not todo:
        print(f"{args.task} {args.condition} {args.start}:{args.stop} already done")
        return
    policy = rc365_eval.ServerPolicy(args.host, args.port)
    try:
        records = rc365_eval.run_cell(
            args.task, args.condition, todo, policy, args.base_seed, args.n_act, results,
            video_dir=args.out / "videos", videos_per_cell=args.videos_per_cell,
            probe_names=[name for name in args.probes.split(",") if name],
        )
    finally:
        policy.close()
    wins = sum(r["success"] for r in records)
    print(f"{args.task} {args.condition} {args.start}:{args.stop}: {wins}/{len(records)} succeeded")


if __name__ == "__main__":
    main()
