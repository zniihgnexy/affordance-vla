"""Check the evaluation conditions A/B/C and that a seed rebuilds the same scene (CPU only, robocasa365 environment).

For each condition: which kitchens are drawn from, which object-instance split is set, and a fingerprint of the
scene after `reset(seed=s)` (layout, style, the initial simulator state, the episode meta text). The same seed is
reset twice in one process; run the script twice (separate processes) and compare the printed fingerprints.

  source envs/server/robocasa365_env.sh
  "$RC365_PYTHON" scripts/rc365/check_env_conditions.py --task OpenDrawer --seeds 11 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import rc365_eval  # noqa: E402


def fingerprint(env) -> str:
    inner = env.env
    state = np.round(np.asarray(inner.sim.get_state().flatten()), 6)
    meta = json.dumps(inner.get_ep_meta(), sort_keys=True, default=str)
    text = f"{inner.layout_id}/{inner.style_id}/" + hashlib.md5(state.tobytes() + meta.encode()).hexdigest()[:12]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="OpenDrawer")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    args = parser.parse_args()
    for condition in sorted(rc365_eval.CONDITIONS):
        env = rc365_eval.make_env(args.task, condition)
        inner = env.env
        kitchens = sorted({tuple(pair) for pair in inner.layout_and_style_ids})
        print(f"{condition}: obj_instance_split={inner.obj_instance_split} kitchens={len(kitchens)} "
              f"layouts={sorted({layout for layout, _ in kitchens})}", flush=True)
        for seed in args.seeds:
            first = (env.reset(seed=seed), fingerprint(env))[1]
            second = (env.reset(seed=seed), fingerprint(env))[1]
            print(f"  seed {seed}: {first} | again: {second} | {'SAME' if first == second else 'DIFFERENT'}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
