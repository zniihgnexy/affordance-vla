"""AFFO-18, part A: the joint-derived "where it moves next" label of the four part tasks against the old gripper
label, and the stove-knob frame diagnosis. Reads the events of `scripts/data/rc365_extract.py` (extract_v1: it already
holds joint type, axis, anchor and values), writes `joint_parts.json` and `knob_diagnosis.json` under <out>/. CPU only.

  source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
  "$RC365_PYTHON" scripts/data/rc365_joint_check.py --config configs/rc365_labels/joint_check.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from affordance_vla.config import load_config, resolved
from affordance_vla.plan import rc365_events as ev
from affordance_vla.plan import rc365_joint_labels as jl
from affordance_vla.plan import rc365_knob_diagnosis as knob
from affordance_vla.sim import rc365_episodes as eps
from affordance_vla.sim import robocasa365 as rc

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"


def task_touches(cfg: dict, work: Path, results: list[dict], task: str):
    """(touches, layout ids, instructions) of one task, pretrain and target together."""
    touches = jl.joint_touches([r for r in results if r["task"] == task])
    ds = {s: rc.dataset_dir(work / cfg["data_root"], task, s) for s in ("pretrain", "target")}
    metas = [eps.read_meta(ds[t.split], t.episode) for t in touches]
    return touches, [m["layout_id"] for m in metas], [m.get("lang", "") for m in metas]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config, args.set)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    out = work / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "joint_check_config.json").write_text(json.dumps(resolved(cfg, args.config, args.set, REPO), indent=2))
    results = ev.load_results(work / cfg["extract"])
    report = {}
    for task in cfg["tasks"]:
        touches, layouts, langs = task_touches(cfg, work, results, task)
        groups = [knob.stove_type(t.part) for t in touches] if task == cfg["knob_task"] else None
        report[task] = jl.joint_report(touches, layouts, groups)
        print(task, json.dumps(report[task]["contact"])[:300], flush=True)
        if task == cfg["knob_task"]:
            diagnosis = knob.knob_diagnosis(touches, layouts, langs)
            (out / "knob_diagnosis.json").write_text(json.dumps(diagnosis, indent=1))
            print(json.dumps(diagnosis, indent=1), flush=True)
    (out / "joint_parts.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
