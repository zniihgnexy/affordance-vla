"""Step 1 (1) of Pipeline v4 (AFFO-15): what the extracted RoboCasa365 events say.

Stages (each reads <extract>/ of scripts/data/rc365_extract.py and writes to <out>/; CPU only):
  counts    what the demonstrations hold: manipulated objects per task and split, picks found per category
  objects   mug / cup / bowl: azimuth of the grasps (object frame vs robot-facing), transfer from the pretrain
            instances to the target instances with three treatments (plain, ring, robot frame), own-demo reference
  parts     drawer handle, door handle, stove knob, faucet handle: contact, closing line, motion, consistency
  all       the three in order

  source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
  "$RC365_PYTHON" scripts/data/rc365_label_check.py --config configs/rc365_labels/check.yaml --stage all
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path

from affordance_vla.config import load_config, resolved
from affordance_vla.plan import rc365_events as ev
from affordance_vla.plan import rc365_parts as parts_lib
from affordance_vla.plan import rc365_symmetry as sym
from affordance_vla.sim import rc365_episodes as eps
from affordance_vla.sim import robocasa365 as rc

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"
PART_TASKS = ("OpenDrawer", "OpenCabinet", "TurnOffStove", "TurnOnSinkFaucet")


def _jsonable(value):
    if is_dataclass(value):
        extra = {k: getattr(value, k) for k in ("landing_pass", "axis_pass", "passed") if hasattr(value, k)}
        return _jsonable({**asdict(value), **extra})
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def stage_counts(cfg: dict, work: Path, out: Path) -> dict:
    root, results = work / cfg["data_root"], ev.load_results(work / cfg["extract"])
    table = {}
    for split in ("pretrain", "target"):
        for ds in sorted(root.glob(f"v1.0/{split}/atomic/*/*/lerobot")):
            task = ds.parts[-3]
            count = eps.manipulated_counts(ds) if task.startswith(("PickPlace", "Coffee")) else None
            table[f"{split}/{task}"] = {
                "episodes": eps.episode_count(ds),
                "manipulated": dict(count.most_common()) if count else None,
            }
    picks = ev.object_grasps(results)
    found = defaultdict(lambda: defaultdict(Counter))
    for g in picks:
        found[g.category][g.split][g.task] += 1
    instances = defaultdict(lambda: defaultdict(set))
    for g in picks:
        instances[g.category][g.split].add(g.instance)
    summary = {
        "tasks": table,
        "lifted_picks": {c: {s: dict(t) for s, t in v.items()} for c, v in found.items()},
        "instances": {c: {s: len(i) for s, i in v.items()} for c, v in instances.items()},
        "approach_z_mean": float(sum(g.approach_world_z for g in picks) / max(len(picks), 1)),
    }
    (out / "counts.json").write_text(json.dumps(summary, indent=1))
    return summary


def stage_objects(cfg: dict, work: Path, out: Path) -> dict:
    folder, results = work / cfg["extract"], ev.load_results(work / cfg["extract"])
    picks = ev.object_grasps(results)
    report = {}
    for category, options in cfg["categories"].items():
        own = [g for g in picks if g.category == category]
        side = {s: [g for g in own if g.split == s] for s in ("pretrain", "target")}
        entry = {
            "picks": {s: len(v) for s, v in side.items()},
            "instances": {s: len({g.instance for g in v}) for s, v in side.items()},
        }
        if min(len(side["pretrain"]), len(side["target"])) < cfg["min_grasps"]:
            entry["status"] = f"too few picks (need {cfg['min_grasps']} on each side)"
            report[category] = entry
            continue
        surfaces = {g.instance: ev.load_surface(folder, g.instance) for g in own}
        sources, targets = sym.build_items(side["pretrain"], surfaces), sym.build_items(side["target"], surfaces)
        entry["azimuth"] = {
            s: sym.azimuth_distributions(items) for s, items in (("pretrain", sources), ("target", targets))
        }
        entry["transfer"] = {t: sym.evaluate(t, sources, targets, options["handle"]) for t in cfg["treatments"]}
        entry["own_demos"] = {t: sym.evaluate_own(t, targets, options["handle"]) for t in cfg["treatments"]}
        entry["status"] = "checked"
        report[category] = entry
        print(category, {t: _jsonable(r["pooled"]) for t, r in entry["transfer"].items()}, flush=True)
    (out / "objects.json").write_text(json.dumps(_jsonable(report), indent=1))
    return report


def stage_parts(cfg: dict, work: Path, out: Path) -> dict:
    root, results = work / cfg["data_root"], ev.load_results(work / cfg["extract"])
    report = {}
    for task in PART_TASKS:
        for split in ("pretrain", "target", "both"):
            sel = [r for r in results if r["task"] == task and split in (r["split"], "both")]
            touches = ev.part_touches(sel)
            if len(touches) < 3:
                continue
            ds = {s: rc.dataset_dir(root, task, s) for s in ("pretrain", "target")}
            layouts = [eps.read_meta(ds[t.split], t.episode)["layout_id"] for t in touches]
            names = Counter(t.part.split("_")[-1] if False else t.fixture for t in touches)
            report[f"{task}/{split}"] = {
                **parts_lib.part_report(touches, layouts),
                "episodes_replayed": len(sel),
                "no_touch_found": len(sel) - len(touches),
                "most_touched_fixtures": names.most_common(3),
            }
    (out / "parts.json").write_text(json.dumps(_jsonable(report), indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["counts", "objects", "parts", "all"], default="all")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config, args.set)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    out = work / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_config.json").write_text(json.dumps(resolved(cfg, args.config, args.set, REPO), indent=2))
    for stage in ("counts", "objects", "parts"):
        if args.stage in (stage, "all"):
            print(f"== {stage}", flush=True)
            print(json.dumps(_jsonable(globals()[f"stage_{stage}"](cfg, work, out)), indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
