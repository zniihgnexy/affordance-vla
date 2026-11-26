"""AFFO-18, part B: the held-out mug check of AFFO-15 with the handle aligned before the transfer. Per alignment
(`none` is AFFO-15's own rows, `length`, `registered`) and treatment (plain, ring): landing rate and closing-axis
median pooled and per held-out mug. The alignment reads the held-out mugs' surface points only; their
demonstrations are used to score. Writes <out>/mug_align.json. CPU only.

  source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
  "$RC365_PYTHON" scripts/data/rc365_mug_align.py --config configs/rc365_labels/mug_align.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from affordance_vla.config import load_config, resolved
from affordance_vla.plan import handle_align as align
from affordance_vla.plan import rc365_events as ev
from affordance_vla.plan import rc365_symmetry as sym

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"


def aligned(kind: str, sources: list, targets: list) -> tuple[list, list]:
    """Sources and targets after the named alignment."""
    if kind == "none":
        return sources, targets
    if kind == "length":
        both = align.align_length(sources + targets, align.median_reach(sources))
        return both[: len(sources)], both[len(sources) :]
    if kind == "registered":
        return align.align_registered(sources, targets)
    raise ValueError(f"unknown alignment {kind!r}")


def row(result: dict) -> dict:
    pooled = result["pooled"]
    return {
        "pooled": {"landing": pooled.landing_rate, "axis_deg": pooled.axis_error_median_deg, "passed": pooled.passed},
        "per_instance": {
            name.split("/")[-1]: {"landing": r.landing_rate, "axis_deg": r.axis_error_median_deg, "grasps": r.grasps}
            for name, r in result["per_instance"].items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config, args.set)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    align.HANDLE_X0 = cfg.get("handle_x0", align.HANDLE_X0)  # where the handle starts, category units
    folder, out = work / cfg["extract"], work / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "mug_align_config.json").write_text(json.dumps(resolved(cfg, args.config, args.set, REPO), indent=2))
    own = [g for g in ev.object_grasps(ev.load_results(folder)) if g.category == cfg["category"]]
    surfaces = {g.instance: ev.load_surface(folder, g.instance) for g in own}
    sources = sym.build_items([g for g in own if g.split == "pretrain"], surfaces)
    targets = sym.build_items([g for g in own if g.split == "target"], surfaces)
    reaches = {
        i.name.split("/")[-1]: align.handle_reach(align.instance_points(i.frame, i.surface.points))
        for i in sources + targets
    }
    report = {"handle_reach": reaches, "rows": {}}
    for kind in cfg["alignments"]:
        src, tgt = aligned(kind, sources, targets)
        for treatment in cfg["treatments"]:
            result = sym.evaluate(treatment, src, tgt, True)
            report["rows"][f"{kind}/{treatment}"] = row(result)
            print(kind, treatment, json.dumps(report["rows"][f"{kind}/{treatment}"]["pooled"]), flush=True)
            print(
                "   ",
                {
                    k: (round(v["landing"], 2), round(v["axis_deg"], 1))
                    for k, v in report["rows"][f"{kind}/{treatment}"]["per_instance"].items()
                },
                flush=True,
            )
    (out / "mug_align.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
