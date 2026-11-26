"""AFFO-18, part B figure: the held-out mug's own demonstrated grasps and the map carried to it, before and after the
handle is aligned (3D, category frame of the mug, matplotlib). Same data and treatment as rc365_mug_align.py; the map
shown is the one used for the instance's first grasp (ring treatment: cut to the half facing that grasp's robot).

  source envs/server/libero_env.sh; export PYTHONPATH=$PWD:$PYTHONPATH
  "$LIBERO_PYTHON" scripts/data/rc365_mug_figure.py --config configs/rc365_labels/mug_align.yaml \
      --out report/media/weekly_2026_11_23/label_followups   # mug_8, alignment length, treatment ring
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from affordance_vla.config import load_config  # noqa: E402
from affordance_vla.plan import handle_align as align  # noqa: E402
from affordance_vla.plan import rc365_events as ev  # noqa: E402
from affordance_vla.plan import rc365_symmetry as sym  # noqa: E402
from affordance_vla.plan.category_frame import CategoryFrame  # noqa: E402
from affordance_vla.plan.transfer_check import high_region  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"


def _runner():
    """`aligned` of scripts/data/rc365_mug_align.py (a script, so loaded by path)."""
    spec = importlib.util.spec_from_file_location("rc365_mug_align", REPO / "scripts/data/rc365_mug_align.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aligned


def panel(ax, target, treatment, title: str) -> None:
    """Target surface in its own category frame, coloured by the carried map; its own grasp centres in red."""
    cmap = next(iter(treatment.maps(target, [0])))[1]
    plain = CategoryFrame(target.frame.origin, target.frame.rotation, target.frame.scale)
    pts = plain.point_to_category(cmap.points)
    shown = high_region(cmap)  # the best-scoring fifth of the surface: where the check says a grasp should land
    ax.scatter(*pts[~shown].T, s=2, c="#cccccc", alpha=0.35, linewidths=0)
    sc = ax.scatter(*pts[shown].T, s=9, c=cmap.score[shown], cmap="viridis", vmin=0, vmax=1, linewidths=0)
    centres = plain.point_to_category(np.stack([g.centre for g in target.grasps]))
    ax.scatter(*centres.T, s=22, c="#e63946", marker="*", label=f"own demonstrated grasps ({len(centres)})")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (handle)")
    ax.set_ylabel("y")
    ax.set_zlabel("z (up)")
    ax.view_init(elev=12, azim=-90)
    ax.set_box_aspect((1, 1, 1))
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--instance", default="objaverse/mug/mug_8")
    ap.add_argument("--alignment", default="length")
    ap.add_argument("--treatment", default="ring")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config)
    align.HANDLE_X0 = cfg.get("handle_x0", align.HANDLE_X0)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    folder = work / cfg["extract"]
    own = [g for g in ev.object_grasps(ev.load_results(folder)) if g.category == cfg["category"]]
    surfaces = {g.instance: ev.load_surface(folder, g.instance) for g in own}
    sources = sym.build_items([g for g in own if g.split == "pretrain"], surfaces)
    targets = sym.build_items([g for g in own if g.split == "target"], surfaces)
    after_sources, after_targets = _runner()(args.alignment, sources, targets)
    pick = lambda items: next(i for i in items if i.name == args.instance)  # noqa: E731
    fig = plt.figure(figsize=(11, 5))
    for k, (srcs, tgt, title) in enumerate(
        (
            (sources, pick(targets), "before: category frame as is"),
            (after_sources, pick(after_targets), f"after: handle {args.alignment}"),
        )
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        sc = panel(ax, tgt, sym.TREATMENTS[args.treatment](srcs, True), f"{args.instance.split('/')[-1]} {title}")
    fig.colorbar(sc, ax=fig.axes, shrink=0.6, label="carried map score (best fifth of the surface)")
    fig.axes[0].legend(fontsize=7, loc="upper left")
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.instance.split('/')[-1]}_align_before_after.png", dpi=150, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
