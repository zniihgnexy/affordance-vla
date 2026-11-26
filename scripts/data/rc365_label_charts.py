"""Charts of step 1 (1) (AFFO-15) from the check run: grasp azimuth, transfer results, 3D maps on mugs, part
consistency. matplotlib only (any env that has it, e.g. envs/server/libero_env.sh); CPU.

  source envs/server/libero_env.sh; export PYTHONPATH=$PWD
  "$LIBERO_PYTHON" scripts/data/rc365_label_charts.py --config configs/rc365_labels/check.yaml \
      --out report/media/weekly_2026_11_23/interaction_labels
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from affordance_vla.config import load_config  # noqa: E402
from affordance_vla.plan import rc365_events as ev  # noqa: E402
from affordance_vla.plan import rc365_symmetry as sym  # noqa: E402
from affordance_vla.plan.transfer_check import PASS_AXIS_DEG, PASS_LANDING  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"
COLOURS = {"plain": "#7a7a7a", "ring": "#1b7f79", "robot": "#c4552d"}
LABELS = {"plain": "plain (handle frame)", "ring": "(1) ring fill + robot side", "robot": "(2) robot-facing frame"}


def azimuth_chart(report: dict, out: Path) -> None:
    cats = [c for c, e in report.items() if "azimuth" in e]
    fig, axes = plt.subplots(len(cats), 2, figsize=(9, 2.6 * len(cats)), squeeze=False)
    for row, cat in zip(axes, cats):
        for ax, key, title in zip(
            row,
            ("object_frame", "robot_facing_frame"),
            ("in the object's frame (handle at 0°)", "in the robot-facing frame (robot at 0°)"),
        ):
            for split, colour in (("pretrain", "#8a8a8a"), ("target", "#1b7f79")):
                stats = report[cat]["azimuth"][split][key]
                ax.bar(
                    np.arange(-165, 180, 30) + (7 if split == "target" else -7),
                    stats["histogram_30deg"],
                    width=14,
                    color=colour,
                    label=f"{split} (n={report[cat]['azimuth'][split]['grasps']}, R={stats['resultant']:.2f})",
                )
            ax.set(
                title=f"{cat}: grasp azimuth {title}",
                xlabel="azimuth of the grasp centre (deg)",
                ylabel="grasps",
                xticks=range(-180, 181, 60),
            )
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "azimuth.png", dpi=140)
    plt.close(fig)


def transfer_chart(report: dict, out: Path) -> None:
    cats = [c for c, e in report.items() if "transfer" in e]
    fig, axes = plt.subplots(len(cats), 2, figsize=(9, 3 * len(cats)), squeeze=False)
    for row, cat in zip(axes, cats):
        treatments = list(report[cat]["transfer"])
        for ax, key, line, label in (
            (row[0], "landing_rate", PASS_LANDING, "landing rate (pass ≥ 0.8)"),
            (row[1], "axis_error_median_deg", PASS_AXIS_DEG, "median closing-axis error, deg (pass < 20)"),
        ):
            for k, t in enumerate(treatments):
                result = report[cat]["transfer"][t]
                ax.bar(k, result["pooled"][key], color=COLOURS[t], alpha=0.8, label=LABELS[t])
                ax.scatter(
                    [k] * len(result["per_instance"]),
                    [r[key] for r in result["per_instance"].values()],
                    s=14,
                    color="k",
                    zorder=3,
                )
                own = report[cat]["own_demos"][t]["pooled"]
                if own:
                    ax.hlines(own[key], k - 0.4, k + 0.4, colors="k", linestyles=":", linewidth=1)
            ax.axhline(line, color="r", linewidth=1)
            ax.set(title=f"{cat}: {label}", xticks=range(len(treatments)), xticklabels=treatments)
        row[0].legend(fontsize=7)
    fig.text(
        0.01,
        0.005,
        "bar = pooled over held-out instances; dots = per instance; "
        "dotted = own-demonstration reference; red = pass line",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out / "transfer.png", dpi=140)
    plt.close(fig)


def maps_chart(cfg: dict, work: Path, out: Path, category: str = "mug", treatment: str = "ring") -> None:
    folder = work / cfg["extract"]
    own = [g for g in ev.object_grasps(ev.load_results(folder)) if g.category == category]
    surfaces = {g.instance: ev.load_surface(folder, g.instance) for g in own}
    sources = sym.build_items([g for g in own if g.split == "pretrain"], surfaces)
    targets = sym.build_items([g for g in own if g.split == "target"], surfaces)
    model = sym.TREATMENTS[treatment](sources, cfg["categories"][category]["handle"])
    fig = plt.figure(figsize=(3.2 * len(targets), 6.4))
    for col, target in enumerate(targets):
        k = 0
        cmap = next(iter(model.maps(target, [k])))[1]
        centres = np.stack([g.centre for g in target.grasps])
        for row, (title, score) in enumerate((("carried map", cmap.score), ("own demonstrated grasps", None))):
            ax = fig.add_subplot(2, len(targets), row * len(targets) + col + 1, projection="3d")
            p = target.surface.points
            if score is not None:
                ax.scatter(*p.T, c=score, s=2, cmap="viridis", vmin=0, vmax=1)
            else:
                ax.scatter(*p.T, c="#cccccc", s=2)
                ax.scatter(*centres.T, c="#d1451b", s=6)
            ax.set_title(f"{target.name.split('/')[-1]}: {title}", fontsize=8)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-50)
            ax.set_axis_off()
    fig.suptitle(
        f"{category}: map carried from pretrain instances ({LABELS[treatment]}) vs held-out demonstrations", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "maps_3d.png", dpi=140)
    plt.close(fig)


def parts_chart(report: dict, out: Path) -> None:
    tasks = [k for k in report if k.endswith("/both")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, name, tol in (
        (axes[0], "closing", "within 20° of the other\nkitchens' mean line"),
        (axes[1], "motion", "within 30° of the other\nkitchens' mean direction"),
    ):
        for k, frame in enumerate(("part", "world", "base")):
            values = [report[t][f"{name}_{frame}_frame"].get("across_kitchens_within_tolerance") or 0 for t in tasks]
            ax.bar(np.arange(len(tasks)) + 0.27 * (k - 1), values, 0.27, label=f"{frame} frame")
        ax.set(
            xticks=range(len(tasks)),
            xticklabels=[t.split("/")[0] for t in tasks],
            ylim=(0, 1.05),
            title=f"{name}: share of touches\n{tol}",
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "parts_consistency.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--treatment", default="ring", help="treatment drawn in the 3D map figure")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config)
    work, out = Path(os.environ.get("W", DEFAULT_WORK_DIR)), REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    run = work / cfg["out"]
    objects = json.loads((run / "objects.json").read_text())
    azimuth_chart(objects, out)
    transfer_chart(objects, out)
    maps_chart(cfg, work, out, treatment=args.treatment)
    parts_chart(json.loads((run / "parts.json").read_text()), out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
