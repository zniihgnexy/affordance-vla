"""AFFO-18, part A charts (matplotlib, no simulator): the stove knob's contact positions before and after the frame
fix, and the old and new "where it moves next" label of the four part tasks side by side. Reads extract_v1 and the
`joint_parts.json` of scripts/data/rc365_joint_check.py.

  source envs/server/libero_env.sh; export PYTHONPATH=$PWD
  "$LIBERO_PYTHON" scripts/data/rc365_joint_charts.py --config configs/rc365_labels/joint_check.yaml \
      --out report/media/weekly_2026_11_23/label_followups
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
from affordance_vla.plan import rc365_joint_labels as jl  # noqa: E402
from affordance_vla.plan.rc365_knob_diagnosis import stove_type  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"
FAMILY_COLOURS = {"stove": "#1b6ca8", "stovetop": "#d1495b", "widestove": "#6a994e"}
TASK_LABELS = {
    "OpenDrawer": "drawer handle\n(slide)",
    "OpenCabinet": "door handle\n(hinge)",
    "TurnOnSinkFaucet": "faucet handle\n(hinge)",
    "TurnOffStove": "stove knob\n(hinge)",
}


def knob_scatter(touches: list[jl.JointTouch], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=False)
    for ax, get, title in (
        (axes[0], jl.JointTouch.contact_from_body, "origin = the knob body's origin (old)"),
        (axes[1], jl.JointTouch.contact_from_anchor, "origin = the knob's joint anchor (new)"),
    ):
        for family, colour in FAMILY_COLOURS.items():
            pts = np.array([100 * get(t) for t in touches if stove_type(t.part) == family])
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 2], s=14, alpha=0.7, c=colour, label=f"{family} ({len(pts)})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x in the part frame (cm)")
        ax.set_ylabel("z in the part frame (cm)")
        ax.axhline(0, c="#bbb", lw=0.6)
        ax.axvline(0, c="#bbb", lw=0.6)
    axes[0].legend(fontsize=8, title="stove design")
    fig.suptitle(
        "TurnOffStove: where the gripper first touches the knob, 259 demonstrations in 57 kitchens", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_bars(report: dict, path: Path, window: str) -> None:
    tasks = list(report)
    x = np.arange(len(tasks))
    rows = {
        "defined: old (gripper)": [report[t]["window_" + window]["old_defined"] for t in tasks],
        "defined: new (joint)": [report[t]["window_" + window]["new_defined"] for t in tasks],
        "consistent across kitchens: old": [
            report[t]["window_" + window]["old_direction_part_frame"].get("across_kitchens_within_tolerance") or 0
            for t in tasks
        ],
        "consistent across kitchens: new": [
            report[t]["window_" + window]["new_direction"]["part"].get("across_kitchens_within_tolerance") or 0
            for t in tasks
        ],
    }
    colours = ["#aab7c4", "#1b6ca8", "#e0b3ba", "#d1495b"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, ((name, values), colour) in enumerate(zip(rows.items(), colours)):
        ax.bar(x + (k - 1.5) * 0.2, values, 0.2, label=name, color=colour)
    ax.set_xticks(x, [TASK_LABELS[t] for t in tasks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of demonstrations")
    ax.set_title(
        f"Where the part moves after the touch: gripper label vs joint label (sign window {window}, tolerance 30°)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", default="1s")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = ev.load_results(work / cfg["extract"])
    stove = jl.joint_touches([r for r in results if r["task"] == cfg["knob_task"]])
    knob_scatter(stove, out / "knob_frame_before_after.png")
    report = json.loads((work / cfg["out"] / "joint_parts.json").read_text())
    comparison_bars(report, out / "part_label_comparison.png", args.window)
    print(out)


if __name__ == "__main__":
    main()
