"""Figures of the AFFO-19 base-policy checks for the weekly review.

  success_vs_steps.png     success in condition A at 16 000 / 32 000 / 48 000 training steps, total and per task
  part_distance.png        gripper-to-part distance over the episode, TurnOffStove and TurnOnSinkFaucet, n_act 8 and 4
  seen_vs_unseen.png       seen (A) against unseen (B) object instances on the four pick tasks, 80 episodes per bar

Usage (repo root; matplotlib needed, e.g. the starvla env; run scripts/rc365/summarise_checks.py first):
  python scripts/report/make_base_checks_figures.py --summary <dir with steps.json stuck.json instances.json> \
      --nact8 <run> --nact4 <run> --out report/media/weekly_2026_11_23/base_checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.analysis.rc365_stalls import CLASSES as STALL_CLASSES  # noqa: E402
from affordance_vla.analysis.rc365_stalls import NEAR_CM, classify  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#1f4e79", "#e08a1e", "#4f9d69", "#5b7083"
CLASS_COLOUR = {
    "success": GREEN,
    "not_near": "#b5483a",
    "near_no_contact": "#e0a43a",
    "contact_no_motion": "#8e6bb0",
    "moved_not_enough": "#5b7083",
}
CLASS_LABEL = {
    "success": "succeeded",
    "not_near": "never within 8 cm",
    "near_no_contact": "within 8 cm, never touched",
    "contact_no_motion": "touched, joint did not move",
    "moved_not_enough": "joint moved, not enough",
}
TASKS_FLOOR = ("TurnOffStove", "TurnOnSinkFaucet")


def errors(entries: list[dict]) -> list[list[float]]:
    return [
        [100 * (e["rate"] - e["wilson95"][0]) for e in entries],
        [100 * (e["wilson95"][1] - e["rate"]) for e in entries],
    ]


def success_vs_steps(report: dict, out: Path) -> None:
    steps = list(report["success"])
    tasks = [t for t in report["success"][steps[0]] if t != "all"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 3]})
    total = [report["success"][s]["all"] for s in steps]
    left.errorbar(
        range(len(steps)), [100 * e["rate"] for e in total], yerr=errors(total), fmt="o-", color=BLUE, capsize=3
    )
    left.set_xticks(range(len(steps)), steps)
    left.set_xlabel("training steps")
    left.set_ylabel("success rate (%), condition A")
    left.set_ylim(0, 100)
    left.set_title("8 tasks x 20 episodes (n = 160)", fontsize=9)
    width = 0.8 / len(steps)
    shades = ["#9fb8cc", "#5b87ad", BLUE]
    for k, s in enumerate(steps):
        entries = [report["success"][s][t] for t in tasks]
        right.bar(
            [i + (k - 1) * width for i in range(len(tasks))],
            [100 * e["rate"] for e in entries],
            width,
            yerr=errors(entries),
            color=shades[k % 3],
            label=f"{s} steps",
            error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333"},
        )
    right.set_xticks(range(len(tasks)), [t.replace("PickPlace", "PP ") for t in tasks], fontsize=7, rotation=15)
    right.set_ylim(0, 100)
    right.legend(fontsize=8, frameon=False, ncol=3)
    right.set_title("per task, 20 episodes per bar, bars = Wilson 95 %", fontsize=9)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "success_vs_steps.png", dpi=160)


def part_distance(runs: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(len(TASKS_FLOOR), len(runs), figsize=(12, 6.4), sharey=True, squeeze=False)
    for col, (name, rows) in enumerate(runs.items()):
        for row, task in enumerate(TASKS_FLOOR):
            ax = axes[row][col]
            mine = [r for r in rows if r["task"] == task]
            for r in mine:
                kind = classify(r)
                ax.plot(r["part"]["dist_cm"], color=CLASS_COLOUR[kind], lw=0.8, alpha=0.7)
            ax.axhline(NEAR_CM, color="#333", ls="--", lw=0.8)
            ax.set_title(f"{task}, {name}: {sum(r['success'] for r in mine)}/{len(mine)} succeeded", fontsize=9)
            ax.set_xlabel("environment step")
            if col == 0:
                ax.set_ylabel("gripper to joint anchor (cm)")
            ax.set_ylim(0, 120)
            ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], color=CLASS_COLOUR[c], lw=2) for c in ("success", *STALL_CLASSES)]
    fig.legend(
        handles,
        [CLASS_LABEL[c] for c in ("success", *STALL_CLASSES)],
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "part_distance.png", dpi=160)


def seen_vs_unseen(report: dict, out: Path) -> None:
    items = {**report["per_task"], "all four": report["pooled"]}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.38
    for k, (side, name, colour) in enumerate(
        (("seen", "A  seen instances", BLUE), ("unseen", "B  unseen instances", ORANGE))
    ):
        entries = [r[side] for r in items.values()]
        ax.bar(
            [i + (k - 0.5) * width for i in range(len(items))],
            [100 * e["rate"] for e in entries],
            width,
            yerr=errors(entries),
            color=colour,
            label=name,
            error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333"},
        )
    labels = [
        f"{t.replace('PickPlace', 'PP ')}\nonly A {r['counts']['only_first']} / only B {r['counts']['only_second']}"
        for t, r in items.items()
    ]
    ax.set_xticks(range(len(items)), labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("success rate (%)")
    pooled = report["pooled"]
    ax.set_title(
        f"Same seeds, same kitchens; {pooled['pairs'] // len(report['per_task'])} episodes per bar "
        f"(all four: {pooled['pairs']}); "
        f"sign test p = {pooled['p_sign_test']:.2f}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "seen_vs_unseen.png", dpi=160)


def read_rows(folder: str) -> list[dict]:
    return [json.loads(line) for line in (Path(folder) / "episodes.jsonl").read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nact8")
    parser.add_argument("--nact4")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for part, make in (("steps", success_vs_steps), ("instances", seen_vs_unseen)):
        if (args.summary / f"{part}.json").exists():
            make(json.loads((args.summary / f"{part}.json").read_text()), args.out)
    if args.nact8 and args.nact4:
        part_distance({"n_act=8": read_rows(args.nact8), "n_act=4": read_rows(args.nact4)}, args.out)


if __name__ == "__main__":
    main()
