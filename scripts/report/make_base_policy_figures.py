"""Figures of the AFFO-16 base policy (step 1) for the weekly review, from the files of one finished run.

  success_by_task.png    success per task and condition (A seen objects / B unseen objects / C target kitchens)
  failure_classes.png    what the failed episodes looked like at the end, per task (analysis/rc365_failures.py)
  training_loss.png      the action-head loss over the 48 000 training steps

Usage (repo root; needs matplotlib, e.g. the starvla env; run summarise_eval.py and classify_failures.py first):
  python scripts/report/make_base_policy_figures.py --eval $W/runs/robocasa365_eval/<run> \
      --train $W/runs/robocasa365_starvla/<run> --out report/media/weekly_2026_11_23/base_policy
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
from affordance_vla.analysis.starvla_log import read_series, smoothed  # noqa: E402

CONDITIONS = {"A": ("A  seen objects, pretrain kitchens", "#1f4e79"),
              "B": ("B  unseen objects, pretrain kitchens", "#e08a1e"),
              "C": ("C  unseen objects, target kitchens", "#4f9d69")}
CLASSES = {"closed_empty": ("gripper shut on nothing", "#b5483a"),
           "holding": ("holding the object at the end", "#8e6bb0"),
           "open_end": ("gripper open at the end", "#e0a43a"), "stalled": ("arm stopped moving", "#5b7083"),
           "kept_moving": ("kept moving until the timeout", "#9fb8cc")}
SHORT = {"CoffeeSetupMug": "CoffeeSetup\nMug", "PickPlaceCounterToCabinet": "PickPlace\nCounter→Cabinet",
         "PickPlaceCounterToStove": "PickPlace\nCounter→Stove", "PickPlaceSinkToCounter": "PickPlace\nSink→Counter",
         "OpenDrawer": "Open\nDrawer", "OpenCabinet": "Open\nCabinet", "TurnOffStove": "TurnOff\nStove",
         "TurnOnSinkFaucet": "TurnOn\nSinkFaucet"}


def success_by_task(summary: dict, tasks: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    width = 0.26
    for k, (cond, (name, colour)) in enumerate(CONDITIONS.items()):
        entries = [summary["cells"][f"{t}/{cond}"] for t in tasks]
        rates = [100 * e["rate"] for e in entries]
        err = [[100 * (e["rate"] - e["wilson95"][0]) for e in entries],
               [100 * (e["wilson95"][1] - e["rate"]) for e in entries]]
        ax.bar([i + (k - 1) * width for i in range(len(tasks))], rates, width, yerr=err, color=colour, label=name,
               error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#333"})
    ax.set_xticks(range(len(tasks)), [SHORT[t] for t in tasks], fontsize=8)
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    total = {c: summary["by_condition"][c] for c in CONDITIONS}
    ax.set_title("Base policy (QwenOFT + LoRA, 48 000 steps), 20 episodes per bar, bars = Wilson 95 % interval\n"
                 + "   ".join(f"{c} all tasks {t['wins']}/{t['n']}" for c, t in total.items()), fontsize=10)
    ax.legend(fontsize=8, loc="upper right", ncol=1, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "success_by_task.png", dpi=160)


def failure_classes(failures: dict, tasks: list[str], out: Path) -> None:
    pooled = {t: {} for t in tasks}
    for key, counts in failures["classes"].items():
        for cls, n in counts.items():
            pooled[key.split("/")[0]][cls] = pooled[key.split("/")[0]].get(cls, 0) + n
    fig, ax = plt.subplots(figsize=(11, 3.8))
    bottoms = [0.0] * len(tasks)
    for cls, (name, colour) in CLASSES.items():
        values = [pooled[t].get(cls, 0) for t in tasks]
        ax.bar(range(len(tasks)), values, 0.6, bottom=bottoms, color=colour, label=name)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_xticks(range(len(tasks)), [SHORT[t] for t in tasks], fontsize=8)
    ax.set_ylabel("failed episodes (A+B+C, of 60)")
    ax.set_title(f"{failures['failed']} failed episodes, {failures['timeouts']} of them timeouts; "
                 "class = gripper and arm at the end", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "failure_classes.png", dpi=160)


def training_loss(train: Path, out: Path) -> None:
    loss = read_series(train / "train.log", "action_dit_loss")
    steps = [10 * (i + 1) for i in range(len(loss))]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(steps, loss, color="#bbb", lw=0.6, label="every 10 steps")
    ax.plot(steps, smoothed(loss, 100), color="#1f4e79", lw=1.6, label="mean of the last 100 records")
    ax.set_xlabel("training step (batch 16)")
    ax.set_ylabel("action-head L1 loss")
    ax.set_ylim(0, 0.4)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "training_loss.png", dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.eval / "summary.json").read_text())
    failures = json.loads((args.eval / "failures.json").read_text())
    tasks = json.loads((args.eval / "config.json").read_text())["tasks"]
    success_by_task(summary, tasks, args.out)
    failure_classes(failures, tasks, args.out)
    training_loss(args.train, args.out)


if __name__ == "__main__":
    main()
