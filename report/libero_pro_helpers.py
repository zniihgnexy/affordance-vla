"""Tables and figures for report/2026-11-12_libero_pro_object.ipynb.

Inputs: the folders written by scripts/remote/run_libero_pro_gpu.sh, one per policy:
runs/libero_pro/<name>/<suite>/task<k>_ep<n>.npz. A rollout counts as solved when the benchmark's own
goal check held at some step (`success` in the log's meta), which is LIBERO's convention.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from affordance3d.rollout_log import RolloutLog, load_rollout_log
from affordance3d.rollout_outcome import classify
from report.plan_following_helpers import rollout_stages
from report.vla_pipeline_demo_helpers import pretty

SUITES = {"position: the target swaps places with another object": "libero_object_swap",
          "task: the instruction and the goal name another object": "libero_object_task"}
EPISODE_FILE = re.compile(r"task(\d+)_ep(\d+)\.npz$")
# LIBERO-PRO numbers reported in other papers, as (Spatial, Object, Goal, LIBERO-10); see docs/vla R9.
PUBLISHED = (
    ("π0.5, as reported in the GAM paper", "official checkpoint", (.20, .17, .38, .08), (.01, .01, .00, .01)),
    ("π0.5, as reported in the CounterAlign paper", "trained on LIBERO", (.53, .19, .29, .06), (.55, .10, .17, .17)),
    ("GAM (arXiv 2609.23863)", "not a VLA; 3D detector, simulator depth", (.60, .47, .19, .05), (.88, .50, .16, .08)),
    ("CounterAlign (arXiv 2608.21740)", "π0.5 + offline RL", (.60, .51, .41, .11), (.63, .26, .46, .29)),
    ("π0.5 + Affordance Field Intervention (arXiv 2512.07472)", "test-time depth and GPT-4o", (.757, .732, None, None),
     (None, None, None, None)),
)


def episode_paths(folder: Path, suite: str) -> list[Path]:
    """Logs of one suite, ordered by task and start state."""
    paths = [path for path in (Path(folder) / suite).glob("task*_ep*.npz") if EPISODE_FILE.search(path.name)]
    if not paths:
        raise FileNotFoundError(f"no rollout logs in {Path(folder) / suite}")
    return sorted(paths, key=lambda path: tuple(int(n) for n in EPISODE_FILE.search(path.name).groups()))


def load_suite(folder: Path, suite: str) -> list[RolloutLog]:
    return [load_rollout_log(path, images=False) for path in episode_paths(folder, suite)]


def task_of(log: RolloutLog) -> int:
    return int(EPISODE_FILE.search(log.path.name).group(1))


def solved(logs: list[RolloutLog]) -> tuple[int, int]:
    return sum(bool(log.meta["success"]) for log in logs), len(logs)


def score_table(runs: dict[str, dict[str, list[RolloutLog]]]) -> pd.DataFrame:
    """Per perturbation and policy: rollouts in which the benchmark's goal was reached."""
    rows = {}
    for label, suite in SUITES.items():
        rows[label] = {}
        for name, run in runs.items():
            done, total = solved(run[suite])
            rows[label][name] = f"{done} / {total} = {done / total:.2f}"
    return pd.DataFrame.from_dict(rows, orient="index")


def per_task_table(runs: dict[str, dict[str, list[RolloutLog]]], suite: str) -> pd.DataFrame:
    """One row per task of a suite: the instruction given, and the solved rollouts of each policy."""
    rows = {}
    for name, run in runs.items():
        for log in run[suite]:
            row = rows.setdefault(task_of(log), {"instruction given": log.meta["instruction"]})
            done, total = row.get(name, (0, 0))
            row[name] = (done + bool(log.meta["success"]), total + 1)
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    for name in runs:
        table[name] = [f"{done} / {total}" for done, total in table[name]]
    table.index.name = "task"
    return table


def outcome_counts(runs: dict[str, dict[str, list[RolloutLog]]], suite: str) -> pd.DataFrame:
    """What happened to the object named by the instruction: counts of the outcome labels per policy."""
    rows = {}
    for name, run in runs.items():
        labels = [classify(log.names, log.grasped, log.target, log.delivered).label for log in run[suite]]
        rows[name] = pd.Series(labels).value_counts().to_dict() | {"rollouts": len(labels)}
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)


def stage_counts(folder: Path, suite: str) -> pd.Series:
    """For a policy that was given plans: how many rollouts reached the planned grasp point, held, delivered."""
    stages = [rollout_stages(path) for path in episode_paths(folder, suite)]
    return pd.Series({"rollouts": len(stages),
                      "gripper came within 2 cm of the planned grasp point": sum(s.reached for s in stages),
                      "object knocked over before it was held": sum(s.toppled for s in stages),
                      "object held": sum(s.held for s in stages),
                      "object in the basket at the end": sum(s.delivered for s in stages),
                      "held, then let go outside the basket":
                          sum(s.held and not s.delivered and not s.holding_at_end for s in stages),
                      "held, still in the gripper at the end":
                          sum(s.held and not s.delivered and s.holding_at_end for s in stages)})


def published_table() -> pd.DataFrame:
    """LIBERO-PRO numbers from other papers: position and task perturbation on the four suites."""
    show = lambda values: " / ".join("–" if v is None else f"{v:.2f}" for v in values)  # noqa: E731
    rows = {name: {"what it is": what, "position (Spatial / Object / Goal / LIBERO-10)": show(position),
                   "task (Spatial / Object / Goal / LIBERO-10)": show(task)}
            for name, what, position, task in PUBLISHED}
    return pd.DataFrame.from_dict(rows, orient="index")


def starts_figure(standard: RolloutLog, logs: dict[str, RolloutLog]) -> plt.Figure:
    """The first camera frame of the unperturbed scene and of the same task under each perturbation."""
    panels = [("standard LIBERO-Object", standard)] + list(logs.items())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.0), squeeze=False)
    for axis, (title, log) in zip(axes[0], panels):
        axis.imshow(log.agentview[0])
        axis.set_title(f"{title}\n“{log.meta['instruction']}”\nasked for: {pretty(log.target)}", fontsize=9)
        axis.set_axis_off()
    fig.tight_layout()
    return fig
