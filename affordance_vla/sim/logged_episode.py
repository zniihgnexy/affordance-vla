"""What a logged test episode recorded about the policy, and a stand-in policy that repeats it.

A rollout log of the evaluation (scripts/eval/rollout_libero_logged.py; affordance_vla/sim/rollout_log.py)
holds the executed actions and, for every query to the policy server, the step, the action chunk it
returned and the plan input. Replaying the actions through affordance_vla/sim/episode.py reruns the test
without the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Query:
    """One request the policy answered on the server: the step, the chunk it returned, the plan it was given."""

    step: int
    chunk: np.ndarray  # (H, 7) as the policy returned it
    plan: np.ndarray | None  # (10,) plan input; None when the run sent none
    predicted: bool  # the server's plan head predicted this plan at this query (False: sent along or kept)


class LoggedPolicy:
    """Answers each step with the action the policy returned on the server (no model runs)."""

    stop_on_success = True  # as the policy server in the evaluation

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=float)

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, image: np.ndarray, wrist: np.ndarray, step: int, plan: np.ndarray | None = None,
            ee_pos: np.ndarray | None = None) -> np.ndarray:
        return self.actions[step]


def read_queries(path: Path) -> list[Query]:
    """The queries of a log, in order; empty for a log without them (a replayed demonstration)."""
    with np.load(path) as data:
        steps = data["chunk_steps"] if "chunk_steps" in data.files else np.zeros(0, dtype=int)
        chunks = data["chunks"] if "chunks" in data.files else np.zeros((0, 0, 7))
        plans = data["plans"] if "plans" in data.files and data["plans"].size else None
        predicted = data["plan_predicted"] if "plan_predicted" in data.files else np.zeros(len(steps), dtype=bool)
    return [Query(int(step), chunks[k], None if plans is None or k >= len(plans) else plans[k],
                  bool(predicted[k]) if k < len(predicted) else False) for k, step in enumerate(steps)]


def task_index(path: Path, meta: dict) -> int:
    """The task's index in its suite: from the name the evaluation gives a log (task<k>_ep<n>.npz), else the
    suite's task with the logged task text."""
    match = re.fullmatch(r"task(\d+)_ep\d+", Path(path).stem)
    if match:
        return int(match.group(1))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[meta["suite"]]()
    for index in range(suite.n_tasks):
        if suite.get_task(index).language == meta["task"]:
            return index
    raise SystemExit(f"task {meta['task']!r} is not in {meta['suite']}")
