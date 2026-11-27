"""Coarse classes of failed RoboCasa365 evaluation episodes, read from the trace the evaluation records.

The trace holds, every policy query (8 environment steps), [step, end-effector x, y, z relative to the base, gripper
opening]. It has no object positions, so "did not reach" cannot be told apart from "did not try to grasp"; the
classes are what the gripper and the arm did, and a rule, not a label of what went wrong in the scene.

Grasp tasks (the object is picked and moved), by the gripper at the end of the episode:
  closed_empty   the fingers are shut on nothing (< EMPTY): a grasp closed on air
  holding        the fingers are closed on something (EMPTY..CLOSED): grasped, not placed
  open_end       the gripper is open (>= CLOSED): never grasped, dropped, or let go at the wrong place; the trace
                 cannot tell these apart (the gripper also starts at about 0.02, so "never closed" is not a class)
Articulated tasks (a door, drawer, knob or faucet is operated):
  stalled        the end effector hardly moved in the last 40 % of the episode (< STALL_PATH m of path)
  kept_moving    it kept moving until the time ran out
Every failure of the base policy is a timeout (no episode ended by itself), so "timeout" is not a class here.
"""
from __future__ import annotations

import math
from collections import Counter

GRASP_TASKS = ("CoffeeSetupMug", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceSinkToCounter")
CLOSED = 0.03      # gripper opening (m per finger) below this = closed; open is about 0.04
EMPTY = 0.006      # below this the fingers touch each other: nothing is between them
STALL_PATH = 0.1   # metres of end-effector path in the last 40 % of the trace below which the arm counts as stalled
LATE_FRACTION = 0.6


def classify_failure(task: str, trace: list[list[float]]) -> str:
    """The class of one failed episode (see the module docstring)."""
    opening = [point[4] for point in trace]
    if task in GRASP_TASKS:
        if opening[-1] >= CLOSED:
            return "open_end"
        return "closed_empty" if opening[-1] < EMPTY else "holding"
    late = [point[1:4] for point in trace[int(len(trace) * LATE_FRACTION):]]
    path = sum(math.dist(a, b) for a, b in zip(late[:-1], late[1:]))
    return "stalled" if path < STALL_PATH else "kept_moving"


def count_failures(rows: list[dict]) -> dict[str, dict[str, int]]:
    """`{"Task/Cond": {class: n}}` over the failed episodes of an episodes.jsonl."""
    counts: dict[str, Counter] = {}
    for row in rows:
        if not row["success"]:
            key = f"{row['task']}/{row['condition']}"
            counts.setdefault(key, Counter())[classify_failure(row["task"], row["trace"])] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(counts.items())}
