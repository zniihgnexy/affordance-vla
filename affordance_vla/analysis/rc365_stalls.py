"""Why the arm stops at the stove knob or the faucet handle (AFFO-19, question 2): each episode of TurnOffStove or
TurnOnSinkFaucet, with the part probe's per-step distance / contact / joint value, goes into one class.

Classes, in the order of the chain "reach it, touch it, turn it, turn it enough" (a failure takes the first that holds):
  not_near           the gripper never came within NEAR_CM of the part's joint anchor
  near_no_contact    it came within NEAR_CM but no gripper geom ever touched the part
  contact_no_motion  it touched the part but the joint never moved MOVED_RAD from where it started
  moved_not_enough   the joint moved but the task did not succeed
  success            the episode succeeded
`stalled` is the AFFO-16 failure rule on the same episode (the end effector covered < 0.1 m in the last 40 % of its
trace; affordance_vla.analysis.rc365_failures), so the classes can be read for "the arm stopped" episodes alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from statistics import median
from typing import Any

from affordance_vla.analysis.rc365_failures import classify_failure

NEAR_CM = 8.0  # gripper centre to joint anchor: the stove knob and the faucet handle are 2-6 cm across; touches in the
# demonstrations are within 2 cm (knob) and 6 cm (handle) of the anchor (AFFO-18)
MOVED_RAD = 0.05  # about 3 degrees of the part's joint
CLASSES = ("not_near", "near_no_contact", "contact_no_motion", "moved_not_enough")


def episode_features(row: dict[str, Any]) -> dict[str, Any]:
    """Distances, contact and joint motion of one episode, from its part probe."""
    part = row["part"]
    dist, touch, q = part["dist_cm"], part["touch"], part["q"]
    closest = min(range(len(dist)), key=dist.__getitem__)
    first_touch = next((i for i, t in enumerate(touch) if t), None)
    return {
        "min_dist_cm": dist[closest],
        "step_of_min_dist": closest,
        "first_touch_step": first_touch,
        "touch_steps": int(sum(touch)),
        "joint_motion_rad": max(abs(v - part["q0"]) for v in q),
        "final_dist_cm": dist[-1],
    }


def classify(row: dict[str, Any]) -> str:
    """The class of one episode (see the module docstring)."""
    if row["success"]:
        return "success"
    f = episode_features(row)
    if f["min_dist_cm"] > NEAR_CM:
        return "not_near"
    if f["touch_steps"] == 0:
        return "near_no_contact"
    return "contact_no_motion" if f["joint_motion_rad"] < MOVED_RAD else "moved_not_enough"


def stall_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per task: episodes, successes, failures by class, how many failures the AFFO-16 rule calls stalled and the
    classes of those, and the median closest distance of the failures."""
    out: dict[str, Any] = {}
    for task in sorted({r["task"] for r in rows}):
        mine = [r for r in rows if r["task"] == task]
        failed = [r for r in mine if not r["success"]]
        stalled = [r for r in failed if classify_failure(task, r["trace"]) == "stalled"]
        classes = Counter(classify(r) for r in failed)
        out[task] = {
            "episodes": len(mine),
            "successes": len(mine) - len(failed),
            "failures": len(failed),
            "failure_classes": {c: classes.get(c, 0) for c in CLASSES},
            "stalled": len(stalled),
            "stalled_classes": {c: sum(classify(r) == c for r in stalled) for c in CLASSES},
            "median_min_dist_cm": median(episode_features(r)["min_dist_cm"] for r in failed) if failed else None,
            "median_joint_motion_rad": median(episode_features(r)["joint_motion_rad"] for r in failed)
            if failed
            else None,
        }
    return out
