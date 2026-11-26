"""The recorded events of `scripts/data/rc365_extract.py` as grasps in the object's frame and touches in a part's
frame (affordance_vla/sim/rc365_scan.py stores raw world-frame poses; this is the one place that turns them into
frames). Only numpy is used.

Frames: a RoboCasa gripper site has x = closing line, z = approach (a line and a direction; checked in the record).
For an object, the grasp is expressed in the object's body frame (the frame of the surface file); for a part, in the
part body's frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affordance_vla.plan.interaction_labels import Grasp
from affordance_vla.sim.object_surface import Surface

WORLD_UP = np.array([0.0, 0.0, 1.0])
MIN_LIFT_M = 0.03  # an object grasp counts as a pick when the object rises this much while held
MIN_MOTION_M = 0.02  # the gripper's displacement over the motion window below which "where it moves next" is unset


def _mat(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(3, 3)


def _vec(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def load_results(folder: Path) -> list[dict]:
    """Every episode result of an extraction run, in file-name order."""
    return [json.loads(p.read_text()) for p in sorted((Path(folder) / "episodes").glob("*.json"))]


def load_surface(folder: Path, instance: str) -> Surface:
    with np.load(Path(folder) / "surfaces" / f"{instance.replace('/', '__')}.npz") as data:
        return Surface(data["points"], data["normals"])


@dataclass(frozen=True)
class ObjectGrasp:
    """One pick of an object instance: the grasp in its body frame, where the robot stood, where the object rests."""

    task: str
    split: str
    episode: int
    category: str
    instance: str
    grasp: Grasp
    base_in_object: np.ndarray  # (3,) the robot base position in the object's body frame
    up: np.ndarray  # (3,) world up in the object's body frame at rest (step 0)
    lift_m: float
    approach_world_z: float  # z component of the approach direction in the world: -1 is straight down


def object_grasps(results: list[dict], min_lift: float = MIN_LIFT_M) -> list[ObjectGrasp]:
    """The first lifted grasp of every (episode, object); results of `kind == "object"` events only."""
    found = []
    for result in results:
        seen = set()
        for e in result["events"]:
            if e["kind"] != "object" or e["lift_m"] < min_lift or e["object"] in seen:
                continue
            seen.add(e["object"])
            rot, ee = _mat(e["obj_rot"]), _mat(e["ee_rot"])
            grasp = Grasp(
                rot.T @ (_vec(e["grasp_centre"]) - _vec(e["obj_pos"])), unit(rot.T @ ee[:, 0]), unit(rot.T @ ee[:, 2])
            )
            found.append(
                ObjectGrasp(
                    result["task"],
                    result["split"],
                    result["episode"],
                    e["category"],
                    e["instance"],
                    grasp,
                    rot.T @ (_vec(e["base_pos"]) - _vec(e["obj_pos"])),
                    _mat(e["rest_rot"]).T @ WORLD_UP,
                    e["lift_m"],
                    float(ee[2, 2]),
                )
            )
    return found


def mean_up(ups: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Mean of unit up vectors, normalised, and the largest angle (degrees) of any of them from it."""
    up = unit(np.mean(ups, axis=0))
    return up, float(np.degrees(np.arccos(np.clip(np.min(np.array(ups) @ up), -1, 1))))


@dataclass(frozen=True)
class PartTouch:
    """One touch of a part in the part body's frame, with the motion that follows."""

    task: str
    split: str
    episode: int
    part: str
    fixture: str  # the part's name without its trailing handle / knob word
    contact: np.ndarray  # (3,) mean contact point of the fingers on the part
    closing: np.ndarray  # (3,) the gripper's closing line
    approach: np.ndarray  # (3,)
    both_fingers: bool
    motion: np.ndarray | None  # (3,) unit direction the gripper moves over the next second; None if it hardly moves
    motion_m: float
    joint_type: str | None  # 'slide' or 'hinge'
    joint_axis: np.ndarray | None  # (3,) in the part frame
    joint_sign: float  # sign of the joint's change over the next second (0 when it did not move)
    world: dict  # the same vectors in the world frame and the robot base frame, for comparing frames


def part_touches(results: list[dict]) -> list[PartTouch]:
    found = []
    for result in results:
        for e in result["events"]:
            if e["kind"] != "part":
                continue
            rot, ee, base = _mat(e["part_rot"]), _mat(e["ee_rot"]), _mat(e["base_rot"])
            points = np.array(e["points"]) if e["points"] else _vec(e["grasp_centre"])[None]
            centre = points.mean(axis=0)
            move = _vec(e["ee_later"]) - _vec(e["ee_pos"])
            long_enough = np.linalg.norm(move) >= MIN_MOTION_M
            joint = e.get("joint_type")
            axis = unit(rot.T @ _vec(e["joint_axis"])) if joint else None
            dq = (e["q_later"] - e["q_start"]) if joint else 0.0
            found.append(
                PartTouch(
                    result["task"],
                    result["split"],
                    result["episode"],
                    e["part"],
                    e["part"].rsplit("_", 1)[0],
                    rot.T @ (centre - _vec(e["part_pos"])),
                    unit(rot.T @ ee[:, 0]),
                    unit(rot.T @ ee[:, 2]),
                    bool(all(e["fingers"])),
                    unit(rot.T @ move) if long_enough else None,
                    float(np.linalg.norm(move)),
                    {2: "slide", 3: "hinge"}.get(joint),
                    axis,
                    float(np.sign(dq)) if abs(dq) > 1e-3 else 0.0,
                    {
                        "contact": centre - _vec(e["part_pos"]),
                        "closing": unit(ee[:, 0]),
                        "closing_base": unit(base.T @ ee[:, 0]),
                        "contact_base": base.T @ (centre - _vec(e["base_pos"])),
                        "motion": unit(move) if long_enough else None,
                        "motion_base": unit(base.T @ move) if long_enough else None,
                    },
                )
            )
    return found
