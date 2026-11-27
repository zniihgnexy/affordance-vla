"""Optional per-step recorders for the RoboCasa365 evaluation (AFFO-19). Off unless asked for (`--probes`), so the
records of an ordinary evaluation keep exactly the fields they had.

A probe is built once per environment and called around every episode: `reset()` after `env.reset`, `step()` after every
`env.step`, `result()` at the end; the result is added to the episode record under the probe's `name`.

  part     the stove knob or sink faucet handle the task operates: per environment step, the distance from the gripper
           (centre of the two finger tips) to the part's joint anchor, whether any gripper geom touches the part,
           and the joint value. The knob is the one the instruction names (`env.knob`); the anchor, not the body
           origin, because a knob body's origin is ~50 cm from the knob (the AFFO-18 record, 旋钮标签与把手对齐).
  objects  category and instance of every object the scene holds (which object the task moves is the one named `obj`).

Needs the robocasa365 environment (envs/server/robocasa365_env.sh).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from affordance_vla.sim.rc365_episodes import instance_key

GRIPPER = "gripper0_right"
TIPS = (f"{GRIPPER}_finger_joint1_tip", f"{GRIPPER}_finger_joint2_tip")


def part_joint_name(env: Any, task: str) -> str:
    """The joint the task has to turn: the named knob of the stove, or the sink's handle."""
    if task == "TurnOffStove":
        return f"{env.stove.naming_prefix}knob_{env.knob}_joint"
    if task == "TurnOnSinkFaucet":
        return f"{env.sink.naming_prefix}handle_joint"
    raise ValueError(f"the part probe knows TurnOffStove and TurnOnSinkFaucet, not {task}")


def subtree_bodies(model: Any, root: int) -> np.ndarray:
    """`root` and every body below it."""
    parent = np.asarray(model.body_parentid)
    inside = np.zeros(model.nbody, dtype=bool)
    inside[root] = True
    for body in range(root + 1, model.nbody):  # bodies are stored parent-first
        inside[body] = inside[parent[body]]
    return np.flatnonzero(inside)


class PartProbe:
    name = "part"

    def __init__(self, env: Any, task: str) -> None:
        self.env, self.task = env, task
        self.rows: list[list[float]] = []

    def reset(self) -> None:
        model, data = self.env.sim.model, self.env.sim.data
        self.joint_name = part_joint_name(self.env, self.task)
        self.joint = model.joint_name2id(self.joint_name)
        self.address = int(model.jnt_qposadr[self.joint])
        bodies = subtree_bodies(model, int(model.jnt_bodyid[self.joint]))
        geoms = np.arange(model.ngeom)
        gripper_body = np.array([(model.body_id2name(b) or "").startswith(GRIPPER) for b in range(model.nbody)])
        self.part_geoms = geoms[np.isin(model.geom_bodyid, bodies)]
        self.gripper_geoms = geoms[gripper_body[model.geom_bodyid]]
        self.tips = [model.body_name2id(n) for n in TIPS]
        self.anchor0 = data.xanchor[self.joint].copy()
        self.q0 = float(data.qpos[self.address])
        self.rows = []
        self.step()

    def step(self) -> None:
        data = self.env.sim.data
        centre = np.mean([data.xpos[t] for t in self.tips], axis=0)
        n = data.ncon
        g1, g2 = data.contact.geom1[:n], data.contact.geom2[:n]
        touch = (np.isin(g1, self.gripper_geoms) & np.isin(g2, self.part_geoms)) | (
            np.isin(g2, self.gripper_geoms) & np.isin(g1, self.part_geoms)
        )
        distance = float(np.linalg.norm(centre - data.xanchor[self.joint]))
        self.rows.append([round(100 * distance, 1), int(touch.any()), round(float(data.qpos[self.address]), 3)])

    def result(self) -> dict[str, Any]:
        distance, touch, q = (list(column) for column in zip(*self.rows))
        return {
            "joint": self.joint_name,
            "anchor0": [round(float(x), 4) for x in self.anchor0],
            "q0": round(self.q0, 3),
            "dist_cm": distance,
            "touch": touch,
            "q": q,
        }


class ObjectProbe:
    name = "objects"

    def __init__(self, env: Any, task: str) -> None:
        self.env = env

    def reset(self) -> None:
        self.found = {
            str(cfg["name"]): {"category": str(cfg["info"]["cat"]), "instance": instance_key(cfg["info"]["mjcf_path"])}
            for cfg in self.env.object_cfgs
            if cfg.get("info", {}).get("cat")
        }

    def step(self) -> None:
        pass

    def result(self) -> dict[str, Any]:
        return self.found


PROBES: dict[str, Callable[[Any, str], Any]] = {PartProbe.name: PartProbe, ObjectProbe.name: ObjectProbe}


def make_probes(names: Sequence[str], env: Any, task: str) -> list[Any]:
    """The probes called `names` for `env` (the gym environment; its unwrapped robocasa environment is read)."""
    unknown = sorted(set(names) - set(PROBES))
    if unknown:
        raise ValueError(f"unknown probes {unknown}; choose from {sorted(PROBES)}")
    return [PROBES[name](env.unwrapped, task) for name in names]
