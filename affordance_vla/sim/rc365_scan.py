"""Contact scans of replayed RoboCasa365 demonstrations: where and how the gripper takes hold of an object or touches
a part (drawer or door handle, stove knob, faucet handle), and where it moves next. No rendering, CPU only.

An episode is replayed state by state (`robocasa365.set_state`); in every state the simulator's contacts say which
gripper geoms touch which object or part geoms. Raw poses (world frame) are stored, so any frame can be computed
later (`affordance_vla/plan/rc365_events.py`):

- object grasp: both fingers touch the object for `MIN_HOLD_STEPS` steps in a row; the event is the first step of
  the span (as the LIBERO labels do);
- part touch: a finger touches a body whose name says handle or knob for `MIN_TOUCH_STEPS` steps; the
  event is the first step of the part touched most often, plus the motion of the gripper and of the part's joint
  over the next `MOTION_STEPS` steps.

Needs the robocasa365 env (envs/server/robocasa365_env.sh).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from affordance_vla.sim import robocasa365 as rc
from affordance_vla.sim.object_surface import Surface, body_surface
from affordance_vla.sim.rc365_episodes import instance_key

MIN_HOLD_STEPS = 5  # both fingers on the object this many steps in a row: a held grasp (0.25 s at 20 Hz)
MIN_TOUCH_STEPS = 3  # a part counts as touched after this many steps of contact in a row
MOTION_STEPS = 20  # the motion after a touch is read over this many steps (1 s at 20 Hz)
PART_BODY = re.compile(r"handle|knob")
GRIPPER = "gripper0_right"
TIPS = (f"{GRIPPER}_finger_joint1_tip", f"{GRIPPER}_finger_joint2_tip")
SLIDE, HINGE = 2, 3  # mjtJoint


def instance_split(key: str, category: str) -> str:
    """'pretrain' or 'target': the official instance split of a registry key (robocasa365.category_instances)."""
    for registry, parts in rc.category_instances(category).items():
        for split in ("pretrain", "target"):
            if any(instance_key(p) == key for p in parts[split]):
                return split
    raise KeyError(f"{key} is not an instance of {category}")


class Rig:
    """Ids of one environment's gripper, robot and (for parts) the part bodies, looked up once."""

    def __init__(self, env):
        self.env, self.model, self.data = env, env.sim.model, env.sim.data
        names = [self.model.geom_id2name(g) or "" for g in range(self.model.ngeom)]
        bodies = [self.model.body_id2name(self.model.geom_bodyid[g]) or "" for g in range(self.model.ngeom)]
        self.finger = [
            np.array([g for g, n in enumerate(names) if n.startswith(f"{GRIPPER}_finger{k}")]) for k in (1, 2)
        ]
        self.fingers = np.concatenate(self.finger)
        self.part_geoms = np.array(
            [g for g, b in enumerate(bodies) if PART_BODY.search(b) and not b.startswith(("robot0", GRIPPER))]
        )
        self.site = env.robots[0].eef_site_id["right"]
        self.tips = [self.model.body_name2id(n) for n in TIPS]
        self.base = self.model.body_name2id("robot0_base")

    def contacts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.data.ncon
        return self.data.contact.geom1[:n], self.data.contact.geom2[:n], self.data.contact.pos[:n]

    def pose(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[body].copy(), self.data.xmat[body].copy()

    def gripper_pose(self) -> dict:
        tips = np.stack([self.data.xpos[t] for t in self.tips])
        return {
            "ee_pos": self.data.site_xpos[self.site].copy(),
            "ee_rot": self.data.site_xmat[self.site].copy(),
            "grasp_centre": tips.mean(axis=0),
        }


def _between(g1: np.ndarray, g2: np.ndarray, side_a: np.ndarray, side_b: np.ndarray) -> np.ndarray:
    """Contact mask: one geom in side_a and the other in side_b."""
    return (np.isin(g1, side_a) & np.isin(g2, side_b)) | (np.isin(g2, side_a) & np.isin(g1, side_b))


def runs(flags: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """(start, end) of every run of True at least `min_length` long; end is exclusive."""
    padded = np.concatenate([[False], np.asarray(flags, dtype=bool), [False]])
    starts, ends = np.flatnonzero(~padded[:-1] & padded[1:]), np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_length]


def scan_object_grasps(env, episode: dict, tracked: dict[str, str]) -> list[dict]:
    """Held grasps of the tracked objects ({env object name: category}) in one episode."""
    rig = Rig(env)
    objects = {
        name: (
            env.objects[name].root_body,
            np.array([rig.model.geom_name2id(g) for g in env.objects[name].contact_geoms]),
        )
        for name in tracked
    }
    bodies = {name: rig.model.body_name2id(root) for name, (root, _) in objects.items()}
    steps = len(episode["states"])
    held = {name: np.zeros((steps, 2), dtype=bool) for name in tracked}
    track = {"gripper": [], "object": {name: [] for name in tracked}, "base": []}
    for t in range(steps):
        rc.set_state(env, episode["states"][t])
        g1, g2, _ = rig.contacts()
        for name, (_, geoms) in objects.items():
            held[name][t] = [_between(g1, g2, rig.finger[k], geoms).any() for k in (0, 1)]
            track["object"][name].append(rig.pose(bodies[name]))
        track["gripper"].append(rig.gripper_pose())
        track["base"].append(rig.pose(rig.base))
    events = []
    for name, category in tracked.items():
        for start, end in runs(held[name].all(axis=1), MIN_HOLD_STEPS):
            pos, rot = track["object"][name][start]
            height = [p[2] for p, _ in track["object"][name][start:end]]
            events.append(
                {
                    "kind": "object",
                    "object": name,
                    "category": category,
                    "step": start,
                    "hold_steps": end - start,
                    "lift_m": float(max(height) - height[0]),
                    "obj_pos": pos,
                    "obj_rot": rot,
                    "rest_rot": track["object"][name][0][1],
                    "base_pos": track["base"][start][0],
                    "base_rot": track["base"][start][1],
                    **track["gripper"][start],
                }
            )
    return events


def _joint_above(model, body: int) -> int | None:
    """The first hinge or slide joint on the body or above it (the joint that moves the part)."""
    while body > 0:
        for joint in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
            if model.jnt_type[joint] in (SLIDE, HINGE):
                return int(joint)
        body = int(model.body_parentid[body])
    return None


def scan_part_touch(env, episode: dict) -> dict | None:
    """The first touch of the part (handle or knob) the gripper touches most, with the motion that follows."""
    rig = Rig(env)
    steps = len(episode["states"])
    touched, centroid, finger_flags, track = {}, {}, [], []
    for t in range(steps):
        rc.set_state(env, episode["states"][t])
        g1, g2, pos = rig.contacts()
        mask = _between(g1, g2, rig.fingers, rig.part_geoms)
        found = {}
        for k in np.flatnonzero(mask):
            geom = g2[k] if g1[k] in rig.fingers else g1[k]
            body = int(rig.model.geom_bodyid[geom])
            found.setdefault(body, []).append(pos[k])
        for body, points in found.items():
            touched.setdefault(body, np.zeros(steps, dtype=bool))[t] = True
            centroid.setdefault(body, {})[t] = np.mean(points, axis=0)
        finger_flags.append([_between(g1, g2, rig.finger[k], rig.part_geoms).any() for k in (0, 1)])
        track.append({**rig.gripper_pose(), "base_pos": rig.pose(rig.base)[0], "base_rot": rig.pose(rig.base)[1]})
        for body in touched:
            track[-1].setdefault("part", {})[body] = rig.pose(body)
        track[-1]["q"] = rig.data.qpos.copy()
    candidates = [(int(touched[b].sum()), b, runs(touched[b], MIN_TOUCH_STEPS)) for b in touched]
    candidates = [c for c in candidates if c[2]]
    if not candidates:
        return None
    count, body, spans = max(candidates)
    start = spans[0][0]
    stop = min(start + MOTION_STEPS, steps - 1)
    joint = _joint_above(rig.model, body)
    part_pos, part_rot = track[start]["part"][body]
    event = {
        "kind": "part",
        "part": rig.model.body_id2name(body),
        "step": start,
        "touch_steps": count,
        "points": [centroid[body][t] for t in range(start, min(start + MIN_TOUCH_STEPS, steps)) if t in centroid[body]],
        "fingers": np.any(finger_flags[start : start + MIN_TOUCH_STEPS], axis=0),
        "part_pos": part_pos,
        "part_rot": part_rot,
        **{k: track[start][k] for k in ("ee_pos", "ee_rot", "grasp_centre", "base_pos", "base_rot")},
        "ee_later": track[stop]["ee_pos"],
        "motion_steps": stop - start,
        "ee_path": [track[t]["ee_pos"] for t in range(start, stop + 1)],
        "part_later": track[stop]["part"].get(body, (part_pos, part_rot))[0],
    }
    if joint is not None:
        address = rig.model.jnt_qposadr[joint]
        rc.set_state(env, episode["states"][start])
        event |= {
            "joint": rig.model.joint_id2name(joint),
            "joint_type": int(rig.model.jnt_type[joint]),
            "joint_axis": rig.data.xaxis[joint].copy(),
            "joint_anchor": rig.data.xanchor[joint].copy(),
            "joint_range": rig.model.jnt_range[joint].copy(),
            "q_start": float(track[start]["q"][address]),
            "q_later": float(track[stop]["q"][address]),
            "q_end": float(track[-1]["q"][address]),
        }
    return event


def object_surface(env, name: str, count: int = 3000) -> Surface:
    """Surface points of an environment object in its body frame, the frame the object events are given in."""
    model = env.sim.model
    return body_surface(model, model.body_name2id(env.objects[name].root_body), count)


def save_surface(path: Path, surface: Surface) -> None:
    """Write a surface once (temp file, then rename, so parallel workers never see half a file)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp.npz")
    np.savez_compressed(tmp, points=surface.points, normals=surface.normals)
    tmp.rename(path)
