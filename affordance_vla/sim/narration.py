"""The stages of a LIBERO-PRO test, printed in plain English while the episode runs in the viewer.

`explain_scene` and `explain_loop` say what the test is before it starts (task file, goal check, instruction,
start state, how the policy is queried); `Narrator` is the `on_step` callback of
affordance_vla/sim/episode.run_episode: it prints each query, gripper command, grasp and goal event, and
draws the step in the viewer; `compare` sets the replay against the server's record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from affordance_vla.plan.plan_input import EVENT_COLUMN, GRASP_CODE, plan_contact
from affordance_vla.sim.live_view import LiveView, Marker
from affordance_vla.sim.logged_episode import Query

PERTURBATIONS = {"swap": "position perturbation: the target swapped places with another object",
                 "task": "task perturbation: instruction and target changed to another object in the scene",
                 "object": "object appearance perturbation", "lan": "instruction wording perturbation"}
PLAN_KINDS = {"none": "no plan", "oracle": "each query carries the plan computed from simulator truth (ORACLE)",
              "zero": "an empty plan", "predicted": "the plan head on the server predicts the plan from the images"}


class WindowClosed(Exception):
    """The user closed the viewer during an episode."""


def goal_text(env) -> str:
    return " and ".join(f"({' '.join(term)})" for term in env.env.parsed_problem["goal_state"])


def explain_scene(suite: str, task, env, names: list[str], instruction: str, episode: int, seed: int) -> None:
    from libero.libero import get_libero_path

    kind = suite.rsplit("_", 1)[-1]
    print(f"[suite] {suite} ({PERTURBATIONS.get(kind, 'original task, no perturbation')})")
    print(f"  scene file: {Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file}")
    print(f"  goal check: {goal_text(env)}")
    print(f"  movable objects: {', '.join(names)}")
    print(f"[instruction] \"{instruction}\"")
    if instruction != task.language:
        print(f"  (the original task says \"{task.language}\"; the perturbed task file changed it)")
    print(f"[start state] recorded initial state {episode} of the suite; random seed {seed}")


def explain_loop(meta: dict, queries: list[Query], max_steps: int) -> None:
    horizon = len(queries[0].chunk) if queries else 0
    hold = "; a predicted plan is kept until the gripper has carried out its event" if meta.get("hold_plan") else ""
    print("[loop] 10 no-op steps let the objects settle; then each query sends the two camera images (third-person and")
    print(f"  wrist, rotated 180 degrees as in training) and the instruction; the policy returns {horizon} actions")
    print("  (one chunk), which run before the next query.")
    print(f"  policy: {meta['policy']}; {PLAN_KINDS.get(meta.get('plan', 'none'), meta.get('plan'))}{hold}.")
    print("  No model runs here: every step executes the action the policy gave on the server.")
    print(f"  At most {max_steps} steps.")


def plan_source(query: Query, run_plan: str) -> str:
    if run_plan == "oracle":
        return "computed from simulator truth (ORACLE)"
    if run_plan == "predicted":
        return "newly predicted by the plan head" if query.predicted else "kept from an earlier prediction"
    return "sent with the query"


def describe_plan(query: Query, ee_pos: np.ndarray, run_plan: str) -> tuple[str, np.ndarray | None]:
    """A line about the plan of a query, and the contact point it points at (None without a plan)."""
    if query.plan is None:
        return "no plan input (this run gives the policy no plan)", None
    contact = plan_contact(query.plan, ee_pos)
    if contact is None:
        return "plan: no gripper event left", None
    event = "grasp" if query.plan[EVENT_COLUMN] == GRASP_CODE else "release"
    reach_cm = 100 * float(np.linalg.norm(contact - ee_pos))
    text = (f"plan: next gripper event {event}, contact {np.round(contact, 3)}, {reach_cm:.1f} cm from the gripper"
            f" ({plan_source(query, run_plan)})")
    return text, contact


class Narrator:
    """`on_step` callback: prints the events of the loop and shows each step in the viewer."""

    def __init__(self, queries: list[Query], names: list[str], run_plan: str, env, view: LiveView | None) -> None:
        self.queries = {query.step: (k, query) for k, query in enumerate(queries)}
        self.names, self.run_plan, self.env, self.view = names, run_plan, env, view
        self.contact: np.ndarray | None = None
        self.gripper = -1.0
        self.grasped = np.zeros(len(names), dtype=bool)
        self.in_goal = np.zeros(len(names), dtype=bool)

    @property
    def markers(self) -> tuple[Marker, ...]:
        return () if self.contact is None else (Marker(self.contact),)

    def __call__(self, step: int, obs: dict, log) -> None:
        if step in self.queries:
            k, query = self.queries[step]
            text, contact = describe_plan(query, log.rows["ee_pos"][-1], self.run_plan)
            self.contact = contact if contact is not None else self.contact
            print(f"  step {step:3d}: query {k + 1} -> chunk of {len(query.chunk)} actions; {text}", flush=True)
        if log.actions[-1][6] != self.gripper:
            self.gripper = log.actions[-1][6]
            print(f"  step {step:3d}: gripper {'closes' if self.gripper > 0 else 'opens'}", flush=True)
        grasped, in_goal = np.asarray(log.rows["grasped"][-1]), np.asarray(log.rows["in_goal"][-1])
        for index in np.flatnonzero(grasped & ~self.grasped):
            print(f"  step {step:3d}: both fingers touch {self.names[index]} (grasped)", flush=True)
        for index in np.flatnonzero(in_goal & ~self.in_goal):
            print(f"  step {step:3d}: {self.names[index]} meets the goal", flush=True)
        self.grasped, self.in_goal = grasped, in_goal
        if self.view is not None and not self.view.show(self.env, self.markers):
            raise WindowClosed


def compare(meta: dict, logged_ee: np.ndarray, arrays: dict[str, np.ndarray], success: bool) -> None:
    """The replay against the server's record: outcome, steps and the end effector's path."""
    replayed = arrays["ee_pos"]
    steps = min(len(logged_ee), len(replayed))
    drift_mm = 1000 * np.linalg.norm(replayed[:steps] - logged_ee[:steps], axis=1)
    print(f"[compare] server: {'success' if meta['success'] else 'failure'}, {meta['steps']} steps; "
          f"here: {'success' if success else 'failure'}, {len(arrays['actions'])} steps; "
          f"end-effector drift median {np.median(drift_mm):.1f} mm, max {drift_mm.max():.1f} mm")
    if success != meta["success"]:
        print("  Outcomes differ: MuJoCo here is not bit-identical to the server, so the replay drifted; "
              "the server's record counts.")
