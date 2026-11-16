"""Closed-loop LIBERO rollouts with a per-step log (simulator side: runs in the LIBERO env, CPU only).

The episode loop itself (starVLA's `eval_libero.py` with a per-step log) is in
affordance_vla/sim/episode.py, shared with the local viewer. This script adds:
  1. optional home-made perturbations: another instruction for the same scene, or a shifted object;
  2. a replay mode that re-executes a recorded demonstration instead of asking a policy, which gives
     the same log for training data (and tests this script without a GPU);
  3. a state-replay mode that loads the recorded simulator states of demonstrations one by one, with
     no dynamics: the log is then exactly the demonstration, and every demonstration can be used;
  4. an optional plan input sent to the policy server with every query: the oracle plan from simulator
     ground truth (affordance_vla/plan/plan_input.py; needs this repo on PYTHONPATH), or an empty plan;
  5. with `--plan predicted` the server predicts the plan itself; `--hold-plan` then keeps each predicted
     plan until the gripper has carried out its event (affordance_vla/plan/plan_hold.py) instead of asking for a
     new one at every query.

Usage (on the server, after `source envs/server/libero_env.sh`, with starVLA on PYTHONPATH):
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag standard --episodes 0 1 2 --port 6694
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag demo_replay --replay-demo FILE.hdf5 --demo demo_1
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag salad_dressing --replay-states FILE.hdf5 --demos 0 1 2
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag standard --port 6694 --plan oracle --plan-offsets F.json

One `<tag>_ep<k>.npz` is written per episode (`<tag>_demo<k>.npz` in state-replay mode). Arrays that
describe a state have T + 1 rows (before each of the T actions, plus the final state); `actions` has
T rows. Images are stored the way the policy sees them (simulator frame rotated by 180°).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)  # for affordance_vla (numpy-only modules); last, so that it shadows no library
from affordance_vla.plan.plan_hold import Hold, after_query, after_step, plan_to_send  # noqa: E402
from affordance_vla.sim.episode import (  # noqa: E402
    benchmark_instruction,
    camera_matrices,
    env_gripper_command,
    find_task,
    goal_container,
    max_steps_of,
    replay_states,
    run_episode,
    shifted_state,
)
from affordance_vla.sim.episode import make_env as episode_env  # noqa: E402

HOLD_STEPS = 5  # the oracle plan treats the target as held until contact has been absent this long
IMAGE_ARRAYS = ("agentview", "wrist", "seg", "depth")  # what --images-for leaves out of a log




















class ServerPolicy:
    """Actions from a starVLA policy server; keeps every action chunk it receives."""

    stop_on_success = True  # as in starVLA's evaluation loop

    def __init__(self, host: str, port: int) -> None:
        from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

        self.client = ModelClient(host=host, port=port)
        self.name = f"server:{self.client._server_metadata.get('ckpt_path', '?')}"
        self.max_steps: int | None = None
        self.send_ee_pos = False  # set for --plan predicted: the server predicts the plan and needs the gripper
        self.hold_plan = False  # set for --hold-plan: a predicted plan is kept until its event is carried out
        self.answer: dict = {}  # data of the server's latest answer (a plan server also returns its plan inputs)
        ask = self.client.client.predict_action

        def remember(payload: dict) -> dict:
            response = ask(payload)
            self.answer = response.get("data", {})
            return response

        self.client.client.predict_action = remember
        self.reset("")

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.client.reset(task_description=instruction)
        self.chunk_steps: list[int] = []
        self.chunks: list[np.ndarray] = []
        self.seconds: list[float] = []
        self.plans: list[np.ndarray] = []
        # per logged plan: the server predicted it at that query (False: it was sent along, i.e. an oracle or
        # empty plan, or with --hold-plan a predicted plan kept from an earlier query)
        self.plan_predicted: list[bool] = []
        self.hold = Hold()

    def act(self, image: np.ndarray, wrist: np.ndarray, step: int, plan: np.ndarray | None = None,
            ee_pos: np.ndarray | None = None) -> np.ndarray:
        fresh = step % self.client.action_chunk_size == 0
        started = time.perf_counter()
        example = {"image": [image, wrist], "lang": self.instruction}
        if fresh and plan is None and self.hold_plan and ee_pos is not None:
            plan = plan_to_send(self.hold, ee_pos)  # the kept plan, seen from where the gripper is now
        if plan is not None:
            example["plan"] = np.asarray(plan, dtype=np.float32)
        if self.send_ee_pos and ee_pos is not None:
            example["ee_pos"] = np.asarray(ee_pos, dtype=np.float32)
        raw = self.client.step(example=example, step=step)["raw_action"]
        if fresh:
            self.chunk_steps.append(step)
            self.chunks.append(np.asarray(self.client.raw_actions, dtype=np.float32).copy())
            self.seconds.append(time.perf_counter() - started)
            self.remember_plan(example.get("plan"), ee_pos)
        gripper = env_gripper_command(float(np.asarray(raw["open_gripper"]).reshape(-1)[0]))
        if self.hold_plan:
            self.hold = after_step(self.hold, gripper)
        return np.concatenate([np.asarray(raw["world_vector"], dtype=float).reshape(3),
                               np.asarray(raw["rotation_delta"], dtype=float).reshape(3), [gripper]])

    def remember_plan(self, sent: np.ndarray | None, ee_pos: np.ndarray | None) -> None:
        """Log the plan input of the query just made; with --hold-plan a plan the server predicted is kept."""
        if sent is not None:
            self.plans.append(sent)
            self.plan_predicted.append(False)
        elif "plans" in self.answer:  # predicted by the server
            self.plans.append(np.asarray(self.answer["plans"], dtype=np.float32)[0])
            self.plan_predicted.append(True)
            if self.hold_plan and ee_pos is not None:
                self.hold = after_query(self.hold, self.plans[-1], ee_pos)


class ReplayPolicy:
    """Re-executes the actions of one recorded demonstration."""

    stop_on_success = False  # keep the release and the retreat, which come after the goal is first met

    def __init__(self, actions: np.ndarray, name: str) -> None:
        self.actions = np.asarray(actions, dtype=float)
        self.name = name
        self.max_steps: int | None = len(self.actions)
        self.chunk_steps, self.chunks, self.seconds, self.plans, self.plan_predicted = [], [], [], [], []

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, image: np.ndarray, wrist: np.ndarray, step: int, plan: np.ndarray | None = None,
            ee_pos: np.ndarray | None = None) -> np.ndarray:
        return self.actions[step]










def build_planner(args: argparse.Namespace, env, names: list[str],
                  target: str) -> Callable[[dict[str, list]], np.ndarray] | None:
    """None, or a function from the log rows so far to the plan input for the next query."""
    if args.plan in ("none", "predicted"):  # predicted: the server makes the plan itself
        return None
    from affordance_vla.plan.plan_input import no_plan_features, offsets_from_json, oracle_plan

    if args.plan == "zero":
        return lambda rows: no_plan_features()
    if not args.plan_offsets:
        raise SystemExit("--plan oracle needs --plan-offsets (written by build_plan_dataset.py)")
    offsets = offsets_from_json(args.plan_offsets.read_text())
    target_index, container_index = names.index(target), names.index(goal_container(env, names))

    def planner(rows: dict[str, list]) -> np.ndarray:
        positions = rows["object_pos"][-1]
        # a one-step loss of finger contact while carrying must not switch the plan back to the grasp
        held = any(bool(row[target_index]) for row in rows["grasped"][-HOLD_STEPS:])
        return oracle_plan(offsets, rows["ee_pos"][-1], positions[target_index], positions[container_index], target,
                           grasped=held, delivered=bool(rows["in_goal"][-1][target_index]))

    return planner




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="prefix of the output files, e.g. standard")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", default="pick up the salad dressing and place it in the basket")
    parser.add_argument("--task-id", type=int, help="index of the task in the suite (instead of --task)")
    parser.add_argument("--images-for", type=int,
                        help="keep the camera images only in the logs of the first N episodes of this run "
                             "(default: all); the other logs can still be read with images=False")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0],
                        help="indices of LIBERO's initial states (not used with --replay-demo)")
    parser.add_argument("--instruction", help="instruction given to the policy (default: the task's own)")
    parser.add_argument("--target",
                        help="object the instruction refers to (default: the task's first object of interest)")
    parser.add_argument("--shift", nargs=3, action="append", default=[], metavar=("OBJECT", "DX", "DY"),
                        help="move an object in the initial state by DX, DY metres (repeatable)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--replay-demo", type=Path, help="LIBERO HDF5 file: replay a demonstration instead of a policy")
    parser.add_argument("--demo", default="demo_1",
                        help="demo_0 of the salad dressing file does not reproduce on replay")
    parser.add_argument("--replay-states", type=Path,
                        help="LIBERO HDF5 file: log demonstrations from their recorded simulator states")
    parser.add_argument("--demos", type=int, nargs="*", help="demonstration indices for --replay-states (default: all)")
    parser.add_argument("--summary-suffix", default="",
                        help="appended to the summary file name, for processes that share a tag")
    parser.add_argument("--plan", choices=("none", "oracle", "zero", "predicted"), default="none",
                        help="plan input sent with every query: none, the oracle plan, or an empty plan; "
                             "predicted: the gripper position is sent and the server predicts the plan")
    parser.add_argument("--plan-offsets", type=Path, help="contact offsets (JSON) for --plan oracle")
    parser.add_argument("--hold-plan", action="store_true",
                        help="with --plan predicted: keep each predicted plan until the gripper has carried out "
                             "its event, instead of predicting a new plan at every query")
    parser.add_argument("--depth", action="store_true",
                        help="also log the agent view's depth image (metres) at every step, as `depth`")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_policy(args: argparse.Namespace, suite,
                 task_id: int) -> tuple[ServerPolicy | ReplayPolicy, dict[int, np.ndarray]]:
    """(policy, {episode: initial state}): a replay of one demonstration, or a policy server on LIBERO's starts."""
    if args.replay_demo:
        if args.shift:
            raise SystemExit("--shift cannot be combined with --replay-demo: "
                             "recorded actions only fit the recorded start")
        import h5py

        with h5py.File(args.replay_demo, "r") as f:
            group = f["data"][args.demo]
            policy = ReplayPolicy(group["actions"][:], name=f"replay:{args.replay_demo.name}/{args.demo}")
            return policy, {0: group["states"][0]}
    initial_states = suite.get_task_init_states(task_id)
    policy = ServerPolicy(args.host, args.port)
    policy.send_ee_pos = args.plan == "predicted"
    policy.hold_plan = args.hold_plan
    return policy, {episode: initial_states[episode] for episode in args.episodes}


def save_episode(path: Path, meta: dict, arrays: dict[str, np.ndarray], policy, intrinsics: np.ndarray,
                 cam_to_world: np.ndarray) -> None:
    """Write the log under a temporary name and rename it, so an interrupted run leaves no half-written file."""
    partial = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(
        partial, meta=json.dumps(meta), intrinsics=intrinsics, cam_to_world=cam_to_world,
        chunk_steps=np.asarray(policy.chunk_steps, dtype=int),
        chunks=np.stack(policy.chunks) if policy.chunks else np.zeros((0, 0, 7), dtype=np.float32),
        infer_seconds=np.asarray(policy.seconds, dtype=float),
        plans=np.stack(policy.plans) if policy.plans else np.zeros((0, 0), dtype=np.float32),
        plan_predicted=np.asarray(policy.plan_predicted, dtype=bool), **arrays)
    partial.replace(path)


Episode = tuple[str, dict, dict[str, np.ndarray], "ServerPolicy | ReplayPolicy"]  # label, meta, arrays, policy


def policy_episodes(args: argparse.Namespace, env, suite, task_id: int, names: list[str], target: str,
                    instruction: str) -> Iterator[Episode]:
    """Yields (file label, meta, arrays, policy) for each requested start, driven by a policy or a replay."""
    shifts = {name: (float(dx), float(dy)) for name, dx, dy in args.shift}
    policy, starts = build_policy(args, suite, task_id)
    planner = build_planner(args, env, names, target)
    for episode, start in starts.items():
        arrays, success = run_episode(env, policy, instruction, shifted_state(env, start, shifts), names, target,
                                      policy.max_steps or max_steps_of(args.suite), planner)
        yield f"ep{episode}", {"episode": episode, "shifts": shifts, "success": success}, arrays, policy


def state_replays(args: argparse.Namespace, env, names: list[str], target: str) -> Iterator[Episode]:
    """Yields (file label, meta, arrays, policy stand-in) for each requested demonstration of the HDF5 file."""
    import h5py

    with h5py.File(args.replay_states, "r") as f:
        count = len(f["data"])
        for demo in (args.demos if args.demos is not None else range(count)):
            if not 0 <= demo < count:
                raise SystemExit(f"demonstration {demo} is not in {args.replay_states.name} (it has {count})")
            group = f["data"][f"demo_{demo}"]
            arrays, success = replay_states(env, group["states"][:], group["actions"][:], names, target)
            stand_in = ReplayPolicy(arrays["actions"], name=f"states:{args.replay_states.name}/demo_{demo}")
            yield f"demo{demo}", {"episode": demo, "shifts": {}, "success": success}, arrays, stand_in


def make_env(args: argparse.Namespace) -> tuple:
    """(suite, task index, task, simulator) for the task named on the command line."""
    from libero.libero import benchmark

    task_id = args.task_id
    if task_id is None:
        task_id = find_task(benchmark.get_benchmark_dict()[args.suite](), args.task)
    suite, task, env = episode_env(args.suite, task_id, args.seed, args.depth)
    return suite, task_id, task, env


def main() -> None:
    args = parse_args()
    if args.replay_states and (args.replay_demo or args.shift or args.plan != "none"):
        raise SystemExit("--replay-states reads recorded states: it takes no policy, shift or plan")
    if args.hold_plan and (args.plan != "predicted" or args.replay_demo):
        raise SystemExit("--hold-plan keeps plans the server predicts: it needs --plan predicted and a policy server")
    np.random.seed(args.seed)
    suite, task_id, task, env = make_env(args)
    names = sorted(env.env.objects_dict)
    target = args.target or env.obj_of_interest[0]
    if target not in names:
        raise SystemExit(f"target {target!r} is not a movable object; choose one of {names}")
    instruction = (task.language if args.replay_states
                   else args.instruction or benchmark_instruction(args.suite, task))
    intrinsics, cam_to_world = camera_matrices(env)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = (state_replays(args, env, names, target) if args.replay_states
                else policy_episodes(args, env, suite, task_id, names, target, instruction))
    summary = []
    started = time.perf_counter()
    try:
        for label, extra, arrays, policy in episodes:
            meta = {"suite": args.suite, "task": task.language, "instruction": instruction, "target": target,
                    "objects_of_interest": list(env.obj_of_interest), "seed": args.seed, "names": names,
                    "policy": policy.name, "plan": args.plan, "hold_plan": args.hold_plan,
                    "steps": int(len(arrays["actions"])),
                    "delivered": bool(arrays["in_goal"][-1, names.index(target)]),
                    "wall_seconds": round(time.perf_counter() - started, 1), **extra}
            path = args.out / f"{args.tag}_{label}.npz"
            if args.images_for is not None and len(summary) >= args.images_for:
                arrays = {key: value for key, value in arrays.items() if key not in IMAGE_ARRAYS}
            save_episode(path, meta, arrays, policy, intrinsics, cam_to_world)
            summary.append({**meta, "file": path.name})
            print(f"[{args.tag} {label}] steps={meta['steps']} success={meta['success']} "
                  f"delivered({target})={meta['delivered']} wall={meta['wall_seconds']}s -> {path}", flush=True)
            started = time.perf_counter()
    finally:  # keep what finished, and release the simulator, even if an episode fails
        (args.out / f"{args.tag}_summary{args.summary_suffix}.json").write_text(json.dumps(summary, indent=2))
        env.close()

if __name__ == "__main__":
    main()
