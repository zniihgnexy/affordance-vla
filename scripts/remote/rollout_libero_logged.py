"""Closed-loop LIBERO rollouts with a per-step log (simulator side: runs in the LIBERO env, CPU only).

This is the loop of starVLA's `eval_libero.py` with three additions:
  1. a log of every step: end effector, each object's position, grasp and touch flags from simulator
     contacts, visible pixels per object (instance segmentation), and the goal check per object;
  2. optional home-made perturbations: another instruction for the same scene, or a shifted object;
  3. a replay mode that re-executes a recorded demonstration instead of asking a policy, which gives
     the same log for training data (and tests this script without a GPU).

Usage (on the server, after `source scripts/remote/libero_env.sh`, with starVLA on PYTHONPATH):
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag standard --episodes 0 1 2 --port 6694
    $LIBERO_PYTHON rollout_libero_logged.py --out DIR --tag demo_replay --replay-demo FILE.hdf5 --demo demo_1

One `<tag>_ep<k>.npz` is written per episode. Arrays that describe a state have T + 1 rows (before each
of the T actions, plus the final state); `actions` has T rows. Images are stored the way the policy
sees them (simulator frame rotated by 180°).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open: lets the objects settle
RESOLUTION = 256  # resolution of the training images
WAIT_STEPS = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
CAMERA = "agentview"
ROBOT_SEG_ID = 255  # keep in step with affordance3d/rollout_log.py (this script runs in another environment)
ROBOT_INSTANCES = ("Panda0", "RethinkMount0", "PandaGripper0")
GRIPPER_OPEN_THRESHOLD = 0.5  # the policy's gripper output is 1 for open and 0 for close


def model_view(image: np.ndarray) -> np.ndarray:
    """The 180° rotation applied to LIBERO frames when the training data was made."""
    return np.ascontiguousarray(image[::-1, ::-1])


def env_gripper_command(open_value: float) -> float:
    """Policy output (1 = open, 0 = close) to the simulator's command (-1 = open, +1 = close)."""
    return 1.0 - 2.0 * float(open_value > GRIPPER_OPEN_THRESHOLD)


def find_task(suite, language: str) -> int:
    for task_id in range(suite.n_tasks):
        if suite.get_task(task_id).language.lower() == language.lower():
            return task_id
    known = [suite.get_task(i).language for i in range(suite.n_tasks)]
    raise SystemExit(f"task {language!r} is not in this suite; choose one of {known}")


def goal_template(env) -> tuple[str, str] | None:
    """(predicate, region) if the goal is one binary predicate on one object, e.g. ("in", "basket_1_contain_region")."""
    goal = env.env.parsed_problem["goal_state"]
    if len(goal) == 1 and len(goal[0]) == 3:
        return goal[0][0], goal[0][2]
    return None


def shifted_state(env, state: np.ndarray, shifts: dict[str, tuple[float, float]]) -> np.ndarray:
    """A copy of a flattened simulator state (time, qpos, qvel) with objects moved in x and y (metres)."""
    state = np.array(state, dtype=float, copy=True)
    for name, (dx, dy) in shifts.items():
        if name not in env.env.objects_dict:
            raise SystemExit(f"cannot shift {name!r}; movable objects are {sorted(env.env.objects_dict)}")
        start, _ = env.sim.model.get_joint_qpos_addr(env.env.objects_dict[name].joints[0])
        state[1 + start] += dx
        state[2 + start] += dy
    return state


class ServerPolicy:
    """Actions from a starVLA policy server; keeps every action chunk it receives."""

    stop_on_success = True  # as in starVLA's evaluation loop

    def __init__(self, host: str, port: int) -> None:
        from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

        self.client = ModelClient(host=host, port=port)
        self.name = f"server:{self.client._server_metadata.get('ckpt_path', '?')}"
        self.max_steps: int | None = None
        self.reset("")

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.client.reset(task_description=instruction)
        self.chunk_steps: list[int] = []
        self.chunks: list[np.ndarray] = []
        self.seconds: list[float] = []

    def act(self, image: np.ndarray, wrist: np.ndarray, step: int) -> np.ndarray:
        fresh = step % self.client.action_chunk_size == 0
        started = time.perf_counter()
        raw = self.client.step(example={"image": [image, wrist], "lang": self.instruction}, step=step)["raw_action"]
        if fresh:
            self.chunk_steps.append(step)
            self.chunks.append(np.asarray(self.client.raw_actions, dtype=np.float32).copy())
            self.seconds.append(time.perf_counter() - started)
        gripper = env_gripper_command(float(np.asarray(raw["open_gripper"]).reshape(-1)[0]))
        return np.concatenate([np.asarray(raw["world_vector"], dtype=float).reshape(3),
                               np.asarray(raw["rotation_delta"], dtype=float).reshape(3), [gripper]])


class ReplayPolicy:
    """Re-executes the actions of one recorded demonstration."""

    stop_on_success = False  # keep the release and the retreat, which come after the goal is first met

    def __init__(self, actions: np.ndarray, name: str) -> None:
        self.actions = np.asarray(actions, dtype=float)
        self.name = name
        self.max_steps: int | None = len(self.actions)
        self.chunk_steps, self.chunks, self.seconds = [], [], []

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, image: np.ndarray, wrist: np.ndarray, step: int) -> np.ndarray:
        return self.actions[step]


class StepLog:
    """Per-step records of one episode."""

    KEYS = ("agentview", "wrist", "seg", "ee_pos", "ee_quat", "gripper_qpos", "object_pos", "grasped", "touched",
            "visible_px", "in_goal")

    def __init__(self, env, names: list[str], target: str) -> None:
        self.env, self.names, self.target = env, names, target
        self.template = goal_template(env)
        instances = list(env.env.model.instances_to_ids.keys())
        unknown = [name for name in names if name not in instances]
        if unknown:  # such an object would silently count as never visible
            raise RuntimeError(f"objects without a segmentation instance: {unknown}")
        self.seg_values = {instances.index(name) + 1: k + 1 for k, name in enumerate(names)}
        self.robot_values = [instances.index(name) + 1 for name in ROBOT_INSTANCES if name in instances]
        self.rows: dict[str, list] = {key: [] for key in self.KEYS}
        self.actions: list[np.ndarray] = []

    def in_goal(self, name: str) -> bool:
        if self.template is None:  # goal is not a single object predicate: only the task's own check is known
            return bool(name == self.target and self.env.check_success())
        predicate, region = self.template
        if region.startswith(f"{name}_"):  # the object that owns the goal region, e.g. basket_1_contain_region
            return False
        return bool(self.env.env._eval_predicate([predicate, name, region]))

    def record(self, obs: dict) -> None:
        inner, gripper = self.env.env, self.env.env.robots[0].gripper
        raw_seg = obs[f"{CAMERA}_segmentation_instance"][..., 0]
        seg = np.zeros(raw_seg.shape, dtype=np.uint8)
        for value, index in self.seg_values.items():
            seg[raw_seg == value] = index
        seg[np.isin(raw_seg, self.robot_values)] = ROBOT_SEG_ID
        objects = [inner.objects_dict[name] for name in self.names]
        row = {
            "agentview": model_view(obs[f"{CAMERA}_image"]), "wrist": model_view(obs["robot0_eye_in_hand_image"]),
            "seg": model_view(seg), "ee_pos": obs["robot0_eef_pos"].copy(), "ee_quat": obs["robot0_eef_quat"].copy(),
            "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
            "object_pos": np.array([inner.sim.data.body_xpos[inner.obj_body_id[name]] for name in self.names]),
            "grasped": np.array([inner._check_grasp(gripper=gripper, object_geoms=obj) for obj in objects]),
            "touched": np.array([inner.check_contact(gripper, obj) for obj in objects]),
            "visible_px": np.array([(seg == k + 1).sum() for k in range(len(self.names))]),
            "in_goal": np.array([self.in_goal(name) for name in self.names]),
        }
        for key, value in row.items():
            self.rows[key].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {key: np.stack(values) for key, values in self.rows.items()}
        out["actions"] = np.stack(self.actions) if self.actions else np.zeros((0, 7))
        return out


def run_episode(env, policy, instruction: str, init_state: np.ndarray, names: list[str], target: str,
                max_steps: int) -> tuple[dict[str, np.ndarray], bool]:
    """One episode from `init_state`; returns the log arrays and whether the task's own goal was reached.

    A policy episode ends when the task's goal is met (as in starVLA's evaluation) or when the object named
    by the instruction satisfies the goal, which differs from the task's goal under a swapped instruction.
    """
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    policy.reset(instruction)
    log = StepLog(env, names, target)
    success = False
    for step in range(max_steps):
        log.record(obs)
        action = policy.act(log.rows["agentview"][-1], log.rows["wrist"][-1], step)
        log.actions.append(np.asarray(action, dtype=float))
        obs, _, done, _ = env.step(action.tolist())  # LIBERO returns its goal check as `done`
        success = success or bool(done)
        if policy.stop_on_success and (done or log.in_goal(target)):
            break
    log.record(obs)
    return log.arrays(), success


def camera_matrices(env) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    return (get_camera_intrinsic_matrix(env.sim, CAMERA, RESOLUTION, RESOLUTION),
            get_camera_extrinsic_matrix(env.sim, CAMERA))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tag", required=True, help="prefix of the output files, e.g. standard")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", default="pick up the salad dressing and place it in the basket")
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
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_policy(args: argparse.Namespace, suite, task_id: int):
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
    return ServerPolicy(args.host, args.port), {episode: initial_states[episode] for episode in args.episodes}


def save_episode(path: Path, meta: dict, arrays: dict[str, np.ndarray], policy, intrinsics: np.ndarray,
                 cam_to_world: np.ndarray) -> None:
    np.savez_compressed(
        path, meta=json.dumps(meta), intrinsics=intrinsics, cam_to_world=cam_to_world,
        chunk_steps=np.asarray(policy.chunk_steps, dtype=int),
        chunks=np.stack(policy.chunks) if policy.chunks else np.zeros((0, 0, 7), dtype=np.float32),
        infer_seconds=np.asarray(policy.seconds, dtype=float), **arrays)


def main() -> None:
    args = parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = find_task(suite, args.task)
    task = suite.get_task(task_id)
    env = SegmentationRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=RESOLUTION, camera_widths=RESOLUTION)
    env.seed(args.seed)  # the seed affects object placement even with a fixed initial state
    env.reset()
    names = sorted(env.env.objects_dict)
    target = args.target or env.obj_of_interest[0]
    if target not in names:
        raise SystemExit(f"target {target!r} is not a movable object; choose one of {names}")
    instruction = args.instruction or task.language
    shifts = {name: (float(dx), float(dy)) for name, dx, dy in args.shift}
    intrinsics, cam_to_world = camera_matrices(env)
    policy, starts = build_policy(args, suite, task_id)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    try:
        for episode, start in starts.items():
            started = time.perf_counter()
            arrays, success = run_episode(env, policy, instruction, shifted_state(env, start, shifts), names, target,
                                          policy.max_steps or MAX_STEPS[args.suite])
            meta = {"suite": args.suite, "task": task.language, "instruction": instruction, "target": target,
                    "objects_of_interest": list(env.obj_of_interest), "episode": episode, "shifts": shifts,
                    "seed": args.seed, "names": names, "policy": policy.name, "steps": int(len(arrays["actions"])),
                    "success": success, "delivered": bool(arrays["in_goal"][-1, names.index(target)]),
                    "wall_seconds": round(time.perf_counter() - started, 1)}
            path = args.out / f"{args.tag}_ep{episode}.npz"
            save_episode(path, meta, arrays, policy, intrinsics, cam_to_world)
            summary.append({**meta, "file": path.name})
            print(f"[{args.tag} ep{episode}] steps={meta['steps']} success={success} "
                  f"delivered({target})={meta['delivered']} wall={meta['wall_seconds']}s -> {path}", flush=True)
    finally:  # keep what finished, and release the simulator, even if an episode fails
        (args.out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))
        env.close()

if __name__ == "__main__":
    main()
