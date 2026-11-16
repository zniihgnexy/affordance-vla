"""Closed-loop LIBERO rollouts with a per-step log (simulator side: runs in the LIBERO env, CPU only).

This is the loop of starVLA's `eval_libero.py` with three additions:
  1. a log of every step: end effector, each object's position, grasp and touch flags from simulator
     contacts, visible pixels per object (instance segmentation), and the goal check per object;
  2. optional home-made perturbations: another instruction for the same scene, or a shifted object;
  3. a replay mode that re-executes a recorded demonstration instead of asking a policy, which gives
     the same log for training data (and tests this script without a GPU);
  4. a state-replay mode that loads the recorded simulator states of demonstrations one by one, with
     no dynamics: the log is then exactly the demonstration, and every demonstration can be used;
  5. an optional plan input sent to the policy server with every query: the oracle plan from simulator
     ground truth (affordance_vla/plan/plan_input.py; needs this repo on PYTHONPATH), or an empty plan;
  6. with `--plan predicted` the server predicts the plan itself; `--hold-plan` then keeps each predicted
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

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open: lets the objects settle
RESOLUTION = 256  # resolution of the training images
WAIT_STEPS = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
CAMERA = "agentview"
ROBOT_SEG_ID = 255  # keep in step with affordance_vla/sim/rollout_log.py (this script runs in another environment)
ROBOT_INSTANCES = ("Panda0", "RethinkMount0", "PandaGripper0")
GRIPPER_OPEN_THRESHOLD = 0.5  # the policy's gripper output is 1 for open and 0 for close
HOLD_STEPS = 5  # the oracle plan treats the target as held until contact has been absent this long
IMAGE_ARRAYS = ("agentview", "wrist", "seg", "depth")  # what --images-for leaves out of a log


def owns_region(name: str, region: str) -> bool:
    """Whether the goal region is the object itself (plate_1 in LIBERO-Spatial's "On bowl plate_1") or one of
    its regions (basket_1_contain_region in LIBERO-Object)."""
    return region == name or region.startswith(f"{name}_")


def model_view(image: np.ndarray) -> np.ndarray:
    """The 180° rotation applied to LIBERO frames when the training data was made."""
    return np.ascontiguousarray(image[::-1, ::-1])


def metric_depth(sim, buffer: np.ndarray) -> np.ndarray:
    """The simulator's depth buffer (0 = near plane, 1 = far plane) as metres along the optical axis."""
    from robosuite.utils.camera_utils import get_real_depth_map

    return get_real_depth_map(sim, buffer).astype(np.float32)


def env_gripper_command(open_value: float) -> float:
    """Policy output (1 = open, 0 = close) to the simulator's command (-1 = open, +1 = close)."""
    return 1.0 - 2.0 * float(open_value > GRIPPER_OPEN_THRESHOLD)


def max_steps_of(suite_name: str) -> int:
    """Step limit of a suite; a variant such as libero_object_swap takes the limit of the suite it is built on."""
    known = [name for name in MAX_STEPS if suite_name == name or suite_name.startswith(f"{name}_")]
    if not known:
        raise SystemExit(f"no step limit is known for suite {suite_name!r}")
    return MAX_STEPS[max(known, key=len)]


def benchmark_instruction(suite_name: str, task) -> str:
    """The task's instruction; for a perturbed suite, with the words that its task file changed.

    LIBERO takes the instruction from the file name, which LIBERO-PRO's task perturbation leaves as it
    was: the changed instruction is only in the file's (:language ...) line (affordance_vla/sim/benchmark_text.py).
    """
    from libero.libero import get_libero_path

    from affordance_vla.sim.benchmark_text import bddl_language, perturbed_instruction

    folder = Path(get_libero_path("bddl_files"))
    bases = [name for name in MAX_STEPS if suite_name.startswith(f"{name}_")]
    original = folder / max(bases, key=len) / task.bddl_file if bases else None
    if original is None or not original.is_file():
        return task.language
    perturbed = folder / task.problem_folder / task.bddl_file
    return perturbed_instruction(task.language, bddl_language(original.read_text()),
                                 bddl_language(perturbed.read_text()))


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
        if owns_region(name, region):  # the container itself, e.g. basket_1 for basket_1_contain_region
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
        if f"{CAMERA}_depth" in obs:  # only when the simulator renders depth (--depth)
            row["depth"] = model_view(metric_depth(inner.sim, obs[f"{CAMERA}_depth"][..., 0]))
        for key, value in row.items():
            self.rows.setdefault(key, []).append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {key: np.stack(values) for key, values in self.rows.items()}
        out["actions"] = np.stack(self.actions) if self.actions else np.zeros((0, 7))
        return out


def run_episode(env, policy, instruction: str, init_state: np.ndarray, names: list[str], target: str,
                max_steps: int, planner=None) -> tuple[dict[str, np.ndarray], bool]:
    """One episode from `init_state`; returns the log arrays and whether the task's own goal was reached.

    A policy episode ends when the task's goal is met (as in starVLA's evaluation) or when the object named
    by the instruction satisfies the goal, which differs from the task's goal under a swapped instruction.
    `planner` maps the log rows so far to the plan input sent with the query (None: no plan is sent).
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
        plan = planner(log.rows) if planner else None
        action = policy.act(log.rows["agentview"][-1], log.rows["wrist"][-1], step, plan, log.rows["ee_pos"][-1])
        log.actions.append(np.asarray(action, dtype=float))
        obs, _, done, _ = env.step(action.tolist())  # LIBERO returns its goal check as `done`
        success = success or bool(done)
        if policy.stop_on_success and (done or log.in_goal(target)):
            break
    log.record(obs)
    return log.arrays(), success


def replay_states(env, states: np.ndarray, actions: np.ndarray, names: list[str],
                  target: str) -> tuple[dict[str, np.ndarray], bool]:
    """The log of a demonstration read off its recorded simulator states; no dynamics are run.

    State t is the one in which action t was issued, so T states give T rows and T - 1 actions.
    Returns the arrays and whether the task's goal holds in the last state.
    """
    log = StepLog(env, names, target)
    for state in states:
        log.record(env.set_init_state(state))
    log.actions = [np.asarray(action, dtype=float) for action in actions[:len(states) - 1]]
    return log.arrays(), bool(env.check_success())


def goal_container(env, names: list[str]) -> str:
    """The object that owns the goal region, e.g. basket_1 for basket_1_contain_region, plate_1 for plate_1."""
    template = goal_template(env)
    owners = [name for name in names if template and owns_region(name, template[1])]
    if len(owners) != 1:
        raise SystemExit(f"cannot tell which object the goal region belongs to (goal: {template})")
    return owners[0]


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
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id = args.task_id if args.task_id is not None else find_task(suite, args.task)
    if not 0 <= task_id < suite.n_tasks:
        raise SystemExit(f"suite {args.suite} has tasks 0-{suite.n_tasks - 1}, not {task_id}")
    task = suite.get_task(task_id)
    env = SegmentationRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=RESOLUTION, camera_widths=RESOLUTION, camera_depths=args.depth)
    env.seed(args.seed)  # the seed affects object placement even with a fixed initial state
    env.reset()
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
