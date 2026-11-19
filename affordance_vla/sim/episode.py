"""One LIBERO / LIBERO-PRO test episode on the simulator side, as the evaluations run it.

This is the loop of starVLA's `eval_libero.py`, with a log of every step (end effector, each object's
position, grasp and touch flags from simulator contacts, visible pixels per object, the goal check per
object). It is shared by the server's evaluation script (scripts/eval/rollout_libero_logged.py, which asks a
policy server for actions) and by the local viewer (scripts/local/show_test_episode.py, which replays the
actions of a logged episode), so both run exactly the same steps:

  1. the task: a suite (e.g. libero_object_swap) and a task index give a BDDL file and its goal predicate;
  2. the instruction: for a perturbed suite, the wording its task file changed (benchmark_instruction);
  3. the start: one of the suite's recorded initial states, then WAIT_STEPS no-op steps to let objects settle;
  4. the loop: at every step the policy gets the two camera images (rotated as in training), the
     instruction, and optionally a plan; it returns one action (a server answers a chunk at a time);
  5. the end: LIBERO's own goal check succeeds, the instructed object meets the goal, or the step limit
     of the suite (MAX_STEPS) is reached.

Needs the LIBERO simulator environment (robosuite, MuJoCo, LIBERO on PYTHONPATH); LIBERO and robosuite are
imported inside the functions, so the module itself imports with numpy only.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from affordance_vla.sim.rollout_log import ROBOT_SEG_ID

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # no motion, gripper open: lets the objects settle
RESOLUTION = 256  # resolution of the training images
WAIT_STEPS = 10
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
CAMERA = "agentview"
ROBOT_INSTANCES = ("Panda0", "RethinkMount0", "PandaGripper0")
GRIPPER_OPEN_THRESHOLD = 0.5  # the policy's gripper output is 1 for open and 0 for close


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


def base_suite(suite_name: str) -> str:
    """The LIBERO suite a variant is built on: libero_object for libero_object_swap."""
    known = [name for name in MAX_STEPS if suite_name == name or suite_name.startswith(f"{name}_")]
    if not known:
        raise SystemExit(f"suite {suite_name!r} is not built on a known LIBERO suite ({', '.join(MAX_STEPS)})")
    return max(known, key=len)


def max_steps_of(suite_name: str) -> int:
    """Step limit of a suite; a variant such as libero_object_swap takes the limit of the suite it is built on."""
    return MAX_STEPS[base_suite(suite_name)]


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


class StepLog:
    """Per-step records of one episode."""

    KEYS = ("agentview", "wrist", "seg", "ee_pos", "ee_quat", "gripper_qpos", "object_pos", "object_quat", "grasped",
            "touched", "visible_px", "in_goal")

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
            # MuJoCo keeps (w, x, y, z); stored as (x, y, z, w) like robosuite's end-effector quaternion
            "object_quat": np.array([inner.sim.data.body_xquat[inner.obj_body_id[name]][[1, 2, 3, 0]]
                                     for name in self.names]),
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


def start_episode(env, init_state: np.ndarray) -> dict:
    """Reset to a recorded initial state and let the objects settle; returns the first observation."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    return obs


def run_episode(env, policy, instruction: str, init_state: np.ndarray, names: list[str], target: str,
                max_steps: int, planner=None,
                on_step: Callable[[int, dict, "StepLog"], None] | None = None) -> tuple[dict[str, np.ndarray], bool]:
    """One episode from `init_state`; returns the log arrays and whether the task's own goal was reached.

    A policy episode ends when the task's goal is met (as in starVLA's evaluation) or when the object named
    by the instruction satisfies the goal, which differs from the task's goal under a swapped instruction.
    `planner` maps the log rows so far to the plan input sent with the query (None: no plan is sent).
    `on_step(step, obs, log)` is called once the action of a step is known, before it is executed (the
    local viewer draws the frame there); it does not change the episode.
    """
    obs = start_episode(env, init_state)
    policy.reset(instruction)
    log = StepLog(env, names, target)
    success = False
    for step in range(max_steps):
        log.record(obs)
        plan = planner(log.rows) if planner else None
        action = policy.act(log.rows["agentview"][-1], log.rows["wrist"][-1], step, plan, log.rows["ee_pos"][-1])
        log.actions.append(np.asarray(action, dtype=float))
        if on_step is not None:
            on_step(step, obs, log)
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


def camera_matrices(env) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    return (get_camera_intrinsic_matrix(env.sim, CAMERA, RESOLUTION, RESOLUTION),
            get_camera_extrinsic_matrix(env.sim, CAMERA))


def make_env(suite_name: str, task_id: int, seed: int, depth: bool = False, cameras: bool = True) -> tuple:
    """(suite, task, simulator) for task `task_id` of a suite, with the cameras the policy was trained on.

    `cameras=False` builds the simulator without any rendering (for the viewer, which draws the scene itself).
    The seed is set before the first reset because it affects object placement even with a fixed initial state.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    if not 0 <= task_id < suite.n_tasks:
        raise SystemExit(f"suite {suite_name} has tasks 0-{suite.n_tasks - 1}, not {task_id}")
    task = suite.get_task(task_id)
    bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    if cameras:
        env = SegmentationRenderEnv(bddl_file_name=bddl, camera_heights=RESOLUTION, camera_widths=RESOLUTION,
                                    camera_depths=depth)
    else:
        env = ControlEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False)
    env.seed(seed)
    env.reset()
    return suite, task, env
