"""Local MuJoCo simulator for LIBERO-PRO: open a task's scene, show the 3D interaction maps (L) on its objects,
run a scripted grasp in it, or replay a test episode that the evaluation on the server logged, all live in the
MuJoCo viewer.

Run from the repository root; scripts/local/sim.sh starts this under mjpython, which the viewer needs on macOS:
    bash scripts/local/sim.sh scene  --suite libero_object_swap --task 2 --episode 0
    bash scripts/local/sim.sh map    --suite libero_10 --task 4
    bash scripts/local/sim.sh grasp  --suite libero_object_swap --task 2 --object ketchup_1
    bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
    bash scripts/local/sim.sh list   runs/libero_pro/predicted_g1_hold_s10
Defaults come from configs/sim/local.yaml; --config picks another file, --set key=value overrides an entry
(e.g. --set viewer.realtime=0.5). The data comes from scripts/local/fetch_sim_data.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from affordance_vla.config import load_config  # noqa: E402
from affordance_vla.sim.live_view import LiveView, Marker, offscreen_without_windows  # noqa: E402

BLUE = (0.2, 0.45, 0.9, 0.8)
MAP_LOW, MAP_HIGH = (1.0, 0.85, 0.35, 0.45), (0.8, 0.05, 0.05, 0.95)  # score 0 -> pale yellow, 1 -> red
SETTLE_STEPS = 30
GRASP_PHASES = {"above": "move above the grasp point", "descend": "descend to the grasp point",
                "close": "close the gripper", "lift": "lift", "carry": "move above the place point",
                "lower": "lower to the place point", "release": "open the gripper", "retreat": "move up and away",
                "done": "done"}


def prepare_libero(config: dict) -> None:
    """LIBERO reads its paths at import: point it at the local checkout before anything imports it."""
    checkout, settings = REPO / config["libero_pro"], REPO / config["libero_config"]
    if not (settings / "config.yaml").is_file() or not checkout.is_dir():
        raise SystemExit(f"no LIBERO-PRO at {checkout}; run scripts/local/fetch_sim_data.sh first")
    os.environ["LIBERO_CONFIG_PATH"] = str(settings)
    sys.path.insert(0, str(checkout))


def open_scene(config: dict, cameras: bool) -> tuple:
    """(env, names, first observation) of the configured scene, reset to its initial state and settled."""
    from affordance_vla.sim.episode import benchmark_instruction, make_env, start_episode
    from affordance_vla.sim.narration import explain_scene

    scene = config["scene"]
    suite, task, env = make_env(scene["suite"], int(scene["task"]), int(scene["seed"]), cameras=cameras)
    obs = start_episode(env, suite.get_task_init_states(int(scene["task"]))[int(scene["episode"])])
    names = sorted(env.env.objects_dict)
    instruction = benchmark_instruction(scene["suite"], task)
    explain_scene(scene["suite"], task, env, names, instruction, int(scene["episode"]), int(scene["seed"]))
    return env, names, obs


def contact_offsets(config: dict, suite: str, names: list[str], required: bool = True):
    """Grasp / release offsets measured on the training demonstrations of the suite's base (None if absent)."""
    from affordance_vla.plan.plan_input import offsets_from_json, with_same_kind
    from affordance_vla.sim.episode import base_suite

    path = REPO / config["offsets"].get(base_suite(suite), "")
    if not path.is_file():
        if required:
            raise SystemExit(f"no contact offsets at {path}; run scripts/local/fetch_sim_data.sh first")
        return None
    return with_same_kind(offsets_from_json(path.read_text()), names)


def positions(env, names: list[str]) -> dict[str, np.ndarray]:
    inner = env.env
    return {name: inner.sim.data.body_xpos[inner.obj_body_id[name]].copy() for name in names}


def cmd_scene(config: dict, view: LiveView) -> None:
    from affordance_vla.plan.plan_input import grasp_contact
    from affordance_vla.sim.episode import DUMMY_ACTION

    env, names, _ = open_scene(config, cameras=False)
    offsets = contact_offsets(config, config["scene"]["suite"], names, required=False)
    print("[window] orange dots: each object's grasp point (where the gripper closed in the demonstrations).")
    print("  Drag an object: double-click it, then Ctrl + right mouse button; space pauses; close the window to end.")
    while True:
        where = positions(env, names)
        markers = () if offsets is None else tuple(
            Marker(grasp_contact(offsets, name, where[name])) for name in names if name in offsets.grasp)
        if not view.show(env, markers):
            break
        env.step(DUMMY_ACTION)


def category_of(name: str) -> str:
    """'porcelain_mug_1' -> 'porcelain_mug', the category the interaction maps are keyed by."""
    return re.sub(r"_\d+$", "", name)


def map_markers(maps: dict, env, names: list[str], settings: dict) -> tuple[Marker, ...]:
    """Each object's 3D interaction map at the object's current pose: its highest-scoring surface points coloured
    by score, and the closing axis through the hottest point as a short blue line."""
    from affordance_vla.plan.interaction_labels import map_in_world
    from affordance_vla.sim.object_surface import quat_wxyz_matrix

    inner, markers = env.env, []
    for name in names:
        cmap = maps.get(category_of(name))
        if cmap is None:
            continue
        body = inner.obj_body_id[name]
        points, axes = map_in_world(cmap, inner.sim.data.body_xpos[body],
                                    quat_wxyz_matrix(inner.sim.data.body_xquat[body]))
        order = np.argsort(-cmap.score)[:int(settings["points"])]
        order = order[cmap.score[order] >= float(settings["min_score"])]
        for k in order:
            score = float(cmap.score[k])
            rgba = tuple(low + score * (high - low) for low, high in zip(MAP_LOW, MAP_HIGH))
            markers.append(Marker(points[k], rgba=rgba, radius=float(settings["radius"])))
        if len(order):
            half = float(settings["axis_m"]) / 2
            markers += [Marker(points[order[0]] + t * axes[order[0]], rgba=BLUE, radius=0.003)
                        for t in np.linspace(-half, half, 9)]
    return tuple(markers)


def cmd_map(config: dict, view: LiveView) -> None:
    from affordance_vla.plan.interaction_labels import load_maps
    from affordance_vla.sim.episode import DUMMY_ACTION, base_suite

    env, names, _ = open_scene(config, cameras=False)
    path = REPO / config["maps"].get(base_suite(config["scene"]["suite"]), "")
    if not path.is_file():
        raise SystemExit(f"no interaction maps at {path}; run scripts/local/fetch_sim_data.sh first")
    maps = load_maps(path)
    shown = [name for name in names if category_of(name) in maps]
    print(f"[map] 3D interaction maps (L, from the demonstrations' grasps) on: {', '.join(shown) or 'no object'}")
    print("  yellow -> red: how much grasp contact lands on that surface point; blue line: the closing axis of the")
    print("  gripper at the hottest point. The maps follow the objects: drag one (double-click, Ctrl + right drag).")
    while True:
        if not view.show(env, map_markers(maps, env, names, config["map"])):
            break
        env.step(DUMMY_ACTION)


def grasp_targets(offsets, env, names: list[str], target: str):
    from affordance_vla.plan.plan_input import grasp_contact, release_contact
    from affordance_vla.sim.episode import goal_container
    from affordance_vla.sim.scripted_grasp import Targets

    where = positions(env, names)
    container = goal_container(env, names)
    return Targets(grasp_contact(offsets, target, where[target]), release_contact(offsets, where[container]))


def cmd_grasp(config: dict, view: LiveView) -> None:
    from affordance_vla.sim.episode import DUMMY_ACTION, goal_template
    from affordance_vla.sim.scripted_grasp import GraspSettings, GraspState, next_action

    env, names, obs = open_scene(config, cameras=False)
    settings = GraspSettings(**{key: value for key, value in config["grasp"].items() if key != "object"})
    target = config["grasp"]["object"] or env.env.parsed_problem["goal_state"][0][1]
    if target not in names:
        raise SystemExit(f"{target!r} is not a movable object here; choose one of {names}")
    offsets = contact_offsets(config, config["scene"]["suite"], names)
    print(f"[grasp] object {target}; orange = grasp point, blue = place point (the same points the ORACLE plan uses)")
    state, targets, phase = GraspState(), grasp_targets(offsets, env, names, target), None
    gripper, obj = env.env.robots[0].gripper, env.env.objects_dict[target]
    held_after_lift = None
    while state.phase != "done":
        if state.phase in ("above", "descend"):  # follow the object until the gripper closes on it
            targets = grasp_targets(offsets, env, names, target)
        if state.phase != phase:
            if phase == "lift":
                held_after_lift = bool(env.env._check_grasp(gripper=gripper, object_geoms=obj))
                print(f"    after the lift {target} is {'held' if held_after_lift else 'not held'}", flush=True)
            phase = state.phase
            print(f"  {GRASP_PHASES[phase]}", flush=True)
        action, state = next_action(state, obs["robot0_eef_pos"], targets, settings)
        obs = env.step(action.tolist())[0]
        if not view.show(env, (Marker(targets.grasp), Marker(targets.release, rgba=BLUE))):
            return
    if held_after_lift is None:  # without placing, the run ends right after the lift
        held_after_lift = bool(env.env._check_grasp(gripper=gripper, object_geoms=obj))
    for _ in range(SETTLE_STEPS):  # let the object come to rest before the goal check
        env.step(DUMMY_ACTION)
        view.show(env, (Marker(targets.grasp), Marker(targets.release, rgba=BLUE)))
    template = goal_template(env)
    placed = bool(template and env.env._eval_predicate([template[0], target, template[1]]))
    task_goal = "met" if env.check_success() else "not met"
    print(f"[result] {'held' if held_after_lift else 'not held'} at the lift; {target} is "
          f"{'inside' if placed else 'outside'} the goal region ({template[1] if template else '-'}); "
          f"the task's own goal check: {task_goal}")
    if config["viewer"]["hold"]:
        view.hold((Marker(targets.grasp), Marker(targets.release, rgba=BLUE)))


def cmd_replay(log_path: Path, view: LiveView, hold: bool) -> None:
    from affordance_vla.sim.episode import make_env, max_steps_of, run_episode
    from affordance_vla.sim.logged_episode import LoggedPolicy, read_queries, task_index
    from affordance_vla.sim.narration import Narrator, WindowClosed, compare, explain_loop, explain_scene
    from affordance_vla.sim.rollout_log import load_rollout_log

    log = load_rollout_log(log_path, images=False)
    meta, queries, task_id = log.meta, read_queries(log_path), task_index(log_path, log.meta)
    suite, task, env = make_env(meta["suite"], task_id, int(meta["seed"]))
    names = sorted(env.env.objects_dict)
    if names != list(log.names):
        raise SystemExit(f"the scene's objects {names} differ from the log's {list(log.names)}")
    explain_scene(meta["suite"], task, env, names, meta["instruction"], int(meta["episode"]), int(meta["seed"]))
    explain_loop(meta, queries, max_steps_of(meta["suite"]))
    narrator = Narrator(queries, names, meta.get("plan", "none"), env, view)
    init_state = suite.get_task_init_states(task_id)[int(meta["episode"])]
    try:
        arrays, success = run_episode(env, LoggedPolicy(log.actions), meta["instruction"], init_state, names,
                                      meta["target"], min(len(log.actions), max_steps_of(meta["suite"])),
                                      on_step=narrator)
    except WindowClosed:
        return
    print(f"[end] step {len(arrays['actions'])}, {'LIBERO goal check succeeded' if success else 'no success'}")
    compare(meta, log.ee_pos, arrays, success)
    if hold:
        view.hold(narrator.markers)


def cmd_list(folder: Path) -> None:
    rows = [entry | {"suite": summary.parent.name} for summary in sorted(folder.glob("*/*_summary*.json"))
            for entry in json.loads(summary.read_text())]
    if not rows:
        raise SystemExit(f"no task summaries under {folder}")
    for row in rows:
        changed = "" if row["instruction"] == row["task"] else f"  instruction: {row['instruction']}"
        print(f"{row['suite']}/{row['file']:16s} {'success' if row['success'] else 'failure'} {row['steps']:3d} steps  "
              f"{row['task']}{changed}")
    print(f"{len(rows)} episodes, {sum(bool(row['success']) for row in rows)} successful")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("scene", "map", "grasp", "replay", "list"))
    parser.add_argument("path", nargs="?", type=Path, help="replay: a rollout log; list: a run folder")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "sim" / "local.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--suite", help="= --set scene.suite=...")
    parser.add_argument("--task", type=int, help="= --set scene.task=...")
    parser.add_argument("--episode", type=int, help="= --set scene.episode=...")
    parser.add_argument("--object", help="= --set grasp.object=... (grasp)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shortcuts = {"scene.suite": args.suite, "scene.task": args.task, "scene.episode": args.episode,
                 "grasp.object": args.object}
    overrides = args.overrides + [f"{key}={value}" for key, value in shortcuts.items() if value is not None]
    config = load_config(args.config, overrides)
    if args.command in ("replay", "list") and args.path is None:
        raise SystemExit(f"{args.command} needs a path (a log for replay, a run folder for list)")
    if args.command == "list":
        return cmd_list(args.path)
    prepare_libero(config)
    offscreen_without_windows()
    view = LiveView(float(config["viewer"]["realtime"]))
    try:
        {"scene": lambda: cmd_scene(config, view), "map": lambda: cmd_map(config, view),
         "grasp": lambda: cmd_grasp(config, view),
         "replay": lambda: cmd_replay(args.path, view, bool(config["viewer"]["hold"]))}[args.command]()
    finally:
        view.close()


if __name__ == "__main__":
    main()
