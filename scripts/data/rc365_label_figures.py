"""Project the labels of step 1 (1) onto rendered RoboCasa365 frames, one figure per task and episode (AFFO-15).

Each figure: three frames of one demonstration (approach, touch, a second later), the same camera. For an object
task the held-out object's map carried from the pretrain instances (mug: the treatment named in the config), or,
for an object without a category map, the contact patch of the demonstrated grasp; for a part task the contact
point, closing line (white) and where the gripper moves next (green arrow). Needs the extraction of
scripts/data/rc365_extract.py (surfaces, pretrain grasps) and the robocasa365 env; CPU only.

  source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
  "$RC365_PYTHON" scripts/data/rc365_label_figures.py --config configs/rc365_labels/figures.yaml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from affordance_vla.config import load_config
from affordance_vla.plan import rc365_events as ev
from affordance_vla.plan import rc365_symmetry as sym
from affordance_vla.plan.interaction_labels import contact_weights
from affordance_vla.sim import rc365_project as proj
from affordance_vla.sim import rc365_scan as scan
from affordance_vla.sim import robocasa365 as rc
from affordance_vla.sim.rc365_episodes import episode_objects, read_meta

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"
WHITE, GREEN, RED = (255, 255, 255), (60, 220, 60), (230, 40, 40)
CLOSING_HALF_M = 0.05


def mug_treatment(cfg: dict, work: Path):
    """The pretrain-built treatment and the held-out items, for the categories that have a category map."""
    folder, picks = work / cfg["extract"], ev.object_grasps(ev.load_results(work / cfg["extract"]))
    out = {}
    for category, handle in cfg["map_categories"].items():
        own = [g for g in picks if g.category == category]
        surfaces = {g.instance: ev.load_surface(folder, g.instance) for g in own}
        sources = sym.build_items([g for g in own if g.split == "pretrain"], surfaces)
        targets = {i.name: i for i in sym.build_items([g for g in own if g.split == "target"], surfaces)}
        out[category] = (sym.TREATMENTS[cfg["treatment"]](sources, handle), targets, own)
    return out


def object_pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    body = env.sim.model.body_name2id(env.objects[name].root_body)
    return env.sim.data.body_xpos[body].copy(), env.sim.data.body_xmat[body].reshape(3, 3).copy()


def object_labels(env, episode: dict, meta: dict, split: str, ep: int, mugs: dict, cfg: dict):
    """(event, points in the object frame, scores, axes) for the first lifted pick of the episode, or None."""
    tracked = episode_objects(meta, None, {"obj"})
    events = [
        e
        for e in scan.scan_object_grasps(env, episode, {n: o["category"] for n, o in tracked.items()})
        if e["lift_m"] >= ev.MIN_LIFT_M
    ]
    if not events:
        return None
    event, info = events[0], tracked[events[0]["object"]]
    surface = scan.object_surface(env, event["object"])
    pick = ev.object_grasps(
        [
            {
                "task": "",
                "split": split,
                "episode": ep,
                "events": [{**event, "instance": info["instance"], "kind": "object"}],
            }
        ]
    )[0]
    if info["category"] in mugs and info["instance"] in mugs[info["category"]][1]:
        treatment, targets, _ = mugs[info["category"]]
        target = replace(targets[info["instance"]], grasps=(pick.grasp,), robot=(pick.base_in_object,))
        cmap = next(iter(treatment.maps(target, [0])))[1]
        return event, pick, cmap.points, cmap.score, cmap.axis, "map carried from pretrain mugs"
    weights = contact_weights(surface, pick.grasp)
    return event, pick, surface.points, weights / max(weights.max(), 1e-9), None, "contact patch of this grasp"


def panel(env, cam_name: str, size: int, step: int, episode: dict, draw) -> np.ndarray:
    rc.set_state(env, episode["states"][step])
    view = rc.render_view(env, cam_name, size, depth=True)
    cam = proj.camera_from(env, cam_name, size)
    image = np.ascontiguousarray(view["rgb"])
    return draw(image, cam, view["depth"])


def draw_object(env, labels, name: str):
    event, pick, points, score, axis, _ = labels

    def draw(image, cam, depth):
        pos, rot = object_pose(env, name)
        world = points @ rot.T + pos
        uv, z = proj.project(cam, world)
        keep = proj.visible(uv, z, depth)
        image = proj.draw_map(image, uv[keep], score[keep])
        centre = np.asarray(event["grasp_centre"])
        closing = np.asarray(event["ee_rot"]).reshape(3, 3)[:, 0]
        proj.draw_segment(image, cam, centre - CLOSING_HALF_M * closing, centre + CLOSING_HALF_M * closing, WHITE, 2)
        return image

    return draw


def draw_part(event: dict, step: int, later: bool):
    def draw(image, cam, depth):
        centre = np.mean(event["points"], axis=0) if event["points"] else np.asarray(event["grasp_centre"])
        closing = np.asarray(event["ee_rot"]).reshape(3, 3)[:, 0]
        proj.draw_point(image, cam, centre, RED)
        proj.draw_segment(
            image,
            cam,
            np.asarray(event["grasp_centre"]) - CLOSING_HALF_M * closing,
            np.asarray(event["grasp_centre"]) + CLOSING_HALF_M * closing,
            WHITE,
            2,
        )
        proj.draw_segment(image, cam, np.asarray(event["ee_pos"]), np.asarray(event["ee_later"]), GREEN, 3, arrow=True)
        return image

    return draw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--only", nargs="*", help="task names (default: all in the config)")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config if not Path(args.config).is_absolute() else args.config, args.set)
    work = Path(os.environ.get("W", DEFAULT_WORK_DIR))
    out = REPO / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    mugs = mug_treatment(cfg, work)
    size, cam_name = cfg["size"], cfg["camera"]
    for task, spec in cfg["tasks"].items():
        if args.only and task not in args.only:
            continue
        ds = rc.dataset_dir(work / cfg["data_root"], task, spec["split"])
        env, made = rc.make_env(ds), 0
        for ep in range(spec.get("first", 0), spec.get("first", 0) + spec["tries"]):
            if made >= spec["episodes"]:
                break
            episode, meta = rc.load_episode(ds, ep), read_meta(ds, ep)
            rc.reset_to_episode(env, episode)
            if spec["kind"] == "object":
                labels = object_labels(env, episode, meta, spec["split"], ep, mugs, cfg)
                if labels is None:
                    continue
                step, name = labels[0]["step"], labels[0]["object"]
                draws = [draw_object(env, labels, name)] * 3
                note = f"{task} ep{ep}: {labels[5]}"
            else:
                event = scan.scan_part_touch(env, episode)
                if event is None:
                    continue
                step = event["step"]
                draws, note = (
                    [draw_part(event, step, k == 2) for k in range(3)],
                    f"{task} ep{ep}: contact, closing line, next motion",
                )
            steps = [max(0, step - cfg["before"]), step, min(len(episode["states"]) - 1, step + cfg["after"])]
            frames = [panel(env, cam_name, size, s, episode, d) for s, d in zip(steps, draws)]
            frames = [
                proj.caption(f, f"step {s}" + (" (touch)" if k == 1 else ""))
                for k, (f, s) in enumerate(zip(frames, steps))
            ]
            figure = proj.caption(np.concatenate(frames, axis=1), note + f" [{meta['lang']}]")
            cv2.imwrite(str(out / f"{task}_ep{ep}.png"), figure[..., ::-1])
            print(out / f"{task}_ep{ep}.png", flush=True)
            made += 1
        env.close()


if __name__ == "__main__":
    main()
