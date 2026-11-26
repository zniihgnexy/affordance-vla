"""AFFO-18, part A: the joint-derived label drawn on rendered frames of the four part tasks. Per demonstration: the
contact point (red), where the part moves at the contact by the joint (green arrow), where the gripper moved by the old
label (orange arrow, both 8 cm long: only the direction is compared), the joint axis (cyan line through the anchor,
magenta dot). Replays the listed episodes (CPU, software rendering).

  source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
  "$RC365_PYTHON" scripts/data/rc365_joint_figures.py --config configs/rc365_labels/joint_figures.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from affordance_vla.config import load_config
from affordance_vla.plan import rc365_joint_labels as jl
from affordance_vla.sim import rc365_panels as panels
from affordance_vla.sim import rc365_project as proj
from affordance_vla.sim import rc365_scan as scan
from affordance_vla.sim import robocasa365 as rc
from affordance_vla.sim.rc365_episodes import read_meta

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = "/mnt/nas/xinyizheng/affordance_vla"
RED, GREEN, ORANGE, CYAN, MAGENTA = (230, 40, 40), (60, 220, 60), (255, 150, 0), (0, 200, 220), (220, 0, 220)
ARROW_M, AXIS_HALF_M = 0.08, 0.10


def draw_labels(touch: jl.JointTouch, window: str):
    def draw(image, cam, depth):
        old = touch.old.world["motion"]
        new = touch.direction(window)
        proj.draw_segment(
            image, cam, touch.anchor - AXIS_HALF_M * touch.axis, touch.anchor + AXIS_HALF_M * touch.axis, CYAN, 2
        )
        proj.draw_point(image, cam, touch.anchor, MAGENTA, 3)
        if old is not None:
            proj.draw_segment(image, cam, touch.contact, touch.contact + ARROW_M * old, ORANGE, 3, arrow=True)
        if new is not None:
            proj.draw_segment(image, cam, touch.contact, touch.contact + ARROW_M * new, GREEN, 3, arrow=True)
        proj.draw_point(image, cam, touch.contact, RED)
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
    for task, spec in cfg["tasks"].items():
        if args.only and task not in args.only:
            continue
        ds = rc.dataset_dir(work / cfg["data_root"], task, spec["split"])
        env = rc.make_env(ds)
        for ep in spec["episodes"]:
            episode, meta = rc.load_episode(ds, ep), read_meta(ds, ep)
            rc.reset_to_episode(env, episode)
            event = scan.scan_part_touch(env, episode)
            touch = jl.joint_touches([{"task": task, "split": spec["split"], "episode": ep, "events": [event]}])[0]
            sign = {1.0: "+", -1.0: "-", 0.0: "0"}[touch.sign(cfg["window"])]
            title = (
                f"{task} ep{ep} [{meta['lang']}] {touch.joint_type} joint, sign {sign}, "
                f"lever {100 * touch.radius:.0f} cm; "
                "red contact, green joint direction, orange gripper direction, cyan axis"
            )
            figure = panels.touch_strip(
                env,
                cfg["camera"],
                cfg["size"],
                episode,
                event["step"],
                cfg["before"],
                cfg["after"],
                draw_labels(touch, cfg["window"]),
                title,
            )
            cv2.imwrite(str(out / f"{task}_ep{ep}.png"), figure[..., ::-1])
            print(out / f"{task}_ep{ep}.png", flush=True)
        env.close()


if __name__ == "__main__":
    main()
