"""Replay RoboCasa365 demonstrations from their saved states and compare with the recorded videos.

For each task and episode: rebuild the scene from extras/ (model + ep_meta), then
  1. state replay: set every `--stride`-th recorded state, render the 3 cameras at 256x256 and compare the RGB
     with the video frame (PSNR; the videos are h264, so the comparison is to the compressed frame);
     depth and instance segmentation are rendered for the first camera;
  2. action replay: from the first state apply the recorded actions open loop and compare the simulator state
     with the recorded one at every step (the demonstration was recorded with different library versions).
Also times reset, step and render. CPU only (software rendering, robocasa365 env).

Usage (repo root, server):
  source envs/server/robocasa365_env.sh
  "$RC365_PYTHON" scripts/data/replay_robocasa365.py --task OpenDrawer --episodes 5 \
      --out $W/runs/robocasa365_replay/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import robocasa365 as rc  # noqa: E402


def _t(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def state_replay(env, ds, ep, episode, stride, size, keep) -> tuple[dict, dict]:
    states = episode["states"]
    videos = {cam: rc.read_video(ds, cam, ep) for cam in rc.CAMERAS}
    n = min(len(states), *(len(v) for v in videos.values()))
    ps = {cam: {d: [] for d in (-1, 0, 1)} for cam in rc.CAMERAS}
    t_rgb, t_aux, samples = [], [], {}
    for t in range(0, n, stride):
        rc.set_state(env, states[t])
        for cam in rc.CAMERAS:
            view, dt = _t(rc.render_view, env, cam, size)
            t_rgb.append(dt)
            for d in (-1, 0, 1):
                if 0 <= t + d < n:
                    ps[cam][d].append(rc.psnr(view["rgb"], videos[cam][t + d]))
            if cam == rc.CAMERAS[0]:
                aux, dt = _t(rc.render_view, env, cam, size, True, True)
                t_aux.append(dt)
                if t in keep:
                    samples[t] = dict(
                        video=videos[cam][t],
                        rgb=view["rgb"],
                        depth=aux["depth"],
                        geom=aux["geom"],
                        body=rc.body_of_geom(env, aux["geom"]),
                        names=rc.instance_names(env),
                    )
    summary = {cam: {str(d): float(np.mean(v)) for d, v in by.items()} for cam, by in ps.items()}
    summary["frames"] = len(range(0, n, stride))
    summary["min_psnr_aligned"] = min(min(ps[cam][0]) for cam in rc.CAMERAS)
    return dict(psnr=summary, render_rgb_s=float(np.mean(t_rgb)), render_rgb_depth_seg_s=float(np.mean(t_aux))), samples


def action_replay(env, episode) -> dict:
    states, actions = episode["states"], episode["actions"]
    rc.set_state(env, states[0])
    err, t_step = [], []
    for t in range(len(actions) - 1):
        _, dt = _t(env.step, actions[t])
        t_step.append(dt)
        err.append(float(np.linalg.norm(env.sim.get_state().flatten() - states[t + 1])))
    err = np.array(err)
    return dict(
        steps=len(err),
        step_s=float(np.mean(t_step)),
        state_err_first=float(err[0]),
        state_err_mid=float(err[len(err) // 2]),
        state_err_last=float(err[-1]),
        state_err_max=float(err.max()),
        exact_steps=int((err == 0).sum()),
    )


def save_figure(path: Path, samples: dict) -> None:
    import cv2

    def depth_img(d):
        lo, hi = np.percentile(d, [1, 99])
        return cv2.applyColorMap(
            (255 * (1 - np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8), cv2.COLORMAP_VIRIDIS
        )[..., ::-1]

    def seg_img(body, names):
        out = np.zeros(body.shape + (3,), np.uint8)
        for b in np.unique(body):
            if b >= 0:
                out[body == b] = np.random.default_rng(zlib.crc32(names[int(b)].encode())).integers(40, 255, 3)
        return out

    rows = [
        np.concatenate(
            [
                s["video"],
                s["rgb"],
                np.abs(s["video"].astype(int) - s["rgb"]).clip(0, 255).astype(np.uint8) * 4,
                depth_img(s["depth"]),
                seg_img(s["body"], s["names"]),
            ],
            axis=1,
        )
        for _, s in sorted(samples.items())
    ]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0)[..., ::-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--root", default=None, help="dataset root (default: $W/data/robocasa365)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5, help="compare every n-th state")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = args.root or str(Path(os.environ.get("W", "/mnt/nas/xinyizheng/affordance_vla")) / "data/robocasa365")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in args.task:
        ds = rc.dataset_dir(root, task)
        env, t_make = _t(rc.make_env, ds)
        results = dict(task=task, dataset=str(ds), episodes=[], make_env_s=t_make, args=vars(args))
        for ep in range(args.episodes):
            episode = rc.load_episode(ds, ep)
            _, t_reset = _t(rc.reset_to_episode, env, episode)
            keep = {0, len(episode["states"]) // 2, len(episode["states"]) - 1}
            keep = {t - t % args.stride for t in keep}
            res, samples = state_replay(env, ds, ep, episode, args.stride, args.size, keep)
            res.update(
                episode=ep,
                frames=len(episode["states"]),
                reset_s=t_reset,
                actions=action_replay(env, episode),
                lang=episode["ep_meta"].get("lang"),
            )
            results["episodes"].append(res)
            if ep == 0:
                save_figure(out / f"{task}_ep0_video_replay_diff_depth_seg.png", samples)
                names = {int(b): n for b, n in next(iter(samples.values()))["names"].items()}
                (out / f"{task}_ep0_instances.json").write_text(json.dumps(sorted(set(names.values())), indent=1))
            print(
                task,
                ep,
                json.dumps({k: res[k] for k in ("psnr", "reset_s", "render_rgb_s")}, default=float)[:300],
                flush=True,
            )
            (out / f"{task}_replay.json").write_text(json.dumps(results, indent=1, default=float))
        env.close()


if __name__ == "__main__":
    main()
