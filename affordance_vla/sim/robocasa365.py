"""RoboCasa365 demonstrations: rebuild an episode's kitchen from its saved model and state, render it.

A RoboCasa365 bundle (LeRobot v2.1 layout) holds, besides parquet data and 3 mp4 videos per episode, an
`extras/episode_XXXXXX/` folder with the MuJoCo model (`model.xml.gz`), the flattened simulator state of every
step (`states.npz`) and the episode meta (`ep_meta.json`). That is enough to put the simulator in any recorded
state and render what the videos do not contain: depth and instance segmentation.

Needs the robocasa365 env (envs/server/robocasa365_env.sh): software rendering, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
VIDEO_KEY = "observation.images.{cam}"
_GEOM = 5  # mjOBJ_GEOM


def dataset_dir(root: str | Path, task: str, split: str = "target") -> Path:
    """The `lerobot` folder of a downloaded task, e.g. <root>/v1.0/target/atomic/OpenDrawer/<date>/lerobot."""
    hits = sorted(Path(root).glob(f"v1.0/{split}/*/{task}/*/lerobot"))
    if not hits:
        raise FileNotFoundError(f"no {split} bundle of {task} under {root}")
    return hits[-1]


def make_env(ds: Path):
    """The task's environment with the arguments the demonstrations were recorded with, offscreen, no obs."""
    import robosuite
    import robocasa.utils.lerobot_utils as lu

    meta = lu.get_env_metadata(ds)
    kwargs = dict(meta["env_kwargs"], env_name=meta["env_name"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False)
    return robosuite.make(**kwargs)


def load_episode(ds: Path, ep: int) -> dict:
    import robocasa.utils.lerobot_utils as lu

    return dict(
        states=lu.get_episode_states(ds, ep),
        model=lu.get_episode_model_xml(ds, ep),
        ep_meta=lu.get_episode_meta(ds, ep),
        actions=lu.get_episode_actions(ds, ep),
    )


def reset_to_episode(env, episode: dict) -> None:
    """Rebuild the episode's scene (same layout, style, objects) and set its first state."""
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    reset_to(env, dict(states=episode["states"][0], model=episode["model"], ep_meta=json.dumps(episode["ep_meta"])))


def set_state(env, flat_state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(flat_state)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()


def instance_names(env) -> dict[int, str]:
    """Body id -> name of the top-level body it belongs to (robot base, a fixture or an object)."""
    model = env.sim.model
    names = {}
    for body in range(model.nbody):
        top = body
        while top != 0 and model.body_parentid[top] != 0:
            top = int(model.body_parentid[top])
        names[body] = model.body_id2name(top) if top else "world"
    return names


def render_view(env, cam: str, size: int = 256, depth: bool = False, seg: bool = False) -> dict:
    """RGB (uint8 HxWx3, upright), optionally metric depth (HxW, metres) and the geom id (HxW, -1 = none)."""
    from robosuite.utils.camera_utils import get_real_depth_map

    out = {"rgb": env.sim.render(height=size, width=size, camera_name=cam)[::-1]}
    if depth:
        _, d = env.sim.render(height=size, width=size, camera_name=cam, depth=True)
        out["depth"] = get_real_depth_map(env.sim, d)[::-1]
    if seg:
        s = env.sim.render(height=size, width=size, camera_name=cam, segmentation=True)[::-1]
        out["geom"] = np.where(s[..., 0] == _GEOM, s[..., 1], -1)
    return out


def body_of_geom(env, geom: np.ndarray) -> np.ndarray:
    """Per-pixel body id from a geom-id image (-1 stays -1)."""
    table = np.append(env.sim.model.geom_bodyid, -1)
    return table[geom]


def read_video(ds: Path, cam: str, ep: int) -> np.ndarray:
    """All frames of one recorded camera of one episode, (T, H, W, 3) uint8."""
    import av

    path = next(ds.glob(f"videos/*/{VIDEO_KEY.format(cam=cam)}/episode_{ep:06d}.mp4"))
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))
