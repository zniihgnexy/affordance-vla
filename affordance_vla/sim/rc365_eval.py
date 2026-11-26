"""Closed-loop evaluation of a starVLA policy server in RoboCasa365 kitchens (CPU simulator, software rendering).

Three conditions separate "instances the policy has seen" from "instances it has not":
  A  pretrain kitchens, pretrain object instances (everything seen)
  B  pretrain kitchens, target object instances   (only the object instances are new)
  C  target kitchens,   target object instances   (the official evaluation setting)
The inputs follow training (affordance_vla.sim.rc365_policy_io); every episode has its own seed, so the
same (base seed, task, condition, index) always rebuilds the same kitchen and objects.
Success is RoboCasa's own: the task check holds at some step. An episode ends at the first success or at
the task horizon (the official loop runs successful episodes to the horizon too; the metric is the same).
Needs the robocasa365 environment (source envs/server/robocasa365_env.sh) and the starVLA snapshot on PYTHONPATH.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from affordance_vla.sim.rc365_policy_io import (
    CAMERAS,
    LANGUAGE_KEY,
    action_steps,
    build_example,
    episode_seed,
)

CONDITIONS: dict[str, dict[str, Any]] = {
    "A": {"split": "pretrain"},
    "B": {"split": None, "obj_instance_split": "target", "layout_ids": -2, "style_ids": -2},
    "C": {"split": "target"},
}
VIDEO_EVERY = 2  # one video frame per two environment steps (10 Hz, as the starVLA example)


def task_horizon(task: str) -> int:
    """RoboCasa v1.0.1's own episode length of an atomic task."""
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS

    return int(ATOMIC_TASK_DATASETS[task]["horizon"])


def make_env(task: str, condition: str):
    """The gym environment of `task` under one condition; the scene is drawn at each `reset(seed=...)`."""
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<Task>)

    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition!r}")
    return gym.make(f"robocasa/{task}", enable_render=True, **CONDITIONS[condition])


class ServerPolicy:
    """A starVLA websocket policy server as a function from an example to an un-normalised action chunk."""

    def __init__(self, host: str, port: int) -> None:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()

    def __call__(self, example: dict) -> np.ndarray:
        response = self.client.predict_action({"examples": [example], "do_sample": False, "unnorm_key": None})
        return np.asarray(response["data"]["actions"])[0]

    def close(self) -> None:
        self.client.close()


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python number it holds (JSON cannot write numpy integers)."""
    return value.item() if isinstance(value, np.generic) else value


def record_line(record: dict[str, Any]) -> str:
    """An episode record as one JSON line (numpy scalars written as plain numbers)."""
    import json

    return json.dumps(record, default=_plain)


def _write_video(frames: list[np.ndarray], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=20 // VIDEO_EVERY, codec="libx264", quality=6)


def run_episode(
    env: Any,
    policy: Callable[[dict], np.ndarray],
    seed: int,
    horizon: int,
    n_act: int,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """One episode from the scene drawn by `seed`; returns its record (success, steps, a coarse trace)."""
    started = time.time()
    obs, _ = env.reset(seed=seed)
    steps, first_success, queries, policy_seconds = 0, None, 0, 0.0
    trace: list[list[float]] = []
    frames: list[np.ndarray] = []
    while steps < horizon and first_success is None:
        asked = time.perf_counter()
        chunk = action_steps(policy(build_example(obs)), n_act)
        policy_seconds += time.perf_counter() - asked
        queries += 1
        eef = np.round(obs["state.end_effector_position_relative"], 3)
        trace.append([steps, *eef, float(obs["state.gripper_qpos"][0])])
        for action in chunk:
            if video_path is not None and steps % VIDEO_EVERY == 0:
                frames.append(obs[f"video.{CAMERAS[0]}"])
            obs, _, _, _, info = env.step(action)
            steps += 1
            if info["success"]:
                first_success = steps
                break
            if steps >= horizon:
                break
    if video_path is not None:
        _write_video(frames, video_path)
    inner = getattr(env, "env", None)
    return {
        "seed": seed,
        "success": first_success is not None,
        "first_success_step": first_success,
        "steps": steps,
        "horizon": horizon,
        "seconds": round(time.time() - started, 1),
        "queries": queries,
        "policy_seconds": round(policy_seconds, 1),
        "lang": str(obs.get(LANGUAGE_KEY, "")),
        "layout_id": _plain(getattr(inner, "layout_id", None)),
        "style_id": _plain(getattr(inner, "style_id", None)),
        "trace": trace,
    }


def run_cell(
    task: str, condition: str, indices: list[int], policy: Callable[[dict], np.ndarray], base_seed: int,
    n_act: int, out_jsonl: Path, video_dir: Path | None = None, videos_per_cell: int = 3,
    horizon: int | None = None,
) -> list[dict[str, Any]]:
    """Episodes `indices` of one (task, condition) cell; each record is appended to `out_jsonl` as it finishes."""
    env = make_env(task, condition)
    horizon = horizon or task_horizon(task)
    records = []
    try:
        for index in indices:
            video = video_dir / f"{task}_{condition}_{index:03d}.mp4" if video_dir and index < videos_per_cell else None
            record = run_episode(env, policy, episode_seed(base_seed, task, condition, index), horizon, n_act, video)
            record |= {"task": task, "condition": condition, "index": index}
            records.append(record)
            with out_jsonl.open("a") as handle:
                handle.write(record_line(record) + "\n")
    finally:
        env.close()
    return records
