"""Rendered panels of a replayed RoboCasa365 demonstration with labels drawn on them (AFFO-18). The same three-frame
strip as `scripts/data/rc365_label_figures.py` (approach, touch, a second later), for scripts that draw their own
labels. Needs the robocasa365 env (rendering through `robocasa365.render_view`)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from affordance_vla.sim import rc365_project as proj
from affordance_vla.sim import robocasa365 as rc

Draw = Callable[[np.ndarray, dict, np.ndarray], np.ndarray]  # (image, camera, depth) -> image


def panel(env, cam_name: str, size: int, step: int, episode: dict, draw: Draw) -> np.ndarray:
    """The frame at `step` seen from the named camera, with `draw` applied."""
    rc.set_state(env, episode["states"][step])
    view = rc.render_view(env, cam_name, size, depth=True)
    return draw(np.ascontiguousarray(view["rgb"]), proj.camera_from(env, cam_name, size), view["depth"])


def touch_strip(
    env, cam_name: str, size: int, episode: dict, step: int, before: int, after: int, draw: Draw, title: str
):
    """Three panels side by side: `before` steps before the touch, the touch, `after` steps later; a title above."""
    steps = [max(0, step - before), step, min(len(episode["states"]) - 1, step + after)]
    frames = [panel(env, cam_name, size, s, episode, draw) for s in steps]
    frames = [
        proj.caption(f, f"step {s}" + (" (touch)" if k == 1 else "")) for k, (f, s) in enumerate(zip(frames, steps))
    ]
    return proj.caption(np.concatenate(frames, axis=1), title)
