"""The MuJoCo viewer on a running LIBERO simulator, so an episode is watched as it is simulated.

The viewer is passive: the caller steps the simulator (env.step) and calls `show` after each step; the
window shows the simulator's own state (orbit, zoom, pause, and drag objects with the mouse as in any
MuJoCo viewer). Markers are drawn as small spheres, e.g. the contact point of a plan. On macOS the viewer
needs the script to run under `mjpython` (installed with the mujoco package).

LIBERO rebuilds its MuJoCo model at every reset, so the window is opened on the first `show`, after the
episode's reset; a new episode needs a new LiveView.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import numpy as np

CONTROL_HZ = 20  # LIBERO's control frequency: one env.step is 1/20 s of simulated time


@dataclass(frozen=True)
class Marker:
    position: np.ndarray  # (3,) world coordinates
    rgba: tuple[float, float, float, float] = (0.93, 0.49, 0.19, 0.8)  # orange
    radius: float = 0.012


def offscreen_without_windows() -> None:
    """On macOS, let robosuite render its cameras through CGL instead of a hidden GLFW window.

    robosuite 1.4 renders offscreen through GLFW on macOS, which must create a window on the main thread;
    under mjpython the main thread runs the viewer, so the first camera render aborts the process. MuJoCo's
    own CGL context needs no window. Call this before the simulator is built; elsewhere it does nothing.
    """
    if platform.system() != "Darwin":
        return
    import mujoco.cgl
    import robosuite.utils.binding_utils as binding

    class CglContext(mujoco.cgl.GLContext):
        def __init__(self, max_width: int, max_height: int, device_id: int = -1) -> None:
            super().__init__(max_width, max_height)

    binding.GLContext = CglContext


class LiveView:
    """One viewer window on one simulator; `realtime` 1.0 paces steps at 20 Hz, 0 runs as fast as it can."""

    def __init__(self, realtime: float = 1.0) -> None:
        if realtime < 0:
            raise ValueError(f"realtime must be >= 0, got {realtime}")
        self.realtime = realtime
        self.viewer = None
        self.last = 0.0

    def _open(self, env) -> None:
        import mujoco.viewer

        try:
            self.viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
        except RuntimeError as error:  # macOS without mjpython
            raise SystemExit(f"{error}\nRun it as `mjpython ...` (scripts/local/sim.sh does this).") from error

    @property
    def is_open(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def show(self, env, markers: tuple[Marker, ...] = ()) -> bool:
        """Draw the simulator's current state; False once the user has closed the window."""
        if self.viewer is None:
            self._open(env)
        if not self.viewer.is_running():
            return False
        self._draw_markers(markers)
        self.viewer.sync()
        if self.realtime > 0:
            pause = 1.0 / (CONTROL_HZ * self.realtime) - (time.perf_counter() - self.last)
            if pause > 0:
                time.sleep(pause)
        self.last = time.perf_counter()
        return True

    def _draw_markers(self, markers: tuple[Marker, ...]) -> None:
        import mujoco

        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            for marker in markers[:len(scene.geoms)]:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.array([marker.radius, 0.0, 0.0]),
                                    pos=np.asarray(marker.position, dtype=float), mat=np.eye(3).ravel(),
                                    rgba=np.asarray(marker.rgba, dtype=np.float32))
                scene.ngeom += 1

    def hold(self, markers: tuple[Marker, ...] = ()) -> None:
        """Keep the window on the final state (simulation paused) until the user closes it."""
        if self.viewer is None:
            return
        print("(final state shown in the MuJoCo window; close the window to finish)", flush=True)
        while self.viewer.is_running():
            self._draw_markers(markers)
            self.viewer.sync()
            time.sleep(0.05)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
