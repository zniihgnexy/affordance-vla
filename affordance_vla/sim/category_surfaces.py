"""The surface of one object of each category, read from the simulator of LIBERO tasks."""
from __future__ import annotations

from typing import Iterable

from affordance_vla.analysis.grasp_orientation import object_category
from affordance_vla.sim.object_surface import Surface, body_surface


def sample_surfaces(suites: Iterable[str], categories: set[str], count: int, seed: int) -> dict[str, Surface]:
    """Surface points (object frame) of one object of each category, taken from the first task of the given suites,
    in order, that has an object of it. Categories that no task has are left out of the result."""
    from affordance_vla.sim.episode import make_env

    surfaces: dict[str, Surface] = {}
    for suite in suites:
        task, n_tasks = 0, 1
        while task < n_tasks and not categories <= set(surfaces):
            benchmark, _, env = make_env(suite, task, seed, cameras=False)
            n_tasks, inner = benchmark.n_tasks, env.env
            for name in sorted(inner.objects_dict):
                category = object_category(name)
                if category in categories and category not in surfaces:
                    surfaces[category] = body_surface(inner.sim.model, inner.obj_body_id[name], count, seed)
            env.close()
            task += 1
    return surfaces
