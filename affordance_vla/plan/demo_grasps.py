"""The held grasps of replayed demonstrations, in the frame of the object that is held.

Shared by the per-object maps (scripts/data/build_interaction_labels.py) and the category-level transfer check
(scripts/data/mug_transfer_check.py). A demonstration log must have been replayed with object orientations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from affordance_vla.analysis.grasp_orientation import object_category
from affordance_vla.plan.interaction_labels import Grasp, grasp_in_object, held_grasp_steps
from affordance_vla.sim.rollout_log import load_rollout_log


@dataclass(frozen=True)
class DemoGrasp:
    category: str
    grasp: Grasp  # in the held object's frame
    log: Path
    object_index: int  # column of the object in the log's per-object arrays
    step: int  # the step the grasp starts to register


@dataclass(frozen=True)
class DemoScan:
    grasps: list[DemoGrasp]  # in log order, then time order
    rest_quats: dict[str, np.ndarray]  # category -> (n, 4) orientations (x, y, z, w) of its objects at step 0


def scan_demos(paths: Iterable[Path], categories: set[str] | None = None) -> DemoScan:
    """Held grasps of every object (of the given categories, default all) in the logs, and the objects' orientations
    at step 0, when they stand at rest on the table."""
    grasps, rest = [], defaultdict(list)
    for path in paths:
        log = load_rollout_log(path, images=False)
        if log.object_quat is None:
            raise SystemExit(f"{path.name} has no object orientations: replay the demonstrations again")
        for k, name in enumerate(log.names):
            if categories is None or object_category(name) in categories:
                rest[object_category(name)].append(log.object_quat[0, k])
        for k, step in held_grasp_steps(log.grasped):
            category = object_category(log.names[k])
            if categories is None or category in categories:
                grasp = grasp_in_object(log.ee_pos[step], log.ee_quat[step], log.object_pos[step, k],
                                        log.object_quat[step, k])
                grasps.append(DemoGrasp(category, grasp, Path(path), k, step))
    return DemoScan(grasps, {category: np.stack(quats) for category, quats in rest.items()})
