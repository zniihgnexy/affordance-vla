"""Why the stove knob's contact position scattered 32.9 cm in the part body's frame (AFFO-18, question 1): the numbers
that show where the body's origin is, which knob the demonstrator turned, and what the scatter is with the joint
anchor as the origin. Only numpy is used.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from affordance_vla.plan.rc365_joint_labels import JointTouch
from affordance_vla.plan.rc365_parts import position_stats

BURNER = re.compile(r"the ([a-z ]+?) burner")


def burner_of(lang: str) -> str | None:
    """'front_left' from 'Turn off the front left burner of the stove.'"""
    found = BURNER.search(lang.lower())
    return found.group(1).replace(" ", "_") if found else None


def knob_slot(part: str) -> str:
    """'front_left' from 'stovetop_main_group_knob_front_left'."""
    return part.split("knob_", 1)[-1]


def stove_type(part: str) -> str:
    """'stove' or 'stovetop': the asset family in the body's name."""
    return part.split("_", 1)[0]


def _pair_distances(touches: Sequence[JointTouch], layouts: Sequence, get) -> list[float]:
    """Distance of `get(touch)` between two different knobs of the same stove (same layout, same body-name prefix)."""
    groups = defaultdict(dict)
    for t, layout in zip(touches, layouts):
        groups[(layout, t.part.split("knob_")[0])][t.part] = get(t)
    return [
        float(np.linalg.norm(a - b))
        for knobs in groups.values()
        for i, a in enumerate(knobs.values())
        for b in list(knobs.values())[i + 1 :]
    ]


def _median_cm(distances: Sequence[float]) -> float | None:
    return float(100 * np.median(distances)) if len(distances) else None


def _by_type(touches: Sequence[JointTouch], layouts: np.ndarray) -> dict:
    """Contact scatter for each asset family of stove, with either origin."""
    out = {}
    for kind in sorted({stove_type(t.part) for t in touches}):
        keep = [k for k, t in enumerate(touches) if stove_type(t.part) == kind]
        stack = lambda get: np.stack([get(touches[k]) for k in keep])  # noqa: E731
        out[kind] = {
            "touches": len(keep),
            "from_body_origin": position_stats(stack(JointTouch.contact_from_body), layouts[keep]),
            "from_joint_anchor": position_stats(stack(JointTouch.contact_from_anchor), layouts[keep]),
        }
    return out


def knob_diagnosis(touches: Sequence[JointTouch], layouts: Sequence, langs: Sequence[str]) -> dict:
    """The evidence about the stove-knob frame, from the touches of one task (TurnOffStove), one layout id and one
    instruction per touch."""
    layouts = np.asarray(layouts)
    per_layout = defaultdict(set)
    for t, layout in zip(touches, layouts):
        per_layout[layout].add(t.part)
    asked = [burner_of(lang) for lang in langs]
    own_hinge = [t.joint == f"{t.part}_joint" and t.joint_type == "hinge" for t in touches]
    return {
        "touches": len(touches),
        "kitchens": int(len(per_layout)),
        "distinct_knobs": int(sum(len(v) for v in per_layout.values())),
        "kitchens_with_more_than_one_knob": int(sum(len(v) > 1 for v in per_layout.values())),
        "knobs_per_kitchen": {str(k): v for k, v in sorted(Counter(len(v) for v in per_layout.values()).items())},
        "touched_knob_is_the_one_asked_for": float(np.mean([a == knob_slot(t.part) for a, t in zip(asked, touches)])),
        "instruction_without_burner_name": int(sum(a is None for a in asked)),
        "joint_is_the_knobs_own_hinge": float(np.mean(own_hinge)),
        "contact_to_body_origin_cm_median": _median_cm([np.linalg.norm(t.contact - t.part_pos) for t in touches]),
        "contact_to_joint_anchor_cm_median": _median_cm([np.linalg.norm(t.contact - t.anchor) for t in touches]),
        "body_origin_to_joint_anchor_cm_median": _median_cm([np.linalg.norm(t.anchor - t.part_pos) for t in touches]),
        "other_knob_of_same_stove_body_origin_cm_median": _median_cm(
            _pair_distances(touches, layouts, lambda t: t.part_pos)
        ),
        "other_knob_of_same_stove_joint_anchor_cm_median": _median_cm(
            _pair_distances(touches, layouts, lambda t: t.anchor)
        ),
        "by_stove_type": _by_type(touches, layouts),
    }
