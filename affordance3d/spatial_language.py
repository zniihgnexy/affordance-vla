"""Where the objects of a LIBERO-Spatial scene start, in the words its instructions use.

In LIBERO-Spatial every scene has two identical black bowls, and the instruction says which one by where it
stands ("the black bowl on the stove"). To pair a frame with an instruction that names the other bowl, the
other bowl's place is read from the task file's initial state and put into the instruction instead of the
target's. Only plain text processing: the simulator environment can import this.
"""
from __future__ import annotations

import re

# initial region (or supporting object) of a bowl -> the words the instructions of LIBERO-Spatial use for it
REGION_PHRASES = {
    "main_table_between_plate_ramekin_region": "between the plate and the ramekin",
    "main_table_table_center": "from table center",
    "wooden_cabinet_1_top_region": "in the top drawer of the wooden cabinet",
    "main_table_next_to_box_region": "next to the cookie box",
    "main_table_next_to_plate_region": "next to the plate",
    "main_table_next_to_ramekin_region": "next to the ramekin",
    "cookies_1": "on the cookie box",
    "glazed_rim_porcelain_ramekin_1": "on the ramekin",
    "flat_stove_1_cook_region": "on the stove",
    "wooden_cabinet_1_top_side": "on the wooden cabinet",
}
_PLACED = re.compile(r"\((?:On|In)\s+(\w+)\s+(\w+)\)")


def initial_regions(bddl: str) -> dict[str, str]:
    """Object -> the region or object it is placed on or in, from the (:init ...) block of a task file."""
    start = bddl.find("(:init")
    end = bddl.find("(:goal", start)
    if start < 0:
        raise ValueError("the task file has no (:init ...) block")
    return dict(_PLACED.findall(bddl[start:end if end > 0 else None]))


def location_phrases(bddl: str) -> dict[str, str]:
    """Object -> the words for its place, for the objects whose initial region the instructions describe."""
    return {name: REGION_PHRASES[region] for name, region in initial_regions(bddl).items() if region in REGION_PHRASES}


def instruction_naming(instruction: str, phrases: dict[str, str], target: str, other: str) -> str | None:
    """The instruction with the target's place replaced by the other object's place; None if either has no
    phrase, the target's phrase is not in the instruction, or both stand at a place with the same words."""
    if target not in phrases or other not in phrases or phrases[target] not in instruction:
        return None
    renamed = instruction.replace(phrases[target], phrases[other], 1)
    return renamed if renamed != instruction else None


def object_words(name: str) -> str:
    """'salad_dressing_1' -> 'salad dressing'."""
    return name.rsplit("_", 1)[0].replace("_", " ")


def named_instruction(instruction: str, target: str, other: str,
                      phrases: dict[str, str] | None = None) -> str | None:
    """The instruction made to ask for `other` instead of `target`: by its place where `phrases` describe both
    (LIBERO-Spatial), otherwise by its name (LIBERO-Object). None if no different instruction results, e.g.
    for two identical objects without phrases."""
    by_place = instruction_naming(instruction, phrases or {}, target, other)
    if by_place is not None:
        return by_place
    renamed = instruction.replace(object_words(target), object_words(other))
    return renamed if renamed != instruction else None
