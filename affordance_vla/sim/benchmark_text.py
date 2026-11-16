"""The instruction of a perturbed benchmark task, in the wording the policy was trained on.

LIBERO derives a task's instruction from its file name. LIBERO-PRO's task perturbation keeps the file
name and changes the `(:language ...)` line inside the BDDL file, so the changed instruction has to be
read from there. The file's wording differs from the trained one ("Pick the ..." against "pick up
the ..."), so only the words that changed between the original and the perturbed file are carried over.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

LANGUAGE_LINE = re.compile(r"\(:language\s+([^)]*)\)")


def bddl_language(text: str) -> str:
    """The instruction written in a BDDL task file."""
    found = LANGUAGE_LINE.search(text)
    if not found:
        raise ValueError("the task file has no (:language ...) line")
    return " ".join(found.group(1).split())


def _place(position: list[int | None], first: int, last: int) -> tuple[int, int] | None:
    """Where the original words first..last sit in the trained instruction, as (start, end) word indices.

    Replaced or removed words must all be present side by side. Added words (first == last) go right
    after the word before them, or right before the word after them. None if that cannot be told.
    """
    if first < last:
        places = position[first:last]
        if None in places or places != list(range(places[0], places[0] + len(places))):
            return None
        return places[0], places[-1] + 1
    if first > 0 and position[first - 1] is not None:
        return position[first - 1] + 1, position[first - 1] + 1
    if first < len(position) and position[first] is not None:
        return position[first], position[first]
    return None


def perturbed_instruction(trained: str, original: str, perturbed: str) -> str:
    """`trained` with the word changes from `original` to `perturbed` applied.

    `original` and `perturbed` are the languages of the task file before and after the perturbation.
    Whole words are compared, ignoring case, and every change is placed where the original words stand
    in `trained`. If a change cannot be placed, the perturbed language is returned lower-cased.
    """
    words = trained.split()
    before, after = original.lower().split(), perturbed.lower().split()
    position: list[int | None] = [None] * len(before)  # index in `words` of each word of `original`
    aligned = SequenceMatcher(a=before, b=[word.lower() for word in words], autojunk=False)
    for block in aligned.get_matching_blocks():
        for k in range(block.size):
            position[block.a + k] = block.b + k
    edits = []
    for tag, a0, a1, b0, b1 in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        place = _place(position, a0, a1)
        if place is None:
            return " ".join(after)
        edits.append((place[0], place[1], after[b0:b1]))
    for start, end, new in sorted(edits, reverse=True):  # from the right, so earlier indices stay valid
        words[start:end] = new
    return " ".join(words)
