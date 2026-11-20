"""Does an 8 x 8 map over the agent view's image tokens point at the object the instruction names? (E1 of Pipeline v3)

Two kinds of map are scored the same way: the cosine similarity between instruction tokens and the cell tokens
of the frozen VLM (merged over the instruction's tokens by their mean or their maximum), and the cell scores
of a trained grid plan head. A map points at the object that covers most of its best cell in the logged
segmentation; a best cell on no object points at the background. Only numpy is used.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

CELLS = 8  # image tokens per side of the agent view (affordance_vla/heads/plan_grid.py)
MIN_COVER = 0.05  # share of a cell an object must cover for a map's best cell to count as on it
MERGES = ("mean", "max")


def object_phrase(name: str) -> str:
    """The words for a scene object: 'tomato_sauce_1' -> 'tomato sauce'."""
    return re.sub(r"_\d+$", "", name).replace("_", " ")


def cell_coverage(seg: np.ndarray, objects: int, cells: int = CELLS) -> np.ndarray:
    """(objects, cells²) share of each image cell, row by row, covered by object k (segmentation value k + 1)."""
    height, width = seg.shape
    if height % cells or width % cells:
        raise ValueError(f"a {width} x {height} segmentation does not split into {cells} x {cells} cells")
    blocks = seg.reshape(cells, height // cells, cells, width // cells).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(cells * cells, -1)
    return np.stack([(blocks == k + 1).mean(axis=1) for k in range(objects)])


def merge_similarity(text: np.ndarray, cells: np.ndarray, how: str) -> np.ndarray:
    """(cells,) cosine similarity of each cell token (cells, D) to the text tokens (n, D), merged over the n tokens."""
    if how not in MERGES:
        raise ValueError(f"merge must be one of {MERGES}, got {how!r}")
    if len(text) == 0:
        raise ValueError("no text tokens to compare with")
    unit_text = text / np.linalg.norm(text, axis=1, keepdims=True)
    unit_cells = cells / np.linalg.norm(cells, axis=1, keepdims=True)
    similarity = unit_text @ unit_cells.T
    return similarity.mean(axis=0) if how == "mean" else similarity.max(axis=0)


def object_at_peak(scores: np.ndarray, coverage: np.ndarray, min_cover: float = MIN_COVER) -> int | None:
    """Index of the object covering most of the best-scoring cell; None when no object covers min_cover of it."""
    best = int(np.argmax(scores))
    k = int(np.argmax(coverage[:, best]))
    return k if coverage[k, best] >= min_cover else None


def object_means(scores: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """(objects,) coverage-weighted mean of the cell scores over each object; -inf for objects not in view.

    This is what a multiplicative gate sees: the interaction map already limits the target to objects, so the
    question is whether the named object scores higher than the other objects, wherever the single best cell is.
    """
    weight = coverage.sum(axis=1)
    means = np.full(len(coverage), -np.inf)
    seen = weight > 0
    means[seen] = coverage[seen] @ scores / weight[seen]
    return means


def text_token_indices(pieces: Sequence[str], text: str) -> np.ndarray:
    """Indices of the tokens, given as their decoded text in sequence order, that overlap the first occurrence of
    `text` in their concatenation, case-insensitive; empty when it does not occur.

    Matching characters rather than token ids does not depend on how the surrounding prompt splits the words:
    after "Your task is" the instruction starts with the token " pick", on its own with "pick".
    """
    joined = "".join(pieces).lower()
    begin = joined.find(text.lower()) if text else -1
    if begin < 0:
        return np.zeros(0, dtype=int)
    end = begin + len(text)
    bounds = np.cumsum([0] + [len(piece) for piece in pieces])
    return np.flatnonzero((bounds[:-1] < end) & (bounds[1:] > begin))
