"""One file for the tracks of many replayed demonstrations.

The images of a training set live in separate arrays with one row per recorded state; `frame_offsets`
gives the first row of each demonstration, so state `t` of demonstration `d` is row `offsets[d] + t`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from affordance_vla.training.plan_samples import DemoTrack, track_from_arrays

IMAGE_SIZE = 224  # what starVLA's LIBERO client feeds the policy (bilinear resize of the 256 x 256 frame)


def resized(frames: np.ndarray) -> np.ndarray:
    """Camera frames (N, H, W, 3) at the size the policy is given, as starVLA's LIBERO client resizes them."""
    from PIL import Image

    return np.stack([np.asarray(Image.fromarray(frame).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
                     for frame in frames])


def frame_offsets(tracks: Sequence[DemoTrack]) -> np.ndarray:
    """(D + 1,) first state row of each demonstration when all are stacked; the last entry is the total."""
    return np.concatenate([[0], np.cumsum([len(track.ee_pos) for track in tracks])]).astype(np.int64)


def save_tracks(path: Path, tracks: Sequence[DemoTrack]) -> None:
    """Write the tracks as stacked arrays. Objects are padded to the largest count with NaN positions."""
    if not tracks:
        raise ValueError("no tracks to save")
    offsets = frame_offsets(tracks)
    width = max(len(track.names) for track in tracks)
    object_pos = np.full((offsets[-1], width, 3), np.nan)
    actions = np.zeros((offsets[-1], 7))  # the row of each demonstration's final state stays zero
    for track, first in zip(tracks, offsets):
        object_pos[first:first + len(track.ee_pos), :len(track.names)] = track.object_pos
        actions[first:first + len(track.actions)] = track.actions
    meta = [{"names": list(track.names), "target": track.target, "instruction": track.instruction}
            for track in tracks]
    np.savez_compressed(path, ee_pos=np.concatenate([track.ee_pos for track in tracks]), actions=actions,
                        grasped=np.concatenate([track.grasped for track in tracks]), object_pos=object_pos,
                        demo_start=offsets, meta=json.dumps(meta))


def load_tracks(path: Path) -> list[DemoTrack]:
    """Read tracks written by save_tracks; the labels are derived again from the contact flags."""
    with np.load(path) as data:
        meta, starts = json.loads(str(data["meta"])), data["demo_start"]
        if len(meta) + 1 != len(starts):
            raise ValueError(f"{path}: {len(meta)} demonstrations but {len(starts)} offsets")
        tracks = []
        for info, first, last in zip(meta, starts[:-1], starts[1:]):
            tracks.append(track_from_arrays(
                ee_pos=data["ee_pos"][first:last], actions=data["actions"][first:last - 1],
                grasped=data["grasped"][first:last], object_pos=data["object_pos"][first:last, :len(info["names"])],
                names=info["names"], target=info["target"], instruction=info["instruction"]))
    return tracks
