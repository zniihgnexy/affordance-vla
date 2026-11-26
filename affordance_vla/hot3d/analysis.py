"""Touch and opposition of the hands on one object through a clip, and the held grasps found in them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from affordance_vla.hot3d import contact as C
from affordance_vla.hot3d.clips import Clip
from affordance_vla.plan.interaction_labels import Grasp, line_angle_deg

MIN_HOLD_FRAMES = 5  # an opposition counts as a grasp when it lasts this many frames (1/6 s at 30 fps)
MAX_AXIS_DRIFT_DEG = 20.0  # a run ends when the closing axis (object frame) turns this far from the run's first


@dataclass
class ClipContact:
    clip: str
    object_id: str
    frames: list[int]
    touched: dict[float, np.ndarray]  # threshold (m) -> (T, N) bool, surface points touched by either hand
    oppositions: list[tuple[str, C.Opposition, np.ndarray] | None]  # per frame: hand, opposition, wrist (object frame)
    object_position: np.ndarray  # (T, 3) world

    def opposed(self) -> np.ndarray:
        return np.array([o is not None for o in self.oppositions])


def analyse_clip(clip: Clip, object_id: str, surface: np.ndarray, samples: C.HandSamples, triangles: np.ndarray,
                 thresholds: tuple[float, ...] = (C.THRESHOLD_M,), stride: int = 1) -> ClipContact:
    """Per frame: which surface points the hands touch (at each threshold), and the thumb-fingers opposition at
    the first threshold. `surface` is (N, 3) in the object's frame."""
    frames = clip.frames[::stride]
    touched = {t: np.zeros((len(frames), len(surface)), bool) for t in thresholds}
    oppositions, positions = [], []
    for k, frame in enumerate(frames):
        pose = clip.object_pose(frame, object_id)
        positions.append(pose[:3, 3])
        best = None
        for hand in clip.hands(frame):
            points = C.to_object_frame(samples.points(hand.vertices, triangles), pose)
            for t in thresholds:
                touched[t][k] |= C.touched_points(surface, points, t)
            near = C.touching_hand_points(surface, points, thresholds[0])
            pair = C.opposition(points, samples.finger, near)
            if pair is not None and (best is None or near.sum() > best[3]):
                best = (hand.side, pair, C.to_object_frame(hand.wrist[None], pose)[0], near.sum())
        oppositions.append(None if best is None else best[:3])
    return ClipContact(clip.path.name, object_id, frames, touched, oppositions, np.array(positions))


def steady_runs(contact: ClipContact, min_frames: int, max_drift_deg: float) -> list[tuple[int, int]]:
    """Runs (start, end exclusive, in frame indices) of opposed frames whose closing axis stays within
    `max_drift_deg` of the run's first axis, as a line; a run is cut where it turns further, and where the
    opposition is lost."""
    runs, start = [], None
    for k, item in enumerate(contact.oppositions):
        drifted = start is not None and item is not None and \
            line_angle_deg(item[1].closing, contact.oppositions[start][1].closing) > max_drift_deg
        if start is not None and (item is None or drifted):
            runs.append((start, k))
            start = None
        if item is not None and start is None:
            start = k
    if start is not None:
        runs.append((start, len(contact.oppositions)))
    return [(a, b) for a, b in runs if b - a >= min_frames]


def held_grasps(contact: ClipContact, min_frames: int = MIN_HOLD_FRAMES,
                max_drift_deg: float = MAX_AXIS_DRIFT_DEG) -> list[dict]:
    """One grasp per steady run of opposed frames: the run's middle frame as a `Grasp`, with the largest angle
    between the run's closing axes and the middle one's, and how far the object moved during the run."""
    found = []
    for start, end in steady_runs(contact, min_frames, max_drift_deg):
        middle = (start + end) // 2
        side, pair, wrist = contact.oppositions[middle]
        spread = max(line_angle_deg(contact.oppositions[i][1].closing, pair.closing) for i in range(start, end))
        moved = float(np.linalg.norm(contact.object_position[start:end] - contact.object_position[start], axis=1).max())
        found.append({"clip": contact.clip, "hand": side, "first_frame": contact.frames[start],
                      "last_frame": contact.frames[end - 1], "frame": contact.frames[middle],
                      "span_m": pair.span, "fingers": pair.fingers, "axis_spread_deg": spread, "moved_m": moved,
                      "grasp": C.as_grasp(pair, wrist)})
    return found


def grasps_of(found: list[dict]) -> list[Grasp]:
    return [item["grasp"] for item in found]
