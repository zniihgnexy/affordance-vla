"""One HOT3D-Clips tar (webdataset: 150 frames of images, camera, hand and object poses; documented in the
HOT3D toolkit's hot3d/clips/README.md) and the object meshes of the dataset's GLB files.

The hand is the UmeTrack hand annotation, whose mesh needs no MANO licence: its skin is in the clip's
`__hand_shapes.json__`. Poses are 4x4 matrices `T_world_from_x`; lengths are metres.
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from hand_tracking_toolkit import camera as htk_camera
from hand_tracking_toolkit.dataset import HandSide, decode_hand_pose
from hand_tracking_toolkit.hand_models import umetrack_hand_model as umetrack
from hand_tracking_toolkit.math_utils import quat_trans_to_matrix

RGB_STREAM = "214-1"  # Aria's RGB camera, 1408 x 1408 fisheye


@dataclass(frozen=True)
class HandMesh:
    side: str  # "left" or "right"
    vertices: np.ndarray  # (V, 3) world frame
    faces: np.ndarray  # (F, 3)
    wrist: np.ndarray  # (3,) world position of the wrist


def pose_matrix(pose: dict) -> np.ndarray:
    return quat_trans_to_matrix(*pose["quaternion_wxyz"], *pose["translation_xyz"])


class Clip:
    """Random access to the frames of one clip tar."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._tar = tarfile.open(self.path)
        names = set(self._tar.getnames())
        self.frames = sorted({int(n.split(".")[0]) for n in names if n[0].isdigit()})
        self.hand_model = umetrack.from_json(self._read_json("__hand_shapes.json__")["umetrack"])
        info = self.json(self.frames[0], "info")
        self.sequence, self.participant, self.device = info["sequence_id"], info["participant_id"], info["device"]

    def _read_json(self, name: str):
        return json.load(self._tar.extractfile(name))

    def json(self, frame: int, kind: str):
        return self._read_json(f"{frame:06d}.{kind}.json")

    def timestamp_s(self, frame: int) -> float:
        return self.json(frame, "info")["ref_timestamp_ns"] * 1e-9

    def image(self, frame: int, stream: str = RGB_STREAM) -> np.ndarray:
        """RGB (or grey, for the monochrome streams) uint8 image."""
        raw = np.frombuffer(self._tar.extractfile(f"{frame:06d}.image_{stream}.jpg").read(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def camera(self, frame: int, stream: str = RGB_STREAM):
        return htk_camera.from_json(self.json(frame, "cameras")[stream])

    def object_ids(self, frame: int) -> list[str]:
        return sorted(self.json(frame, "objects"), key=int)

    def object_pose(self, frame: int, object_id: str) -> np.ndarray:
        """T_world_from_object; the object's frame is the one of the GLB model."""
        return pose_matrix(self.json(frame, "objects")[object_id][0]["T_world_from_object"])

    def hands(self, frame: int) -> list[HandMesh]:
        """The UmeTrack hand meshes of one frame (a hand missing from the annotation is left out)."""
        poses = decode_hand_pose(self.json(frame, "hands"))
        meshes = []
        for side, collection in poses.items():
            if collection.umetrack is None:
                continue
            _, vertices, faces = umetrack.forward_kinematics(collection.umetrack, self.hand_model, requires_mesh=True)
            wrist = collection.umetrack.wrist_xform[:3, 3].numpy()
            meshes.append(HandMesh(side.value, vertices.numpy().astype(float), faces.numpy().astype(int), wrist))
        return meshes


def load_object_mesh(models_dir: str | Path, object_id: str | int) -> trimesh.Trimesh:
    """The scanned mesh of a HOT3D object (obj_<id:06d>.glb, metres) in the frame `T_world_from_object` refers to."""
    return trimesh.load(Path(models_dir) / f"obj_{int(object_id):06d}.glb", force="mesh")


def object_name(models_dir: str | Path, object_id: str | int) -> str:
    info = json.loads((Path(models_dir) / "models_info.json").read_text())
    return info[str(int(object_id))]["name"]


__all__ = ["Clip", "HandMesh", "HandSide", "load_object_mesh", "object_name", "pose_matrix"]
