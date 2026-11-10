"""Read LIBERO demonstration files (robomimic-style HDF5) and their BDDL scene descriptions.

Each task file holds ~50 demos under data/demo_<i>: actions (T, 7) as OSC_POSE end-effector deltas
plus a gripper command (-1 open, +1 close), flattened MuJoCo states for replay, and observations.
Files written with `macros_image_convention = "opengl"` store images bottom row first, so they are
flipped vertically here (a 180° rotation, as some VLA pipelines use, would also mirror left/right).
The BDDL scene text is not stored in the file, only its name (`bddl_file_name`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DEMO_NAME = re.compile(r"demo_(\d+)")
_COMMENT = re.compile(r";[^\n]*")
REQUIRED_OBS = ("ee_pos", "gripper_states", "agentview_rgb", "eye_in_hand_rgb")


@dataclass(frozen=True)
class BddlScene:
    language: str
    fixtures: dict[str, str]
    objects: dict[str, str]
    of_interest: tuple[str, ...]
    goal: str

    @property
    def distractors(self) -> tuple[str, ...]:
        return tuple(name for name in self.objects if name not in self.of_interest)


def _block(text: str, name: str) -> str | None:
    """Body of the first `(:name ...)` block (exact name), matched by parenthesis depth."""
    match = re.search(rf"\(:{re.escape(name)}(?=[\s)])", text)
    if match is None:
        return None
    depth = 0
    for i in range(match.start(), len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[match.end():i].strip()
    raise ValueError(f"unbalanced parentheses in the :{name} block")


def _typed_names(body: str | None) -> dict[str, str]:
    """`a - t` and grouped `a b - t` declarations, one per line, as {name: type}."""
    names: dict[str, str] = {}
    for line in (body or "").splitlines():
        if " - " in line:
            left, kind = line.split(" - ", 1)
            names.update({name: kind.strip() for name in left.split()})
    return names


def parse_bddl(text: str) -> BddlScene:
    text = _COMMENT.sub("", text)
    objects = _block(text, "objects")
    if objects is None:
        raise ValueError("BDDL text has no (:objects ...) block")
    return BddlScene(language=" ".join((_block(text, "language") or "").split()),
                     fixtures=_typed_names(_block(text, "fixtures")), objects=_typed_names(objects),
                     of_interest=tuple((_block(text, "obj_of_interest") or "").split()),
                     goal=" ".join((_block(text, "goal") or "").split()))


@dataclass(frozen=True)
class LiberoDemo:
    name: str
    actions: np.ndarray  # (T, 7): Δxyz, Δaxis-angle, gripper command
    ee_pos: np.ndarray  # (T, 3) end-effector position
    gripper: np.ndarray  # (T, 2) finger joint positions
    agentview: np.ndarray | None  # (T, H, W, 3) uint8, upright; None when read with images=False
    wrist: np.ndarray | None  # (T, H, W, 3) uint8, upright; None when read with images=False
    has_replay_state: bool  # flattened MuJoCo states + model XML stored, so the demo can be re-rendered
    extra_obs: tuple[str, ...] = field(default=())


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("reading LIBERO files needs h5py (pip install h5py)") from error
    return h5py


def _text(value) -> str:
    """HDF5 string attributes may come back as str, bytes or numpy bytes."""
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def list_demos(path: Path | str) -> list[str]:
    """Demo group names in numeric order (demo_2 before demo_10); other keys are ignored."""
    with _h5py().File(path, "r") as f:
        names = [name for name in f["data"].keys() if _DEMO_NAME.fullmatch(name)]
    return sorted(names, key=lambda name: int(_DEMO_NAME.fullmatch(name).group(1)))


def read_task_info(path: Path | str) -> tuple[dict, str]:
    """(problem_info, BDDL file name) stored as attributes of the file's `data` group."""
    with _h5py().File(path, "r") as f:
        attrs = f["data"].attrs
        return json.loads(_text(attrs.get("problem_info", "{}"))), _text(attrs.get("bddl_file_name", ""))


def _flip_from_attrs(attrs, path: Path) -> bool:
    if "macros_image_convention" not in attrs:
        raise ValueError(f"{path.name} has no macros_image_convention attribute; pass flip=True/False explicitly")
    return _text(attrs["macros_image_convention"]) == "opengl"


def read_demo(path: Path | str, demo: str, images: bool = True, flip: bool | None = None) -> LiberoDemo:
    """One demo's arrays. `flip=None` reads the image convention from the file and fails if it is absent."""
    path = Path(path)
    with _h5py().File(path, "r") as f:
        if demo not in f["data"]:
            raise KeyError(f"{demo} not in {path.name}")
        flip = _flip_from_attrs(f["data"].attrs, path) if flip is None else flip
        group = f["data"][demo]
        obs = group["obs"]
        missing = [key for key in ("actions",) if key not in group] + [key for key in REQUIRED_OBS if key not in obs]
        if missing:
            raise KeyError(f"{path.name}/{demo} is missing {', '.join(missing)}")
        lengths = {"actions": group["actions"].shape[0], **{key: obs[key].shape[0] for key in REQUIRED_OBS}}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path.name}/{demo}: arrays have different numbers of steps {lengths}")

        def upright(key: str) -> np.ndarray | None:  # h5py cannot slice with negative steps: read, then flip
            if not images:
                return None
            frames = obs[key][:]
            return np.ascontiguousarray(frames[:, ::-1]) if flip else frames

        return LiberoDemo(name=demo, actions=group["actions"][:], ee_pos=obs["ee_pos"][:],
                          gripper=obs["gripper_states"][:], agentview=upright("agentview_rgb"),
                          wrist=upright("eye_in_hand_rgb"),
                          has_replay_state="states" in group and "model_file" in group.attrs,
                          extra_obs=tuple(sorted(set(obs.keys()) - set(REQUIRED_OBS))))
