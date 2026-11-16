"""The grid plan predictor: which image cell shows the contact, where in the cell, and how deep.

The VLM turns the agent view into CELLS x CELLS image tokens. Instead of regressing the contact's
coordinates from the whole token sequence (plan_predictor.py, which memorises where objects usually
stand), this head scores every cell: "is the contact of the next gripper event seen here?" It adds the
position inside the cell and the depth along the viewing ray. The 3D contact follows from the camera
calibration; no depth image is needed. The cell scores are an affordance map at the VLM's own resolution.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from affordance_vla.geometry import transform_points
from affordance_vla.plan.plan_input import PLAN_DIM, no_plan_features, plan_features
from affordance_vla.sim.libero_camera import project_points
from affordance_vla.training.plan_targets import GRASP, NONE

CELLS = 8  # image tokens per side of one view
IMAGE_SIZE = 224  # side of the image the VLM is given, in pixels
CELL_PX = IMAGE_SIZE / CELLS
DEPTH_REFERENCE_M, DEPTH_SCALE_M = 1.0, 0.10  # depth is predicted as (depth - 1 m) in units of 10 cm
MIN_DEPTH_M, MAX_DEPTH_M = 0.2, 3.0  # a decoded contact is kept in front of the camera, inside the scene
NO_CELL = -1  # cell target of an event whose contact no cell shows (it lies outside the image)
CENTRE_TOLERANCE_PX = 1.0
OFFSET_DELTA, DEPTH_DELTA = 0.1, 0.2  # Huber thresholds: a tenth of a cell (about 1 cm), and 2 cm
OFFSET_WEIGHT, DEPTH_WEIGHT, DIRECTION_WEIGHT = 10.0, 10.0, 0.5


@dataclass(frozen=True)
class Camera:
    """Calibration of the agent view as logged (the policy sees the logged image resized to IMAGE_SIZE)."""

    intrinsics: np.ndarray  # (3, 3) for the logged image
    cam_to_world: np.ndarray  # (4, 4)
    logged_size: int = 256  # side of the logged image in pixels


def camera_of_log(log) -> Camera:
    """Calibration of the agent view of a rollout log (affordance_vla/sim/rollout_log.py) read with its images.

    The size of the logged images is part of the calibration, so it is read from them. Mirroring and the
    scaling to IMAGE_SIZE assume square images with the principal point at their centre.
    """
    if log.agentview is None:
        raise ValueError("the camera needs a log read with its images: their size is part of the calibration")
    height, width = log.agentview.shape[1:3]
    if height != width:
        raise ValueError(f"the agent view must be square, got {width} x {height}")
    centre = float(log.intrinsics[0, 2])
    if abs(centre - width / 2) > CENTRE_TOLERANCE_PX:
        raise ValueError(f"the principal point ({centre:.1f}) is not at the centre of a {width} px image")
    return Camera(np.asarray(log.intrinsics, dtype=float), np.asarray(log.cam_to_world, dtype=float),
                  logged_size=int(width))


def contact_pixel(contact: np.ndarray, camera: Camera) -> tuple[float, float, float]:
    """(u, v) of a world point in the IMAGE_SIZE image the policy sees, and its depth along the optical axis."""
    pixels, depth = project_points(np.asarray(contact, dtype=float)[None], camera.intrinsics, camera.cam_to_world,
                                   width=camera.logged_size, mirrored=True)
    scale = IMAGE_SIZE / camera.logged_size
    return float(pixels[0, 0] * scale), float(pixels[0, 1] * scale), float(depth[0])


def pixel_contact(u: float, v: float, depth: float, camera: Camera) -> np.ndarray:
    """The world point seen at pixel (u, v) of the IMAGE_SIZE image at the given depth; inverse of contact_pixel."""
    scale = camera.logged_size / IMAGE_SIZE
    column = (camera.logged_size - 1) - u * scale  # the policy's image is mirrored left to right
    ray = np.linalg.inv(np.asarray(camera.intrinsics, dtype=float)) @ np.array([column, v * scale, 1.0])
    return transform_points(np.asarray(camera.cam_to_world, dtype=float), (ray * depth)[None])[0]


def grid_target(u: float, v: float) -> tuple[int, np.ndarray] | None:
    """Cell index (row-major) of a pixel and its position inside the cell in [0, 1); None outside the image."""
    if not (0.0 <= u < IMAGE_SIZE and 0.0 <= v < IMAGE_SIZE):
        return None
    column, row = int(u // CELL_PX), int(v // CELL_PX)
    return row * CELLS + column, np.array([u / CELL_PX - column, v / CELL_PX - row])


def shifted_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The image moved by (dx, dy) pixels (right and down are positive); uncovered pixels are zero."""
    height, width = image.shape[:2]
    moved = np.zeros_like(image)
    rows = slice(max(dy, 0), height + min(dy, 0))
    columns = slice(max(dx, 0), width + min(dx, 0))
    source_rows = slice(max(-dy, 0), height + min(-dy, 0))
    source_columns = slice(max(-dx, 0), width + min(-dx, 0))
    moved[rows, columns] = image[source_rows, source_columns]
    return moved


def image_token_cells(input_ids: torch.Tensor, image_token_id: int, views: int = 2) -> torch.Tensor:
    """(B, CELLS²) positions of the first view's image tokens in each token sequence, row by row.

    `input_ids` (B, T) is the VLM's input; every view contributes CELLS² image tokens, the agent view first.
    """
    cells = []
    for ids in input_ids:
        positions = torch.nonzero(ids == image_token_id).squeeze(1)
        if len(positions) != views * CELLS * CELLS:
            raise ValueError(f"expected {views * CELLS * CELLS} image tokens for {views} views, "
                             f"found {len(positions)}")
        cells.append(positions[:CELLS * CELLS])
    return torch.stack(cells)


class PlanGridPredictor(nn.Module):
    """Per-cell contact scores, in-cell offsets and depths, plus the event class and approach direction."""

    def __init__(self, width: int, layers: int, hidden: int = 512, heads: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(layers))  # softmax weights over the VLM's layers
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden))
        self.summary = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)  # a token that reads the whole sequence
        block = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4 * hidden, dropout=0.0,
                                           batch_first=True, norm_first=True)
        # the VLM is causal and puts the images before the instruction; here the cells can read the instruction
        self.encoder = nn.TransformerEncoder(block, depth, enable_nested_tensor=False)
        self.cell_out = nn.Linear(hidden, 4)
        self.global_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Linear(hidden, 6))

    def forward(self, states: Sequence[torch.Tensor], mask: torch.Tensor | None,
                cells: torch.Tensor) -> dict[str, torch.Tensor]:
        """`states`: one (B, T, width) tensor per VLM layer; `mask` (B, T) True for real tokens; `cells`
        (B, CELLS²) positions of the agent view's image tokens in the sequence, row by row.

        Returns `event` (B, 3) logits, `approach` (B, 3) unit vectors, and per cell: `cell` (B, CELLS²)
        logits, `offset` (B, CELLS², 2) in [0, 1] and `depth` (B, CELLS²) in units of 10 cm from 1 m.
        """
        if len(states) != len(self.mix):
            raise ValueError(f"expected hidden states of {len(self.mix)} layers, got {len(states)}")
        if cells.shape[1] != CELLS * CELLS:
            raise ValueError(f"expected the positions of {CELLS * CELLS} image tokens, got {cells.shape[1]}")
        weights = torch.softmax(self.mix, dim=0)
        tokens = self.project(sum(weight * layer.float() for weight, layer in zip(weights, states)))
        batch = tokens.shape[0]
        sequence = torch.cat([self.summary.expand(batch, -1, -1), tokens], dim=1)
        padding = None
        if mask is not None:
            padding = ~torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        read = self.encoder(sequence, src_key_padding_mask=padding)
        index = (cells + 1).unsqueeze(-1).expand(-1, -1, read.shape[-1])  # +1: the summary token comes first
        per_cell = self.cell_out(read.gather(1, index))
        overall = self.global_out(read[:, 0])
        return {"event": overall[:, :3], "approach": F.normalize(overall[:, 3:6], dim=-1),
                "cell": per_cell[..., 0], "offset": torch.sigmoid(per_cell[..., 1:3]), "depth": per_cell[..., 3]}


def grid_loss(output: dict[str, torch.Tensor], event: torch.Tensor, cell: torch.Tensor, offset: torch.Tensor,
              depth: torch.Tensor, approach: torch.Tensor) -> dict[str, torch.Tensor]:
    """Training loss and its parts. `cell` (B,) is the true cell or NO_CELL, `offset` (B, 2) the position
    inside it, `depth` (B,) in units of 10 cm from 1 m. The event counts everywhere, the direction where an
    event is ahead, and the cell, offset and depth where an event is ahead and a cell shows its contact."""
    ahead = event != NONE
    classes = F.cross_entropy(output["event"], event)
    zero = output["cell"].sum() * 0.0
    direction, where, inside, deep = zero, zero, zero, zero
    if ahead.any():
        direction = (1.0 - (output["approach"][ahead] * approach[ahead]).sum(dim=-1)).mean()
    rows = torch.nonzero(ahead & (cell != NO_CELL)).squeeze(1)
    if len(rows):
        where = F.cross_entropy(output["cell"][rows], cell[rows])
        inside = F.huber_loss(output["offset"][rows, cell[rows]], offset[rows], delta=OFFSET_DELTA)
        deep = F.huber_loss(output["depth"][rows, cell[rows]], depth[rows], delta=DEPTH_DELTA)
    total = classes + where + OFFSET_WEIGHT * inside + DEPTH_WEIGHT * deep + DIRECTION_WEIGHT * direction
    return {"total": total, "event": classes, "cell": where, "offset": inside, "depth": deep, "direction": direction}


def grid_contacts(output: dict[str, torch.Tensor], camera: Camera) -> np.ndarray:
    """(B, 3) world contacts: the best cell of each example with its offset and depth, through the camera.

    Cells are only trained where an event is ahead: the contact of an example whose event is NONE means nothing.
    """
    best = output["cell"].detach().argmax(dim=-1)
    rows = torch.arange(len(best), device=best.device)
    offsets = output["offset"].detach()[rows, best].float().cpu().numpy()
    depths = DEPTH_REFERENCE_M + DEPTH_SCALE_M * output["depth"].detach()[rows, best].float().cpu().numpy()
    depths = np.clip(depths, MIN_DEPTH_M, MAX_DEPTH_M)
    best = best.cpu().numpy()
    contacts = np.zeros((len(best), 3))
    for k, cell in enumerate(best):
        u = (cell % CELLS + offsets[k, 0]) * CELL_PX
        v = (cell // CELLS + offsets[k, 1]) * CELL_PX
        contacts[k] = pixel_contact(float(u), float(v), float(depths[k]), camera)
    return contacts


def grid_plan_inputs(output: dict[str, torch.Tensor], ee_pos: np.ndarray, camera: Camera) -> np.ndarray:
    """(B, PLAN_DIM) plan inputs of the action head from predictions and the gripper positions (B, 3)."""
    events = output["event"].detach().argmax(dim=-1).cpu().numpy()
    contacts = grid_contacts(output, camera)
    approaches = output["approach"].detach().float().cpu().numpy()
    ee_pos = np.asarray(ee_pos, dtype=float)
    if ee_pos.shape != (len(events), 3):
        raise ValueError(f"need one gripper position per example, got {ee_pos.shape} for {len(events)}")
    features = np.zeros((len(events), PLAN_DIM), dtype=np.float32)
    for k, event in enumerate(events):
        features[k] = (no_plan_features() if event == NONE
                       else plan_features(ee_pos[k], contacts[k], approaches[k], closing=bool(event == GRASP)))
    return features
