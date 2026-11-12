"""The plan predictor: a small readout of the VLM's hidden states.

It answers, from what the VLM made of the images and the instruction: what is the next gripper event
(grasp, release, none), where should the gripper be at that event, and from which direction should it
arrive. A few learned queries attend to the token sequence (a learned mix of the VLM's layers) and an
MLP turns them into the three outputs. The contact is regressed directly, in the robot's base frame;
no depth image or camera calibration is used.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from affordance3d.plan_input import PLAN_DIM, no_plan_features, plan_features
from affordance3d.plan_targets import GRASP, NONE

CONTACT_SCALE_M = 0.10  # the contact is predicted in units of 10 cm
HUBER_DELTA = 0.2  # in those units: errors beyond 2 cm count linearly
CONTACT_WEIGHT, DIRECTION_WEIGHT = 10.0, 0.5


class PlanPredictor(nn.Module):
    """Event class, contact point and approach direction from layer-wise VLM hidden states."""

    def __init__(self, width: int, layers: int, hidden: int = 512, heads: int = 8, depth: int = 2,
                 queries: int = 4) -> None:
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(layers))  # softmax weights over the VLM's layers
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden))
        self.queries = nn.Parameter(torch.randn(queries, hidden) * 0.02)
        block = nn.TransformerDecoderLayer(hidden, heads, dim_feedforward=4 * hidden, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.readout = nn.TransformerDecoder(block, depth)
        self.out = nn.Sequential(nn.LayerNorm(queries * hidden), nn.Linear(queries * hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 9))

    def forward(self, states: Sequence[torch.Tensor], mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """`states`: one (B, T, width) tensor per VLM layer; `mask` (B, T) is True for real tokens.

        Returns `event` (B, 3) logits for none / grasp / release, `contact` (B, 3) in units of 10 cm and
        `approach` (B, 3) unit vectors.
        """
        if len(states) != len(self.mix):
            raise ValueError(f"expected hidden states of {len(self.mix)} layers, got {len(states)}")
        weights = torch.softmax(self.mix, dim=0)
        tokens = sum(weight * layer.float() for weight, layer in zip(weights, states))
        memory = self.project(tokens)
        queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        read = self.readout(queries, memory, memory_key_padding_mask=None if mask is None else ~mask)
        raw = self.out(read.flatten(1))
        return {"event": raw[:, :3], "contact": raw[:, 3:6], "approach": F.normalize(raw[:, 6:9], dim=-1)}


def predictor_loss(output: dict[str, torch.Tensor], event: torch.Tensor, contact: torch.Tensor,
                   approach: torch.Tensor) -> dict[str, torch.Tensor]:
    """Training loss and its parts. `contact` (B, 3) is in metres, `event` (B,) holds the classes.

    Contact and direction count only where an event is ahead. `contact_cm` is the mean distance between
    predicted and true contact in centimetres (0 when no example has an event).
    """
    has = event != NONE
    classes = F.cross_entropy(output["event"], event)
    zero = output["contact"].sum() * 0.0
    if not has.any():
        return {"total": classes, "event": classes, "contact": zero, "contact_cm": zero, "direction": zero}
    predicted, wanted = output["contact"][has], contact[has] / CONTACT_SCALE_M
    huber = F.huber_loss(predicted, wanted, delta=HUBER_DELTA)
    distance = (predicted - wanted).norm(dim=-1).mean() * (100.0 * CONTACT_SCALE_M)
    direction = (1.0 - (output["approach"][has] * approach[has]).sum(dim=-1)).mean()
    total = classes + CONTACT_WEIGHT * huber + DIRECTION_WEIGHT * direction
    return {"total": total, "event": classes, "contact": huber, "contact_cm": distance.detach(),
            "direction": direction}


def plan_inputs(output: dict[str, torch.Tensor], ee_pos: np.ndarray) -> np.ndarray:
    """(B, PLAN_DIM) plan inputs of the action head from predictions and the gripper positions (B, 3)."""
    events = output["event"].detach().argmax(dim=-1).cpu().numpy()
    contacts = output["contact"].detach().float().cpu().numpy() * CONTACT_SCALE_M
    approaches = output["approach"].detach().float().cpu().numpy()
    ee_pos = np.asarray(ee_pos, dtype=float)
    if ee_pos.shape != (len(events), 3):
        raise ValueError(f"need one gripper position per example, got {ee_pos.shape} for {len(events)}")
    features = np.zeros((len(events), PLAN_DIM), dtype=np.float32)
    for k, event in enumerate(events):
        features[k] = (no_plan_features() if event == NONE
                       else plan_features(ee_pos[k], contacts[k], approaches[k], closing=bool(event == GRASP)))
    return features
