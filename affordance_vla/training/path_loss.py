"""Plan–action consistency as a training loss: the path a chunk asks for against the path of the plan.

The tensor version of plan_consistency.chunk_path, plus a robust distance to target points. It is
differentiable with respect to the actions, so it can supervise a flow-matching head through its
one-step estimate of the clean chunk.
"""
from __future__ import annotations

import torch

QUADRATIC_ZONE_CM = 0.5  # errors below this are penalised quadratically, larger ones linearly
_METRES_TO_CM = 100.0
_MIN_SQUARED_M = 1e-16  # keeps the square root differentiable at zero distance


def chunk_path(start: torch.Tensor, actions_xyz: torch.Tensor, scale: float) -> torch.Tensor:
    """(B, H, 3) positions after each step: start + scale · Σ of the first k translation actions.

    `start` is (B, 3), `actions_xyz` (B, H, 3) in action units, `scale` metres per action unit.
    """
    return start[:, None, :] + scale * torch.cumsum(actions_xyz, dim=1)


def path_loss(path: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean Huber distance in centimetres between path and target points over the valid steps.

    `path` and `target` are (B, H, 3) in metres, `valid` (B, H) bool. Zero if no step is valid.
    """
    distance = _METRES_TO_CM * torch.sqrt(((path - target) ** 2).sum(dim=-1) + _MIN_SQUARED_M)
    huber = torch.where(distance < QUADRATIC_ZONE_CM, 0.5 * distance ** 2 / QUADRATIC_ZONE_CM,
                        distance - 0.5 * QUADRATIC_ZONE_CM)
    weights = valid.to(huber.dtype)
    return (huber * weights).sum() / weights.sum().clamp(min=1.0)
