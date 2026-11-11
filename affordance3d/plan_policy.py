"""Condition starVLA's flow-matching action head on a plan.

The released QwenPI LIBERO checkpoint runs its DiT without self-attention: every action token only
cross-attends to the VLM's hidden states, so tokens placed beside it are never read. The plan
therefore enters by addition: its embedding is added to each action token before the first block.
The embedding starts at zero, which leaves the released policy unchanged until training moves it.
For the same reason the head's 32 learned tokens cannot influence the action tokens and are left out.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from affordance3d.lora import add_lora
from affordance3d.plan_input import PLAN_DIM


def dit_lora_match(name: str) -> bool:
    """The linear layers of the action DiT that get a low-rank update: attention and feed-forward of every block."""
    return name.startswith("model.transformer_blocks.") and (".attn1.to_" in name or ".ff.net." in name)


class PlanEncoder(nn.Module):
    """Plan features (PLAN_DIM,) to one vector of the head's width; the output layer starts at zero."""

    def __init__(self, width: int, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PLAN_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def head_velocity(head: nn.Module, noisy: torch.Tensor, timesteps: torch.Tensor, states: list[torch.Tensor],
                  mask: torch.Tensor | None, plan_embedding: torch.Tensor) -> torch.Tensor:
    """Predicted flow velocity (B, H, D) for the noisy chunk, with the plan added to every action token.

    `timesteps` are the (B,) integer time buckets, `states` the layer-wise VLM hidden states, `mask`
    their attention mask, `plan_embedding` (B, width).
    """
    if not hasattr(head.model, "_interleave_self_attention"):
        raise ValueError("cannot tell whether this head's tokens attend to each other "
                         "(its DiT has no _interleave_self_attention flag); the plan input needs to know")
    if head.model._interleave_self_attention:
        raise ValueError("this head uses self-attention between its tokens, so its learned tokens cannot be "
                         "left out; the plan input was written for the all-cross-attention head")
    features = head.action_encoder(noisy, timesteps)
    if head.config.add_pos_embed:
        positions = torch.arange(features.shape[1], device=features.device)
        features = features + head.position_embedding(positions).unsqueeze(0)
    features = features + plan_embedding.to(features.dtype).unsqueeze(1)
    output = head.model(hidden_states=features, encoder_hidden_states=states, timestep=timesteps,
                        encoder_attention_mask=mask, return_pre_output=True)
    return head.action_decoder(output)


def sample_actions(head: nn.Module, states: list[torch.Tensor], mask: torch.Tensor | None,
                   plan_embedding: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Euler integration from `noise` (B, H, D) to an action chunk, as in the head's own sampler."""
    actions, steps = noise, head.num_inference_timesteps
    for k in range(steps):
        bucket = int(k / float(steps) * head.num_timestep_buckets)
        timesteps = torch.full((actions.shape[0],), bucket, device=actions.device, dtype=torch.long)
        actions = actions + head_velocity(head, actions, timesteps, states, mask, plan_embedding) / steps
    return actions


class PlanConditionedHead(nn.Module):
    """A frozen action head with what is trained on top of it: the plan encoder, low-rank updates of
    the layers chosen by `lora_match`, and the head's action decoder (kept in float32)."""

    def __init__(self, head: nn.Module, lora_rank: int, lora_alpha: float, lora_match: Callable[[str], bool],
                 hidden: int = 512) -> None:
        super().__init__()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self.lora_layers = add_lora(head, lora_rank, lora_alpha, lora_match)
        head.action_decoder.float()
        for parameter in head.action_decoder.parameters():
            parameter.requires_grad_(True)
        self.head = head
        self.plan_encoder = PlanEncoder(head.input_embedding_dim, hidden)
        self.plan_encoder.to(next(head.action_decoder.parameters()).device)

    def velocity(self, noisy: torch.Tensor, timesteps: torch.Tensor, states: list[torch.Tensor],
                 mask: torch.Tensor | None, plan: torch.Tensor) -> torch.Tensor:
        return head_velocity(self.head, noisy, timesteps, states, mask, self.plan_encoder(plan))

    def sample(self, states: list[torch.Tensor], mask: torch.Tensor | None, plan: torch.Tensor,
               noise: torch.Tensor) -> torch.Tensor:
        return sample_actions(self.head, states, mask, self.plan_encoder(plan), noise)

    def _trained_names(self) -> set[str]:
        return {name for name, parameter in self.named_parameters() if parameter.requires_grad}

    def adapter_state(self) -> dict[str, torch.Tensor]:
        """The trained parameters only (a few tens of megabytes instead of the whole head)."""
        names = self._trained_names()
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items() if name in names}

    def load_adapter(self, state: dict[str, torch.Tensor]) -> None:
        expected = self._trained_names()
        if set(state) != expected:
            raise ValueError(f"adapter state does not fit this head: missing {sorted(expected - set(state))[:5]}, "
                             f"unexpected {sorted(set(state) - expected)[:5]}")
        self.load_state_dict(state, strict=False)
