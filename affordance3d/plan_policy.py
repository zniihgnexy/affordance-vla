"""Condition starVLA's flow-matching action head on a plan.

The released QwenPI LIBERO checkpoint runs its DiT without self-attention: every action token only
cross-attends to the VLM's hidden states, so tokens placed beside it are never read. The plan
therefore enters by addition: its embedding is added to each action token before the first block.
The embedding starts at zero, which leaves the released policy unchanged until training moves it.
For the same reason the head's 32 learned tokens cannot influence the action tokens and are left out.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

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


class DeepPlan(nn.Module):
    """The plan added to the input of every DiT block after the first, each through its own linear map.

    "Fast Plans, Faithful Actions" (arXiv 2609.30833) injects the plan into every layer of the action head,
    so that the head cannot bypass it; this is that injection for the all-cross-attention DiT. Block 0
    already receives the plan with the action tokens. The maps have no bias and start at zero, which
    leaves the head unchanged until training moves them. The plan embedding is handed over for one pass
    with `given`; outside it the blocks get their inputs unchanged. A pass that runs the blocks again
    later (gradient checkpointing) would therefore miss the plan: do not combine the two.
    """

    def __init__(self, blocks: nn.ModuleList, width: int) -> None:
        super().__init__()
        later = list(blocks)[1:]
        self.maps = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in later)
        self._embedding: torch.Tensor | None = None
        for layer, block in zip(self.maps, later):
            nn.init.zeros_(layer.weight)
            block.register_forward_pre_hook(self._adder(layer))

    def _adder(self, layer: nn.Linear) -> Callable[[nn.Module, tuple], tuple | None]:
        def add(_block: nn.Module, args: tuple) -> tuple | None:
            if self._embedding is None:
                return None  # the head is used without a plan
            if not args:
                raise RuntimeError("the DiT gave a block its hidden states by keyword; the deep plan can only "
                                   "be added to hidden states passed as the first positional argument")
            hidden = args[0]
            return (hidden + layer(self._embedding).to(hidden.dtype).unsqueeze(1), *args[1:])

        return add

    @contextmanager
    def given(self, embedding: torch.Tensor) -> Iterator[None]:
        """While open, every pass through the blocks adds this plan embedding (B, width)."""
        before = self._embedding
        self._embedding = embedding
        try:
            yield
        finally:
            self._embedding = before


class PlanConditionedHead(nn.Module):
    """A frozen action head with what is trained on top of it: the plan encoder, low-rank updates of
    the layers chosen by `lora_match`, and the head's action decoder (kept in float32). With `deep` the
    plan also enters every later block of the DiT (DeepPlan)."""

    def __init__(self, head: nn.Module, lora_rank: int, lora_alpha: float, lora_match: Callable[[str], bool],
                 hidden: int = 512, deep: bool = False) -> None:
        super().__init__()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self.lora_layers = add_lora(head, lora_rank, lora_alpha, lora_match)
        head.action_decoder.float()
        for parameter in head.action_decoder.parameters():
            parameter.requires_grad_(True)
        self.head = head
        device = next(head.action_decoder.parameters()).device
        self.plan_encoder = PlanEncoder(head.input_embedding_dim, hidden).to(device)
        self.deep_plan = (DeepPlan(head.model.transformer_blocks, head.input_embedding_dim).to(device)
                          if deep else None)

    def _in_blocks(self, embedding: torch.Tensor) -> AbstractContextManager[None]:
        """Context in which the DiT's later blocks receive the plan too, if this head is a deep one."""
        return nullcontext() if self.deep_plan is None else self.deep_plan.given(embedding)

    def velocity(self, noisy: torch.Tensor, timesteps: torch.Tensor, states: list[torch.Tensor],
                 mask: torch.Tensor | None, plan: torch.Tensor) -> torch.Tensor:
        embedding = self.plan_encoder(plan)
        with self._in_blocks(embedding):
            return head_velocity(self.head, noisy, timesteps, states, mask, embedding)

    def sample(self, states: list[torch.Tensor], mask: torch.Tensor | None, plan: torch.Tensor,
               noise: torch.Tensor) -> torch.Tensor:
        embedding = self.plan_encoder(plan)
        with self._in_blocks(embedding):
            return sample_actions(self.head, states, mask, embedding, noise)

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
