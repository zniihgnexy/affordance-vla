"""Low-rank adapters for frozen linear layers.

Written out here (about forty lines) because the pinned starVLA environment has no adapter library
and the frozen weights are bf16 while the trained update must stay in float32.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update: base(x) + (alpha / rank) · up(down(x))."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # the wrapped layer starts out equal to the base layer
        self.scale = alpha / rank
        self.down.to(base.weight.device)
        self.up.to(base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.up(self.down(x.to(self.down.weight.dtype)))
        return out + self.scale * update.to(out.dtype)


def add_lora(module: nn.Module, rank: int, alpha: float, match: Callable[[str], bool]) -> list[str]:
    """Wrap every linear layer of `module` whose dotted name satisfies `match`; returns those names."""
    wrapped = [name for name, child in module.named_modules() if isinstance(child, LoRALinear)]
    names = [name for name, child in module.named_modules() if isinstance(child, nn.Linear) and match(name)
             and not any(name.startswith(f"{outer}.") for outer in wrapped)]  # never wrap a wrapper's parts
    if not names:
        raise ValueError("the pattern matches no linear layer of the module")
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha))
    return names
