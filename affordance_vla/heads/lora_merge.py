"""Fold LoRA adapters into the base weights of a saved state dict, so a stock model class can load it."""
from __future__ import annotations

import torch

_DOWN, _UP, _BASE = ".down.weight", ".up.weight", ".base."


def merge_lora_state(state: dict[str, torch.Tensor], alpha: float, rank: int) -> dict[str, torch.Tensor]:
    """A state dict with each `LoRALinear` (`<name>.base.*`, `<name>.down/up.weight`) folded back into `<name>.*`.

    Matches `LoRALinear`: weight = base + (alpha / rank) · up @ down, computed in float32 and stored in the base
    dtype. Every other entry is copied unchanged. An adapter without its base weight (or the reverse) is an error.
    """
    scale = alpha / rank
    prefixes = {key[: -len(_DOWN)] for key in state if key.endswith(_DOWN)}
    if prefixes != {key[: -len(_UP)] for key in state if key.endswith(_UP)}:
        raise KeyError("some adapter has only one of its two matrices")
    merged: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        prefix = next((p for p in prefixes if key.startswith(p + ".")), None)
        if prefix is None:
            merged[key] = tensor
        elif key == f"{prefix}.base.weight":
            delta = scale * (state[prefix + _UP].float() @ state[prefix + _DOWN].float())
            merged[f"{prefix}.weight"] = (tensor.float() + delta).to(tensor.dtype)
        elif key.startswith(prefix + _BASE):
            merged[f"{prefix}.{key[len(prefix + _BASE):]}"] = tensor
    missing = [p for p in prefixes if f"{p}.weight" not in merged]
    if missing:
        raise KeyError(f"{missing[0]} has an adapter but no base weight")
    return merged
