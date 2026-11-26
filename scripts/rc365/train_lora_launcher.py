"""Run starVLA's train_starvla.py with LoRA adapters on the VLM's language-model layers (starVLA itself is not changed).

`framework.lora` in the training config switches this on (absent = the stock run): rank, alpha, optional `pattern`
(a regex on the full dotted name of a linear layer). Before training starts this launcher
  1. wraps `build_framework`: adds adapters (affordance_vla.heads.lora, the LIBERO-stage code) and freezes every
     other VLM weight, so that the stock `freeze_modules` need not freeze the adapters with the VLM;
  2. wraps `build_param_lr_groups`: leaves frozen weights out of the optimiser, so DeepSpeed keeps no states for them.
It also prints the CUDA peak memory when the process ends (as peak_memory_launcher.py).
Checkpoints keep the adapters as separate tensors (resumable with `trainer.is_resume=true`); serve them after
scripts/rc365/merge_lora_checkpoint.py.

Usage (under `accelerate launch`): python train_lora_launcher.py <train_starvla.py> [its arguments]
"""
from __future__ import annotations

import atexit
import re
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.heads.lora import LoRALinear, add_lora  # noqa: E402

# language-model attention and MLP projections of Qwen3-VL as wrapped by starVLA's QwenVL interface
DEFAULT_PATTERN = r"qwen_vl_interface\..*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"


def add_adapters_and_freeze(model: torch.nn.Module, rank: int, alpha: float, pattern: str) -> int:
    """Adapters on the matching layers, every other `qwen_vl_interface` weight frozen; returns the layers wrapped."""
    regex = re.compile(pattern)
    names = add_lora(model, rank, alpha, lambda name: regex.fullmatch(name) is not None)
    for parameter in model.qwen_vl_interface.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for parameter in (*module.down.parameters(), *module.up.parameters()):
                parameter.requires_grad_(True)
    return len(names)


def _report_peak() -> None:
    if torch.cuda.is_available():
        gib = 2**30
        allocated, reserved = torch.cuda.max_memory_allocated() / gib, torch.cuda.max_memory_reserved() / gib
        print(f"PEAK_MEMORY allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
              f"device={torch.cuda.current_device()}", flush=True)


def patch_starvla() -> None:
    import starVLA.model.framework.base_framework as base_framework
    import starVLA.training.trainer_utils.trainer_tools as trainer_tools

    build_framework, build_groups = base_framework.build_framework, trainer_tools.build_param_lr_groups

    def build_with_lora(cfg):
        model = build_framework(cfg)
        lora = cfg.framework.get("lora")
        if lora:
            rank, alpha = int(lora["rank"]), float(lora["alpha"])
            count = add_adapters_and_freeze(model, rank, alpha, lora.get("pattern", DEFAULT_PATTERN))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"LoRA: {count} layers, rank {rank}, alpha {alpha}; trainable {trainable / 1e6:.1f} M", flush=True)
        return model

    def groups_without_frozen(model, cfg):
        kept = [{**g, "params": [p for p in g["params"] if p.requires_grad]} for g in build_groups(model, cfg)]
        return [g for g in kept if g["params"]]

    base_framework.build_framework = build_with_lora
    trainer_tools.build_param_lr_groups = groups_without_frozen


if __name__ == "__main__":
    script, sys.argv = sys.argv[1], sys.argv[1:]
    patch_starvla()
    atexit.register(_report_peak)
    runpy.run_path(script, run_name="__main__")
