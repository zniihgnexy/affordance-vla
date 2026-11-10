"""CPU-only smoke test for the starVLA LIBERO recipe: build the model and one training batch.

Catches path, config and dependency errors before a GPU slot is spent, and prints the parameter
count per top-level module (which decides whether full fine-tuning can fit on a 48 GB card).

Usage on the server (inside the starvla env, from the starVLA repo root):
    CUDA_VISIBLE_DEVICES= python /data/xinyizheng/affordance3d_vla/scripts/smoke_starvla_cpu.py \
        --vlm /data/xinyizheng/affordance3d_vla/models/Qwen3-VL-2B-Instruct \
        --data-root /data/xinyizheng/affordance3d_vla/data/libero_lerobot
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from omegaconf import OmegaConf

CONFIG_YAML = "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml"
BYTES_PER_PARAM_ADAMW_FP32 = 16  # weight + gradient + two Adam moments, 4 bytes each


def build_config(args: argparse.Namespace):
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    overrides = OmegaConf.from_dotlist([
        f"framework.name={args.framework}",
        f"framework.qwenvl.base_vlm={args.vlm}",
        f"datasets.vla_data.data_root_dir={args.data_root}",
        f"datasets.vla_data.data_mix={args.data_mix}",
        f"datasets.vla_data.per_device_batch_size={args.batch_size}",
        "trainer.vla_data.video_backend=torchvision_av",
        f"output_dir={args.out}",  # the trainer sets this in setup_directories(); the dataloader writes stats there
    ])
    cfg = OmegaConf.merge(OmegaConf.load(CONFIG_YAML), overrides)
    return wrap_config(apply_config_compat(cfg))


def report_parameters(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total / 1e6:,.1f} M total")
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count:
            print(f"  {name:28s} {count / 1e6:10,.1f} M  ({100 * count / total:5.1f} %)")
    gib = total * BYTES_PER_PARAM_ADAMW_FP32 / 2**30
    print(f"fp32 weights + gradients + AdamW moments on ONE card, before activations: {gib:.1f} GiB")


def describe(value) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}] of {describe(value[0]) if value else '-'}"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--framework", default="QwenPI")
    parser.add_argument("--data-mix", default="libero_all")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", default="/data/xinyizheng/affordance3d_vla/runs/smoke_cpu")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework.base_framework import build_framework

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = build_config(args)
    started = time.perf_counter()
    model = build_framework(cfg)
    print(f"built {args.framework} in {time.perf_counter() - started:.0f} s")
    report_parameters(model)

    if args.skip_data:
        return
    started = time.perf_counter()
    loader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    batch = next(iter(loader))
    print(f"first batch in {time.perf_counter() - started:.0f} s · {len(loader):,} batches per epoch "
          f"at batch size {args.batch_size}")
    sample = batch[0] if isinstance(batch, (list, tuple)) else batch
    for key, value in (sample.items() if isinstance(sample, dict) else []):
        print(f"  {key:24s} {describe(value)}")
    print("SMOKE-OK")


if __name__ == "__main__":
    main()
