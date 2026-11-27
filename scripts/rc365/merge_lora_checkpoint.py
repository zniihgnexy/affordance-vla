"""Turn a LoRA training checkpoint into one a stock starVLA policy server loads (adapters folded into the weights).

Writes <out_run>/{checkpoints/<same file name>, config.yaml, dataset_statistics.json}; the server finds the config
and the statistics next to the `checkpoints` folder. Nothing in the training run is changed.

  python scripts/rc365/merge_lora_checkpoint.py --ckpt <run>/checkpoints/steps_N_pytorch_model.pt --out <out_run>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.heads.lora_merge import merge_lora_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.ckpt.parent.parent
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    lora = cfg["framework"]["lora"]
    if (args.out / "checkpoints" / args.ckpt.name).exists():
        raise SystemExit(f"{args.out / 'checkpoints' / args.ckpt.name} exists; choose another --out")
    state = torch.load(args.ckpt, map_location="cpu")
    merged = merge_lora_state(state, float(lora["alpha"]), int(lora["rank"]))
    print(f"{len(state)} tensors -> {len(merged)} (rank {lora['rank']}, alpha {lora['alpha']})")
    (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out / "checkpoints" / args.ckpt.name)
    cfg["framework"].pop("lora")
    (args.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(run / "dataset_statistics.json", args.out / "dataset_statistics.json")


if __name__ == "__main__":
    main()
