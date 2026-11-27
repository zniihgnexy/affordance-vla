"""Build the starVLA dataset objects of a mixture (writing their cache files) and print what they hold.

Run in the starvla environment from the starVLA checkout via scripts/rc365/prepare_train_data.sh.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mix", required=True)
    args = parser.parse_args()

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = OmegaConf.load(args.config).datasets.vla_data
    cfg.data_root_dir, cfg.data_mix = Path(args.data_root), args.mix
    started = time.time()
    dataset = get_vla_dataset(data_cfg=cfg)
    print(f"mixture {args.mix}: {len(dataset.datasets)} datasets, built in {time.time() - started:.0f} s")
    for part, weight in zip(dataset.datasets, dataset.dataset_sampling_weights):
        print(f"  {str(part.dataset_path).split('robocasa365/')[-1]}: {len(part.trajectory_ids)} trajectories, "
              f"{len(part.all_steps)} steps, sampling weight {weight:.3f}")


if __name__ == "__main__":
    main()
