"""Check that evaluation feeds the model what training fed it (starvla environment, CPU).

For random steps of the human pretrain demonstrations the same raw step (three decoded 256 x 256 camera frames, state,
instruction) goes through (a) starVLA's training data path (`transforms` + `_pack_sample`, what `__getitem__` returns)
and (b) the evaluation path (affordance_vla.sim.rc365_policy_io.build_example, on an observation dict shaped like the
gym environment's). The model inputs must be identical: number and order of images, pixels, instruction text, and no
state when the run's config has include_state false. Also prints the action chunk length trained and the instruction
strings of the training data. Run from the starVLA checkout (scripts/rc365/check_train_eval_alignment.sh).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from affordance_vla.sim import rc365_policy_io as io  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["CoffeeSetupMug", "OpenDrawer", "TurnOffStove"])
    parser.add_argument("--steps", type=int, default=6, help="random steps per task")
    args = parser.parse_args()

    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    cfg = OmegaConf.load(args.config).datasets.vla_data
    include_state = cfg.get("include_state", False) not in ("False", False)
    rng = np.random.default_rng(0)
    failures = 0
    for task in args.tasks:
        name = f"v1.0/pretrain/atomic/{task}/20250819/lerobot"
        ds = make_LeRobotSingleDataset(Path(args.data_root), name, "panda_omron_robocasa365", data_cfg=cfg)
        for index in rng.choice(len(ds.all_steps), args.steps, replace=False):
            trajectory, base = ds.all_steps[int(index)]
            raw = ds.get_step_data(trajectory, base)
            sample = ds._pack_sample(ds.transforms(raw))
            obs = {key: np.asarray(raw[key][0]) for key in (*io.VIDEO_KEYS, *io.STATE_KEYS)}
            obs[io.LANGUAGE_KEY] = raw[io.LANGUAGE_KEY][0]
            example = io.build_example(obs, include_state=False)
            train_images = [np.asarray(im) for im in sample["image"]]
            same_images = len(train_images) == len(example["image"]) and all(
                np.array_equal(a, b) for a, b in zip(train_images, example["image"]))
            same_lang = sample["lang"] == example["lang"]
            same_state = ("state" in sample) == ("state" in example) == include_state
            ok = same_images and same_lang and same_state
            failures += not ok
            print(f"{task} traj {trajectory} step {base}: images {len(train_images)}x{train_images[0].shape} "
                  f"identical={same_images} lang identical={same_lang} state absent-as-trained={same_state} "
                  f"action chunk {sample['action'].shape} -> {'OK' if ok else 'MISMATCH'}")
        tasks_file = Path(args.data_root) / name / "meta" / "tasks.jsonl"
        print(f"  instruction strings of {task}: {tasks_file.read_text().count(chr(10))} lines in tasks.jsonl, "
              f"e.g. {sample['lang']!r}")
    print(f"include_state={include_state}; mismatches: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
