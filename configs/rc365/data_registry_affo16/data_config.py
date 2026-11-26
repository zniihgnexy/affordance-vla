"""Pretrain-split mixture of the eight AFFO-16 atomic tasks: MimicGen demonstrations plus the human ones.

Copied by scripts/rc365/run_train_base_gpu.sh into the starVLA checkout as
examples/simBenchmarks/AFFO16_rc365/train_files/data_registry/data_config.py, where starVLA's registry discovers it.
The robot tag is the one registered by examples/simBenchmarks/Robocasa_365. Only `pretrain` folders are listed:
tests/rc365/test_data_registry.py fails if a `target` folder ever appears (the target demonstrations were recorded
on the held-out object instances, so training on them would make "unseen instances" meaningless).
Sampling is proportional to the weights (the loader does not balance by length): MimicGen 1.0 and human 0.4 per
task, so the 100 human demonstrations of a task make up 29 % of its samples.
"""

_ROBOT_TAG = "panda_omron_robocasa365"
_DATE = "20250819"
HUMAN_WEIGHT = 0.4

# task -> time stamp of its MimicGen folder (robocasa.utils.dataset_registry.ATOMIC_TASK_DATASETS[task]["pretrain"]["mg_path"])
_MG_STAMP = {
    "CoffeeSetupMug": "2025-08-20-21-54-35",
    "PickPlaceCounterToCabinet": "2025-08-20-21-56-25",
    "PickPlaceCounterToStove": "2025-08-20-22-33-04",
    "PickPlaceSinkToCounter": "2025-08-20-22-34-07",
    "OpenDrawer": "2025-08-20-21-55-00",
    "OpenCabinet": "2025-08-20-21-54-43",
    "TurnOffStove": "2025-08-21-10-12-48",
    "TurnOnSinkFaucet": "2025-08-21-12-24-03",
}


def _entries(tasks, human_weight):
    entries = []
    for task in tasks:
        base = f"v1.0/pretrain/atomic/{task}/{_DATE}"
        entries.append((f"{base}/mg/demo/{_MG_STAMP[task]}/lerobot", 1.0, _ROBOT_TAG))
        if human_weight > 0:
            entries.append((f"{base}/lerobot", human_weight, _ROBOT_TAG))
    return entries


DATASET_NAMED_MIXTURES = {
    **{f"affo16_rc365_pretrain_{task}": _entries([task], HUMAN_WEIGHT) for task in _MG_STAMP},  # one task: cache building
    "affo16_rc365_pretrain_8tasks": _entries(_MG_STAMP, HUMAN_WEIGHT),
    "affo16_rc365_pretrain_coffee_mg": _entries(["CoffeeSetupMug"], 0.0),   # trial: one task, one source
    "affo16_rc365_pretrain_3tasks": _entries(["CoffeeSetupMug", "OpenDrawer", "TurnOffStove"], HUMAN_WEIGHT),
}
