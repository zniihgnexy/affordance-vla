"""Mixture of three RoboCasa365 target/human tasks for the AFFO-10 pipeline check.

Copied by scripts/rc365/run_train_gpu.sh into the starVLA checkout as
examples/simBenchmarks/AFFO10_rc365/train_files/data_registry/data_config.py, where starVLA's registry
discovers it. The robot tag is the one registered by examples/simBenchmarks/Robocasa_365.
"""

_ROBOT_TAG = "panda_omron_robocasa365"

DATASET_NAMED_MIXTURES = {
    "affo10_rc365_3tasks": [
        ("v1.0/target/atomic/CoffeeSetupMug/20250813/lerobot", 1.0, _ROBOT_TAG),
        ("v1.0/target/atomic/OpenDrawer/20250816/lerobot", 1.0, _ROBOT_TAG),
        ("v1.0/target/atomic/TurnOffStove/20250812/lerobot", 1.0, _ROBOT_TAG),
    ],
}
