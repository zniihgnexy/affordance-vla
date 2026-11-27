import importlib.util
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "configs/rc365/data_registry_affo16/data_config.py"
TASKS = {"CoffeeSetupMug", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceSinkToCounter",
         "OpenDrawer", "OpenCabinet", "TurnOffStove", "TurnOnSinkFaucet"}


def _mixtures() -> dict:
    spec = importlib.util.spec_from_file_location("affo16_data_config", REGISTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DATASET_NAMED_MIXTURES


def test_no_mixture_contains_a_target_split_demonstration():
    for name, entries in _mixtures().items():
        assert entries, name
        for path, _, _ in entries:
            assert path.startswith("v1.0/pretrain/") and "/target/" not in path, (name, path)


def test_the_eight_task_mixture_has_one_mimicgen_and_one_human_set_per_task():
    entries = _mixtures()["affo16_rc365_pretrain_8tasks"]
    assert len(entries) == 16
    assert {path.split("/")[3] for path, _, _ in entries} == TASKS
    mg = [e for e in entries if "/mg/" in e[0]]
    assert len(mg) == 8 and all(weight == 1.0 for _, weight, _ in mg)
    assert all(weight == 0.4 for path, weight, _ in entries if "/mg/" not in path)
