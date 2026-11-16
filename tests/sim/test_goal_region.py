"""Which object owns the goal region of a LIBERO task (scripts/eval/rollout_libero_logged.py)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "rollout_libero_logged.py"
spec = importlib.util.spec_from_file_location("rollout_libero_logged", SCRIPT)
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)


def test_an_object_owns_its_named_regions():
    assert rollout.owns_region("basket_1", "basket_1_contain_region")


def test_an_object_that_is_itself_the_goal_owns_it():
    assert rollout.owns_region("plate_1", "plate_1")


def test_an_object_does_not_own_a_region_of_another_object_with_a_longer_name():
    assert not rollout.owns_region("plate_1", "plate_10_contain_region")
