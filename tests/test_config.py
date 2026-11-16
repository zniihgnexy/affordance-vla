"""Config files with a base they extend, and command-line overrides (affordance_vla/config.py)."""
import pytest

from affordance_vla.config import load_config, merged, resolved, with_override


def test_nested_mappings_merge_key_by_key_and_leave_the_inputs_alone():
    base = {"viewer": {"realtime": 1.0, "hold": True}, "seed": 7}
    out = merged(base, {"viewer": {"realtime": 0.5}})
    assert out == {"viewer": {"realtime": 0.5, "hold": True}, "seed": 7}
    assert base["viewer"]["realtime"] == 1.0


def test_an_override_sets_a_dotted_key_to_a_yaml_value():
    config = with_override({"grasp": {"place": True}}, "grasp.place=false")
    assert config["grasp"]["place"] is False
    assert with_override({}, "scene.task=3")["scene"]["task"] == 3
    assert with_override({}, "grasp.object=ketchup_1")["grasp"]["object"] == "ketchup_1"


def test_an_override_without_a_value_is_refused():
    with pytest.raises(ValueError):
        with_override({}, "grasp.place")


def test_a_config_extends_its_base_and_overrides_come_last(tmp_path):
    (tmp_path / "base.yaml").write_text("scene: {suite: libero_object, task: 0}\nviewer: {realtime: 1.0}\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\nscene: {task: 2}\n")
    config = load_config(tmp_path / "exp.yaml", ["viewer.realtime=0"])
    assert config == {"scene": {"suite": "libero_object", "task": 2}, "viewer": {"realtime": 0}}


def test_bases_that_form_a_loop_are_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n")
    with pytest.raises(ValueError, match="loop"):
        load_config(tmp_path / "a.yaml")


def test_the_resolved_config_records_its_source(tmp_path):
    out = resolved({"seed": 7}, "configs/x.yaml", ["seed=8"], tmp_path)
    assert out["_source"] == {"config": "configs/x.yaml", "overrides": ["seed=8"], "commit": "unknown"}
