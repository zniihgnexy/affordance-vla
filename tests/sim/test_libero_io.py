"""Reading LIBERO demonstration files: BDDL scene descriptions and per-demo arrays."""
import json

import numpy as np
import pytest

from affordance_vla.sim.libero_io import parse_bddl

BDDL = """(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language Pick up the salad dressing and place it in the basket)
    (:regions
      (bin_region
          (:target floor)
          (:ranges ((-0.01 0.25 0.01 0.27)))
      )
    )
  (:fixtures
    floor - floor
  )
  (:objects
    alphabet_soup_1 - alphabet_soup
    salad_dressing_1 - salad_dressing
    basket_1 - basket
  )
  (:obj_of_interest
    salad_dressing_1
    basket_1
  )
  (:init
    (On alphabet_soup_1 floor_object_region)
  )
  (:goal
    (And (In salad_dressing_1 basket_1_contain_region))
  )
)
"""


def test_parse_bddl_reads_language_objects_and_targets():
    scene = parse_bddl(BDDL)

    assert scene.language == "Pick up the salad dressing and place it in the basket"
    assert scene.fixtures == {"floor": "floor"}
    assert scene.objects == {"alphabet_soup_1": "alphabet_soup", "salad_dressing_1": "salad_dressing",
                             "basket_1": "basket"}
    assert scene.of_interest == ("salad_dressing_1", "basket_1")
    assert scene.goal == "(And (In salad_dressing_1 basket_1_contain_region))"


def test_parse_bddl_distractors_are_objects_not_of_interest():
    assert parse_bddl(BDDL).distractors == ("alphabet_soup_1",)


def test_parse_bddl_rejects_text_without_objects():
    with pytest.raises(ValueError, match=":objects"):
        parse_bddl("(define (problem x) (:language hi))")


def _write_demo_file(path, steps=6):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "do it"})
        data.attrs["bddl_file_name"] = "libero/bddl_files/task.bddl"
        data.attrs["macros_image_convention"] = "opengl"
        demo = data.create_group("demo_0")
        demo.attrs["model_file"] = "<mujoco/>"
        demo.attrs["init_state"] = np.zeros(4)
        demo.create_dataset("actions", data=np.tile(np.arange(7.0), (steps, 1)))
        demo.create_dataset("states", data=np.zeros((steps, 4)))
        obs = demo.create_group("obs")
        image = np.zeros((steps, 2, 3, 3), dtype=np.uint8)
        image[:, 0, 0] = 255  # mark the stored top-left pixel
        obs.create_dataset("agentview_rgb", data=image)
        obs.create_dataset("eye_in_hand_rgb", data=image)
        obs.create_dataset("ee_pos", data=np.ones((steps, 3)))
        obs.create_dataset("gripper_states", data=np.zeros((steps, 2)))


def test_read_demo_returns_arrays_and_flips_opengl_images_vertically(tmp_path):
    from affordance_vla.sim.libero_io import list_demos, read_demo, read_task_info

    path = tmp_path / "task_demo.hdf5"
    _write_demo_file(path)

    demo = read_demo(path, "demo_0")
    info, bddl_name = read_task_info(path)

    assert list_demos(path) == ["demo_0"]
    assert info["language_instruction"] == "do it" and bddl_name.endswith("task.bddl")
    assert demo.actions.shape == (6, 7) and demo.ee_pos.shape == (6, 3)
    assert demo.agentview.shape == (6, 2, 3, 3)
    assert demo.agentview[0, -1, 0, 0] == 255  # bottom-left after the flip; columns keep their order
    assert demo.agentview[0, 0, 0, 0] == 0
    assert demo.has_replay_state


def test_read_demo_rejects_unknown_demo(tmp_path):
    from affordance_vla.sim.libero_io import read_demo

    path = tmp_path / "task_demo.hdf5"
    _write_demo_file(path)
    with pytest.raises(KeyError, match="demo_9"):
        read_demo(path, "demo_9")


def test_parse_bddl_assigns_type_to_every_name_in_a_grouped_declaration():
    text = BDDL.replace("alphabet_soup_1 - alphabet_soup", "bowl_1 bowl_2 - bowl")
    assert parse_bddl(text).objects == {"bowl_1": "bowl", "bowl_2": "bowl", "salad_dressing_1": "salad_dressing",
                                        "basket_1": "basket"}


def test_parse_bddl_block_names_must_match_exactly():
    text = BDDL.replace("(:objects", "(:objects_extra\n    x_1 - x\n  )\n  (:objects", 1)
    assert "x_1" not in parse_bddl(text).objects


def test_read_demo_requires_image_convention_or_explicit_flip(tmp_path):
    h5py = pytest.importorskip("h5py")
    from affordance_vla.sim.libero_io import read_demo

    path = tmp_path / "task_demo.hdf5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        del f["data"].attrs["macros_image_convention"]

    with pytest.raises(ValueError, match="macros_image_convention"):
        read_demo(path, "demo_0")
    assert read_demo(path, "demo_0", flip=False).agentview[0, 0, 0, 0] == 255


def test_read_demo_keeps_opencv_images_and_skips_images_on_request(tmp_path):
    h5py = pytest.importorskip("h5py")
    from affordance_vla.sim.libero_io import read_demo

    path = tmp_path / "task_demo.hdf5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        f["data"].attrs["macros_image_convention"] = "opencv"

    assert read_demo(path, "demo_0").wrist[0, 0, 0, 0] == 255
    no_images = read_demo(str(path), "demo_0", images=False)
    assert no_images.agentview is None and no_images.wrist is None


def test_read_demo_rejects_mismatched_lengths(tmp_path):
    h5py = pytest.importorskip("h5py")
    from affordance_vla.sim.libero_io import read_demo

    path = tmp_path / "task_demo.hdf5"
    _write_demo_file(path)
    with h5py.File(path, "a") as f:
        del f["data/demo_0/obs/ee_pos"]
        f["data/demo_0/obs"].create_dataset("ee_pos", data=np.ones((5, 3)))

    with pytest.raises(ValueError, match="steps"):
        read_demo(path, "demo_0")


def test_list_demos_sorts_numerically_and_skips_other_keys(tmp_path):
    h5py = pytest.importorskip("h5py")
    from affordance_vla.sim.libero_io import list_demos

    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as f:
        for name in ["demo_10", "demo_2", "mask"]:
            f.create_group(f"data/{name}")

    assert list_demos(path) == ["demo_2", "demo_10"]


def test_read_task_info_decodes_bytes_attributes(tmp_path):
    h5py = pytest.importorskip("h5py")
    from affordance_vla.sim.libero_io import read_task_info

    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = np.bytes_(json.dumps({"language_instruction": "go"}))
        data.attrs["bddl_file_name"] = np.bytes_("bddl/task.bddl")

    assert read_task_info(path) == ({"language_instruction": "go"}, "bddl/task.bddl")
