import numpy as np
import pytest

from affordance_vla.plan.category_data import MeshData, collect_mesh_data, load_mesh_data, save_mesh_data
from affordance_vla.plan.interaction_labels import Grasp
from affordance_vla.sim.object_surface import Surface
from tests.plan.test_category_transfer import tube, waist_grasp


def test_mesh_data_survive_a_round_trip(tmp_path):
    surface = tube(0.04, 0.09, count=80)
    mesh = MeshData("mug", surface, np.array([0, 0, 1.0]), (waist_grasp(0.09), waist_grasp(0.09, at=0.3)),
                    ("a_demo0.npz", "b_demo1.npz"), (0, 1), (12, 40))
    save_mesh_data(tmp_path / "m.npz", [mesh])
    (back,) = load_mesh_data(tmp_path / "m.npz")
    assert back.name == "mug" and back.logs == mesh.logs and back.objects == (0, 1) and back.steps == (12, 40)
    assert np.allclose(back.surface.points, surface.points) and np.allclose(back.up, [0, 0, 1])
    assert all(isinstance(g, Grasp) for g in back.grasps)
    assert np.allclose(back.grasps[1].centre, mesh.grasps[1].centre)
    assert np.allclose(back.frame.scale, mesh.frame.scale)


def test_a_mesh_without_a_held_grasp_stops_the_collection(tmp_path):
    with pytest.raises(SystemExit):
        collect_mesh_data([], {"mug": Surface(np.zeros((1, 3)), np.zeros((1, 3)))}, ["mug"])
