from types import SimpleNamespace

import numpy as np
import pytest

from affordance_vla.sim.object_surface import body_meshes, body_surface, quat_wxyz_matrix, sample_faces

SQUARE_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
SQUARE_FACES = np.array([[0, 1, 2], [0, 2, 3]])


def test_quat_wxyz_matrix_turns_x_into_y_for_a_quarter_turn_about_z():
    half = np.sqrt(0.5)
    assert np.allclose(quat_wxyz_matrix([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(quat_wxyz_matrix([half, 0, 0, half]) @ [1, 0, 0], [0, 1, 0])


def test_sample_faces_puts_points_on_the_mesh_with_its_face_normal():
    surface = sample_faces(SQUARE_VERTICES, SQUARE_FACES, 500, np.random.default_rng(0))
    assert surface.points.shape == (500, 3) and surface.normals.shape == (500, 3)
    assert np.all((surface.points[:, :2] >= 0) & (surface.points[:, :2] <= 1))
    assert np.allclose(surface.points[:, 2], 0)
    assert np.allclose(np.abs(surface.normals[:, 2]), 1)


def test_sample_faces_rejects_a_mesh_without_area():
    with pytest.raises(ValueError):
        sample_faces(np.zeros((3, 3)), np.array([[0, 1, 2]]), 10, np.random.default_rng(0))


def fake_model():
    """Body 1 has a visual square lifted by 0.5 m and a colliding box; body 2 has nothing."""
    return SimpleNamespace(
        ngeom=2, geom_bodyid=np.array([1, 1]), geom_type=np.array([7, 6]), geom_contype=np.array([0, 1]),
        geom_conaffinity=np.array([0, 1]), geom_dataid=np.array([0, -1]), geom_quat=np.array([[1, 0, 0, 0]] * 2),
        geom_pos=np.array([[0, 0, 0.5], [0, 0, 0]]), mesh_vertadr=np.array([0]), mesh_vertnum=np.array([4]),
        mesh_faceadr=np.array([0]), mesh_facenum=np.array([2]), mesh_vert=SQUARE_VERTICES, mesh_face=SQUARE_FACES)


def test_body_meshes_takes_only_visual_meshes_and_places_them_in_the_body_frame():
    vertices, faces = body_meshes(fake_model(), 1)
    assert vertices.shape == (4, 3) and np.allclose(vertices[:, 2], 0.5)
    assert faces.tolist() == SQUARE_FACES.tolist()
    with pytest.raises(ValueError):
        body_meshes(fake_model(), 2)


def test_body_surface_is_reproducible_for_a_seed():
    a, b = body_surface(fake_model(), 1, count=50, seed=3), body_surface(fake_model(), 1, count=50, seed=3)
    assert np.array_equal(a.points, b.points)
