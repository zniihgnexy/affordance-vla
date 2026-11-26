"""The parts that need the hot3d environment: hand sampling, grasp runs, drawing, and one real clip if present."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hand_tracking_toolkit")
pytest.importorskip("cv2")

from affordance_vla.hot3d import contact as C  # noqa: E402
from affordance_vla.hot3d import render  # noqa: E402
from affordance_vla.hot3d.analysis import steady_runs  # noqa: E402
from affordance_vla.hot3d.clips import Clip  # noqa: E402

CLIP = Path(os.environ.get("HOT3D_DIR", "/mnt/nas/xinyizheng/affordance_vla/data/hot3d")) / "clips/clip-003345.tar"


def _model():
    """Two triangles; vertex v is bound to bone group v: thumb (bone 2), index (5), palm (1), wrist (0)."""
    weights = torch.zeros(4, 17)
    for vertex, bone in enumerate((2, 5, 1, 0)):
        weights[vertex, bone] = 1.0
    return SimpleNamespace(mesh_triangles=torch.tensor([[0, 1, 2], [1, 2, 3]]).float(), dense_bone_weights=weights,
                           mesh_vertices=torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]))


def test_vertex_fingers_follow_the_bone_groups():
    assert C.vertex_fingers(_model()).tolist() == [C.THUMB, 1, C.PALM, C.PALM]


def test_sample_hand_vertices_reproduce_the_mesh():
    samples, triangles = C.sample_hand(_model(), 0)
    posed = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]) * 2 + 1
    assert np.allclose(samples.points(posed, triangles), posed)


def test_sample_hand_points_lie_on_the_mesh_and_carry_a_finger():
    samples, triangles = C.sample_hand(_model(), 500, seed=1)
    points = samples.points(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), triangles)
    assert np.allclose(points[:, 2], 0) and points.min() >= 0 and points.max() <= 1
    assert set(samples.finger) <= {C.THUMB, 1, C.PALM}


def _contact(closings):
    pair = lambda c: ("left", SimpleNamespace(closing=np.asarray(c, float)), None)  # noqa: E731
    return SimpleNamespace(oppositions=[None if c is None else pair(c) for c in closings])


def test_steady_runs_split_where_the_closing_axis_turns():
    x, y = (1, 0, 0), (0, 1, 0)
    contact = _contact([x, x, (-1, 0.1, 0), x, y, y, y, None, y, y])
    assert steady_runs(contact, 3, 20.0) == [(0, 4), (4, 7)]
    assert steady_runs(contact, 2, 20.0) == [(0, 4), (4, 7), (8, 10)]


def test_view_rotation_is_a_right_handed_frame():
    rotation = render.view_rotation(35, 20)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_paint_object_draws_the_nearer_face_over_the_farther_one():
    vertices = np.array([[-1.0, -1, 0], [1, -1, 0], [0, 1, 0], [-1, -1, 1], [1, -1, 1], [0, 1, 1]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])  # the second triangle is nearer to a viewer on +z
    image = render.paint_object(vertices, faces, np.array([0.0, 0, 0, 1, 1, 1]), render.view_rotation(0, 0), 64)
    hot = render.colormap(np.array([1.0]))[0] * 255
    centre = image[32, 32].astype(float)
    assert np.abs(centre - hot).max() < 130 and centre[0] > centre[2]  # reddish, not the grey of value 0


@pytest.mark.skipif(not CLIP.exists(), reason="the demo clip is not downloaded")
def test_real_clip_loads_hands_and_the_mug():
    clip = Clip(CLIP)
    assert len(clip.frames) == 150 and clip.device == "Aria" and "9" in clip.object_ids(0)
    hands = clip.hands(clip.frames[100])
    assert {h.side for h in hands} == {"left", "right"} and all(h.vertices.shape == (788, 3) for h in hands)
    assert np.allclose(clip.object_pose(0, "9")[3], [0, 0, 0, 1])
    assert clip.image(0).shape == (1408, 1408, 3)
