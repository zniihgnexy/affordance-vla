"""Touch, opposition and the conversion of a human grasp into a map of interaction_labels.py (numpy and scipy only)."""
import numpy as np
import pytest

from affordance_vla.hot3d import contact as C
from affordance_vla.plan.interaction_labels import canonical_map, load_maps, save_maps
from affordance_vla.sim.object_surface import sample_faces


def _pads(thumb_x=-0.045, finger_x=0.045, fingers=(1, 2, 3), thumb_points=5):
    """Hand points: a thumb pad at x = thumb_x and one pad per finger at x = finger_x, all on the line y = z = 0."""
    labels = [C.THUMB] * thumb_points + [f for f in fingers for _ in range(thumb_points)]
    x = np.array([thumb_x] * thumb_points + [finger_x] * (len(labels) - thumb_points))
    points = np.stack([x, np.zeros(len(labels)), np.zeros(len(labels))], axis=1)
    return points, np.array(labels)


def test_touched_points_uses_the_threshold():
    surface = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.020]])
    hand = np.array([[0.003, 0.0, 0.0], [0.0, 0.0, 0.030]])
    assert C.touched_points(surface, hand, 0.005).tolist() == [True, False]
    assert C.touched_points(surface, hand, 0.011).tolist() == [True, True]


def test_to_object_frame_inverts_the_pose():
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
    pose[:3, 3] = [1.0, 2.0, 3.0]
    world = np.array([[1.0, 3.0, 3.0]])  # the object's +x axis, one unit along
    assert np.allclose(C.to_object_frame(world, pose), [[1.0, 0.0, 0.0]])


def test_hold_segments_keeps_long_runs():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1], bool)
    assert C.hold_segments(flags, 3) == [(3, 8)]
    assert C.hold_segments(flags, 2) == [(0, 2), (3, 8)]


def test_opposition_runs_from_the_thumb_to_the_fingers():
    points, labels = _pads()
    pair = C.opposition(points, labels, np.ones(len(labels), bool))
    assert pair is not None
    assert np.allclose(pair.closing, [1.0, 0.0, 0.0], atol=1e-6)
    assert pair.span == pytest.approx(0.09)
    assert pair.fingers == ("index", "middle", "ring")
    assert np.allclose(pair.centre, [0.0, 0.0, 0.0], atol=1e-3)


def test_opposition_needs_thumb_and_fingers_and_room_between_them():
    points, labels = _pads()
    touching = np.ones(len(labels), bool)
    assert C.opposition(points, labels, touching & (labels != C.THUMB)) is None  # the thumb does not touch
    assert C.opposition(points, labels, touching & (labels == C.THUMB)) is None  # no other finger touches
    close_points, close_labels = _pads(-0.005, 0.005)
    assert C.opposition(close_points, close_labels, np.ones(len(close_labels), bool)) is None  # pads 1 cm apart


def test_as_grasp_makes_the_approach_square_to_the_closing_line():
    points, labels = _pads()
    pair = C.opposition(points, labels, np.ones(len(labels), bool))
    grasp = C.as_grasp(pair, wrist_obj=np.array([0.02, 0.0, 0.15]))
    assert np.linalg.norm(grasp.approach) == pytest.approx(1.0)
    assert abs(grasp.approach @ grasp.closing) < 1e-9
    assert grasp.approach[2] < 0  # the fingers point from the wrist (above) down to the pads
    with pytest.raises(ValueError):
        C.as_grasp(pair, wrist_obj=np.array([0.2, 0.0, 0.0]))


def test_human_grasp_becomes_a_map_in_the_format_of_the_demonstration_maps(tmp_path):
    """A box 9 cm wide held between thumb and fingers: the map's score is highest on the two pressed faces."""
    corners = np.array([[x, y, z] for x in (-0.045, 0.045) for y in (-0.04, 0.04) for z in (0.0, 0.1)])
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    surface = sample_faces(corners, faces, 3000, np.random.default_rng(0))
    points, labels = _pads(-0.045, 0.045)
    pair = C.opposition(points + [0, 0, 0.05], labels, np.ones(len(labels), bool))
    grasp = C.as_grasp(pair, np.array([0.0, 0.0, 0.2]))
    cmap = canonical_map("box", surface, [grasp])
    pressed = np.abs(np.abs(surface.points[:, 0]) - 0.045) < 1e-6
    at_pads = pressed & (np.linalg.norm(surface.points[:, 1:] - [0.0, 0.05], axis=1) < 0.015)
    assert cmap.score[at_pads].mean() > 0.5
    assert cmap.score[~pressed].max() < 1e-9  # faces parallel to the closing line are not under a finger
    assert np.allclose(np.abs(cmap.axis[cmap.score > 0.5] @ [1.0, 0.0, 0.0]), 1.0, atol=1e-6)
    save_maps(tmp_path / "maps.npz", [cmap])
    loaded = load_maps(tmp_path / "maps.npz")["box"]
    assert np.array_equal(loaded.score, cmap.score) and loaded.grasps == 1
