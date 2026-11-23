from dataclasses import replace

import numpy as np
import pytest

from affordance_vla.plan.interaction_labels import (
    CanonicalMap,
    Grasp,
    axis_groups,
    canonical_map,
    contact_weights,
    grasp_in_object,
    held_grasp_steps,
    line_angle_deg,
    load_maps,
    map_in_world,
    project_map,
    save_maps,
)
from affordance_vla.sim.object_surface import Surface

X, Y, Z = np.eye(3)
QUARTER_Z = np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])  # x, y, z, w: 90 degrees about z
IDENTITY = np.array([0, 0, 0, 1.0])


def cube(half=0.03, per_face=200, seed=0):
    """Points on the six faces of a cube centred at the origin, with their outward normals."""
    rng = np.random.default_rng(seed)
    points, normals = [], []
    for axis in range(3):
        for sign in (-1, 1):
            p = rng.uniform(-half, half, (per_face, 3))
            p[:, axis] = sign * half
            n = np.zeros((per_face, 3))
            n[:, axis] = sign
            points.append(p)
            normals.append(n)
    return Surface(np.concatenate(points), np.concatenate(normals))


def test_grasp_in_object_undoes_the_object_rotation():
    grasp = grasp_in_object([1, 0, 0.1], IDENTITY, [1, 0, 0], QUARTER_Z)
    assert np.allclose(grasp.centre, [0, 0, 0.1])
    assert np.allclose(grasp.closing, X)  # the gripper's y axis is the x axis of an object turned by 90 degrees
    assert np.allclose(grasp.approach, Z)


def test_contact_weights_fall_on_the_faces_across_the_closing_line():
    surface = cube()
    weights = contact_weights(surface, Grasp(np.zeros(3), X, -Z))
    on_x_faces = np.abs(surface.normals[:, 0]) == 1
    assert weights[on_x_faces].max() > 0.9
    assert weights[~on_x_faces].max() == 0  # the other faces do not face a finger
    far = contact_weights(surface, Grasp(np.array([0.2, 0, 0]), X, -Z))  # cube beyond the fingers' reach
    assert far.max() == 0


def test_canonical_map_agrees_where_grasps_share_a_line_and_not_where_they_cross():
    surface = cube()
    same = canonical_map("cube", surface, [Grasp(np.zeros(3), X, -Z), Grasp(np.zeros(3), -X, -Z)])
    assert same.score.max() == pytest.approx(1.0)
    best = int(np.argmax(same.score))
    assert line_angle_deg(same.axis[best], X) < 1 and same.agreement[best] == pytest.approx(1.0)
    untouched = same.score == 0
    assert np.all(same.axis[untouched] == 0) and np.all(same.agreement[untouched] == 0)
    crossed = canonical_map("cube", surface, [Grasp(np.zeros(3), X, -Z), Grasp(np.zeros(3), Y, -Z)])
    assert crossed.agreement[crossed.score > 0.5].min() > 0.9  # each face is touched by one grasp only
    with pytest.raises(ValueError):
        canonical_map("cube", surface, [])


def test_axis_groups_count_lines_not_arrows():
    tilted = np.array([1, 0.1, 0]) / np.linalg.norm([1, 0.1, 0])
    assert axis_groups([X, -X, tilted, Y]) == [3, 1]


def test_held_grasp_steps_keeps_grasps_that_last_in_time_order():
    grasped = np.zeros((60, 2), bool)
    grasped[40:, 0] = True  # held to the end, however late after the close command
    grasped[10:30, 1] = True  # held, then released
    grasped[50:52, 1] = True  # a touch of two steps does not count
    assert held_grasp_steps(grasped) == [(1, 10), (0, 40)]
    assert held_grasp_steps(np.zeros((5, 1), bool)) == []


def flat_patch():
    """A small square facing a camera at the origin that looks along +z, 1 m away."""
    xs, ys = np.meshgrid(np.linspace(-0.01, 0.01, 11), np.linspace(-0.01, 0.01, 11))
    points = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    normals = np.tile(-Z, (len(points), 1))
    return Surface(points, normals)


def test_project_map_marks_the_cell_under_the_object_with_its_axis_direction():
    surface = flat_patch()
    cmap = canonical_map("patch", surface, [Grasp(np.array([0, 0, 1.0]), X, Z)])
    cmap = replace(cmap, score=np.ones(len(surface.points)), axis=np.tile(X, (len(surface.points), 1)))
    intrinsics = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    seg = np.full((64, 64), 3, np.uint8)
    lifted = np.array([0, 0.05, 0])  # v = 32 + 2.5 px, well inside row 4
    label = project_map(cmap, lifted, IDENTITY, intrinsics, np.eye(4), seg, seg_id=3)
    assert label.visible == len(surface.points)
    assert label.score[4, 3] == 1 and label.score.sum() == 1  # u is mirrored: 63 - 32 = 31 -> column 3
    assert np.allclose(label.direction[4, 3], [1, 0])  # a horizontal line: cos 2θ = 1
    hidden = project_map(cmap, lifted, IDENTITY, intrinsics, np.eye(4), np.zeros((64, 64), np.uint8), 3)
    assert hidden.visible == 0 and hidden.score.sum() == 0


def test_project_map_drops_points_behind_the_object_itself():
    front = flat_patch()
    back = Surface(front.points + [0, 0, 0.05], front.normals)
    surface = Surface(np.concatenate([front.points, back.points]), np.concatenate([front.normals, back.normals]))
    cmap = canonical_map("slab", surface, [Grasp(np.array([0, 0, 1.0]), X, Z)])
    cmap = replace(cmap, score=np.ones(len(surface.points)))
    intrinsics = np.array([[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]])
    label = project_map(cmap, np.zeros(3), IDENTITY, intrinsics, np.eye(4), np.full((64, 64), 1, np.uint8), 1)
    assert label.visible == len(front.points)


def test_maps_survive_saving_and_loading(tmp_path):
    cmap = canonical_map("cube", cube(per_face=20), [Grasp(np.zeros(3), X, -Z)])
    save_maps(tmp_path / "maps.npz", [cmap])
    loaded = load_maps(tmp_path / "maps.npz")["cube"]
    assert loaded.grasps == 1 and np.array_equal(loaded.score, cmap.score) and np.array_equal(loaded.axis, cmap.axis)


def test_map_in_world_moves_points_and_turns_axes_with_the_object():
    cmap = CanonicalMap("mug", points=np.array([[0.1, 0.0, 0.0]]), normals=np.array([[1.0, 0.0, 0.0]]),
                        score=np.array([1.0]), axis=np.array([[1.0, 0.0, 0.0]]), agreement=np.array([1.0]), grasps=1)
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 degrees about z
    points, axes = map_in_world(cmap, np.array([1.0, 2.0, 3.0]), quarter_turn)
    np.testing.assert_allclose(points, [[1.0, 2.1, 3.0]])
    np.testing.assert_allclose(axes, [[0.0, 1.0, 0.0]])
