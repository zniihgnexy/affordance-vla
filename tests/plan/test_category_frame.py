import numpy as np

from affordance_vla.plan.category_frame import CategoryFrame, fit_frame, up_at_rest
from affordance_vla.sim.object_surface import sample_faces

QUARTER_X = np.array([np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)])  # x, y, z, w: 90 degrees about x


def mug(radius=0.04, height=0.09, handle=0.03, seed=0, count=4000):
    """A tube (lateral wall and base disc) along z with a thin handle sticking out along +x."""
    rng = np.random.default_rng(seed)
    angle, z = rng.uniform(0, 2 * np.pi, count), rng.uniform(0, height, count)
    wall = np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)
    base_radius, base_angle = radius * np.sqrt(rng.random(count // 4)), rng.uniform(0, 2 * np.pi, count // 4)
    base = np.stack([base_radius * np.cos(base_angle), base_radius * np.sin(base_angle), np.zeros(count // 4)], axis=1)
    reach = rng.uniform(radius, radius + handle, count // 8)
    handle_z = rng.uniform(0.3 * height, 0.8 * height, count // 8)
    grip = np.stack([reach, np.zeros(count // 8), handle_z], axis=1)
    return np.concatenate([wall, base, grip])


def test_the_frame_puts_a_mug_upright_on_its_base_with_unit_size_and_the_handle_on_x():
    points = mug()
    frame = fit_frame(points, [0, 0, 1])
    unit = frame.point_to_category(points)
    assert np.allclose(unit[:, 2].min(), 0, atol=1e-9) and np.allclose(unit[:, 2].max(), 1)
    assert abs(np.median(np.linalg.norm(unit[:, :2], axis=1)) - 1) < 1e-9
    assert unit[:, 0].max() > 1.5 and abs(unit[:, 1].mean()) < 0.1  # the handle reaches out along +x


def test_the_frame_is_the_same_for_a_mug_that_is_moved_turned_and_scaled_in_its_body_frame():
    points = mug()
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # 90 degrees about the vertical axis
    moved = 1.7 * (points @ turn.T) + [0.3, -0.2, 0.1]
    a = fit_frame(points, [0, 0, 1]).point_to_category(points)
    b = fit_frame(moved, [0, 0, 1]).point_to_category(moved)
    assert np.allclose(a, b, atol=0.05)


def test_a_mug_lying_in_its_body_frame_is_straightened_by_its_up_direction():
    points = mug()
    lying = points @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]).T  # (x, y, z) -> (x, -z, y): the mug's up is -y
    straight = fit_frame(lying, [0, -1, 0]).point_to_category(lying)
    upright = fit_frame(points, [0, 0, 1]).point_to_category(points)
    assert np.allclose(straight, upright, atol=1e-9)


def test_up_at_rest_reads_the_body_direction_of_world_up():
    assert np.allclose(up_at_rest([[0, 0, 0, 1.0]]), [0, 0, 1])
    # a body turned 90 degrees about x has its y axis pointing up: the body-frame direction of world up is (0, 1, 0)
    assert np.allclose(up_at_rest([QUARTER_X]), [0, 1, 0], atol=1e-9)


def test_axes_move_like_normals_and_come_back():
    frame = CategoryFrame(np.array([0.1, 0.2, 0.3]), np.eye(3), np.array([10.0, 10.0, 5.0]))
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    there = frame.axis_to_category(axis)
    assert np.isclose(np.linalg.norm(there), 1) and np.allclose(there, np.array([1, 1, 2]) / np.sqrt(6))
    assert np.allclose(frame.axis_from_category(there), axis)
    points = np.random.default_rng(1).normal(size=(5, 3))
    assert np.allclose(frame.point_from_category(frame.point_to_category(points)), points)


def test_a_surface_normal_stays_normal_to_the_scaled_surface():
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5]])
    surface = sample_faces(triangle, np.array([[0, 1, 2]]), 5, np.random.default_rng(0))
    frame = CategoryFrame(np.zeros(3), np.eye(3), np.array([2.0, 3.0, 5.0]))
    normal = frame.axis_to_category(surface.normals[0])
    edges = frame.point_to_category(np.array([[1, 0, 0], [0, 1, 0.5]]))  # the triangle's edges from the origin vertex
    assert np.allclose(edges @ normal, 0, atol=1e-9)
