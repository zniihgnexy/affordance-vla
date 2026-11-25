import numpy as np

from affordance_vla.plan.category_frame import fit_frame
from affordance_vla.plan.category_transfer import Source, category_map, transfer_nearest
from affordance_vla.plan.interaction_labels import Grasp, canonical_map, contact_weights
from affordance_vla.plan.transfer_check import check_transfer, high_region
from affordance_vla.sim.object_surface import Surface

UP = np.array([0.0, 0.0, 1.0])
X, Y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])


def tube(radius, height, handle=0.03, count=6000, seed=0):
    """A tube along z standing on its base at z = 0, with a handle along +x: surface points and their normals."""
    rng = np.random.default_rng(seed)
    angle, z = rng.uniform(0, 2 * np.pi, count), rng.uniform(0, height, count)
    wall = np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)
    wall_normals = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1)
    reach = rng.uniform(radius, radius + handle, count // 8)
    grip_z = rng.uniform(0.3 * height, 0.8 * height, count // 8)
    grip = np.stack([reach, np.zeros(count // 8), grip_z], axis=1)
    return Surface(np.concatenate([wall, grip]), np.concatenate([wall_normals, np.tile(Y, (count // 8, 1))]))


def waist_grasp(height, at=0.5):
    """Fingers closing along y at the given share of the height, on the vertical axis: they pinch the wall."""
    return Grasp(np.array([0.0, 0.0, at * height]), Y, -UP)


def source(name, radius, height, at=0.5):
    surface = tube(radius, height)
    return Source(name, fit_frame(surface.points, UP), surface, [waist_grasp(height, at)] * 3)


def test_a_map_carried_to_a_mug_of_other_proportions_lands_on_that_mugs_own_grasps():
    cmap = category_map([source("a", 0.04, 0.09), source("b", 0.05, 0.12)])
    target = tube(0.035, 0.10, seed=3)
    carried = transfer_nearest(cmap, fit_frame(target.points, UP), target, "c")
    result = check_transfer(carried, target, [waist_grasp(0.10)] * 4)
    assert result.grasps == 4 and result.landing_rate == 1.0 and result.axis_error_median_deg < 5
    assert carried.grasps == 6 and result.passed


def test_grasps_elsewhere_on_the_mug_do_not_land():
    cmap = category_map([source("a", 0.04, 0.09), source("b", 0.05, 0.12)])
    target = tube(0.035, 0.10, seed=3)
    carried = transfer_nearest(cmap, fit_frame(target.points, UP), target, "c")
    result = check_transfer(carried, target, [waist_grasp(0.10, at=0.1)] * 4)
    assert result.landing_rate == 0.0 and not result.passed


def test_the_carried_axis_is_the_closing_axis_in_the_target_body_frame():
    cmap = category_map([source("a", 0.04, 0.09)])
    target = tube(0.035, 0.10, seed=3)
    carried = transfer_nearest(cmap, fit_frame(target.points, UP), target, "c")
    region = high_region(carried)
    assert region.any() and np.allclose(np.abs(carried.axis[region] @ Y), 1, atol=0.05)


def test_points_farther_than_the_reach_score_zero():
    cmap = category_map([source("a", 0.04, 0.09)])
    target = tube(0.04, 0.09, seed=1)
    frame = fit_frame(target.points, UP)
    assert transfer_nearest(cmap, frame, target, "c", max_distance=1e-9).score.max() == 0


def test_a_map_carried_to_its_own_mesh_is_the_map_of_its_grasps():
    surface = tube(0.04, 0.09)
    frame = fit_frame(surface.points, UP)
    grasps = [waist_grasp(0.09), waist_grasp(0.09, at=0.3)]
    own = canonical_map("a", surface, grasps)
    carried = transfer_nearest(category_map([Source("a", frame, surface, grasps)]), frame, surface, "a")
    assert np.allclose(carried.score, own.score)
    touched = own.score > 0
    assert np.allclose(np.abs((carried.axis * own.axis).sum(axis=1))[touched], 1, atol=1e-6)


def test_a_grasp_that_touches_nothing_counts_as_a_miss():
    cmap = category_map([source("a", 0.04, 0.09)])
    target = tube(0.04, 0.09)
    carried = transfer_nearest(cmap, fit_frame(target.points, UP), target, "c")
    far = Grasp(np.array([1.0, 1.0, 1.0]), Y, -UP)
    assert contact_weights(target, far).max() == 0
    result = check_transfer(carried, target, [far])
    assert result.untouched == 1 and result.landing_rate == 0 and result.axis_error_median_deg == 90
