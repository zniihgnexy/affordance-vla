import numpy as np

from affordance_vla.plan.category_frame import fit_frame
from affordance_vla.plan.interaction_labels import Grasp
from affordance_vla.plan.rc365_symmetry import (
    Item,
    PlainTreatment,
    RingTreatment,
    RobotFrameTreatment,
    azimuth,
    azimuth_distributions,
    evaluate,
    own_halves,
    robot_frame,
    rotate_grasp,
    wrap,
)
from affordance_vla.plan.transfer_check import PASS_AXIS_DEG, PASS_LANDING
from tests.plan.test_category_transfer import UP, tube


def radial_grasp(radius, height, degrees, at=0.5):
    """Fingers closing along the radius at the given azimuth, centre just inside the wall."""
    a = np.radians(degrees)
    d = np.array([np.cos(a), np.sin(a), 0.0])
    return Grasp(0.9 * radius * d + np.array([0, 0, at * height]), d, -UP)


def item(name, radius, height, degrees, seed, count=3):
    surface = tube(radius, height, seed=seed)
    robot = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees)), 0.0])
    return Item(
        name, surface, fit_frame(surface.points, UP), (radial_grasp(radius, height, degrees),) * count, (robot,) * count
    )


def test_azimuth_is_measured_from_the_handle_direction():
    surface = tube(0.04, 0.09)
    frame = fit_frame(surface.points, UP)
    assert np.allclose(np.degrees(azimuth(frame, np.array([[0, 0.04, 0.05], [-0.04, 0, 0.05]]))), [90, 180], atol=2)


def test_a_rotated_grasp_keeps_its_height_and_distance_from_the_axis():
    frame = fit_frame(tube(0.04, 0.09).points, UP)
    g = radial_grasp(0.04, 0.09, 0)
    turned = rotate_grasp(frame, g, np.pi / 2)
    assert np.allclose(turned.centre[2], g.centre[2]) and np.allclose(turned.closing, [0, 1, 0], atol=1e-2)
    assert np.isclose(np.linalg.norm((turned.centre - frame.origin)[:2]), np.linalg.norm((g.centre - frame.origin)[:2]))


def test_the_robot_frame_points_its_x_axis_at_the_robot():
    frame = fit_frame(tube(0.04, 0.09).points, UP)
    assert np.allclose(robot_frame(frame, np.array([0.0, 1.0, 0.0])).rotation[0], [0, 1, 0], atol=1e-2)


def test_demonstrations_from_the_robot_side_only_transfer_with_the_ring_and_the_robot_frame_but_not_plainly():
    sources = [item("a", 0.04, 0.09, 190, 1), item("b", 0.05, 0.12, 190, 2)]
    target = item("c", 0.035, 0.10, 100, 3)
    plain = evaluate("plain", sources, [target], handle=True)["pooled"]
    ring = evaluate("ring", sources, [target], handle=True)["pooled"]
    robot = evaluate("robot", sources, [target], handle=True)["pooled"]
    assert plain.landing_rate < 0.5
    for result in (ring, robot):
        assert result.landing_rate >= PASS_LANDING and result.axis_error_median_deg < PASS_AXIS_DEG


def test_the_ring_is_cut_to_the_robot_side_so_the_far_side_does_not_land():
    sources = [item("a", 0.04, 0.09, 190, 1), item("b", 0.05, 0.12, 190, 2)]
    far = item("c", 0.035, 0.10, 100, 3)
    far = Item(far.name, far.surface, far.frame, far.grasps, (np.array([-1.0, -1.0, 0.0]),) * len(far.grasps))
    assert evaluate("ring", sources, [far], handle=True)["pooled"].landing_rate < 0.5


def test_a_category_without_a_handle_zone_fills_the_whole_ring():
    model = RingTreatment([item("a", 0.04, 0.09, 190, 1)], handle=False)
    assert isinstance(model, PlainTreatment) and not isinstance(model, RobotFrameTreatment)
    assert model.cmap.grasps == 3 * 12


def test_azimuth_distributions_tell_the_two_frames_apart():
    items = [item("a", 0.04, 0.09, 190, 1), item("b", 0.05, 0.12, 100, 2)]
    stats = azimuth_distributions(items)
    assert stats["grasps"] == 6 and stats["object_frame"]["resultant"] < 0.8
    assert np.isclose(stats["robot_facing_frame"]["resultant"], 1.0, atol=0.05)
    assert abs(wrap(np.pi + 0.1) + np.pi - 0.1) < 1e-9


def test_own_halves_split_the_grasps_alternately():
    a, b = own_halves(item("a", 0.04, 0.09, 0, 1, count=5))
    assert (len(a.grasps), len(b.grasps)) == (3, 2) and len(a.robot) == 3
