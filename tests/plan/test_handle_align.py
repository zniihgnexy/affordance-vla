from dataclasses import replace

import numpy as np

from affordance_vla.plan import handle_align as align
from affordance_vla.plan.category_frame import CategoryFrame
from affordance_vla.plan.rc365_symmetry import Item
from affordance_vla.sim.object_surface import Surface


def _item(reach: float, grasps=()):
    """A unit cylinder body with a handle bar along +x that reaches `reach` beyond x = HANDLE_X0."""
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, 400)
    body = np.stack([np.cos(angle), np.sin(angle), rng.uniform(0, 1, 400)], axis=1)
    bar = np.stack([align.HANDLE_X0 + np.linspace(0, reach, 50), np.zeros(50), np.full(50, 0.5)], axis=1)
    points = np.concatenate([body, bar])
    frame = CategoryFrame(np.zeros(3), np.eye(3), np.ones(3))
    return Item("m", Surface(points, np.tile([0.0, 0.0, 1.0], (len(points), 1))), frame, tuple(grasps), ())


def test_handle_reach_is_the_extent_beyond_the_body():
    item = _item(0.9)
    assert np.isclose(align.handle_reach(item.surface.points), 0.9)


def test_length_warp_scales_the_handle_and_leaves_the_body():
    item = _item(0.9)
    aligned = align.align_length([item], reference=0.45)[0]
    mapped = aligned.frame.point_to_category(item.surface.points)
    assert np.isclose(align.handle_reach(mapped), 0.45)
    body = item.surface.points[:, 0] <= align.HANDLE_X0
    assert np.allclose(mapped[body], item.surface.points[body])


def test_alignment_reads_surface_points_only_never_the_grasps():
    plain, with_grasps = _item(0.9), _item(0.9, grasps=("anything",))
    a = align.align_length([plain], reference=0.5)[0].frame.point_to_category(plain.surface.points)
    b = align.align_length([with_grasps], reference=0.5)[0].frame.point_to_category(plain.surface.points)
    assert np.array_equal(a, b)


def test_default_reference_is_the_median_reach_so_the_median_instance_is_unchanged():
    items = [_item(0.6), _item(0.8), _item(1.2)]
    aligned = align.align_length(items)
    reaches = [align.handle_reach(i.frame.point_to_category(i.surface.points)) for i in aligned]
    assert np.allclose(reaches, 0.8)


def test_registered_warp_moves_a_stretched_handle_onto_the_template():
    template = _item(0.6).surface.points
    handle = template[align.handle_mask(template)]
    stretched = handle * np.array([1.0, 1.0, 1.0]) + np.array([0.0, 0.0, 0.0])
    stretched[:, 0] = align.HANDLE_X0 + (stretched[:, 0] - align.HANDLE_X0) * 1.4
    warp = align.registered_warp(stretched, handle)
    mapped = warp(stretched)
    assert np.abs(mapped[:, 0].max() - handle[:, 0].max()) < np.abs(stretched[:, 0].max() - handle[:, 0].max())


def test_registered_warp_without_a_handle_is_the_identity():
    points = np.array([[0.2, 0.1, 0.3], [0.5, 0.5, 0.5]])
    assert np.array_equal(align.registered_warp(np.zeros((0, 3)), points)(points), points)


def test_aligned_frame_keeps_the_axes_of_the_plain_frame():
    item = replace(_item(0.9))
    aligned = align.with_warp(item, align.length_warp(0.9, 0.45))
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(aligned.frame.axis_to_category(axes), item.frame.axis_to_category(axes))
