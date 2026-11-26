import numpy as np

from affordance_vla.plan.rc365_events import PartTouch
from affordance_vla.plan.rc365_parts import axial_mean, leave_one_layout_out, part_report


def touch(closing, motion, layout_seed=0):
    rng = np.random.default_rng(layout_seed)
    closing = np.asarray(closing, float)
    world = {
        "closing": rng.normal(size=3),
        "closing_base": rng.normal(size=3),
        "motion": rng.normal(size=3),
        "motion_base": rng.normal(size=3),
    }
    world = {k: v / np.linalg.norm(v) for k, v in world.items()}
    return PartTouch(
        "T",
        "pretrain",
        0,
        "x_handle_main",
        "x_handle",
        np.array([0.0, 0.01, 0.0]),
        closing,
        np.array([0, 0, -1.0]),
        True,
        np.asarray(motion, float),
        0.1,
        "slide",
        np.array([0, 1.0, 0]),
        1.0,
        world,
    )


def test_the_axial_mean_ignores_the_sign_of_each_line():
    lines = np.array([[1, 0, 0], [-1, 0, 0], [0.99, 0.1, 0] / np.linalg.norm([0.99, 0.1, 0])])
    assert abs(axial_mean(lines)[0]) > 0.99


def test_every_vector_is_compared_with_the_mean_of_the_other_layouts():
    vectors = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    angles = leave_one_layout_out(vectors, [0, 0, 1], axial=False)
    assert np.allclose(angles, [90, 90, 90])
    assert np.isnan(leave_one_layout_out(vectors, [0, 0, 0], axial=False)).all()


def test_consistent_part_frame_directions_beat_random_world_directions():
    touches = [touch([1, 0, 0], [0, 1, 0], k) for k in range(12)]
    report = part_report(touches, [k % 4 for k in range(12)])
    assert report["closing_part_frame"]["within_tolerance"] == 1.0 and report["motion_part_frame"]["median_deg"] < 1
    assert report["motion_world_frame"]["median_deg"] > 20 and report["kitchens"] == 4
    assert report["contact_part_frame"]["scatter_cm"] < 1e-6 and report["joint_types"]["slide"] == 12
