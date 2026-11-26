"""Pure-Python parts of affordance_vla/sim/robocasa365.py (no simulator: the robocasa365 env is not needed)."""

import numpy as np
import pytest

from affordance_vla.sim import robocasa365 as rc

CUBE_OBJ = """v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""


def _write_object(folder, scale="0.25 0.25 0.25", refquat="1 0 0 0"):
    (folder / "visual").mkdir()
    (folder / "visual" / "cube.obj").write_text(CUBE_OBJ)
    (folder / "model.xml").write_text(
        f'<mujoco><asset><mesh file="visual/cube.obj" name="c_vis" scale="{scale}" refquat="{refquat}"/>'
        '<mesh file="collision/c.obj" name="c_col" scale="1 1 1"/></asset></mujoco>'
    )
    return folder / "model.xml"


def test_psnr_is_infinite_for_identical_images_and_finite_otherwise():
    a = np.full((4, 4, 3), 100, np.uint8)
    assert rc.psnr(a, a) == float("inf")
    assert rc.psnr(a, a + 1) == pytest.approx(20 * np.log10(255.0), abs=1e-6)  # MSE 1


def test_surface_points_lie_on_the_scaled_cube_and_ignore_collision_meshes(tmp_path):
    points = rc.surface_points(_write_object(tmp_path), n=500, seed=1)
    assert points.shape == (500, 3)
    assert np.abs(points).max() == pytest.approx(0.25, abs=1e-9)  # scale 0.25 of the [-1, 1] cube
    assert np.all(np.isclose(np.abs(points), 0.25).any(axis=1))  # every point sits on a face


def test_surface_points_apply_the_reference_rotation(tmp_path):
    # a quarter turn about x, (w, x, y, z) = (cos 45, sin 45, 0, 0), swaps the y and z extents
    model = _write_object(tmp_path, scale="0.5 0.25 0.125", refquat="0.7071067811865476 0.7071067811865476 0 0")
    extent = np.ptp(rc.surface_points(model, n=4000, seed=0), axis=0)
    np.testing.assert_allclose(extent, [1.0, 0.25, 0.5], atol=0.02)  # y and z extents swapped


def test_surface_points_are_reproducible_for_a_seed(tmp_path):
    model = _write_object(tmp_path)
    np.testing.assert_array_equal(rc.surface_points(model, 50, seed=3), rc.surface_points(model, 50, seed=3))


def test_dataset_dir_names_the_missing_task(tmp_path):
    with pytest.raises(FileNotFoundError, match="OpenDrawer"):
        rc.dataset_dir(tmp_path, "OpenDrawer")


def test_dataset_dir_finds_the_newest_bundle(tmp_path):
    for date in ("20250801", "20250816"):
        (tmp_path / "v1.0/target/atomic/OpenDrawer" / date / "lerobot").mkdir(parents=True)
    assert rc.dataset_dir(tmp_path, "OpenDrawer").parent.name == "20250816"
