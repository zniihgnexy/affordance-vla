"""Projection of world points into LIBERO camera images."""
import numpy as np
import pytest

from affordance_vla.sim.libero_camera import project_points, unproject_depth

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])


def looking_down_from(height: float) -> np.ndarray:
    """Camera-to-world for a camera at (0, 0, height) looking straight down: x right, y down the image, z forward."""
    pose = np.eye(4)
    pose[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    pose[:3, 3] = [0.0, 0.0, height]
    return pose


def test_point_on_the_optical_axis_lands_on_the_principal_point():
    uv, depth = project_points(np.array([[0.0, 0.0, 0.0]]), K, looking_down_from(2.0))

    np.testing.assert_allclose(uv, [[32.0, 24.0]])
    np.testing.assert_allclose(depth, [2.0])


def test_offsets_scale_with_focal_length_over_depth():
    # Arrange: 0.2 m to the camera's right and 0.1 m "up" in the world = up in the image (smaller v)
    points = np.array([[0.2, 0.1, 0.0]])

    # Act
    uv, _ = project_points(points, K, looking_down_from(2.0))

    # Assert: 100 px · 0.2 / 2 = 10 px right, 100 px · 0.1 / 2 = 5 px up
    np.testing.assert_allclose(uv, [[42.0, 19.0]])


def test_mirrored_images_flip_the_column_only():
    points = np.array([[0.2, 0.1, 0.0]])

    uv, _ = project_points(points, K, looking_down_from(2.0), width=64, mirrored=True)

    np.testing.assert_allclose(uv, [[63.0 - 42.0, 19.0]])


def test_points_behind_the_camera_get_nan_pixels():
    uv, depth = project_points(np.array([[0.0, 0.0, 3.0]]), K, looking_down_from(2.0))

    assert depth[0] < 0
    assert np.isnan(uv).all()


def test_mirroring_needs_the_image_width():
    with pytest.raises(ValueError, match="width"):
        project_points(np.zeros((1, 3)), K, looking_down_from(2.0), mirrored=True)


def test_a_rotated_and_shifted_camera_pose_is_inverted_before_projecting():
    # Arrange: camera at (1, 2, 0) looking along world +x; image right = world −y, image down = world −z.
    # Unlike the top-down pose above, this pose is not its own inverse.
    pose = np.eye(4)
    pose[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    pose[:3, 3] = [1.0, 2.0, 0.0]
    point = np.array([[3.0, 1.8, 0.1]])  # 2 m ahead, 0.2 m to the right, 0.1 m up

    # Act
    uv, depth = project_points(point, K, pose)

    # Assert
    np.testing.assert_allclose(depth, [2.0])
    np.testing.assert_allclose(uv, [[32.0 + 100 * 0.2 / 2, 24.0 - 100 * 0.1 / 2]])


@pytest.mark.parametrize("mirrored", [False, True])
def test_unprojecting_a_depth_image_and_projecting_it_back_gives_the_pixel_grid(mirrored):
    # Arrange: a camera that is not axis-aligned, and a depth image with a different depth in every pixel
    pose = np.eye(4)
    pose[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    pose[:3, 3] = [1.0, 2.0, 0.5]
    height, width = 48, 64
    depth = 1.0 + 0.01 * np.arange(height * width, dtype=float).reshape(height, width)

    # Act
    points = unproject_depth(depth, K, pose, mirrored=mirrored)
    uv, back = project_points(points.reshape(-1, 3), K, pose, width=width, mirrored=mirrored)

    # Assert: pixel (u, v) = (column, row) and the same depth along the optical axis
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    assert points.shape == (height, width, 3)
    np.testing.assert_allclose(uv[:, 0].reshape(height, width), columns, atol=1e-9)
    np.testing.assert_allclose(uv[:, 1].reshape(height, width), rows, atol=1e-9)
    np.testing.assert_allclose(back.reshape(height, width), depth)


def test_the_centre_pixel_of_a_camera_looking_down_unprojects_to_the_point_below_it():
    depth = np.full((49, 65), 2.0)  # principal point (32, 24) is a pixel of this image

    points = unproject_depth(depth, K, looking_down_from(2.0))

    np.testing.assert_allclose(points[24, 32], [0.0, 0.0, 0.0], atol=1e-12)


def test_unproject_depth_needs_a_two_dimensional_image():
    with pytest.raises(ValueError, match="depth"):
        unproject_depth(np.ones((4, 4, 1)), K, looking_down_from(2.0))


def test_in_a_mirrored_image_a_point_to_the_right_of_the_camera_sits_left_of_the_centre():
    # Arrange: the world point (0.2, 0, 0) projects to column 42 of the upright image (see above),
    # which is column 64 - 42 = 22 of the mirrored image of width 65
    depth = np.full((49, 65), 2.0)

    # Act
    points = unproject_depth(depth, K, looking_down_from(2.0), mirrored=True)

    # Assert
    np.testing.assert_allclose(points[24, 22], [0.2, 0.0, 0.0], atol=1e-12)
