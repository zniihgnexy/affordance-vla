"""Projection of world points into LIBERO camera images."""
import numpy as np
import pytest

from affordance3d.libero_camera import project_points

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
