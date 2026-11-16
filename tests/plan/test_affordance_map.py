"""An affordance map built from ground truth, and the 3D point such a map stands for."""
import numpy as np
import pytest

from affordance_vla.plan.affordance_map import contact_region_map, map_point

SIGMA = 0.02


def grid_points(height: int = 4, width: int = 5, spacing: float = 0.01) -> np.ndarray:
    """(H, W, 3) points on a horizontal plane at z = 0.9, one per pixel, `spacing` metres apart."""
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([spacing * rows, spacing * columns, np.full(rows.shape, 0.9)], axis=-1).astype(float)


def test_the_map_is_one_at_the_contact_and_falls_off_with_distance():
    # Arrange: the contact is the point of pixel (row 1, column 2); pixel (1, 4) is 2 cm = one sigma away
    points = grid_points()
    mask = np.ones(points.shape[:2], dtype=bool)

    # Act
    heat = contact_region_map(points, mask, contact=points[1, 2], sigma=SIGMA)

    # Assert
    assert heat.shape == (4, 5)
    assert heat[1, 2] == pytest.approx(1.0)
    assert heat[1, 4] == pytest.approx(np.exp(-0.5))
    assert heat[1, 3] > heat[1, 4]


def test_pixels_off_the_object_are_zero():
    points = grid_points()
    mask = np.zeros(points.shape[:2], dtype=bool)
    mask[1, 2] = True

    heat = contact_region_map(points, mask, contact=points[1, 2], sigma=SIGMA)

    assert heat[1, 2] == pytest.approx(1.0)
    assert heat.sum() == pytest.approx(1.0)


def test_a_horizontal_map_ignores_how_far_above_the_surface_the_contact_is():
    # Arrange: a release point 15 cm above the plane, straight over pixel (2, 2)
    points = grid_points()
    mask = np.ones(points.shape[:2], dtype=bool)
    above = points[2, 2] + np.array([0.0, 0.0, 0.15])

    # Act
    flat = contact_region_map(points, mask, contact=above, sigma=SIGMA, horizontal=True)
    full = contact_region_map(points, mask, contact=above, sigma=SIGMA)

    # Assert
    assert flat[2, 2] == pytest.approx(1.0)
    assert full[2, 2] < 1e-6


def test_map_point_is_the_mean_of_the_points_weighted_by_the_map():
    # Arrange: all the weight on two pixels, three times as much on the second
    points = grid_points()
    heat = np.zeros(points.shape[:2])
    heat[0, 0], heat[0, 4] = 0.2, 0.6

    # Act
    point = map_point(points, heat)

    # Assert: three quarters of the way from column 0 to column 4 (4 cm apart)
    np.testing.assert_allclose(point, [0.0, 0.03, 0.9])


def test_an_empty_map_stands_for_no_point():
    points = grid_points()

    assert map_point(points, np.zeros(points.shape[:2])) is None


def test_shapes_must_agree():
    points = grid_points()
    with pytest.raises(ValueError, match="mask"):
        contact_region_map(points, np.ones((3, 3), dtype=bool), contact=np.zeros(3), sigma=SIGMA)
    with pytest.raises(ValueError, match="sigma"):
        contact_region_map(points, np.ones(points.shape[:2], dtype=bool), contact=np.zeros(3), sigma=0.0)
    with pytest.raises(ValueError, match="map"):
        map_point(points, np.zeros((3, 3)))


def test_a_map_with_undefined_weights_stands_for_no_point():
    # Arrange: a pixel without a valid depth gives a weight that is not a number
    points = grid_points()
    heat = np.zeros(points.shape[:2])
    heat[0, 0], heat[1, 1] = 0.5, np.nan

    # Act and assert
    assert map_point(points, heat) is None
