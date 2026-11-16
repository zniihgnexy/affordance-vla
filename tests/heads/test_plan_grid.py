"""The grid plan predictor: which image cell shows the contact, where in the cell, and how deep."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from affordance_vla.heads.plan_grid import (
    CELLS,
    IMAGE_SIZE,
    MIN_DEPTH_M,
    NO_CELL,
    Camera,
    PlanGridPredictor,
    camera_of_log,
    contact_pixel,
    grid_contacts,
    grid_loss,
    grid_target,
    image_token_cells,
    pixel_contact,
    shifted_image,
)
from affordance_vla.training.plan_targets import GRASP, NONE

K = np.array([[300.0, 0.0, 128.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]])
LAYERS, BATCH, TOKENS, WIDTH = 2, 3, 80, 16


def camera() -> Camera:
    """A camera 1.2 m above the origin looking straight down, with 256 x 256 logged images."""
    pose = np.eye(4)
    pose[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    pose[:3, 3] = [0.0, 0.0, 1.2]
    return Camera(intrinsics=K, cam_to_world=pose, logged_size=256)


def test_a_contact_goes_to_a_pixel_of_the_small_image_and_back():
    point = np.array([0.10, -0.05, 0.08])

    u, v, depth = contact_pixel(point, camera())
    back = pixel_contact(u, v, depth, camera())

    assert 0 <= u < IMAGE_SIZE and 0 <= v < IMAGE_SIZE
    assert depth == pytest.approx(1.12)
    np.testing.assert_allclose(back, point, atol=1e-9)


def test_a_pixel_is_split_into_its_cell_and_the_position_inside_it():
    # pixel (u, v) = (70, 35): cells are 28 px wide, so column 2, row 1, half and a quarter of a cell in
    cell, offset = grid_target(70.0, 35.0)

    assert cell == 1 * CELLS + 2
    np.testing.assert_allclose(offset, [0.5, 0.25])


def test_a_pixel_outside_the_image_has_no_cell():
    assert grid_target(-3.0, 10.0) is None
    assert grid_target(10.0, float(IMAGE_SIZE)) is None


def test_shifting_an_image_moves_its_content_and_fills_the_rest_with_zeros():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    moved = shifted_image(image, dx=2, dy=-1)

    np.testing.assert_array_equal(moved[0, 2:], image[1, :3])  # content moved 2 px right and 1 px up
    assert not moved[:, :2].any() and not moved[3].any()
    np.testing.assert_array_equal(shifted_image(image, 0, 0), image)


def hidden_states() -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    states = [torch.randn(BATCH, TOKENS, WIDTH, generator=generator) for _ in range(LAYERS)]
    mask = torch.ones(BATCH, TOKENS, dtype=torch.bool)
    cells = torch.arange(3, 3 + CELLS * CELLS).unsqueeze(0).expand(BATCH, -1)  # where the view's tokens sit
    return states, mask, cells


def test_the_predictor_scores_every_cell_and_gives_offsets_inside_the_cell():
    states, mask, cells = hidden_states()
    model = PlanGridPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1)

    output = model(states, mask, cells)

    assert output["cell"].shape == (BATCH, CELLS * CELLS)
    assert output["offset"].shape == (BATCH, CELLS * CELLS, 2)
    assert output["depth"].shape == (BATCH, CELLS * CELLS)
    assert output["event"].shape == (BATCH, 3)
    assert ((output["offset"] >= 0) & (output["offset"] <= 1)).all()
    torch.testing.assert_close(output["approach"].norm(dim=-1), torch.ones(BATCH))


def test_predictions_decode_to_the_contact_they_were_made_from():
    # Arrange: a contact, its cell target, and an output that names that cell with the right offset and depth
    point = np.array([0.10, -0.05, 0.08])
    u, v, depth = contact_pixel(point, camera())
    cell, offset = grid_target(u, v)
    output = {"cell": torch.full((1, CELLS * CELLS), -5.0), "offset": torch.zeros(1, CELLS * CELLS, 2),
              "depth": torch.zeros(1, CELLS * CELLS)}
    output["cell"][0, cell] = 5.0
    output["offset"][0, cell] = torch.tensor(offset, dtype=torch.float32)
    output["depth"][0, cell] = (depth - 1.0) / 0.1

    # Act
    contacts = grid_contacts(output, camera())

    # Assert
    np.testing.assert_allclose(contacts[0], point, atol=1e-5)


def test_the_loss_ignores_the_cell_where_there_is_no_event():
    # Arrange: the same output scored against "no event" targets with different (unused) cells
    states, mask, cells = hidden_states()
    output = PlanGridPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1)(states, mask, cells)
    event = torch.full((BATCH,), NONE)
    zeros, unit = torch.zeros(BATCH), torch.zeros(BATCH, 3)

    # Act
    first = grid_loss(output, event, torch.zeros(BATCH, dtype=torch.long), torch.zeros(BATCH, 2), zeros, unit)
    second = grid_loss(output, event, torch.full((BATCH,), 7), torch.ones(BATCH, 2), zeros + 3, unit)

    # Assert
    torch.testing.assert_close(first["total"], second["total"])


def test_an_event_whose_contact_is_off_the_image_trains_the_event_and_direction_but_no_cell():
    # Arrange: grasps ahead, but no cell shows their contact
    states, mask, cells = hidden_states()
    output = PlanGridPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1)(states, mask, cells)
    event, nowhere = torch.full((BATCH,), GRASP), torch.full((BATCH,), NO_CELL)
    approach = torch.tensor([[0.0, 0.0, -1.0]] * BATCH)

    # Act
    loss = grid_loss(output, event, nowhere, torch.zeros(BATCH, 2), torch.zeros(BATCH), approach)

    # Assert: the event is still scored as a grasp, the direction counts, the cell terms do not
    expected = torch.nn.functional.cross_entropy(output["event"], event)
    torch.testing.assert_close(loss["event"], expected)
    assert float(loss["cell"]) == 0.0 and float(loss["offset"]) == 0.0 and float(loss["depth"]) == 0.0
    assert float(loss["direction"]) > 0.0
    torch.testing.assert_close(loss["total"], expected + 0.5 * loss["direction"])


def test_a_predicted_depth_behind_the_camera_is_moved_in_front_of_it():
    # Arrange: the best cell asks for a depth of 1 m - 5 m, which lies behind the camera
    output = {"cell": torch.zeros(1, CELLS * CELLS), "offset": torch.full((1, CELLS * CELLS, 2), 0.5),
              "depth": torch.full((1, CELLS * CELLS), -50.0)}

    # Act
    contact = grid_contacts(output, camera())[0]

    # Assert: the camera looks down from 1.2 m, so the contact sits MIN_DEPTH_M below it
    assert contact[2] == pytest.approx(1.2 - MIN_DEPTH_M)


def test_the_camera_takes_its_size_from_the_logged_images():
    log = SimpleNamespace(agentview=np.zeros((2, 256, 256, 3), dtype=np.uint8), intrinsics=K, cam_to_world=np.eye(4))

    found = camera_of_log(log)

    assert found.logged_size == 256
    np.testing.assert_array_equal(found.intrinsics, K)


def test_a_log_without_images_or_with_an_off_centre_camera_gives_no_camera():
    off_centre = K.copy()
    off_centre[0, 2] = 100.0
    images = np.zeros((2, 256, 256, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="images"):
        camera_of_log(SimpleNamespace(agentview=None, intrinsics=K, cam_to_world=np.eye(4)))
    with pytest.raises(ValueError, match="centre"):
        camera_of_log(SimpleNamespace(agentview=images, intrinsics=off_centre, cam_to_world=np.eye(4)))
    with pytest.raises(ValueError, match="square"):
        camera_of_log(SimpleNamespace(agentview=images[:, :200], intrinsics=K, cam_to_world=np.eye(4)))


def test_the_loss_falls_when_the_right_cell_is_scored_higher():
    states, mask, cells = hidden_states()
    output = PlanGridPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1)(states, mask, cells)
    event, cell = torch.full((BATCH,), GRASP), torch.full((BATCH,), 5)
    better = dict(output, cell=output["cell"].detach().clone())
    better["cell"][:, 5] += 4.0
    args = (event, cell, torch.full((BATCH, 2), 0.5), torch.zeros(BATCH), torch.tensor([[0.0, 0.0, -1.0]] * BATCH))

    assert grid_loss(better, *args)["cell"] < grid_loss(output, *args)["cell"]


def test_the_first_view_owns_the_first_block_of_image_tokens():
    # Arrange: two sequences with different padding; token 9 marks an image token, two views of 64 each
    image, count = 9, CELLS * CELLS
    first = torch.tensor([1, 2] + [image] * count + [3] + [image] * count + [4, 5, 0, 0])
    second = torch.tensor([0, 0, 1, 2] + [image] * count + [3] + [image] * count + [4, 5])

    # Act
    cells = image_token_cells(torch.stack([first, second]), image)

    # Assert
    assert cells.shape == (2, count)
    assert cells[0].tolist() == list(range(2, 2 + count))
    assert cells[1].tolist() == list(range(4, 4 + count))


def test_a_sequence_with_the_wrong_number_of_image_tokens_is_rejected():
    ids = torch.tensor([[1] + [9] * 10 + [2]])

    with pytest.raises(ValueError, match="image tokens"):
        image_token_cells(ids, 9)
