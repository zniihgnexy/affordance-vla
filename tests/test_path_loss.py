"""Tests for the differentiable plan–action consistency loss on a chunk's end-effector path."""
import numpy as np
import pytest

from affordance3d import plan_consistency

torch = pytest.importorskip("torch")

from affordance3d.path_loss import chunk_path, path_loss  # noqa: E402

SCALE = 0.0125


def test_chunk_path_matches_the_numpy_measure():
    # Arrange
    rng = np.random.default_rng(0)
    start, actions = rng.normal(size=(2, 3)), rng.uniform(-1, 1, size=(2, 8, 3))

    # Act
    path = chunk_path(torch.as_tensor(start), torch.as_tensor(actions), SCALE)

    # Assert: positions after each of the chunk's steps
    for row in range(2):
        expected = plan_consistency.chunk_path(start[row], actions[row], SCALE)[1:]
        np.testing.assert_allclose(path[row].numpy(), expected, atol=1e-12)


def test_path_loss_is_zero_on_the_target_path():
    # Arrange
    target = torch.rand(2, 8, 3)

    # Act
    loss = path_loss(target.clone(), target, torch.ones(2, 8, dtype=torch.bool))

    # Assert
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


def test_path_loss_grows_linearly_in_centimetres_for_large_errors():
    # Arrange: every point is 3 cm off
    target = torch.zeros(1, 4, 3)
    path = target + torch.tensor([0.03, 0.0, 0.0])

    # Act
    loss = path_loss(path, target, torch.ones(1, 4, dtype=torch.bool))

    # Assert: 3 cm minus half of the 0.5 cm quadratic zone
    assert float(loss) == pytest.approx(2.75, abs=1e-5)


def test_path_loss_ignores_steps_that_are_not_valid():
    # Arrange
    target = torch.zeros(1, 4, 3)
    path = torch.tensor([[[0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [0.5, 0.5, 0.5], [0.9, 0.0, 0.0]]], requires_grad=True)
    valid = torch.tensor([[True, True, False, False]])

    # Act
    loss = path_loss(path, target, valid)
    loss.backward()

    # Assert: 1 cm errors on the two valid steps; no gradient reaches the others
    assert float(loss) == pytest.approx(0.75, abs=1e-5)
    assert not path.grad[0, 2:].any() and path.grad[0, :2].any()


def test_path_loss_without_valid_steps_is_zero_and_differentiable():
    # Arrange
    path = torch.rand(2, 8, 3, requires_grad=True)

    # Act
    loss = path_loss(path, torch.zeros(2, 8, 3), torch.zeros(2, 8, dtype=torch.bool))
    loss.backward()

    # Assert
    assert float(loss) == 0.0 and not path.grad.any()


def test_path_loss_sends_gradient_to_the_actions():
    # Arrange: a chunk that stands still while the target moves along x
    actions = torch.zeros(1, 8, 3, requires_grad=True)
    target = torch.cumsum(torch.tensor([SCALE, 0.0, 0.0]).repeat(8, 1), dim=0)[None]

    # Act
    loss = path_loss(chunk_path(torch.zeros(1, 3), actions, SCALE), target, torch.ones(1, 8, dtype=torch.bool))
    loss.backward()

    # Assert: descending the gradient moves the actions along +x
    assert (actions.grad[0, :, 0] < 0).all()
