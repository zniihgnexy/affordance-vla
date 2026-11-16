"""The plan predictor: a small readout of the VLM's hidden states."""
import numpy as np
import pytest
import torch

from affordance_vla.heads.plan_predictor import PlanPredictor, plan_inputs, predictor_loss
from affordance_vla.plan.plan_input import no_plan_features, plan_features
from affordance_vla.training.plan_targets import GRASP, NONE, RELEASE

LAYERS, BATCH, TOKENS, WIDTH = 3, 4, 9, 16


def hidden_states(seed: int = 0) -> tuple[list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    states = [torch.randn(BATCH, TOKENS, WIDTH, generator=generator) for _ in range(LAYERS)]
    mask = torch.ones(BATCH, TOKENS, dtype=torch.bool)
    mask[0, 6:] = False  # the first example is shorter
    return states, mask


def test_the_predictor_returns_one_event_contact_and_unit_direction_per_example():
    states, mask = hidden_states()
    model = PlanPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1, queries=2)

    output = model(states, mask)

    assert output["event"].shape == (BATCH, 3)
    assert output["contact"].shape == (BATCH, 3)
    torch.testing.assert_close(output["approach"].norm(dim=-1), torch.ones(BATCH))


def test_padded_tokens_do_not_change_the_prediction():
    # Arrange: two inputs that differ only in the tokens masked out for the first example
    states, mask = hidden_states()
    changed = [layer.clone() for layer in states]
    for layer in changed:
        layer[0, 6:] += 5.0
    model = PlanPredictor(width=WIDTH, layers=LAYERS, hidden=32, heads=4, depth=1, queries=2).eval()

    # Act
    with torch.no_grad():
        first, second = model(states, mask), model(changed, mask)

    # Assert
    torch.testing.assert_close(first["contact"][0], second["contact"][0], atol=1e-5, rtol=0)


def test_the_number_of_layers_must_match():
    states, mask = hidden_states()
    model = PlanPredictor(width=WIDTH, layers=LAYERS + 1, hidden=32, heads=4, depth=1, queries=2)

    with pytest.raises(ValueError, match="layers"):
        model(states, mask)


def test_the_loss_ignores_contact_and_direction_where_there_is_no_event():
    # Arrange: all targets are "no event"; the predicted contact is far off in one copy and exact in the other
    output = {"event": torch.zeros(2, 3), "contact": torch.zeros(2, 3), "approach": torch.tensor([[1.0, 0, 0]] * 2)}
    far = dict(output, contact=torch.full((2, 3), 9.0))
    event = torch.full((2,), NONE)
    contact, approach = torch.zeros(2, 3), torch.zeros(2, 3)

    # Act
    near_loss = predictor_loss(output, event, contact, approach)
    far_loss = predictor_loss(far, event, contact, approach)

    # Assert
    assert near_loss["contact_cm"].item() == 0.0
    torch.testing.assert_close(near_loss["total"], far_loss["total"])


def test_the_loss_reports_the_contact_error_in_centimetres():
    # Arrange: one grasp sample whose predicted contact is 3 cm off along x (contacts are in units of 10 cm)
    output = {"event": torch.tensor([[0.0, 9.0, 0.0]]), "contact": torch.tensor([[1.3, 0.0, 0.5]]),
              "approach": torch.tensor([[0.0, 0.0, -1.0]])}

    # Act
    loss = predictor_loss(output, torch.tensor([GRASP]), torch.tensor([[0.10, 0.0, 0.05]]),
                          torch.tensor([[0.0, 0.0, -1.0]]))

    # Assert
    assert loss["contact_cm"].item() == pytest.approx(3.0, abs=1e-4)
    assert loss["direction"].item() == pytest.approx(0.0, abs=1e-6)


def test_predictions_become_the_plan_input_of_the_action_head():
    # Arrange: example 0 predicts a grasp, example 1 a release, example 2 no event
    output = {"event": torch.tensor([[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]]),
              "contact": torch.tensor([[1.0, 0.0, 0.5], [2.0, 3.0, 1.5], [0.0, 0.0, 0.0]]),
              "approach": torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])}
    ee_pos = np.array([[0.0, 0.0, 0.3]] * 3)

    # Act
    features = plan_inputs(output, ee_pos)

    # Assert
    grasp = plan_features(ee_pos[0], [0.10, 0.0, 0.05], [0.0, 0.0, -1.0], closing=True)
    release = plan_features(ee_pos[1], [0.20, 0.30, 0.15], [0.0, 1.0, 0.0], closing=False)
    np.testing.assert_allclose(features[0], grasp, atol=1e-6)
    np.testing.assert_allclose(features[1], release, atol=1e-6)
    np.testing.assert_allclose(features[2], no_plan_features())
    assert RELEASE == 2
